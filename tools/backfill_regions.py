#!/usr/bin/env python3
"""Add newly configured pattern regions (AS, STATUS) to finished analyses.

OCR is not rerun. Each video is decoded once, the same way run_pipeline.py
decodes it, and in that pass the tool

  * confirms the machine type (config/machine_identity.json anchors),
  * matches only the requested pattern regions for that machine,
  * rebuilds frame_clock.json.

The regions are added to the video's stage_analysis.json together with its
"machine_id"; existing regions (STAGE, PEDAL) are left as they are.

    python tools/backfill_regions.py --video D:/batch                 # one video at a time
    python tools/backfill_regions.py --video D:/batch --when-idle      # only while the PC is idle

--when-idle runs each video in a child process at idle priority, starts it only
after the rest of the machine has been quiet for a while, and kills it as soon
as other programs need the CPU; that video is retried once the machine is idle
again. Finished videos are skipped, so the command can be stopped and rerun at
any time. Create the file named by --stop-file to make it exit cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import frame_clock  # noqa: E402
from utils.pipeline_utils import duplicate_stems, find_videos  # noqa: E402

DEFAULT_REGIONS = ("AS", "STATUS")
MACHINE_IDS = (1, 2, 3)


def results_dir(video: Path, output_dir: Optional[Path]) -> Path:
    """Results folder: beside the video by default, else the run_pipeline.py rule."""
    if output_dir is None:
        return video.with_suffix("")
    return output_dir if output_dir.name == video.stem else output_dir / video.stem


def machine_regions(machine_id, regions) -> List[str]:
    """The requested regions this machine displays, per its pattern route config."""
    from utils.get_configs import (
        get_machine_key,
        read_surgery_stage_rois,
        resolve_machine_config_path,
    )

    path = REPO / resolve_machine_config_path(
        "surgery_stage_rois.json", get_machine_key(machine_id)
    )
    return [r for r in read_surgery_stage_rois(path, machine_id) if r in regions]


def is_done(folder: Path, regions) -> bool:
    """True when stage_analysis.json already carries the machine and its regions.

    A machine that does not display a region (AS on machine 1) is done without it.
    """
    try:
        stage = json.loads((folder / "stage_analysis.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    machine_id = stage.get("machine_id")
    if machine_id is None or not (folder / frame_clock.FILE_NAME).exists():
        return False
    return all(
        r in stage.get("regions", {}) for r in machine_regions(machine_id, regions)
    )


def known_machine(video: Path, stage: Dict) -> Optional[int]:
    """Machine id already recorded for this video, if any."""
    if stage.get("machine_id") is not None:
        return int(stage["machine_id"])
    try:
        rois = json.loads((REPO / "config" / "rois.json").read_text(encoding="utf-8"))
        mapped = rois.get("video_machine_mapping", {}).get(video.stem)
        return int(mapped) if mapped is not None else None
    except (OSError, ValueError, TypeError):
        return None


def analyse(video: Path, regions, machine_id: Optional[int]):
    """Decode once. Returns (machine_id, {region: matches}, frame times, container frames)."""
    import cv2
    from auto_detect_machine_type import MachineDetector
    from stage_pattern_analysis import StageAnalyzer
    from utils.get_configs import (
        get_machine_key,
        get_pattern_cache_root,
        resolve_machine_config_path,
    )
    from utils.pipeline_utils import iter_video_frames

    # Match the regions for every candidate layout while the machine is still
    # unknown (its anchor can appear minutes into the video), then keep the one
    # the detector confirms. The regions are small, so this costs little next to
    # decoding.
    candidates = [machine_id] if machine_id is not None else list(MACHINE_IDS)
    analyzers = {}
    for mid in candidates:
        key = get_machine_key(mid)
        analyzers[mid] = StageAnalyzer(
            REPO / resolve_machine_config_path("surgery_stage_rois.json", key),
            REPO / get_pattern_cache_root(key),
            diff_rules_path=REPO / resolve_machine_config_path("diff_rule.json", key),
            machine_id=mid,
            regions=list(regions),
        )
    matches = {mid: {r: [] for r in a.roi_dict} for mid, a in analyzers.items()}
    detector = (
        MachineDetector(REPO / "config" / "machine_identity.json")
        if machine_id is None
        else None
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video}")
    times: List[Tuple[int, float]] = []
    try:
        container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        for index, frame, ms in iter_video_frames(cap):
            times.append((index, ms / 1000.0))
            if detector is not None and machine_id is None:
                found = detector.update(frame)
                if found is not None:
                    machine_id = int(found)
                    analyzers = {machine_id: analyzers[machine_id]}
                    matches = {machine_id: matches[machine_id]}
            for mid, analyzer in analyzers.items():
                for region, res in analyzer.process_frame(frame, index).items():
                    matches[mid][region].append((index, res["pattern_id"], res["rmse"]))
    finally:
        cap.release()
    if machine_id is None:
        raise ValueError(
            "machine type not confirmed; add the video to rois.json video_machine_mapping"
        )
    return machine_id, matches[machine_id], times, container_frames


def process(
    video: Path, output_dir: Optional[Path], regions, force: bool = False
) -> Tuple[str, str]:
    """Backfill one video. Returns (status, message); status is ok/skip/fail."""
    from stage_pattern_analysis import region_segments
    from utils.get_configs import (
        get_machine_key,
        load_pattern_name_mapping,
        resolve_machine_config_path,
    )

    folder = results_dir(video, output_dir)
    stage_path = folder / "stage_analysis.json"
    if not stage_path.exists():
        return "skip", f"{video.name}: no {stage_path}; run run_pipeline.py first"
    if not force and is_done(folder, regions):
        return "skip", f"{video.name}: already has {', '.join(regions)}"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))

    try:
        machine_id, matches, times, container_frames = analyse(
            video, regions, known_machine(video, stage)
        )
        clock = frame_clock.build_clock(video.name, times, container_frames)
    except (OSError, ValueError) as e:
        return "fail", f"{video.name}: {e}"

    # The OCR indices came from the run that wrote run_stats.json; a different
    # decoded count means a different decode, and the new regions would not line up.
    stats_path = folder / "run_stats.json"
    if stats_path.exists():
        frames = json.loads(stats_path.read_text(encoding="utf-8")).get("frames")
        if isinstance(frames, int) and frames != clock["decoded_frames"]:
            return "fail", (
                f"{video.name}: run_stats.json reports {frames} decoded frames, "
                f"this decode found {clock['decoded_frames']}; nothing written"
            )

    names = load_pattern_name_mapping(
        REPO
        / resolve_machine_config_path(
            "pattern_name_mapping.json", get_machine_key(machine_id)
        )
    )
    stage["machine_id"] = machine_id
    stage.setdefault("regions", {})
    for region, region_matches in matches.items():
        stage["regions"][region] = region_segments(
            region_matches, names.get(region, {})
        )

    temp = stage_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, stage_path)
    frame_clock.write_clock(folder / frame_clock.FILE_NAME, clock)

    summary = (
        ", ".join(f"{r}={len(stage['regions'][r])} segments" for r in matches)
        or "no such regions"
    )
    return (
        "ok",
        f"{video.name}: machine {machine_id}, {summary}, clock breaks={len(clock['segments']) - 1}",
    )


# --------------------------------------------------------------------------- idle runner


def lower_own_priority() -> None:
    """Run this process at idle priority so any interactive program wins."""
    try:
        import psutil

        me = psutil.Process()
        me.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass


def run_when_idle(videos: List[Path], args, log) -> int:
    import psutil
    from utils.idle_gate import START, STOP, IdleGate, others_percent

    gate = IdleGate(
        start_below=args.start_below,
        stop_above=args.stop_above,
        settle_samples=max(1, round(args.settle / args.sample)),
        stop_samples=2,
    )
    cpus = psutil.cpu_count() or 1
    pending = [
        v
        for v in videos
        if not is_done(results_dir(v, args.output_dir), args.regions)
        and (results_dir(v, args.output_dir) / "stage_analysis.json").exists()
    ]
    log(f"{len(pending)} of {len(videos)} videos to backfill; waiting for an idle CPU")
    failed = 0
    child = None
    current = None
    psutil.cpu_percent(None)
    while pending or child is not None:
        if args.stop_file and Path(args.stop_file).exists():
            log("stop file found; exiting")
            break
        time.sleep(args.sample)
        system = psutil.cpu_percent(None)
        job = 0.0
        if child is not None:
            try:
                job = child.cpu_percent(None)
            except psutil.Error:
                job = 0.0
        decision = gate.observe(
            others_percent(system, job, cpus), running=child is not None
        )

        if child is not None and child.poll() is not None:
            code = child.returncode
            log(f"{'done' if code == 0 else 'FAILED'}: {current.name} (exit {code})")
            failed += code != 0
            pending.remove(current)
            child, current = None, None
            continue
        if child is not None and decision == STOP:
            log(
                f"CPU busy ({system:.0f}% total); stopping {current.name}, will retry when idle"
            )
            child.kill()
            child.wait()
            child, current = None, None
            continue
        if child is None and decision == START and pending:
            current = pending[0]
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                "--video",
                str(current),
                "--regions",
                *args.regions,
                "--child",
            ]
            if args.output_dir is not None:
                cmd += ["--output-dir", str(args.output_dir)]
            log(f"idle; starting {current.name}")
            child = psutil.Popen(cmd, cwd=str(REPO))
            child.cpu_percent(None)
    if child is not None:
        child.kill()
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add new pattern regions to finished analyses without rerunning OCR."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="video file, or folder searched recursively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="results root used with run_pipeline.py (default: folder beside each video)",
    )
    parser.add_argument(
        "--regions", nargs="+", default=list(DEFAULT_REGIONS), help="regions to add"
    )
    parser.add_argument(
        "--force", action="store_true", help="redo videos that already have the regions"
    )
    parser.add_argument(
        "--when-idle",
        action="store_true",
        help="run only while the rest of the PC is idle",
    )
    parser.add_argument(
        "--start-below",
        type=float,
        default=25.0,
        help="idle: start when others use < this %% CPU",
    )
    parser.add_argument(
        "--stop-above",
        type=float,
        default=40.0,
        help="idle: stop when others use > this %% CPU",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=60.0,
        help="idle: seconds of quiet before starting",
    )
    parser.add_argument(
        "--sample", type=float, default=5.0, help="idle: seconds between CPU samples"
    )
    parser.add_argument(
        "--stop-file", type=Path, default=None, help="idle: exit when this file exists"
    )
    parser.add_argument(
        "--log", type=Path, default=None, help="also append progress to this file"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    if not args.video.exists():
        print(f"path not found: {args.video}", file=sys.stderr)
        return 2
    # Config and template paths inside the pipeline are relative to the repo.
    args.video = args.video.resolve()
    args.output_dir = args.output_dir.resolve() if args.output_dir else None
    args.stop_file = args.stop_file.resolve() if args.stop_file else None
    args.log = args.log.resolve() if args.log else None
    os.chdir(REPO)
    videos = find_videos(args.video)
    if not videos:
        print(f"no .mp4 files under {args.video}", file=sys.stderr)
        return 2
    if duplicate_stems(videos) and args.output_dir is not None:
        print(
            "videos share a file name and would share a results folder", file=sys.stderr
        )
        return 2

    if args.when_idle:
        return run_when_idle(videos, args, log)

    if args.child:
        lower_own_priority()
    failed = 0
    for video in videos:
        status, message = process(video, args.output_dir, args.regions, args.force)
        log(f"[{status}] {message}")
        failed += status == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
