#!/usr/bin/env python3
"""Add frame_clock.json to analysis results that predate it, without rerunning OCR.

run_pipeline.py writes frame_clock.json itself. This tool decodes a video once
(the same OpenCV sequence the pipeline uses, so the frame indices match the OCR
jsonl) and writes the clock into the video's existing results folder:

    python tools/frame_clock.py --video D:/batch --jobs 4          # results beside each video
    python tools/frame_clock.py --video data/0816 --output-dir data  # results under data/<stem>/

Without --output-dir the results folder is the one next to the video with the
video's name (the layout drive_batch.py collects into). With --output-dir it is
DIR/<stem>/, as run_pipeline.py writes it. A video whose results folder does not
exist is skipped: run run_pipeline.py for it instead. An existing
frame_clock.json is kept unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import frame_clock  # noqa: E402
from utils.pipeline_utils import duplicate_stems, find_videos  # noqa: E402


def results_dir(video: Path, output_dir: Optional[Path]) -> Path:
    """The folder holding this video's results.

    None means the folder beside the video; otherwise the run_pipeline.py rule.
    """
    if output_dir is None:
        return video.with_suffix("")
    return output_dir if output_dir.name == video.stem else output_dir / video.stem


def frame_count_mismatch(folder: Path, clock: dict) -> Optional[str]:
    """Compare the clock with run_stats.json from the OCR run, when it exists.

    run_stats.json "frames" counts the frames the pipeline decoded. A different
    count means this decode is not the one the OCR indices came from (another
    OpenCV build, a replaced video file), so the clock must not be trusted.
    """
    stats_path = folder / "run_stats.json"
    if not stats_path.exists():
        return None
    try:
        frames = json.loads(stats_path.read_text(encoding="utf-8")).get("frames")
    except (OSError, ValueError):
        return None
    if isinstance(frames, int) and frames != clock["decoded_frames"]:
        return (
            f"run_stats.json reports {frames} decoded frames, this decode found "
            f"{clock['decoded_frames']}"
        )
    return None


def process(video: Path, output_dir: Optional[Path], force: bool) -> Tuple[str, str]:
    """Write one clock. Returns (status, message); status is ok/skip/fail."""
    folder = results_dir(video, output_dir)
    target = folder / frame_clock.FILE_NAME
    if not folder.is_dir():
        return (
            "skip",
            f"{video.name}: no results folder {folder}; run run_pipeline.py first",
        )
    if target.exists() and not force:
        return "skip", f"{video.name}: {target} exists (use --force to rewrite)"
    try:
        frames, container_frames = frame_clock.scan_video(video)
        clock = frame_clock.build_clock(video.name, frames, container_frames)
    except (OSError, ValueError) as e:
        return "fail", f"{video.name}: {e}"
    mismatch = frame_count_mismatch(folder, clock)
    if mismatch:
        return "fail", f"{video.name}: not written, {mismatch}"
    frame_clock.write_clock(target, clock)
    breaks = len(clock["segments"]) - 1
    return "ok", (
        f"{video.name}: decoded={clock['decoded_frames']} "
        f"dropped={clock['dropped_frames']} breaks={breaks} -> {target}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="video file, or a folder searched recursively for .mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="results root used for run_pipeline.py "
        "(default: the folder beside each video, named after it)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rewrite existing frame_clock.json"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="videos decoded in parallel (default: 1)"
    )
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"path not found: {args.video}", file=sys.stderr)
        return 2
    videos = find_videos(args.video)
    if not videos:
        print(f"no .mp4 files under {args.video}", file=sys.stderr)
        return 2
    duplicates = duplicate_stems(videos)
    if duplicates:
        for stem, paths in duplicates.items():
            print(
                f"same file name, results would collide: {stem}: "
                + ", ".join(str(p) for p in paths),
                file=sys.stderr,
            )
        return 2

    failed = 0
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for status, message in pool.map(
            lambda v: process(v, args.output_dir, args.force), videos
        ):
            print(f"[{status}] {message}", flush=True)
            failed += status == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
