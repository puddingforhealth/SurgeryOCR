"""Resolve a viewer's machine from the analysis it displays, without editing mappings."""

import json
from pathlib import Path
import re

from utils.get_configs import get_machine_key, get_video_machine_id


def analysis_machine(analysis_dir, video_name, roi_file):
    """Prefer result metadata, then legacy pipeline evidence, then local configuration."""
    root = Path(analysis_dir)
    for name in ("analysis_layout.json", "_source.json", "stage_analysis.json"):
        source = root / name
        if source.is_file():
            machine = json.loads(source.read_text(encoding="utf-8")).get("machine_id")
            if machine is not None:
                if str(machine) not in ("1", "2", "3"):
                    raise ValueError("Unsupported analysis machine_id")
                return int(machine), "分析結果"
    log = root / "pipeline.log"
    if log.is_file():
        machines = set()
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(
                marker in line
                for marker in ("偵測機型:", "檢測到已知設定:", "使用預設 Type")
            ):
                match = re.search(r"\bType\s+([123])\b", line)
                if match:
                    machines.add(int(match.group(1)))
        if len(machines) > 1:
            raise ValueError("Conflicting machine IDs in analysis log")
        if machines:
            return machines.pop(), "分析日誌"
    machine = get_video_machine_id(video_name, Path(roi_file))
    if machine is not None:
        return int(machine), "本機設定"
    return 1, "預設（無機型紀錄）"


def machine_config(config_root, filename, machine_id):
    root = Path(config_root)
    candidate = root / get_machine_key(machine_id) / filename
    return candidate if candidate.is_file() else root / filename


def numeric_summary_regions(machine_id):
    return (
        ("IOP", "ASP", "VAC")
        if str(machine_id) == "3"
        else ("region1", "region2", "region3")
    )


def scene_stage_changes(records, start_frame, end_frame):
    """Carry observed values into a stage without inventing an OCR observation."""
    from bisect import bisect_right

    events = {}
    for record in records:
        blank = record.get("type") == "deactivate"
        frames = set(record.get("matched_frames", []))
        source = record.get("source_frame", record.get("frame"))
        if source is not None:
            frames.add(source)
        for frame in frames:
            priority = 1 if blank else 0
            if frame not in events or priority >= events[frame][0]:
                events[frame] = (
                    priority,
                    {
                        "frame": frame,
                        "source_frame": source,
                        "text": "" if blank else record.get("ocr_text", ""),
                        "confidence": record.get("confidence", 0.0),
                        "type": "multi_digit"
                        if record.get("type") == "multi_digit_group"
                        else record.get("type", "single_digit"),
                    },
                )
    ordered = sorted(events)
    index = bisect_right(ordered, start_frame) - 1
    changes, previous = [], None
    if index >= 0:
        event = events[ordered[index]][1]
        if event["text"]:
            carried = ordered[index] < start_frame
            changes.append(
                {
                    **event,
                    "frame": start_frame,
                    "carried": carried,
                    "type": "carried" if carried else event["type"],
                }
            )
            previous = event["text"]
    for frame in ordered[index + 1 :]:
        if frame > end_frame:
            break
        event = events[frame][1]
        if not event["text"]:
            previous = None
        elif event["text"] != previous:
            changes.append({**event, "carried": False})
            previous = event["text"]
    return changes


def analysis_rois(analysis_dir, fallback, machine_id, profiles_path=None):
    """Draw the layout used by this analysis, including pre-integration machine 3."""
    root = Path(analysis_dir)
    layout = root / "analysis_layout.json"
    if layout.is_file():
        data = json.loads(layout.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or str(data.get("machine_id")) != str(
            machine_id
        ):
            raise ValueError("Analysis layout version/machine mismatch")
        rois = data["rois"]
        for box in rois.values():
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(v, int) for v in box)
                or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
            ):
                raise ValueError("Invalid ROI in analysis layout")
        return rois
    if str(machine_id) == "3" and profiles_path and Path(profiles_path).is_file():
        stats_path = root / "run_stats.json"
        stats = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.exists()
            else {}
        )
        if stats.get("ocr", {}).get("profile") != "enhanced":
            return json.loads(Path(profiles_path).read_text(encoding="utf-8"))[
                "legacy_machine3_rois"
            ]
    return fallback
