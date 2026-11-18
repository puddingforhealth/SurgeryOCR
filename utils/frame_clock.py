"""Map OCR frame indices to presentation time: frame_clock.json.

OCR results store the index of each decoded picture (0, 1, 2, ... in the order
the pipeline decoded them). A video player seeks by presentation time. The two
agree as ``index / fps`` until the recorder stalls: one picture is then held on
screen for seconds, and pictures that cannot be decoded after the stall still
advance the clock. From that point every ``index / fps`` estimate is off by a
fixed amount (2.96 s in one of our recordings).

frame_clock.json records the real time of every decoded picture, compressed to
the points where the regular cadence breaks:

    time(frame) = segment.time + (frame - segment.frame) / nominal_fps

using the last segment whose ``frame`` is <= the frame being looked up. A video
without stalls has a single segment ``{"frame": 0, "time": 0.0}``.

The format is version 1 of the file PhacoSense reads, so the same reader works
whether the table came from this pipeline or from their slice-header scanner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FILE_NAME = "frame_clock.json"
FORMAT_VERSION = 1
# Times are written in seconds, rounded to a microsecond.
_TIME_DIGITS = 6


def nominal_step(times: Sequence[float]) -> float:
    """Return the regular frame duration in seconds.

    The mean of the gaps within 25 % of the median gap. The median screens out
    stalls (a 2.5 s hold) and the odd short or long gap a recorder writes now and
    then; averaging what is left recovers the true rate when timestamps are
    quantized (33/33/34 ms in a 1 ms timescale), where the median alone is off
    by 1 %. Letting the odd gaps into the mean would bias the rate and make the
    clock re-anchor every few minutes for no real reason.
    """
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if not gaps:
        raise ValueError(
            "need at least two increasing timestamps to find the frame rate"
        )
    median = gaps[len(gaps) // 2]
    regular = [g for g in gaps if abs(g - median) <= 0.25 * median]
    return sum(regular) / len(regular)


def build_segments(
    frames: Sequence[Tuple[int, float]], step: float
) -> List[Dict[str, float]]:
    """Compress (frame index, time in seconds) pairs into clock segments.

    A new segment starts wherever the time predicted from the current segment
    misses the real time by more than half a frame. The prediction runs from the
    segment start rather than the previous frame, so slow drift is caught too.
    Indices may skip (a picture that decoded empty); the formula still holds.
    """
    if not frames:
        raise ValueError("no frames to build a clock from")
    if step <= 0:
        raise ValueError("frame duration must be positive")
    segments: List[Dict[str, float]] = []
    start_frame, start_time = frames[0]
    segments.append(
        {"frame": int(start_frame), "time": round(start_time, _TIME_DIGITS)}
    )
    for frame, time in frames[1:]:
        predicted = start_time + (frame - start_frame) * step
        if abs(time - predicted) > step / 2:
            start_frame, start_time = frame, time
            segments.append({"frame": int(frame), "time": round(time, _TIME_DIGITS)})
    return segments


def build_clock(
    source: str,
    frames: Sequence[Tuple[int, float]],
    container_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the frame_clock.json payload.

    ``frames`` holds (decoded index, presentation time in seconds) for every
    picture the pipeline decoded, in decode order. ``container_frames`` is the
    sample count the container declares; the difference to the decoded count is
    reported as ``dropped_frames`` (pictures in the file that never decode).
    """
    if not frames:
        raise ValueError("no frames to build a clock from")
    step = nominal_step([t for _, t in frames])
    decoded = frames[-1][0] + 1
    dropped = None
    if container_frames is not None and container_frames > 0:
        dropped = max(0, int(container_frames) - decoded)
    return {
        "version": FORMAT_VERSION,
        "source": source,
        "nominal_fps": round(1.0 / step, 6),
        "decoded_frames": decoded,
        "dropped_frames": dropped,
        "segments": build_segments(frames, step),
    }


def frame_time(clock: Dict[str, Any], frame: int) -> float:
    """Return the presentation time in seconds of a decoded frame index."""
    segments = clock["segments"]
    current = segments[0]
    for segment in segments:
        if segment["frame"] > frame:
            break
        current = segment
    return current["time"] + (frame - current["frame"]) / clock["nominal_fps"]


def validate_clock(clock: Any) -> List[str]:
    """Return a list of problems with a frame_clock payload; empty means valid."""
    problems: List[str] = []
    if not isinstance(clock, dict):
        return ["frame_clock must be a JSON object"]
    if clock.get("version") != FORMAT_VERSION:
        problems.append(
            f"version must be {FORMAT_VERSION}, got {clock.get('version')!r}"
        )
    fps = clock.get("nominal_fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        problems.append("nominal_fps must be a positive number")
    decoded = clock.get("decoded_frames")
    if not isinstance(decoded, int) or decoded <= 0:
        problems.append("decoded_frames must be a positive integer")
    segments = clock.get("segments")
    if not isinstance(segments, list) or not segments:
        problems.append("segments must be a non-empty list")
        return problems
    previous = None
    for i, segment in enumerate(segments):
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("frame"), int)
            or not isinstance(segment.get("time"), (int, float))
        ):
            problems.append(f"segments[{i}] needs an integer frame and a numeric time")
            continue
        if previous is not None and segment["frame"] <= previous["frame"]:
            problems.append(f"segments[{i}].frame must increase")
        previous = segment
    return problems


def write_clock(path: Path, clock: Dict[str, Any]) -> None:
    """Write the clock as indented UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(clock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_clock(path: Path) -> Dict[str, Any]:
    """Read and validate a frame_clock.json file."""
    clock = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_clock(clock)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return clock


def scan_video(path: Path) -> Tuple[List[Tuple[int, float]], Optional[int]]:
    """Decode a video the way run_pipeline.py does and return its frame times.

    Returns ((index, seconds) pairs, container frame count). Indices follow the
    pipeline exactly because both use the same iterator. Used to add
    frame_clock.json to results produced before the pipeline wrote it, without
    rerunning OCR.
    """
    import cv2  # Imported here so the pure helpers above need no OpenCV.
    from utils.pipeline_utils import iter_video_frames

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        frames = [(index, ms / 1000.0) for index, _, ms in iter_video_frames(cap)]
        return frames, container_frames
    finally:
        cap.release()
