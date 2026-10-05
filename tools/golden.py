#!/usr/bin/env python3
"""Run the pipeline over the golden test set and compare against a baseline.

    python tools/golden.py baseline [--set fast|full]     # record reference outputs
    python tools/golden.py check    [--set fast|full]     # run again and diff vs baseline

Outputs land in runs/<baseline|candidate>/<video_stem>/ plus a timing.json next to
them, so wall-clock can be compared across phases without a profiler.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tools" / "golden_videos.json"


def run_one(python: str, video: Path, out_root: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [python, str(REPO / "run_pipeline.py"), "--video", str(video),
         "--mode", "ram", "--output-dir", str(out_root)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  pipeline failed (exit {proc.returncode})")
        print((proc.stdout or "")[-1500:])
        print((proc.stderr or "")[-1500:])
        raise SystemExit(1)
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-regression runner")
    ap.add_argument("action", choices=["baseline", "check"])
    ap.add_argument("--set", dest="which", default="fast", choices=["fast", "full"])
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that has torch+easyocr (default: current)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = [REPO / v for v in manifest[args.which]]
    missing = [v for v in videos if not v.exists()]
    if missing:
        print("missing videos:")
        for m in missing:
            print(f"  {m}")
        return 2

    out_root = REPO / "runs" / ("baseline" if args.action == "baseline" else "candidate")
    base_root = REPO / "runs" / "baseline"
    out_root.mkdir(parents=True, exist_ok=True)

    timings = {}
    failures = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}")
        elapsed = run_one(args.python, v, out_root)
        timings[v.stem] = round(elapsed, 2)
        print(f"  wall {elapsed:.1f}s")

        if args.action == "check":
            cmp_proc = subprocess.run(
                [args.python, str(REPO / "tools" / "compare_analysis.py"),
                 str(base_root / v.stem), str(out_root / v.stem), "--rtol", str(args.rtol)],
                cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            print("  " + (cmp_proc.stdout or "").strip().replace("\n", "\n  "))
            if cmp_proc.returncode != 0:
                failures.append(v.stem)

    (out_root / "timing.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(timings.values())
    print(f"\ntotal wall {total:.1f}s  ->  {out_root / 'timing.json'}")

    if args.action == "baseline":
        print("baseline recorded.")
        return 0

    base_timing = base_root / "timing.json"
    if base_timing.exists():
        bt = json.loads(base_timing.read_text(encoding="utf-8"))
        bsum = sum(bt.get(k, 0) for k in timings)
        if bsum:
            print(f"baseline wall {bsum:.1f}s  ->  {(total - bsum) / bsum * 100:+.1f}%")

    if failures:
        print(f"REGRESSION in: {', '.join(failures)}")
        return 1
    print("all outputs match baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
