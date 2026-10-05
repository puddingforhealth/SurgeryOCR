#!/usr/bin/env python3
"""Compare two pipeline output directories for golden-regression testing.

Structural fields (pattern ids, frame boundaries, OCR text) must match exactly;
floating-point scores (avg_rmse, confidence) are compared with a relative
tolerance, because cuDNN kernels are not guaranteed to be bit-reproducible.

Usage:
    python tools/compare_analysis.py <baseline_dir> <candidate_dir> [--rtol 1e-6]
                                     [--max-report N] [--quiet]

Exit code 0 means the two runs are equivalent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Fields compared exactly vs. with tolerance.
FLOAT_FIELDS = {"avg_rmse", "confidence"}
SEGMENT_EXACT = ("pattern", "pattern_name", "start_frame", "end_frame", "frame_count")


class Diff:
    def __init__(self, max_report: int):
        self.items: List[str] = []
        self.count = 0
        self.max_report = max_report

    def add(self, where: str, msg: str) -> None:
        self.count += 1
        if len(self.items) < self.max_report:
            self.items.append(f"{where}: {msg}")

    @property
    def ok(self) -> bool:
        return self.count == 0


def close(a: Any, b: Any, rtol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= rtol * max(1.0, abs(fa), abs(fb))


def compare_stage_analysis(base: Path, cand: Path, d: Diff, rtol: float) -> None:
    bp, cp = base / "stage_analysis.json", cand / "stage_analysis.json"
    if not bp.exists() and not cp.exists():
        return
    if bp.exists() != cp.exists():
        d.add("stage_analysis.json", f"present in baseline={bp.exists()} candidate={cp.exists()}")
        return

    bj = json.loads(bp.read_text(encoding="utf-8"))
    cj = json.loads(cp.read_text(encoding="utf-8"))

    if bj.get("total_frames") != cj.get("total_frames"):
        d.add("stage_analysis.total_frames", f"{bj.get('total_frames')} -> {cj.get('total_frames')}")

    breg, creg = bj.get("regions", {}), cj.get("regions", {})
    for name in sorted(set(breg) | set(creg)):
        if name not in breg or name not in creg:
            d.add(f"stage_analysis.regions[{name}]",
                  f"only in {'baseline' if name in breg else 'candidate'}")
            continue
        bs, cs = breg[name], creg[name]
        if len(bs) != len(cs):
            d.add(f"stage_analysis.regions[{name}]", f"segment count {len(bs)} -> {len(cs)}")
        for i, (b, c) in enumerate(zip(bs, cs)):
            for f in SEGMENT_EXACT:
                if b.get(f) != c.get(f):
                    d.add(f"stage_analysis.regions[{name}][{i}].{f}",
                          f"{b.get(f)!r} -> {c.get(f)!r}")
            if not close(b.get("avg_rmse"), c.get("avg_rmse"), rtol):
                d.add(f"stage_analysis.regions[{name}][{i}].avg_rmse",
                      f"{b.get('avg_rmse')} -> {c.get('avg_rmse')} (rtol {rtol})")


def record_key(rec: Dict) -> Tuple:
    """Stable identity for an OCR record, used to align the two lists."""
    return (rec.get("type"), rec.get("source_frame", rec.get("frame")))


def compare_jsonl(bp: Path, cp: Path, label: str, d: Diff, rtol: float) -> None:
    def load(p: Path) -> List[Dict]:
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    br, cr = load(bp), load(cp)
    if len(br) != len(cr):
        d.add(label, f"record count {len(br)} -> {len(cr)}")

    for i, (b, c) in enumerate(zip(br, cr)):
        if record_key(b) != record_key(c):
            d.add(f"{label}[{i}]", f"record identity {record_key(b)} -> {record_key(c)}")
            continue
        for f in sorted(set(b) | set(c)):
            if f in FLOAT_FIELDS:
                if not close(b.get(f), c.get(f), rtol):
                    d.add(f"{label}[{i}].{f}", f"{b.get(f)} -> {c.get(f)} (rtol {rtol})")
            elif b.get(f) != c.get(f):
                bv, cv = b.get(f), c.get(f)
                if isinstance(bv, list) and isinstance(cv, list) and len(bv) != len(cv):
                    d.add(f"{label}[{i}].{f}", f"list length {len(bv)} -> {len(cv)}")
                else:
                    d.add(f"{label}[{i}].{f}", f"{bv!r} -> {cv!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-regression comparison of two pipeline runs")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance for avg_rmse / confidence (default 1e-6)")
    ap.add_argument("--max-report", type=int, default=25, help="max differences to print")
    ap.add_argument("--quiet", action="store_true", help="print only the verdict line")
    args = ap.parse_args()

    for p in (args.baseline, args.candidate):
        if not p.is_dir():
            print(f"not a directory: {p}")
            return 2

    d = Diff(args.max_report)
    compare_stage_analysis(args.baseline, args.candidate, d, args.rtol)

    names = {p.name for p in args.baseline.glob("*_ocr_testing.jsonl")}
    names |= {p.name for p in args.candidate.glob("*_ocr_testing.jsonl")}
    for n in sorted(names):
        compare_jsonl(args.baseline / n, args.candidate / n,
                      n.replace("_ocr_testing.jsonl", ""), d, args.rtol)

    if d.ok:
        print(f"IDENTICAL  {args.baseline}  ==  {args.candidate}"
              f"  ({len(names)} OCR region files, rtol={args.rtol})")
        return 0

    if not args.quiet:
        for item in d.items:
            print(f"  {item}")
        if d.count > len(d.items):
            print(f"  ... and {d.count - len(d.items)} more")
    print(f"DIFFER     {d.count} difference(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
