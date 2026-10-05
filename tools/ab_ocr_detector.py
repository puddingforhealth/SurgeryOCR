#!/usr/bin/env python3
"""A/B the CRAFT text detector against feeding the whole ROI to the recognizer.

Skipping the detector is the largest remaining win, but unlike the other
optimisations it changes recognition results, so it cannot be validated by a
byte-for-byte golden diff. This records the exact images the pipeline sends to
OCR during a real run (path A, the current behaviour), replays them through the
detector-free path (path B), and reports the speed difference plus every
disagreement, writing the disputed crops out as PNGs for inspection.

    python tools/ab_ocr_detector.py --video data/1131009何素珠od.MP4 [--limit N]

Outputs runs/ab_ocr/<video>/{report.json, disagreements/<region>_<frame>_*.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B the OCR detector stage")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many recorded OCR calls (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "ab_ocr")
    args = ap.parse_args()

    import models.OCR_interface.easyocr_model as em
    from models.OCR_interface import get_ocr_model
    import run_pipeline

    recorded: list = []
    limit_hit = {"flag": False}
    original = em.EasyOCRModel.recognize

    def recording_recognize(self, image, allowlist=None):
        """Record the input and reference result, then delegate to the real path."""
        t0 = time.perf_counter()
        text, conf = original(self, image, allowlist=allowlist)
        elapsed = time.perf_counter() - t0
        if not limit_hit["flag"]:
            recorded.append({
                "image": np.ascontiguousarray(image),
                "allowlist": allowlist,
                "text_a": text, "conf_a": conf, "sec_a": elapsed,
            })
            if args.limit and len(recorded) >= args.limit:
                limit_hit["flag"] = True
        return text, conf

    em.EasyOCRModel.recognize = recording_recognize
    try:
        out_dir = args.out / args.video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[A] running the pipeline with the current OCR path ({args.video.name})")
        run_pipeline.run_pipeline(args.video, out_dir / "_runA", mode="ram", force=False)
    finally:
        em.EasyOCRModel.recognize = original

    if not recorded:
        print("no OCR calls were recorded (nothing changed in any region?)")
        return 1
    print(f"\nrecorded {len(recorded)} OCR calls")

    print("[B] replaying the same inputs with the detector skipped")
    model_b = get_ocr_model(model_type="easyocr", gpu=True, lang_list=["en"],
                            confidence_threshold=0.5, debug_output=False,
                            skip_detector=True)
    # one warm-up so cuDNN autotuning is not charged to the first sample
    model_b.recognize(recorded[0]["image"], allowlist=recorded[0]["allowlist"])

    sec_a = sum(r["sec_a"] for r in recorded)
    sec_b = 0.0
    agree = 0
    disagreements = []
    for r in recorded:
        t0 = time.perf_counter()
        text_b, conf_b = model_b.recognize(r["image"], allowlist=r["allowlist"])
        sec_b += time.perf_counter() - t0
        if text_b == r["text_a"]:
            agree += 1
        else:
            disagreements.append({"a": r["text_a"], "b": text_b,
                                  "conf_a": r["conf_a"], "conf_b": conf_b,
                                  "image": r["image"]})

    n = len(recorded)
    # The first recorded call carries cuDNN autotuning cost; report both figures.
    sec_a_tail = sec_a - recorded[0]["sec_a"]
    print(f"\n{'':22s}{'total':>10s}{'per call':>12s}")
    print(f"{'A current (CRAFT)':22s}{sec_a:9.2f}s{sec_a/n*1000:11.2f} ms")
    print(f"{'A excl. first call':22s}{sec_a_tail:9.2f}s{sec_a_tail/max(1,n-1)*1000:11.2f} ms")
    print(f"{'B whole-ROI':22s}{sec_b:9.2f}s{sec_b/n*1000:11.2f} ms")
    print(f"\nspeedup {sec_a_tail/sec_b:.2f}x   ({(sec_b-sec_a_tail)/sec_a_tail*100:+.1f}% time)")
    print(f"agreement {agree}/{n} = {agree/n*100:.2f}%   disagreements {len(disagreements)}")

    dis_dir = out_dir / "disagreements"
    if disagreements:
        dis_dir.mkdir(parents=True, exist_ok=True)
        for old in dis_dir.glob("*.png"):
            old.unlink()
        print(f"\nfirst disagreements (crops written to {dis_dir}):")
        for i, d in enumerate(disagreements[:40]):
            a = d["a"] or "(empty)"
            b = d["b"] or "(empty)"
            print(f"  {i:3d}  A={a!r:>12s} (conf {d['conf_a']:.2f})   B={b!r:>12s} (conf {d['conf_b']:.2f})")
            safe = lambda s: "".join(c if c.isalnum() or c in "-." else "_" for c in (s or "empty"))
            cv2.imwrite(str(dis_dir / f"{i:04d}_A-{safe(d['a'])}_B-{safe(d['b'])}.png"), d["image"])
        if len(disagreements) > 40:
            print(f"  ... and {len(disagreements)-40} more")

    report = {
        "video": args.video.name,
        "ocr_calls": n,
        "seconds": {"a_craft": round(sec_a, 2), "a_excl_first": round(sec_a_tail, 2),
                    "b_whole_roi": round(sec_b, 2)},
        "ms_per_call": {"a_craft": round(sec_a / n * 1000, 2),
                        "a_excl_first": round(sec_a_tail / max(1, n - 1) * 1000, 2),
                        "b_whole_roi": round(sec_b / n * 1000, 2)},
        "speedup_excl_first": round(sec_a_tail / sec_b, 3) if sec_b else None,
        "agreement": round(agree / n, 5),
        "disagreements": [{"a": d["a"], "b": d["b"],
                           "conf_a": round(float(d["conf_a"]), 4),
                           "conf_b": round(float(d["conf_b"]), 4)}
                          for d in disagreements],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\nreport -> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
