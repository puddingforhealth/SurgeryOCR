"""Classify the ASP/VAC widget display mode per OCR record and cross-tab vs the text."""
import json, cv2, numpy as np, collections, sys

VIDEO = "data/0450713odlisa.mp4"
BOX = {"ASP": [1748, 482, 1915, 516], "VAC": [1748, 568, 1915, 610]}

def records(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fr = r.get("frame", r.get("source_frame"))
            n = len(r.get("matched_frames", [fr]))
            out.append((fr, r["ocr_text"], n, r["type"]))
    return out

def ink_profile(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    w = b.shape[1]
    third = w // 4
    left = b[:, :third].mean() / 255
    right = b[:, -third:].mean() / 255
    mid = b[:, third:-third].mean() / 255
    return left, mid, right

def mode_of(left, mid, right):
    L, R = left > 0.03, right > 0.03
    if L and R:
        return "ramp(3 values)"
    if R and not L:
        return "bar(setting)"
    return "live(single value)"

for region, path in (("ASP", "runs/baseline/0450713odlisa/ASP_ocr_testing.jsonl"),
                     ("VAC", "runs/baseline/0450713odlisa/VAC_ocr_testing.jsonl")):
    recs = records(path)
    cap = cv2.VideoCapture(VIDEO)
    x1, y1, x2, y2 = BOX[region]
    tab = collections.Counter()
    bad = []
    for fr, txt, n, typ in recs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        m = mode_of(*ink_profile(img[y1:y2, x1:x2]))
        multi = " " in txt.strip()
        tab[(m, "multi-number text" if multi else "single-number text")] += 1
        if m != "live(single value)" and not multi:
            bad.append((fr, txt, n))
    cap.release()
    total = sum(tab.values())
    print(f"\n=== {region}  ({total} OCR records) ===")
    for (m, t), c in sorted(tab.items()):
        print(f"  {m:<20} x {t:<20} {c:5d}  ({c/total:5.1%})")
    frames_bad = sum(n for _, _, n in bad)
    print(f"  --> records that LOOK like a live value but are not: {len(bad)}"
          f"  (covering {frames_bad} frames)")
    for fr, txt, n in bad[:12]:
        print(f"        frame {fr:6d}  {txt!r:12s} x{n}")
