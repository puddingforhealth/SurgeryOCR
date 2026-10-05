#!/usr/bin/env python3
"""Compile a machine's capture_log.json (produced by roi_picker.py quick-capture)
into the formal pipeline configs.

Reads  data/roi_img_caches/machine_<id>/capture_log.json and writes:
  - config/machine_<id>/surgery_stage_rois.json   (pattern-region boxes, e.g. STAGE/PEDAL)
  - config/machine_<id>/pattern_name_mapping.json (pattern id -> label)
  - config/machine_<id>/ocr_strategy.json         (default OCR strategy, created once)
  - config/rois.json                              (merges "<id>" entries per OCR region)
  - config/ocr_char_sets.json                     (adds allowlists for new regions)

Idempotent: rerun after adding more captures. Existing values are updated,
unknown/unrelated entries are preserved.

Usage:
    python compile_machine_config.py --machine 3 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# per-region OCR allowlist defaults for newly added regions (fallback: digits+-.)
CHAR_SET_DEFAULTS = {
    "CDE": "0123456789.",
    "IOP": "0123456789",
    "ASP": "0123456789",
    "VAC": "0123456789",
    "PEL": "-0123456789",
    "INFLOW": "0123456789",
    "INCISION": "0123456789.",
    "REFLUX": "0123456789",
}
FALLBACK_CHAR_SET = "-0123456789."

# default per-machine OCR strategy written once (tunable afterwards)
OCR_STRATEGY_DEFAULTS = {
    "3": {
        "ocr_input": "gray",       # feed grayscale crops to OCR (translucent overlay)
        "stable_frames": 4,        # debounce: OCR only after N stable frames
        "min_ocr_height": 32,      # upscale small crops before OCR
        "diff_threshold": 0.02,    # change-detection threshold (binary XOR ratio)
    },
}


def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return default


def save_json(path: Path, data, dry_run: bool):
    if dry_run:
        print(f"[dry-run] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Compile capture_log.json into pipeline configs")
    ap.add_argument("--machine", required=True, help="machine id (e.g. 3)")
    ap.add_argument("--dry-run", action="store_true", help="print without writing")
    args = ap.parse_args()

    mid = str(args.machine)
    mdir = Path("data/roi_img_caches") / f"machine_{mid}"
    log_path = mdir / "capture_log.json"
    if not log_path.exists():
        raise SystemExit(f"capture log not found: {log_path}")
    log = load_json(log_path, {})

    cfg_dir = Path("config") / f"machine_{mid}"
    problems = []

    # --- 1. surgery_stage_rois: pattern-region boxes -------------------------
    pattern_regions = log.get("pattern_regions", {})
    stage_rois = {}
    for region, entry in pattern_regions.items():
        coords = entry.get("coords")
        if not coords:
            problems.append(f"pattern region '{region}' has no coords")
            continue
        stage_rois[region] = coords
        pngs = sorted((mdir / region).glob("*.png")) if (mdir / region).exists() else []
        npys = sorted((mdir / region).glob("*.npy")) if (mdir / region).exists() else []
        if len(pngs) != len(npys):
            problems.append(f"'{region}': {len(pngs)} png vs {len(npys)} npy (mismatch)")
    if stage_rois:
        save_json(cfg_dir / "surgery_stage_rois.json", stage_rois, args.dry_run)

    # --- 2. pattern_name_mapping from labels --------------------------------
    name_map = {}
    for region, entry in pattern_regions.items():
        mapping = {}
        for pid, pat in sorted(entry.get("patterns", {}).items(), key=lambda x: int(x[0])):
            label = pat.get("label")
            if label:
                mapping[pid] = label
            else:
                problems.append(f"'{region}' pattern #{pid} is unlabeled (frame {pat.get('frame')})")
        if mapping:
            name_map[region] = mapping
    if name_map:
        save_json(cfg_dir / "pattern_name_mapping.json", name_map, args.dry_run)

    # --- 3. merge OCR regions into rois.json --------------------------------
    rois_path = Path("config/rois.json")
    rois = load_json(rois_path, {"video_machine_mapping": {}, "regions": {}})
    rois.setdefault("regions", {})
    regions = log.get("regions", {})
    merged = 0
    for name, entry in regions.items():
        value = entry.get("value_coords")
        header = entry.get("header_coords", [])
        if not value:
            problems.append(f"region '{name}' has no value_coords (skipped)")
            continue
        rois["regions"].setdefault(name, {})[mid] = [value, header]
        merged += 1
        header_png = mdir / "roi_headers" / f"{name}.png"
        if header and not header_png.exists():
            problems.append(f"region '{name}' has header coords but no template: {header_png}")
    if merged:
        save_json(rois_path, rois, args.dry_run)
        print(f"  merged {merged} region entries under machine '{mid}'")

    # --- 4. char sets for new regions ---------------------------------------
    char_path = Path("config/ocr_char_sets.json")
    char_sets = load_json(char_path, {})
    added = []
    for name in regions:
        if name not in char_sets:
            char_sets[name] = CHAR_SET_DEFAULTS.get(name, FALLBACK_CHAR_SET)
            added.append(name)
    if added:
        save_json(char_path, char_sets, args.dry_run)
        print(f"  added char sets: {', '.join(added)}")

    # --- 5. ocr_strategy defaults (created once, never overwritten) ---------
    strategy_path = cfg_dir / "ocr_strategy.json"
    if not strategy_path.exists() and mid in OCR_STRATEGY_DEFAULTS:
        save_json(strategy_path, OCR_STRATEGY_DEFAULTS[mid], args.dry_run)

    # --- report --------------------------------------------------------------
    print()
    if problems:
        print("⚠ 需要注意:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("✓ no problems found")


if __name__ == "__main__":
    main()
