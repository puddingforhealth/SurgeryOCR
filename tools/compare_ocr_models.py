#!/usr/bin/env python3
"""Compare original and fine-tuned weights through the same production OCR call."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_training import contained_path, digest, load_dataset, numerical_metrics  # noqa: E402
from tools.ocr_finetune import desktop_gpu_capacity, gpu_idle  # noqa: E402


def comparison_metrics(predictions, bootstrap_samples=2000):
    def metrics(rows):
        old = numerical_metrics([(r["label"], r["baseline"]) for r in rows])
        new = numerical_metrics([(r["label"], r["candidate"]) for r in rows])
        improved = [
            r["id"]
            for r in rows
            if r["baseline"] != r["label"] and r["candidate"] == r["label"]
        ]
        regressed = [
            r["id"]
            for r in rows
            if r["baseline"] == r["label"] and r["candidate"] != r["label"]
        ]
        return {
            "baseline": old,
            "candidate": new,
            "improved": improved,
            "regressed": regressed,
        }

    result = metrics(predictions)
    for key in ["machine_id", "region", "selection", "evaluation_cohort"]:
        result["by_" + key] = {
            str(value): metrics(
                [r for r in predictions if r.get(key, "unspecified") == value]
            )
            for value in sorted(
                {r.get(key, "unspecified") for r in predictions}, key=str
            )
        }
    videos = defaultdict(list)
    for row in predictions:
        videos[row["video_id"]].append(row)
    # Resample whole videos, never treat adjacent crops as independent trials.
    if len(videos) >= 2 and bootstrap_samples:
        groups = list(videos.values())
        rng = random.Random(20261116)
        differences = []
        for _ in range(bootstrap_samples):
            sample = [r for _ in groups for r in rng.choice(groups)]
            change = sum(
                (r["candidate"] == r["label"]) - (r["baseline"] == r["label"])
                for r in sample
            )
            differences.append(change / len(sample))
        differences.sort()
        result["video_cluster_bootstrap_delta_95_interval"] = [
            differences[int(0.025 * (len(differences) - 1))],
            differences[int(0.975 * (len(differences) - 1))],
        ]
    controls = [
        r for r in predictions if r.get("selection") == "high_confidence_control"
    ]
    result["control_regressions"] = metrics(controls)["regressed"]
    result["controls_preserved_in_this_sample"] = not result["control_regressions"]
    return result


def compare(
    data,
    weights,
    output,
    split="val",
    device="cpu",
    desktop_gpu=False,
    base_model_dir=None,
    candidate_detector="craft",
    prediction_policy="raw",
):
    import numpy as np
    import torch
    from PIL import Image
    from models.OCR_interface.easyocr_model import EasyOCRModel
    from utils.cv_processing import trim_black_borders
    from utils.ocr_training import load_recognizer

    rows = [r for r in load_dataset(data, accept_assistant=True) if r["split"] == split]
    if not rows or output.exists():
        raise ValueError("Need a nonempty split and a new output directory")
    if device == "cuda":
        checker = desktop_gpu_capacity if desktop_gpu else gpu_idle
        checker(0, 3072)
        torch.cuda.set_per_process_memory_fraction(
            2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
        )
    torch.set_num_threads(2)
    model = EasyOCRModel(
        gpu=device == "cuda",
        lang_list=["en"],
        quantize=False,
        confidence_threshold=0.5,
        download_enabled=False,
        debug_output=False,
        model_storage_directory=str(base_model_dir) if base_model_dir else None,
    )
    candidate, alphabet = load_recognizer(weights)
    if alphabet != model.reader.character:
        raise ValueError(
            "Candidate vocabulary must match the original English gen2 model"
        )
    candidate.to(device).eval()
    baseline = model.reader.recognizer
    character_sets = json.loads(
        (ROOT / "config/ocr_char_sets.json").read_text(encoding="utf-8")
    )
    results = []
    for i, row in enumerate(rows):
        with Image.open(contained_path(data, row["images"]["binary"])) as image:
            binary = np.array(image.convert("L"))
        prediction = {
            k: row.get(k, "unspecified")
            for k in [
                "id",
                "label",
                "video_id",
                "machine_id",
                "region",
                "selection",
                "evaluation_cohort",
            ]
        }
        for name, recognizer in [("baseline", baseline), ("candidate", candidate)]:
            model.reader.recognizer = recognizer
            model.skip_detector = (
                name == "candidate" and candidate_detector == "whole_roi"
            )
            text, confidence = model.recognize(
                binary, allowlist=character_sets.get(row["region"])
            )
            if not text:
                trimmed = trim_black_borders(binary, max_border=1)
                if trimmed.size and trimmed.shape != binary.shape:
                    text, confidence = model.recognize(
                        trimmed, allowlist=character_sets.get(row["region"])
                    )
            prediction[name] = text
            prediction[name + "_confidence"] = float(confidence)
        if prediction_policy == "conservative":
            from utils.ocr_prediction_policy import conservative_prediction

            chosen = conservative_prediction(
                prediction["baseline"],
                prediction["baseline_confidence"],
                prediction["candidate"],
                prediction["candidate_confidence"],
                row["region"],
                row["machine_id"],
            )
            prediction["candidate_raw"] = prediction["candidate"]
            prediction["candidate_raw_confidence"] = prediction["candidate_confidence"]
            prediction["candidate"] = chosen["text"]
            prediction["candidate_confidence"] = chosen["confidence"]
            prediction["policy_source"] = chosen["source"]
            prediction["needs_review"] = chosen["needs_review"]
        results.append(prediction)
        if (i + 1) % 50 == 0:
            print(f"Compared {i + 1}/{len(rows)}", flush=True)
    output.mkdir(parents=True)
    result = {
        "method": "Baseline uses the existing EasyOCRModel CRAFT call. Both retain field allowlists, 0.5 confidence threshold and empty-result trim retry; no stage/cache/debounce simulation. Candidate detector and weight changes are reported separately.",
        "candidate_detector": candidate_detector,
        "prediction_policy": prediction_policy,
        "candidate_sha256": digest(weights),
        "split": split,
        "device": device,
        "metrics": comparison_metrics(results),
        "predictions": results,
    }
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--desktop-gpu", action="store_true")
    p.add_argument(
        "--candidate-detector", choices=["craft", "whole_roi"], default="craft"
    )
    p.add_argument(
        "--prediction-policy", choices=["raw", "conservative"], default="raw"
    )
    p.add_argument(
        "--base-model-dir",
        type=Path,
        help="directory containing existing english_g2 and CRAFT weights; downloads disabled",
    )
    args = p.parse_args()
    if args.desktop_gpu and args.device != "cuda":
        p.error("--desktop-gpu requires --device cuda")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    result = compare(
        args.data,
        args.weights,
        args.output,
        args.split,
        args.device,
        args.desktop_gpu,
        args.base_model_dir,
        args.candidate_detector,
        args.prediction_policy,
    )
    print(
        json.dumps(
            {
                k: result["metrics"][k]
                for k in ["baseline", "candidate", "controls_preserved_in_this_sample"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
