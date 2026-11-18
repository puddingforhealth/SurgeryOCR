#!/usr/bin/env python3
"""Evaluate the deployed machine-1/2 adapter on a reviewed crop split."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_runtime import ConservativeEasyOCR, load_bundle, file_sha256  # noqa: E402
from utils.ocr_training import load_dataset, contained_path  # noqa: E402
from tools.compare_pipeline_runs import metrics  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--accept-assistant-labels", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new evaluation directory")
    rows = [
        r
        for r in load_dataset(args.data, accept_assistant=args.accept_assistant_labels)
        if r["split"] == args.split
    ]
    if not rows:
        raise ValueError("Empty evaluation split")
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(2)
    root, manifest = load_bundle(args.bundle)
    runtime = ConservativeEasyOCR(root, manifest, 1, args.device)
    character_sets = json.loads(
        (ROOT / "config/ocr_char_sets.json").read_text(encoding="utf-8")
    )
    predictions = []
    for row in rows:
        if str(row["machine_id"]) not in {"1", "2"}:
            raise ValueError("This evaluation requires machine-1/2 crops")
        runtime.machine_id = row["machine_id"]
        with Image.open(contained_path(args.data, row["images"]["binary"])) as image:
            pixels = np.asarray(image.convert("L"))
        text, confidence = runtime.recognize_region(
            pixels, row["region"], character_sets.get(row["region"])
        )
        decision = dict(runtime.last_decision)
        baseline = decision.get("baseline_text", text)
        if (
            manifest.get("machine12_route") == "single_numeric"
            and "candidate_text" in decision
        ):
            text, confidence = (
                decision["candidate_text"],
                runtime.last_candidate_confidence,
            )
            decision.update(
                text=text,
                confidence=confidence,
                source="single_numeric",
                needs_review=False,
            )
        predictions.append(
            {
                "id": row["id"],
                "video_id": row["video_id"],
                "region": row["region"],
                "machine_id": row["machine_id"],
                "label": row["label"],
                "legacy": baseline,
                "enhanced": text,
                "confidence": confidence,
                **decision,
            }
        )
    dataset = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    result = {
        "scope": "Deployed adapter crop regression, excluding full-video activation/trigger/cache. Already consumed research labels are not a fresh acceptance set.",
        "device": args.device,
        "torch_version": torch.__version__,
        "dataset_sha256": file_sha256(args.data / dataset["manifest"]),
        "bundle_version": manifest["version"],
        "metrics": metrics(predictions),
        "predictions": predictions,
    }
    args.output.mkdir(parents=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result["metrics"].items() if k in {"legacy", "enhanced"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
