#!/usr/bin/env python3
"""CPU-only evaluation of RapidOCR/Paddle CTC ONNX recognizers on reviewed crops."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_training import contained_path, ctc_decode, digest, numerical_metrics  # noqa: E402


def paddle_input(bgr, height=48, width=320):
    import cv2
    import numpy as np

    resized_width = math.ceil(height * bgr.shape[1] / bgr.shape[0])
    width = max(width, resized_width)
    image = cv2.resize(bgr, (resized_width, height)).astype("float32")
    image = image.transpose(2, 0, 1) / 127.5 - 1.0
    output = np.zeros((1, 3, height, width), dtype=np.float32)
    output[0, :, :, :resized_width] = image
    return output


def decode_scores(scores, characters, allowed=None):
    import numpy as np

    if scores.shape[-1] != len(characters) + 1:
        raise ValueError("Embedded model dictionary does not match the output head")
    if allowed is not None:
        scores = scores.copy()
        forbidden = [i + 1 for i, char in enumerate(characters) if char not in allowed]
        scores[..., forbidden] = -np.inf
    indices = scores.argmax(-1).tolist()
    return ctc_decode(indices, characters)


def benchmark(datasets, model_path, output, threads=2):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if output.exists() or threads < 1:
        raise ValueError("Choose a new output directory and positive thread count")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1:
        raise ValueError("Expected one CTC recognizer image input")
    metadata = session.get_modelmeta().custom_metadata_map
    if "character" not in metadata:
        raise ValueError("Expected an embedded RapidOCR character dictionary")
    characters = metadata["character"].splitlines() + [" "]
    shape = session.get_inputs()[0].shape
    height = shape[2] if isinstance(shape[2], int) else 48
    samples, seen = [], set()
    for dataset in datasets:
        for line in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] not in {"reviewed_numeric", "reviewed_blank"}:
                continue
            if row["id"] in seen:
                raise ValueError("Duplicate sample ID across datasets")
            seen.add(row["id"])
            path = contained_path(dataset, row["image"])
            if digest(path) != row["image_sha256"]:
                raise ValueError("Crop hash changed")
            with Image.open(path) as source:
                bgr = np.asarray(source.convert("RGB"))[:, :, ::-1]
            samples.append((row, paddle_input(bgr, height)))
    if not samples:
        raise ValueError("No reviewed single-target or blank samples")
    name = session.get_inputs()[0].name
    session.run(None, {name: samples[0][1]})
    predictions, durations = [], []
    for row, tensor in samples:
        start = time.perf_counter()
        scores = session.run(None, {name: tensor})[0][0]
        durations.append(time.perf_counter() - start)
        predictions.append(
            {
                "id": row["id"],
                "region": row["region"],
                "label": row["label"],
                "raw": decode_scores(scores, characters),
                "numeric": decode_scores(scores, characters, "+-0123456789."),
            }
        )
    result = {
        "model_sha256": digest(model_path),
        "model_bytes": model_path.stat().st_size,
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "threads": threads,
        "input": "BGR, no thresholding, height-normalized, width >=320, zero padding after [-1,1] normalization",
        "warm_median_inference_ms": float(np.median(durations) * 1000),
        "raw_metrics": numerical_metrics([(r["label"], r["raw"]) for r in predictions]),
        "numeric_metrics": numerical_metrics(
            [(r["label"], r["numeric"]) for r in predictions]
        ),
        "by_region": {
            region: {
                mode: numerical_metrics(
                    [
                        (r["label"], r[mode])
                        for r in predictions
                        if r["region"] == region
                    ]
                )
                for mode in ["raw", "numeric"]
            }
            for region in sorted({r["region"] for r in predictions})
        },
        "predictions": predictions,
        "limitations": "Selected assistant-labelled crops, not independent human labels or held-out model-selection data. CPU inference timings exclude preprocessing/model load and are machine-specific. Numeric masking is a separate decoder experiment, not a field-format correction.",
    }
    output.mkdir(parents=True)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="review folders containing labels.jsonl and original RGB crops",
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local CTC ONNX with RapidOCR embedded character metadata",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    result = benchmark(args.data, args.model, args.output, args.threads)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "model_bytes",
                    "warm_median_inference_ms",
                    "raw_metrics",
                    "numeric_metrics",
                    "by_region",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
