#!/usr/bin/env python3
"""Create an explicit, local-only rcs OCR job folder; never submit it."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_training import contained_path, digest, load_dataset  # noqa: E402


def prepare(
    data,
    weights,
    output,
    *,
    accept_assistant=False,
    epochs=20,
    batch_size=8,
    max_vram_gib=4,
    remote_python="python3",
):
    rows = load_dataset(data, accept_assistant=accept_assistant)
    if not all(any(r["split"] == s for r in rows) for s in ["train", "val", "test"]):
        raise ValueError("A prepared job requires nonempty train/val/test splits")
    if output.exists() or not weights.is_file():
        raise ValueError("Choose a new output folder and existing English gen2 weights")
    if epochs <= 0 or batch_size <= 0 or not 0 < max_vram_gib < 1024:
        raise ValueError("epochs, batch size and VRAM budget must be positive")
    output.mkdir(parents=True)
    (output / "tools").mkdir()
    (output / "utils").mkdir()
    (output / "dataset").mkdir()
    (output / "weights").mkdir()
    shutil.copy2(ROOT / "tools/ocr_finetune.py", output / "tools/ocr_finetune.py")
    shutil.copy2(ROOT / "utils/ocr_training.py", output / "utils/ocr_training.py")
    (output / "utils/__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(weights, output / "weights/english_g2.pth")
    # Only eligible rows/images cross the transport boundary. No raw videos,
    # original patient filenames, private source maps, or unrelated project files.
    for row in rows:
        for relative in row["images"].values():
            target = contained_path(output / "dataset", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(contained_path(data, relative), target)
    (output / "dataset/records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (output / "dataset/dataset.json").write_text(
        json.dumps({"schema_version": 1, "manifest": "records.jsonl"}), encoding="utf-8"
    )
    acknowledgment = " --accept-assistant-labels" if accept_assistant else ""
    python = shlex.quote(remote_python)
    base = f"{python} -u tools/ocr_finetune.py"
    common = " --data dataset --input binary" + acknowledgment
    command = (
        f"{base} train{common} --weights weights/english_g2.pth --output outputs/train --device cuda"
        f" --epochs {epochs} --batch-size {batch_size} --max-vram-gib {max_vram_gib}"
        f" && {base} evaluate{common} --weights weights/english_g2.pth --output outputs/baseline-test --split test"
        f" && {base} evaluate{common} --weights outputs/train/best/numeric_m12.pth --output outputs/candidate-test --split test"
    )
    config = {
        "version": 1,
        "name": "ocr-m12-finetune",
        "include": ["tools", "utils", "dataset", "weights"],
        "exclude": ["*.pyc", "__pycache__"],
        "setup": [],
        "command": command,
        "outputs": ["outputs"],
        "env": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        "require_gpu": True,
        "remote_root": "/tmp/rcs-jobs",
        "remote_python": remote_python,
    }
    (output / "rcs-job.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "bundle-provenance.json").write_text(
        json.dumps(
            {
                "weights_sha256": digest(weights),
                "records_sha256": digest(output / "dataset/records.jsonl"),
                "samples": len(rows),
                "submitted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Prepared OCR job\n\nThis folder has NOT been submitted. Run `rcs job pack` here to inspect the local archive. "
        "When the selected target is idle, run `rcs --target zettabyte job run`, then `rcs job logs --follow` "
        "and `rcs job fetch`. The target must already provide Python with torch, torchvision, EasyOCR 1.7.2, "
        "NumPy and Pillow. No dependencies are installed into a shared remote environment.\n\n"
        "Training refuses a busy GPU. It checks for foreign compute processes again every ten steps and stops itself "
        "if one appears. The allocator cap is not hardware isolation and excludes CUDA context overhead. "
        "No resource requests, releases, process termination, or automatic submission occur.\n\n"
        "The prepared CPU evaluations compare the untouched weights and validation-selected weights on the held-out test split. "
        "A two-step smoke test is not evidence of improved accuracy. Medical crops are private; do not publish this bundle.\n",
        encoding="utf-8",
    )
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--accept-assistant-labels", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-vram-gib", type=float, default=4)
    p.add_argument(
        "--remote-python",
        default="python3",
        help="existing remote interpreter; no shared environment mutation",
    )
    args = p.parse_args()
    prepare(
        args.data,
        args.weights,
        args.output,
        accept_assistant=args.accept_assistant_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_vram_gib=args.max_vram_gib,
        remote_python=args.remote_python,
    )
    print(f"Prepared locally, not submitted: {args.output}")


if __name__ == "__main__":
    main()
