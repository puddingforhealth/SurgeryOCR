#!/usr/bin/env python3
"""Validate a reviewed OCR pack, fine-tune English gen2 CTC, or evaluate it."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_training import (  # noqa: E402
    contained_path,
    dataset_summary,
    digest,
    encode_labels,
    export_recognizer,
    image_tensor,
    load_dataset,
    load_recognizer,
    numerical_metrics,
    predict,
)


def gpu_idle(index, minimum_free_mib):
    """Refuse to compete with existing compute processes; never terminate them."""

    def query(field, kind="gpu"):
        return subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(index),
                f"--query-{kind}={field}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip()

    processes = query("pid", "compute-apps")
    foreign = [
        p.strip()
        for p in processes.splitlines()
        if p.strip().isdigit() and int(p.strip()) != os.getpid()
    ]
    free, utilization = [
        int(x.strip()) for x in query("memory.free,utilization.gpu").split(",")
    ]
    if foreign or free < minimum_free_mib or utilization >= 10:
        raise RuntimeError(
            f"GPU busy; refusing training. Other PIDs={foreign}, free={free} MiB, utilization={utilization}%"
        )


def gpu_no_foreign_processes(index):
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
    )
    if any(
        p.strip().isdigit() and int(p.strip()) != os.getpid() for p in text.splitlines()
    ):
        raise RuntimeError("Another GPU workload appeared; stopping this training job")


def desktop_gpu_capacity(index, minimum_free_mib):
    """Explicit local Windows policy: WDDM reports ordinary GUI apps as compute."""
    if sys.platform != "win32":
        raise ValueError("--desktop-gpu is only available on local Windows desktops")
    free = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    )
    if free < minimum_free_mib:
        raise RuntimeError(
            f"Insufficient desktop GPU headroom: {free} MiB free, need {minimum_free_mib}"
        )


def evaluate_rows(model, characters, rows, root, mode, device, batch_size, width):
    import torch

    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = torch.stack(
                [
                    image_tensor(contained_path(root, r["images"][mode]), width=width)
                    for r in batch
                ]
            ).to(device)
            texts = predict(model, inputs, characters)
            predictions.extend(
                {
                    "id": r["id"],
                    "label": r["label"],
                    "prediction": t,
                    "machine_id": r["machine_id"],
                    "region": r["region"],
                }
                for r, t in zip(batch, texts)
            )
    metrics = numerical_metrics([(p["label"], p["prediction"]) for p in predictions])
    metrics["by_machine"] = {
        str(m): numerical_metrics(
            [(p["label"], p["prediction"]) for p in predictions if p["machine_id"] == m]
        )
        for m in sorted({p["machine_id"] for p in predictions})
    }
    metrics["by_region"] = {
        r: numerical_metrics(
            [(p["label"], p["prediction"]) for p in predictions if p["region"] == r]
        )
        for r in sorted({p["region"] for p in predictions})
    }
    return {"metrics": metrics, "predictions": predictions}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument(
            "--input",
            choices=["binary", "rgb"],
            default="binary",
            help="rgb is converted to grayscale; no thresholding",
        )
        p.add_argument(
            "--accept-assistant-labels",
            action="store_true",
            help="explicitly accept labels not independently human verified",
        )
        if name == "validate":
            continue
        p.add_argument(
            "--weights",
            type=Path,
            required=True,
            help="English gen2 pretrained or exported numeric_m12.pth",
        )
        p.add_argument(
            "--output",
            type=Path,
            required=True,
            help="new directory; refuses overwrite",
        )
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument(
            "--desktop-gpu",
            action="store_true",
            help="explicit Windows WDDM mode: allow GUI processes, enforce memory headroom; never for a shared Linux GPU",
        )
        p.add_argument(
            "--gpu-index",
            type=int,
            default=0,
            help="physical GPU index, mapped to cuda:0",
        )
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument(
            "--width",
            type=int,
            default=256,
            help="padded width at fixed model height 64; refuses squashing",
        )
        p.add_argument("--threads", type=int, default=2)
        if name == "evaluate":
            p.add_argument("--split", choices=["val", "test"], default="test")
        else:
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument(
                "--max-steps",
                type=int,
                default=0,
                help="0 means all epochs; use a small value for a smoke test",
            )
            p.add_argument("--lr", type=float, default=0.0001)
            p.add_argument("--seed", type=int, default=1111)
            p.add_argument(
                "--max-vram-gib",
                type=float,
                default=4,
                help="PyTorch allocator cap; CUDA context overhead is additional",
            )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = load_dataset(args.data, args.input, args.accept_assistant_labels)
    summary = dataset_summary(rows)
    if args.command == "validate":
        print(json.dumps(summary, indent=2))
        return 0
    for key in ["batch_size", "width", "threads"]:
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if not args.weights.is_file():
        raise ValueError(
            "Pretrained weights not found; provide a local English gen2 .pth"
        )
    if args.command == "train":
        if (
            args.epochs <= 0
            or args.max_steps < 0
            or not math.isfinite(args.lr)
            or args.lr <= 0
            or not math.isfinite(args.max_vram_gib)
            or args.max_vram_gib <= 0
        ):
            raise ValueError(
                "epochs/lr/VRAM must be positive and max-steps nonnegative"
            )
        if not all(any(r["split"] == s for r in rows) for s in ["train", "val"]):
            raise ValueError(
                "Training needs nonempty, source-disjoint train and validation splits"
            )
    if args.output.exists():
        raise ValueError("Output exists; choose a new run directory")
    if args.desktop_gpu and args.device != "cuda":
        raise ValueError("--desktop-gpu requires --device cuda")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        budget = args.max_vram_gib if args.command == "train" else 4
        for _ in range(3):
            checker = desktop_gpu_capacity if args.desktop_gpu else gpu_idle
            checker(args.gpu_index, math.ceil((budget + 1) * 1024))
            time.sleep(1)
    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; no CPU fallback for a requested GPU job"
            )
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(budget * 1024**3 / total, 1.0), 0
        )
    model, characters = load_recognizer(args.weights)
    model.to(args.device)
    args.output.mkdir(parents=True)
    provenance = {
        "dataset_manifest_sha256": digest(
            contained_path(
                args.data,
                json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))[
                    "manifest"
                ],
            )
        ),
        "weights_sha256": digest(args.weights),
        "dataset": summary,
        "options": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "torch": torch.__version__,
        "decoder": "whole ROI greedy CTC; numeric logit allowlist; no confidence threshold or value-format correction",
    }
    (args.output / "run.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    if args.command == "evaluate":
        subset = [r for r in rows if r["split"] == args.split]
        if not subset:
            raise ValueError("Requested evaluation split is empty")
        result = evaluate_rows(
            model,
            characters,
            subset,
            args.data,
            args.input,
            args.device,
            args.batch_size,
            args.width,
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result["metrics"], indent=2))
        return 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
    best = (-1, -float("inf"))
    steps = 0
    # Training never loads or evaluates the held-out test images into the model.
    try:
        for epoch in range(args.epochs):
            random.shuffle(train)
            model.train()
            losses = []
            for start in range(0, len(train), args.batch_size):
                if args.device == "cuda" and steps % 10 == 0:
                    if args.desktop_gpu:
                        desktop_gpu_capacity(args.gpu_index, 512)
                    else:
                        gpu_no_foreign_processes(args.gpu_index)
                batch = train[start : start + args.batch_size]
                inputs = torch.stack(
                    [
                        image_tensor(
                            contained_path(args.data, r["images"][args.input]),
                            width=args.width,
                        )
                        for r in batch
                    ]
                ).to(args.device)
                targets, lengths = encode_labels(
                    [r["label"] for r in batch], characters
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=args.device, enabled=args.device == "cuda"
                ):
                    logits = model(inputs, None)
                repeats = [
                    sum(a == b for a, b in zip(r["label"], r["label"][1:]))
                    for r in batch
                ]
                if any(
                    len(r["label"]) + repeated > logits.shape[1]
                    for r, repeated in zip(batch, repeats)
                ):
                    raise ValueError(
                        "CTC target exceeds available time steps; increase --width"
                    )
                # FP32 CTC avoids overflow under mixed precision.
                loss = criterion(
                    logits.float().log_softmax(2).transpose(0, 1),
                    targets.to(args.device),
                    torch.full((len(batch),), logits.shape[1], dtype=torch.long),
                    lengths,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite CTC loss; refusing an invalid update")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                losses.append(float(loss.detach()))
                if args.max_steps and steps >= args.max_steps:
                    break
            result = evaluate_rows(
                model,
                characters,
                val,
                args.data,
                args.input,
                args.device,
                args.batch_size,
                args.width,
            )
            metrics = result["metrics"]
            score = (metrics["exact_accuracy"], -metrics["character_error_rate"])
            if score > best:
                best = score
                export_recognizer(model, characters, args.output / "best")
                (args.output / "best-validation.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
            event = {
                "epoch": epoch + 1,
                "steps": steps,
                "mean_loss": sum(losses) / len(losses),
                "validation": metrics,
                "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
                if args.device == "cuda"
                else None,
            }
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        export_recognizer(model, characters, args.output / "last")
    except (RuntimeError, KeyboardInterrupt) as error:
        export_recognizer(model, characters, args.output / "interrupted")
        (args.output / "interrupted.json").write_text(
            json.dumps({"steps": steps, "error": str(error)}), encoding="utf-8"
        )
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"OCR job refused/failed: {error}", file=sys.stderr)
        raise SystemExit(1)
