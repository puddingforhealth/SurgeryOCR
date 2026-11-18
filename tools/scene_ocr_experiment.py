#!/usr/bin/env python3
"""Train a compact RGB recognizer with temporal validation or a video holdout."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_training import (  # noqa: E402
    contained_path,
    ctc_decode,
    digest,
    encode_labels,
    numerical_metrics,
)
from utils.scene_ocr import (  # noqa: E402
    CHARACTERS,
    export_scene_onnx,
    load_scene_model,
    load_scene_rows,
    scene_logits,
    scene_tensor,
    scene_partitions,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate(model, rows, tensors, device, batch_size):
    import torch

    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            outputs = model(torch.stack([tensors[r["id"]] for r in batch]).to(device))
            texts = [
                ctc_decode(indices, CHARACTERS)
                for indices in outputs.argmax(-1).cpu().tolist()
            ]
            predictions.extend(
                {
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "region": r["region"],
                    "label": r["label"],
                    "prediction": text,
                }
                for r, text in zip(batch, texts)
            )
    pairs = [(p["label"], p["prediction"]) for p in predictions]
    return {
        "metrics": numerical_metrics(pairs),
        "blank_false_positives": sum(a == "" and b != "" for a, b in pairs),
        "blank_samples": sum(a == "" for a, _ in pairs),
        "by_region": {
            region: numerical_metrics(
                [
                    (p["label"], p["prediction"])
                    for p in predictions
                    if p["region"] == region
                ]
            )
            for region in sorted({r["region"] for r in rows})
        },
        "predictions": predictions,
    }


def make_synthetic(root, train, fonts, count, seed):
    """Use only training-source/time backgrounds; generated labels are exact by construction."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

    rng = random.Random(seed)
    backgrounds = sorted({r["background"] for r in train})
    samples, provenance = [], []
    for _ in range(count):
        background = rng.choice(backgrounds)
        with Image.open(contained_path(root, background)) as source:
            image = source.convert("RGB").resize((160, 64))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.5, 1.25))
        kind = rng.randrange(6)
        label = (
            ""
            if kind == 0
            else (
                str(rng.randint(-20, 750))
                if kind <= 2
                else (
                    f"{rng.uniform(-1, 65):.1f}"
                    if kind <= 4
                    else f"{rng.uniform(0, 15):.2f}"
                )
            )
        )
        font_path = rng.choice(fonts)
        font = ImageFont.truetype(str(font_path), rng.randint(28, 45))
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), label, font=font)
        while box[2] - box[0] > 148:
            font = ImageFont.truetype(str(font_path), font.size - 1)
            box = draw.textbbox((0, 0), label, font=font)
        xy = (80 + rng.randint(-4, 4), 32 + rng.randint(-3, 3))
        draw.text(
            xy,
            label,
            font=font,
            anchor="mm",
            fill=(rng.randint(220, 255),) * 3,
            stroke_width=rng.choice([0, 1]),
            stroke_fill=(60, 60, 60),
        )
        if rng.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.65)))
        samples.append((scene_tensor(image), label))
        provenance.append(
            {"background": background, "label": label, "font_sha256": digest(font_path)}
        )
    return samples, provenance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=["PP-OCRv6_rec_tiny", "PP-OCRv6_rec_small"],
        required=True,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument(
        "--training-video",
        required=True,
        help="Source ID, or combined for final fitting without an independent test",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; refuses overwrite"
    )
    parser.add_argument("--accept-assistant-labels", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=2)
    parser.add_argument(
        "--desktop-gpu",
        action="store_true",
        help="Explicit local Windows WDDM mode; never shared Linux",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=0,
        help="Fixed generated training samples; no held-out backgrounds",
    )
    parser.add_argument(
        "--font",
        action="append",
        type=Path,
        default=[],
        help="Repeat for each local TrueType font",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new experiment directory")
    if (
        min(args.epochs, args.batch_size, args.threads) < 1
        or args.lr <= 0
        or args.gpu_memory_gib <= 0
        or args.synthetic_count < 0
    ):
        raise ValueError("Invalid training parameters")
    if args.synthetic_count and not args.font:
        raise ValueError("Synthetic training needs at least one --font")
    if args.desktop_gpu and args.device != "cuda":
        raise ValueError("Desktop GPU mode requires CUDA")
    rows = load_scene_rows(args.data, args.accept_assistant_labels)
    train, val, test = scene_partitions(rows, args.training_video)
    if args.device == "cuda":
        from tools.ocr_finetune import desktop_gpu_capacity, gpu_idle

        checker = desktop_gpu_capacity if args.desktop_gpu else gpu_idle
        checker(args.gpu_index, int(args.gpu_memory_gib * 1024) + 1024)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    import torch
    import numpy as np
    from PIL import Image

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.gpu_memory_gib / total_gib), 0
        )
        torch.cuda.reset_peak_memory_stats()
    model = load_scene_model(args.architecture, args.weights, args.dictionary).to(
        args.device
    )
    tensors = {}
    for r in rows:
        with Image.open(contained_path(args.data, r["image"])) as image:
            tensors[r["id"]] = scene_tensor(image)
    synthetic, synthesis_provenance = (
        make_synthetic(args.data, train, args.font, args.synthetic_count, args.seed + 1)
        if args.synthetic_count
        else ([], [])
    )
    args.output.mkdir(parents=True)
    write_json(
        args.output / "plan.json",
        {
            "arguments": {
                k: str(v)
                if isinstance(v, Path)
                else [str(p) for p in v]
                if isinstance(v, list)
                else v
                for k, v in vars(args).items()
            },
            "dataset_sha256": digest(args.data / "labels.jsonl"),
            "weights_sha256": digest(args.weights),
            "dictionary_sha256": digest(args.dictionary),
            "train": [r["id"] for r in train],
            "val": [r["id"] for r in val],
            "test": [r["id"] for r in test],
            "synthetic": synthesis_provenance,
            "selection": "Validation exact accuracy, then CER. Epoch zero is eligible; test evaluated only after checkpoint selection.",
            "independent_test_available": bool(test),
            "batch_norm": "Frozen pretrained statistics; all parameters receive gradients",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    best_result = evaluate(model, val, tensors, args.device, args.batch_size)
    best_score = (
        best_result["metrics"]["exact"],
        -best_result["metrics"]["character_error_rate"],
    )
    write_json(args.output / "validation-epoch-0.json", best_result)
    torch.save(model.state_dict(), args.output / "best.pth")
    best_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
    loss_function = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    rng = random.Random(args.seed)
    history = []
    started = time.perf_counter()
    step = 0
    pool = [(tensors[r["id"]], r["label"]) for r in train] + synthetic
    for epoch in range(1, args.epochs + 1):
        model.eval()
        order = list(range(len(pool)))
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            if args.device == "cuda" and step % 10 == 0:
                from tools.ocr_finetune import (
                    desktop_gpu_capacity,
                    gpu_no_foreign_processes,
                )

                if args.desktop_gpu:
                    desktop_gpu_capacity(args.gpu_index, 512)
                else:
                    gpu_no_foreign_processes(args.gpu_index)
            batch = [pool[i] for i in order[offset : offset + args.batch_size]]
            inputs = torch.stack([t for t, _ in batch]).to(args.device)
            targets, lengths = encode_labels([label for _, label in batch], CHARACTERS)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, enabled=args.device == "cuda"):
                logits = scene_logits(model, inputs)
            loss = loss_function(
                logits.float().log_softmax(-1).transpose(0, 1),
                targets,
                torch.full((len(batch),), logits.shape[1], dtype=torch.long),
                lengths,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CTC loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            step += 1
        result = evaluate(model, val, tensors, args.device, args.batch_size)
        score = (result["metrics"]["exact"], -result["metrics"]["character_error_rate"])
        if score > best_score:
            best_score, best_epoch, best_result = score, epoch, result
            torch.save(model.state_dict(), args.output / "best.pth")
        record = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "validation": result["metrics"],
            "best_epoch": best_epoch,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        write_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)
    torch.save(model.state_dict(), args.output / "last.pth")
    model.load_state_dict(
        torch.load(
            args.output / "best.pth", map_location=args.device, weights_only=True
        )
    )
    write_json(args.output / "selected-validation.json", best_result)
    test_result = evaluate(model, test, tensors, args.device, args.batch_size)
    write_json(args.output / "heldout-test.json", test_result)
    export_scene_onnx(model, args.output / "numeric-scene.onnx")
    write_json(
        args.output / "result.json",
        {
            "best_epoch": best_epoch,
            "steps": step,
            "training_seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
            if args.device == "cuda"
            else None,
            "validation": best_result["metrics"],
            "heldout_test": test_result["metrics"],
            "model_sha256": digest(args.output / "numeric-scene.onnx"),
            "production_enabled": False,
        },
    )
    print("Completed:", args.output, flush=True)


if __name__ == "__main__":
    main()
