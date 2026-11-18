"""Local OCR dataset contract, CTC preprocessing, and numerical error metrics."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

NUMERIC = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+)?$")
REVIEWED = {"reviewed_numeric", "reviewed_blank"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Image must stay inside the dataset: {relative}")
    return path


def load_dataset(root, input_mode="binary", accept_assistant=False):
    """Validate schema, provenance, hashes, video splits, and cross-split copies."""
    root = Path(root)
    config = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected OCR dataset schema_version 1")
    manifest = contained_path(root, config["manifest"])
    all_rows = [
        json.loads(s)
        for s in manifest.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    seen, videos, hashes, usable = set(), {}, {}, []
    for row in all_rows:
        sid, split = row["id"], row["split"]
        if sid in seen or split not in {"train", "val", "test"}:
            raise ValueError(f"Duplicate ID or invalid split: {sid}")
        seen.add(sid)
        vid = row["video_id"]
        if vid in videos and videos[vid] != split:
            raise ValueError(f"Source video crosses splits: {vid}")
        videos[vid] = split
        for mode, relative in row["images"].items():
            path = contained_path(root, relative)
            if digest(path) != row["sha256"][mode]:
                raise ValueError(f"Image hash mismatch: {sid}/{mode}")
        if row["status"] not in REVIEWED:
            continue
        label = row.get("label")
        if not isinstance(label, str) or (label != "" and not NUMERIC.fullmatch(label)):
            raise ValueError(f"Expected one exact numerical string: {sid}")
        if (label == "") != (row["status"] == "reviewed_blank"):
            raise ValueError(f"Blank label/status disagree: {sid}")
        if not row.get("annotation_source"):
            raise ValueError(f"Missing review provenance: {sid}")
        if not row.get("human_verified", False) and not accept_assistant:
            raise ValueError(
                "Assistant labels require explicit --accept-assistant-labels"
            )
        if input_mode not in row["images"]:
            raise ValueError(f"Missing {input_mode} input: {sid}")
        for mode, value in row["sha256"].items():
            key = (mode, value)
            if key in hashes and hashes[key] != split:
                raise ValueError(f"Identical {mode} image crosses splits: {sid}")
            hashes[key] = split
        usable.append(row)
    return usable


def dataset_summary(rows):
    return {
        "samples": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "videos": {
            s: sorted({r["video_id"] for r in rows if r["split"] == s})
            for s in ("train", "val", "test")
        },
        "negative_samples": sum(r["label"].startswith("-") for r in rows),
        "decimal_samples": sum("." in r["label"] for r in rows),
        "human_verified": sum(bool(r.get("human_verified")) for r in rows),
    }


def ctc_decode(indices, characters):
    result, previous = [], None
    for index in indices:
        index = int(index)
        if index != 0 and index != previous:
            result.append(characters[index - 1])
        previous = index
    return "".join(result)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def numerical_metrics(pairs):
    """Missing sign/point rates use labels containing those symbols as denominator."""
    n = len(pairs)
    negatives = [(a, b) for a, b in pairs if a.startswith("-")]
    decimals = [(a, b) for a, b in pairs if "." in a]
    return {
        "samples": n,
        "exact": sum(a == b for a, b in pairs),
        "exact_accuracy": sum(a == b for a, b in pairs) / n if n else None,
        "character_error_rate": sum(edit_distance(a, b) for a, b in pairs)
        / max(1, sum(len(a) for a, _ in pairs)),
        "negative_samples": len(negatives),
        "missing_minus": sum(not b.startswith("-") for _, b in negatives),
        "decimal_samples": len(decimals),
        "missing_decimal": sum("." not in b for _, b in decimals),
        "empty_predictions": sum(b == "" for _, b in pairs),
    }


def image_tensor(path, height=64, width=256):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("L")
        target_width = math.ceil(height * image.width / image.height)
        if target_width > width:
            raise ValueError(f"Input width would squash text; increase --width: {path}")
        image = image.resize((target_width, height), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    padded = np.pad(pixels, ((0, 0), (0, width - target_width)), mode="edge")
    return torch.from_numpy(padded.copy()).unsqueeze(0)


def load_recognizer(weights):
    import torch
    from easyocr.config import recognition_models
    from easyocr.model.vgg_model import Model

    characters = recognition_models["gen2"]["english_g2"]["characters"]
    model = Model(1, 256, 256, len(characters) + 1)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, characters


def encode_labels(labels, characters):
    import torch

    mapping = {c: i + 1 for i, c in enumerate(characters)}
    return torch.tensor(
        [mapping[c] for s in labels for c in s], dtype=torch.long
    ), torch.tensor([len(s) for s in labels], dtype=torch.long)


def predict(model, tensors, characters, allowlist="+-0123456789."):
    import torch

    logits = model(tensors, None).float()
    if allowlist:
        forbidden = [i + 1 for i, c in enumerate(characters) if c not in allowlist]
        logits[:, :, forbidden] = -torch.inf
    indices = logits.argmax(2).cpu().tolist()
    return [ctc_decode(row, characters) for row in indices]


def export_recognizer(model, characters, output):
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the original alphabet and output head for weight compatibility.
    torch.save(
        {"module." + k: v.detach().cpu() for k, v in model.state_dict().items()},
        output / "numeric_m12.pth",
    )
    config = {
        "imgH": 64,
        "lang_list": ["en"],
        "character_list": characters,
        "network_params": {
            "input_channel": 1,
            "output_channel": 256,
            "hidden_size": 256,
        },
    }
    (output / "numeric_m12.yaml").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "numeric_m12.py").write_text(
        "from easyocr.model.vgg_model import Model\n", encoding="utf-8"
    )
