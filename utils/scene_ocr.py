"""RGB scene-text experiment helpers; independent from production OCR routing."""

import copy
import json
from pathlib import Path
import re

from utils.ocr_training import contained_path, digest

CHARACTERS = "+-0123456789. "


def load_scene_rows(root, accept_assistant_labels=False):
    root = Path(root)
    rows = [
        json.loads(line)
        for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["id"])
        if row.get("machine_id") != 3:
            raise ValueError("This experiment requires machine 3 samples")
        if row["status"] not in {"reviewed_numeric", "reviewed_blank"}:
            raise ValueError("Every experiment sample must be reviewed")
        if not row.get("human_verified") and not accept_assistant_labels:
            raise ValueError(
                "Explicitly accept assistant labels or independently verify them"
            )
        label = row["label"]
        if (row["status"] == "reviewed_blank") != (label == ""):
            raise ValueError("Blank status does not match label")
        if label and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", label):
            raise ValueError("Expected a single numeric label")
        if digest(contained_path(root, row["image"])) != row["image_sha256"]:
            raise ValueError("Crop hash changed")
        if row.get("time_partition") not in {"train", "val", "purged_gap"}:
            raise ValueError("Missing temporal partition")
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def video_fold(rows, training_video):
    """Never admit held-out video crops or purged temporal samples into training."""
    videos = {r["video_id"] for r in rows}
    if training_video not in videos or len(videos) != 2:
        raise ValueError("Leave-one-video-out requires exactly two source videos")
    train = [
        r
        for r in rows
        if r["video_id"] == training_video and r["time_partition"] == "train"
    ]
    val = [
        r
        for r in rows
        if r["video_id"] == training_video and r["time_partition"] == "val"
    ]
    test = [r for r in rows if r["video_id"] != training_video]
    if not train or not val or not test:
        raise ValueError("Empty fold partition")
    partitions = [train, val, test]
    for i, left in enumerate(partitions):
        for right in partitions[i + 1 :]:
            if {r["image_sha256"] for r in left} & {r["image_sha256"] for r in right}:
                raise ValueError("Duplicate image crosses experiment partitions")
    return train, val, test


def scene_partitions(rows, training_video):
    """A combined final fit has validation but makes no independent test claim."""
    if training_video != "combined":
        return video_fold(rows, training_video)
    train = [r for r in rows if r["time_partition"] == "train"]
    val = [r for r in rows if r["time_partition"] == "val"]
    if not train or not val:
        raise ValueError("Combined fit requires train and validation rows")
    if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in val}:
        raise ValueError("Duplicate image crosses experiment partitions")
    return train, val, []


def numeric_indices(dictionary):
    if len(dictionary) != len(set(dictionary)):
        raise ValueError("Dictionary has duplicate characters")
    return [0] + [dictionary.index(c) + 1 for c in CHARACTERS]


def final_linear(model):
    head = model.head.ctc_head
    return head.fc2 if hasattr(head, "fc2") else head.fc


def load_scene_model(architecture, weights, dictionary):
    """Transfer pretrained numeric classifier rows as well as the RGB backbone."""
    import torch
    import yaml
    import rapidocr
    from rapidocr.inference_engine.pytorch.networks.architectures.base_model import (
        BaseModel,
    )

    configs = yaml.safe_load(
        (
            Path(rapidocr.__file__).parent
            / "inference_engine/pytorch/networks/arch_config.yaml"
        ).read_text(encoding="utf-8")
    )
    if architecture not in {"PP-OCRv6_rec_tiny", "PP-OCRv6_rec_small"}:
        raise ValueError("Unsupported RGB recognizer architecture")
    config = copy.deepcopy(configs[architecture])
    config["Head"]["out_channels_list"]["CTCLabelDecode"] = len(CHARACTERS) + 1
    model = BaseModel(config)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    source_characters = Path(dictionary).read_text(encoding="utf-8").splitlines() + [
        " "
    ]
    indices = numeric_indices(source_characters)
    prefix = (
        "head.ctc_head.fc2"
        if hasattr(model.head.ctc_head, "fc2")
        else "head.ctc_head.fc"
    )
    if state[prefix + ".weight"].shape[0] != len(source_characters) + 1:
        raise ValueError("Source dictionary does not match classifier")
    for suffix in ["weight", "bias"]:
        state[prefix + "." + suffix] = state[prefix + "." + suffix][indices].clone()
    model.load_state_dict(state, strict=True)
    return model.eval()


def scene_logits(model, inputs):
    """Keep frozen BN statistics; RapidOCR's inference-only MultiHead has no SAR branch."""
    captured = []
    handle = final_linear(model).register_forward_hook(
        lambda _module, _args, output: captured.append(output)
    )
    try:
        model(inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected one classifier output")
    return captured[0]


def scene_tensor(image):
    import numpy as np
    import torch
    from tools.benchmark_ocr_onnx import paddle_input

    pixels = np.asarray(image.convert("RGB"))[:, :, ::-1]
    tensor = paddle_input(pixels)
    if tensor.shape[-1] != 320:
        raise ValueError("Crop too wide for fixed training width; do not squash text")
    return torch.from_numpy(tensor[0].copy())


def export_scene_onnx(model, output):
    import onnx
    import torch

    model = copy.deepcopy(model).cpu().eval()
    output = Path(output)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 48, 320),
        str(output),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(
        graph,
        {
            "character": "\n".join(CHARACTERS[:-1]),
            "input": "BGR RGB-source, no thresholding; 48x320; [-1,1]; right zero padding",
            "scope": "experimental machine 3 numeric recognizer",
        },
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
