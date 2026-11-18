"""Versioned local OCR model bundles and production numeric recognizers."""

import hashlib
import json
import os
from pathlib import Path
import re
import threading

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "trained_ocr"
SCENE_FIELDS = {"CDE", "IOP", "ASP", "VAC", "PEL", "INFLOW", "INCISION", "REFLUX"}


def default_ocr_profile():
    config = json.loads((ROOT / "config/ocr_profiles.json").read_text(encoding="utf-8"))
    return os.environ.get(
        "SURGERYOCR_OCR_PROFILE", config.get("default_profile", "enhanced")
    )


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def configured_machine_id(mapping, video_name, override=None):
    chosen = override if override is not None else mapping.get(video_name)
    if chosen is None:
        return None
    if str(chosen) not in {"1", "2", "3"}:
        raise ValueError("Machine override/mapping must be 1, 2 or 3")
    return int(chosen)


def load_bundle(path=None):
    root = Path(
        path or os.environ.get("SURGERYOCR_OCR_BUNDLE", DEFAULT_BUNDLE)
    ).resolve()
    manifest = root / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version not in (1, 2):
        raise ValueError("Unsupported OCR bundle schema")
    if version == 2 and data.get("machine12_route") not in {
        "conservative",
        "single_numeric",
    }:
        raise ValueError("Unsupported machine12_route in OCR bundle")
    if version == 1 and data.get("machine12_route", "conservative") != "conservative":
        raise ValueError("Unsupported machine12_route in OCR bundle")
    for name, entry in data["artifacts"].items():
        if name not in {"machine12", "machine3_onnx", "machine3_torchscript"}:
            raise ValueError(f"Unsupported OCR bundle artifact key: {name}")
        if re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is None:
            raise ValueError(f"Invalid OCR artifact SHA-256: {name}")
        target = (root / entry["file"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(
                f"OCR bundle artifact is missing or escapes its root: {name}"
            )
        if file_sha256(target) != entry["sha256"]:
            raise ValueError(f"OCR bundle SHA-256 mismatch: {name}")
    return root, data


def resolve_ocr_profile(machine_id, profile="auto", bundle=None):
    if profile not in {"auto", "legacy", "enhanced"}:
        raise ValueError("OCR profile must be auto, legacy or enhanced")
    if profile == "legacy":
        return "legacy", None, None
    root = Path(bundle or os.environ.get("SURGERYOCR_OCR_BUNDLE", DEFAULT_BUNDLE))
    if profile == "auto" and not (root / "manifest.json").exists():
        return "legacy", None, None
    if str(machine_id) not in {"1", "2", "3"}:
        raise ValueError("Enhanced OCR supports machines 1, 2 and 3")
    try:
        root, manifest = load_bundle(root)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Enhanced OCR bundle missing. Run: python tools/install_ocr_bundle.py --source BUNDLE_DIR"
        ) from error
    required = "machine3_onnx" if str(machine_id) == "3" else "machine12"
    if required not in manifest["artifacts"]:
        raise ValueError(f"OCR bundle lacks {required}")
    return "enhanced", root, manifest


def scalar_valid(text, region):
    if region not in SCENE_FIELDS:
        raise ValueError(f"Unknown machine-3 field: {region}")
    if text == "":
        return True
    if region == "CDE":
        pattern = r"\d+\.\d{2}"
    elif region == "INCISION":
        pattern = r"\d+\.\d"
    elif region == "INFLOW":
        pattern = r"[+-]?\d+(?:\.\d+)?"
    elif region == "PEL":
        pattern = r"[+-]?\d+"
    else:
        pattern = r"\d+"
    return re.fullmatch(pattern, text) is not None


def scene_input(bgr):
    """Match the training RGB-source/BGR 48x320 normalization exactly."""
    import math

    if bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
        raise ValueError("Scene OCR requires a nonempty BGR crop")
    width = math.ceil(48 * bgr.shape[1] / bgr.shape[0])
    if width > 320:
        raise ValueError("Scene OCR crop too wide; refusing to squash text")
    resized = (
        cv2.resize(bgr, (width, 48)).astype(np.float32).transpose(2, 0, 1) / 127.5 - 1
    )
    result = np.zeros((3, 48, 320), dtype=np.float32)
    result[:, :, :width] = resized
    return result


def decode_scene(probabilities, characters):
    indices = probabilities.argmax(-1)
    text, scores, previous = [], [], 0
    for step, index in enumerate(indices):
        index = int(index)
        if index and index != previous:
            text.append(characters[index - 1])
            scores.append(float(probabilities[step, index]))
        previous = index
    confidence = (
        float(np.exp(np.log(np.maximum(scores, 1e-12)).mean()))
        if scores
        else float(probabilities[:, 0].mean())
    )
    return "".join(text), confidence


class SceneNumericOCR:
    """Batched RGB inference, with explicit blank and invalid-value handling."""

    def __init__(self, root, manifest, device="cpu"):
        entry = manifest["artifacts"]["machine3_onnx"]
        runtime_entry = entry
        self.characters = manifest["scene_characters"]
        self.engine = "onnxruntime-cpu"
        self.model = None
        self.inference_batches = 0
        if device == "cuda" and "machine3_torchscript" in manifest["artifacts"]:
            import torch

            self.model = torch.jit.load(
                str(root / manifest["artifacts"]["machine3_torchscript"]["file"]),
                map_location="cuda",
            ).eval()
            runtime_entry = manifest["artifacts"]["machine3_torchscript"]
            self.engine = "torchscript-cuda"
        else:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError(
                    "Enhanced CPU scene OCR needs Python 3.11+ and project dependencies (onnxruntime==1.25.1); install them or select --ocr-profile legacy"
                ) from error

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(root / entry["file"]), options, providers=["CPUExecutionProvider"]
            )
            dictionary = self.session.get_modelmeta().custom_metadata_map[
                "character"
            ].splitlines() + [" "]
            if "".join(dictionary) != self.characters:
                raise ValueError("Scene OCR dictionary differs from bundle manifest")
            self.input_name = self.session.get_inputs()[0].name
        self.provenance = {
            "backend": self.engine,
            "bundle_version": manifest["version"],
            "weights_sha256": runtime_entry["sha256"],
            "onnx_sha256": entry["sha256"],
        }

    def recognize_regions(self, crops):
        if not crops:
            return {}
        self.inference_batches += 1
        batch = np.stack([scene_input(image) for image in crops.values()])
        if self.model is not None:
            import torch

            with torch.inference_mode():
                probabilities = (
                    self.model(torch.from_numpy(batch).to("cuda")).float().cpu().numpy()
                )
        else:
            probabilities = self.session.run(None, {self.input_name: batch})[0]
        if (
            probabilities.ndim != 3
            or probabilities.shape[-1] != len(self.characters) + 1
        ):
            raise ValueError("Unexpected scene OCR output shape")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Non-finite scene OCR probabilities")
        results = {}
        for region, scores in zip(crops, probabilities):
            text, confidence = decode_scene(scores, self.characters)
            valid = scalar_valid(text, region)
            results[region] = {
                "text": text if valid else "",
                "raw_text": text,
                "confidence": confidence,
                "source": "scene_numeric",
                "needs_review": not valid,
                "reason": "invalid_scalar"
                if not valid
                else "blank"
                if not text
                else "numeric",
            }
        return results


class ConservativeEasyOCR:
    """Original CRAFT reader and fine-tuned direct-ROI recognizer, with provenance."""

    handles_retries = True

    def __init__(self, root, manifest, machine_id, device="cpu"):
        from models.OCR_interface.easyocr_model import EasyOCRModel
        from utils.ocr_training import load_recognizer

        self.machine_id = machine_id
        self.reader = EasyOCRModel(
            gpu=device == "cuda",
            lang_list=["en"],
            quantize=False,
            confidence_threshold=0.5,
            debug_output=False,
        )
        self.baseline = self.reader.reader.recognizer
        entry = manifest["artifacts"]["machine12"]
        self.candidate, characters = load_recognizer(root / entry["file"])
        if characters != self.reader.reader.character:
            raise ValueError("Fine-tuned EasyOCR vocabulary does not match the reader")
        self.candidate.to(device).eval()
        self.lock = threading.RLock()
        self.last_decision = {}
        self.provenance = {
            "backend": "easyocr-conservative",
            "bundle_version": manifest["version"],
            "weights_sha256": entry["sha256"],
            "device": device,
        }

    def _read(self, image, allowlist):
        from utils.cv_processing import trim_black_borders

        self.last_call_count += 1
        text, confidence = self.reader.recognize(image, allowlist=allowlist)
        if not text:
            trimmed = trim_black_borders(image, max_border=1)
            if trimmed.size and trimmed.shape != image.shape:
                self.last_call_count += 1
                text, confidence = self.reader.recognize(trimmed, allowlist=allowlist)
        return text, confidence

    def recognize_region(self, image, region, allowlist=None):
        from utils.ocr_prediction_policy import conservative_prediction

        with self.lock:
            self.last_call_count = 0
            try:
                self.reader.reader.recognizer = self.baseline
                self.reader.skip_detector = False
                old, old_conf = self._read(image, allowlist)
                if region not in {f"region{i}" for i in range(1, 7)}:
                    self.last_decision = {
                        "source": "original_untrained_region",
                        "needs_review": False,
                    }
                    return old, old_conf
                self.reader.reader.recognizer = self.candidate
                self.reader.skip_detector = True
                new, new_conf = self._read(image, allowlist)
                self.last_candidate_confidence = new_conf
                self.last_decision = conservative_prediction(
                    old, old_conf, new, new_conf, region, self.machine_id
                )
                self.last_decision.update({"baseline_text": old, "candidate_text": new})
                return self.last_decision["text"], self.last_decision["confidence"]
            finally:
                self.reader.reader.recognizer = self.baseline
                self.reader.skip_detector = False


class SingleNumericEasyOCR(ConservativeEasyOCR):
    """Run one recognizer per region. BSS retains its original recognition path."""

    def __init__(self, root, manifest, machine_id, device="cpu"):
        super().__init__(root, manifest, machine_id, device)
        self.provenance["backend"] = "easyocr-single-numeric"

    def recognize_region(self, image, region, allowlist=None):
        numeric = region in {
            "region1",
            "region2",
            "region3",
            "region4",
            "region5",
            "region6",
        }
        with self.lock:
            self.last_call_count = 0
            self.reader.reader.recognizer = self.candidate if numeric else self.baseline
            self.reader.skip_detector = numeric
            self.last_decision = {}
            return self._read(image, allowlist)


def default_ocr_sensitivity():
    config = json.loads((ROOT / "config/ocr_profiles.json").read_text(encoding="utf-8"))
    return os.environ.get(
        "SURGERYOCR_OCR_SENSITIVITY", config.get("binary_sensitivity", "standard")
    )
