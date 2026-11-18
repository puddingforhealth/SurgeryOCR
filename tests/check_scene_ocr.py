"""Behavior checks for isolated RGB numeric OCR experiments."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

import torch
from torch import nn

from utils.scene_ocr import (
    CHARACTERS,
    load_scene_rows,
    numeric_indices,
    scene_logits,
    scene_tensor,
    video_fold,
)
from tools.fetch_scene_ocr_models import fetch_artifact


class ToyRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.ctc_head = nn.Module()
        self.head.ctc_head.fc = nn.Linear(3, len(CHARACTERS) + 1)

    def forward(self, inputs):
        return self.head.ctc_head.fc(inputs).softmax(-1)


class SceneExperimentTests(unittest.TestCase):
    def test_model_download_is_verified_and_idempotent(self):
        payload = b"pinned model fixture"
        item = {
            "url": "https://example.invalid/model",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.onnx"
            with patch(
                "urllib.request.urlopen", return_value=io.BytesIO(payload)
            ) as request:
                fetch_artifact(item, target)
                fetch_artifact(item, target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(request.call_count, 1)

    def test_bad_download_never_becomes_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.onnx"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    fetch_artifact(
                        {"url": "https://example.invalid/model", "sha256": "0" * 64},
                        target,
                    )
            self.assertFalse(target.exists())

    def test_existing_wrong_model_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.onnx"
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "Existing file"):
                fetch_artifact({"sha256": "0" * 64}, target)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_numeric_row_transfer_includes_blank_sign_point_space(self):
        dictionary = list("abcdef0123456789.+- ")
        indices = numeric_indices(dictionary)
        self.assertEqual(indices[0], 0)
        self.assertEqual("".join(dictionary[i - 1] for i in indices[1:]), CHARACTERS)

    def test_duplicate_dictionary_rejected(self):
        with self.assertRaises(ValueError):
            numeric_indices(list(CHARACTERS + "0"))

    def test_logits_preserve_gradients_without_training_mode(self):
        model = ToyRecognizer().eval()
        x = torch.ones(2, 4, 3)
        expected = model.head.ctc_head.fc(x)
        actual = scene_logits(model, x)
        torch.testing.assert_close(actual, expected)
        before = model.head.ctc_head.fc.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        actual.square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head.ctc_head.fc.weight))
        self.assertFalse(model.training)

    def test_forward_hooks_removed_after_exception(self):
        model = ToyRecognizer().eval()
        with self.assertRaises(RuntimeError):
            scene_logits(model, torch.ones(2, 4, 9))
        self.assertEqual(len(model.head.ctc_head.fc._forward_hooks), 0)

    def test_rgb_bgr_preprocessing_preserves_channels_and_padding(self):
        from PIL import Image

        tensor = scene_tensor(Image.new("RGB", (40, 20), (255, 127, 0)))
        self.assertEqual(tuple(tensor.shape), (3, 48, 320))
        self.assertEqual(tensor[0, 0, 0].item(), -1)
        self.assertEqual(tensor[2, 0, 0].item(), 1)
        self.assertEqual(tensor[:, :, 96:].abs().sum().item(), 0)

    def test_wide_inputs_are_rejected_instead_of_squashed(self):
        from PIL import Image

        with self.assertRaises(ValueError):
            scene_tensor(Image.new("RGB", (500, 20)))

    def rows(self):
        return [
            {
                "id": str(i),
                "video_id": v,
                "time_partition": part,
                "image_sha256": str(i),
            }
            for i, (v, part) in enumerate(
                [
                    ("a", "train"),
                    ("a", "val"),
                    ("a", "purged_gap"),
                    ("b", "train"),
                    ("b", "val"),
                ]
            )
        ]

    def test_video_holdout_and_temporal_gap(self):
        train, val, test = video_fold(self.rows(), "a")
        self.assertEqual([r["id"] for r in train], ["0"])
        self.assertEqual([r["id"] for r in val], ["1"])
        self.assertEqual([r["id"] for r in test], ["3", "4"])

    def test_duplicate_content_cannot_cross_partitions(self):
        rows = self.rows()
        rows[-1]["image_sha256"] = rows[0]["image_sha256"]
        with self.assertRaises(ValueError):
            video_fold(rows, "a")

    def test_more_than_two_videos_rejected(self):
        rows = self.rows()
        rows[-1]["video_id"] = "c"
        with self.assertRaises(ValueError):
            video_fold(rows, "a")

    def test_manifest_requires_acknowledgement_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "crop.png"
            image.write_bytes(b"fixture")
            row = {
                "id": "x",
                "machine_id": 3,
                "human_verified": False,
                "status": "reviewed_numeric",
                "label": "-0.1",
                "image": "crop.png",
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "time_partition": "train",
            }
            manifest = root / "labels.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Explicitly accept"):
                load_scene_rows(root)
            self.assertEqual(load_scene_rows(root, True)[0]["label"], "-0.1")
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_scene_rows(root, True)


if __name__ == "__main__":
    unittest.main()
