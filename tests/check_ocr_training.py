import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from utils.ocr_training import (
    ctc_decode,
    encode_labels,
    image_tensor,
    load_dataset,
    numerical_metrics,
    export_recognizer,
)
from tools.ocr_finetune import gpu_idle, gpu_no_foreign_processes, desktop_gpu_capacity
from tools.prepare_ocr_job import prepare


class OCRTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i, split in enumerate(["train", "val", "test"]):
            path = self.root / f"{i}.png"
            Image.new("L", (30 + i, 20), 40 + i).save(path)
            self.rows.append(
                {
                    "id": str(i),
                    "video_id": f"v{i}",
                    "split": split,
                    "machine_id": 1,
                    "region": "region6",
                    "images": {"binary": path.name},
                    "sha256": {"binary": hashlib.sha256(path.read_bytes()).hexdigest()},
                    "label": "-1.2",
                    "status": "reviewed_numeric",
                    "annotation_source": "human_review",
                    "human_verified": True,
                }
            )
        (self.root / "dataset.json").write_text(
            json.dumps({"schema_version": 1, "manifest": "records.jsonl"}),
            encoding="utf-8",
        )
        self.save()

    def save(self):
        (self.root / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8"
        )

    def test_valid_source_disjoint_pack(self):
        self.assertEqual(len(load_dataset(self.root)), 3)

    def test_video_leakage_is_rejected(self):
        self.rows[1]["video_id"] = self.rows[0]["video_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "video crosses"):
            load_dataset(self.root)

    def test_identical_input_across_splits_is_rejected(self):
        self.rows[1]["images"] = self.rows[0]["images"]
        self.rows[1]["sha256"] = self.rows[0]["sha256"]
        self.save()
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            load_dataset(self.root)

    def test_changed_crop_is_rejected(self):
        (self.root / "0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_dataset(self.root)

    def test_traversal_is_rejected(self):
        self.rows[0]["images"]["binary"] = "../outside.png"
        self.save()
        with self.assertRaisesRegex(ValueError, "inside"):
            load_dataset(self.root)

    def test_unreviewed_predictions_are_not_training_labels(self):
        self.rows[0]["status"] = "unreviewed"
        self.save()
        self.assertEqual(len(load_dataset(self.root)), 2)

    def test_assistant_provenance_requires_acknowledgment(self):
        self.rows[0].update(
            human_verified=False, annotation_source="assistant_visual_review"
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "accept-assistant"):
            load_dataset(self.root)
        self.assertEqual(len(load_dataset(self.root, accept_assistant=True)), 3)

    def test_multitarget_not_a_scalar_label(self):
        self.rows[0]["label"] = "15 60"
        self.save()
        with self.assertRaisesRegex(ValueError, "one exact"):
            load_dataset(self.root)

    def test_blank_status_must_be_explicit(self):
        self.rows[0]["label"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "Blank"):
            load_dataset(self.root)
        self.rows[0]["status"] = "reviewed_blank"
        self.save()
        self.assertEqual(len(load_dataset(self.root)), 3)

    def test_ctc_keeps_repeats_after_blank_and_punctuation(self):
        self.assertEqual(ctc_decode([1, 1, 0, 1, 2, 3], "1.-"), "11.-")
        values, lengths = encode_labels(["-1.2", ""], "12.-")
        self.assertEqual(values.tolist(), [4, 1, 3, 2])
        self.assertEqual(lengths.tolist(), [4, 0])

    def test_sign_and_point_have_separate_error_counts(self):
        metrics = numerical_metrics([("-1.2", "12"), ("0", "0"), ("2.2", "22")])
        self.assertEqual(
            (metrics["exact"], metrics["missing_minus"], metrics["missing_decimal"]),
            (1, 1, 2),
        )
        self.assertIsNone(numerical_metrics([])["exact_accuracy"])

    def test_resize_preserves_aspect_and_pads_edge(self):
        import torch

        tensor = image_tensor(self.root / "0.png")
        self.assertEqual(tuple(tensor.shape), (1, 64, 256))
        self.assertTrue(torch.equal(tensor[:, :, 95], tensor[:, :, -1]))
        with self.assertRaisesRegex(ValueError, "squash"):
            image_tensor(self.root / "0.png", width=40)

    @patch("tools.ocr_finetune.subprocess.check_output")
    def test_busy_gpu_refuses_without_terminating_processes(self, call):
        call.side_effect = ["1234\n", "20000, 0"]
        with self.assertRaisesRegex(RuntimeError, "GPU busy"):
            gpu_idle(0, 5120)
        self.assertEqual(call.call_count, 2)

    @patch("tools.ocr_finetune.subprocess.check_output")
    def test_new_workload_causes_our_training_to_stop(self, call):
        call.return_value = "1234\n"
        with self.assertRaisesRegex(RuntimeError, "workload appeared"):
            gpu_no_foreign_processes(0)

    @patch("tools.ocr_finetune.sys.platform", "win32")
    @patch("tools.ocr_finetune.subprocess.check_output", return_value="4500")
    def test_desktop_mode_checks_headroom_without_rejecting_gui(self, call):
        desktop_gpu_capacity(0, 3072)
        self.assertIn("--query-gpu=memory.free", call.call_args.args[0])

    @patch("tools.ocr_finetune.sys.platform", "win32")
    @patch("tools.ocr_finetune.subprocess.check_output", return_value="400")
    def test_desktop_mode_stops_when_headroom_is_low(self, call):
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            desktop_gpu_capacity(0, 512)

    @patch("tools.ocr_finetune.sys.platform", "linux")
    def test_desktop_exception_cannot_disable_linux_gpu_protection(self):
        with self.assertRaisesRegex(ValueError, "Windows"):
            desktop_gpu_capacity(0, 3072)

    def test_prepared_job_contains_only_explicit_inputs(self):
        weights = self.root / "weights.pth"
        weights.write_bytes(b"test weight fixture")
        (self.root / "sources.private.json").write_text("private")
        output = self.root / "job"
        config = prepare(self.root, weights, output)
        self.assertEqual(config["include"], ["tools", "utils", "dataset", "weights"])
        self.assertFalse((output / "dataset/sources.private.json").exists())
        self.assertEqual(len(load_dataset(output / "dataset")), 3)
        self.assertIn("--device cuda", config["command"])
        self.assertEqual(config["setup"], [])
        self.assertFalse(
            json.loads((output / "bundle-provenance.json").read_text())["submitted"]
        )
        with self.assertRaisesRegex(ValueError, "new output"):
            prepare(self.root, weights, output)

    def test_export_loads_in_unmodified_easyocr_reader(self):
        import torch
        import easyocr
        from easyocr.model.vgg_model import Model

        torch.set_num_threads(2)
        characters = "01.-"
        model = Model(1, 256, 256, len(characters) + 1)
        output = self.root / "export"
        export_recognizer(model, characters, output)
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            detector=False,
            quantize=False,
            download_enabled=False,
            verbose=False,
            recog_network="numeric_m12",
            model_storage_directory=str(output),
            user_network_directory=str(output),
        )
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(reader.recognizer.state_dict()[key], tensor))

    def test_onnx_decoder_retains_minus_and_decimal(self):
        import numpy as np
        from tools.benchmark_ocr_onnx import decode_scores, paddle_input

        scores = np.zeros((5, 5), dtype=np.float32)
        scores[np.arange(5), [1, 2, 0, 3, 4]] = 1
        self.assertEqual(decode_scores(scores, "-1.2"), "-1.2")
        self.assertEqual(
            paddle_input(np.zeros((20, 40, 3), np.uint8)).shape, (1, 3, 48, 320)
        )
        with self.assertRaisesRegex(ValueError, "dictionary"):
            decode_scores(scores, "01")

    def test_evaluator_records_the_configured_manifest_hash(self):
        from unittest.mock import MagicMock
        from tools.ocr_finetune import main

        (self.root / "records.jsonl").rename(self.root / "custom.jsonl")
        (self.root / "dataset.json").write_text(
            json.dumps({"schema_version": 1, "manifest": "custom.jsonl"}),
            encoding="utf-8",
        )
        weights = self.root / "fixture.pth"
        weights.write_bytes(b"mocked model fixture")
        output = self.root / "evaluation"
        with (
            patch(
                "tools.ocr_finetune.load_recognizer", return_value=(MagicMock(), "01.-")
            ),
            patch(
                "tools.ocr_finetune.evaluate_rows",
                return_value={"metrics": {}, "predictions": []},
            ),
        ):
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--data",
                        str(self.root),
                        "--weights",
                        str(weights),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(
            run["dataset_manifest_sha256"],
            hashlib.sha256((self.root / "custom.jsonl").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
