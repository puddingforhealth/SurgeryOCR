"""Production route, RGB temporal behavior and bundle regression checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
import threading

import numpy as np

from surgery_analysis_process import OCRProcessor
from tools.install_ocr_bundle import install
from utils.ocr_runtime import (
    decode_scene,
    file_sha256,
    load_bundle,
    resolve_ocr_profile,
    scalar_valid,
    scene_input,
    ConservativeEasyOCR,
    SingleNumericEasyOCR,
    SceneNumericOCR,
    configured_machine_id,
)
from utils.scene_ocr import scene_partitions
from tools.compare_pipeline_runs import (
    value_timeline,
    value_at,
    metrics,
    load_plan,
    stage_rule_allows,
)
from utils.pipeline_utils import FrameSpool


class FakeScene:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def recognize_regions(self, crops):
        if not crops:
            return {}
        self.calls += 1
        value = next(self.values)
        return {
            region: {
                "text": value,
                "raw_text": value,
                "confidence": 0.9,
                "source": "scene_numeric",
                "needs_review": False,
            }
            for region in crops
        }


class RGBTemporalTests(unittest.TestCase):
    def test_standard_trigger_retains_original_threshold_with_exact_cache(self):
        reader = SimpleNamespace(recognize_region=Mock(return_value=("10", 0.9)))
        processor = OCRProcessor(
            {},
            {},
            {},
            ocr_runtime=reader,
            exact_cache_regions=["region1"],
            exact_change_regions=[],
        )
        first = np.zeros((32, 32, 3), np.uint8)
        second = first.copy()
        second[0, 0] = 255
        with (
            patch(
                "surgery_analysis_process.binarize",
                side_effect=lambda image: image[:, :, 0],
            ),
            patch("surgery_analysis_process.is_single_digit", return_value=False),
        ):
            processor.process_frame(first, 0, {"region1": [0, 0, 32, 32]})
            processor.process_frame(second, 1, {"region1": [0, 0, 32, 32]})
        self.assertEqual(reader.recognize_region.call_count, 1)

    def test_exact_recognition_cache_also_covers_single_digit_path_without_losing_events(
        self,
    ):
        reader = SimpleNamespace(
            recognize_region=Mock(
                side_effect=lambda image, region, allowlist: (
                    "1" if image.any() else "0",
                    0.9,
                )
            )
        )
        processor = OCRProcessor(
            {}, {}, {}, ocr_runtime=reader, exact_cache_regions=["region1"]
        )
        with (
            patch(
                "surgery_analysis_process.binarize",
                side_effect=lambda image: image[:, :, 0],
            ),
            patch("surgery_analysis_process.is_single_digit", return_value=True),
        ):
            for frame, value in enumerate([0, 255, 0]):
                processor.process_frame(
                    np.full((4, 4, 3), value, np.uint8),
                    frame,
                    {"region1": [0, 0, 4, 4]},
                )
        self.assertEqual(reader.recognize_region.call_count, 2)
        self.assertEqual(
            [r["ocr_text"] for r in processor.single_digit_results["region1"]],
            ["0", "1", "0"],
        )
        self.assertEqual(
            [r["frame"] for r in processor.single_digit_results["region1"]], [0, 1, 2]
        )
        self.assertEqual(processor.exact_prediction_cache_hits, 1)

    def test_legacy_keeps_original_cache_and_event_contract(self):
        reader = SimpleNamespace(
            recognize=lambda image, allowlist=None: ("10" if image.any() else "0", 0.9)
        )
        with (
            patch("surgery_analysis_process.get_ocr_model", return_value=reader),
            patch(
                "surgery_analysis_process.binarize",
                side_effect=lambda image: image[:, :, 0],
            ),
            patch("surgery_analysis_process.is_single_digit", return_value=False),
        ):
            processor = OCRProcessor({}, {}, {})
            for frame, value in enumerate([0, 255, 0]):
                processor.process_frame(
                    np.full((4, 4, 3), value, np.uint8),
                    frame,
                    {"region1": [0, 0, 4, 4]},
                )
        self.assertEqual(
            processor.multi_digit_groups["region1"],
            [
                {
                    "type": "multi_digit_group",
                    "source_frame": 0,
                    "ocr_text": "0",
                    "confidence": 0.9,
                    "matched_frames": [0, 2],
                    "setting": True,
                },
                {
                    "type": "multi_digit_group",
                    "source_frame": 1,
                    "ocr_text": "10",
                    "confidence": 0.9,
                    "matched_frames": [1],
                    "setting": True,
                },
            ],
        )

    def test_empty_enhanced_region_clears_stale_result_file(self):
        processor, _ = self.processor([""])
        processor.process_frame(np.zeros((4, 4, 3), np.uint8), 0, {"ASP": [0, 0, 4, 4]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ASP_ocr_testing.jsonl"
            path.write_text('{"ocr_text":"stale"}\n', encoding="utf-8")
            processor.save_results(Path(folder))
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_invalid_scalar_clears_display_and_preserves_raw_review(self):
        class InvalidScene:
            def recognize_regions(self, crops):
                return {
                    r: {
                        "text": "",
                        "raw_text": "--10",
                        "confidence": 0.9,
                        "source": "scene_numeric",
                        "needs_review": True,
                    }
                    for r in crops
                }

        processor = OCRProcessor({}, {}, {}, ocr_runtime=InvalidScene())
        processor.initialize_regions({"PEL": [0, 0, 4, 4]})
        processor.last_text["PEL"] = "-10"
        result = processor.process_frame(
            np.zeros((4, 4, 3), np.uint8), 5, {"PEL": [0, 0, 4, 4]}
        )
        self.assertEqual(result["PEL"][0]["type"], "deactivate")
        self.assertIsNone(processor.last_text["PEL"])
        self.assertEqual(processor.review_results[0]["raw_text"], "--10")

    def test_reactivation_of_identical_value_emits_a_fresh_event(self):
        class FakeNumeric:
            def recognize_region(self, image, region, allowlist):
                return "10", 0.9

        processor = OCRProcessor(
            {}, {}, {}, ocr_runtime=FakeNumeric(), exact_cache_regions=["region1"]
        )
        with (
            patch.object(processor, "_check_active", side_effect=[True, False, True]),
            patch(
                "surgery_analysis_process.binarize",
                side_effect=lambda image: image[:, :, 0],
            ),
            patch("surgery_analysis_process.is_single_digit", return_value=False),
        ):
            for frame in range(3):
                processor.process_frame(
                    np.zeros((4, 4, 3), np.uint8), frame, {"region1": [0, 0, 4, 4]}
                )
        self.assertEqual(
            [g["source_frame"] for g in processor.multi_digit_groups["region1"]], [0, 2]
        )
        self.assertEqual(processor.single_digit_results["region1"][0]["frame"], 1)

    def test_scene_layout_resolution_must_match_calibration(self):
        processor = OCRProcessor(
            {}, {}, {}, ocr_runtime=FakeScene(["0"]), scene_frame_size=[1920, 1080]
        )
        with self.assertRaisesRegex(ValueError, "calibrate"):
            processor.process_frame(
                np.zeros((4, 4, 3), np.uint8), 0, {"ASP": [0, 0, 4, 4]}
            )

    def test_discovery_spool_retains_exact_pixels_and_order_when_spilling(self):
        import cv2

        spool = FrameSpool(memory_bytes=1)
        try:
            images = [
                np.random.default_rng(i).integers(0, 256, (20, 30, 3), dtype=np.uint8)
                for i in range(2)
            ]
            for i, image in enumerate(images):
                success, encoded = cv2.imencode(".png", image)
                self.assertTrue(success)
                spool.append((i, encoded))
            self.assertLessEqual(spool.used_bytes, 1)
            self.assertEqual(len(spool), 2)
            for index, encoded in spool:
                np.testing.assert_array_equal(
                    cv2.imdecode(encoded, cv2.IMREAD_COLOR), images[index]
                )
        finally:
            spool.clear()
        self.assertEqual(len(spool), 0)

    def processor(self, values):
        runtime = FakeScene(values)
        processor = OCRProcessor({}, {}, {}, ocr_runtime=runtime, stable_frames=4)
        return processor, runtime

    def test_rgb_changes_are_not_hidden_by_binary_or_stability_gate(self):
        processor, runtime = self.processor(["0.00", "0.01"])
        with patch(
            "surgery_analysis_process.binarize",
            side_effect=AssertionError("RGB path must not binarize"),
        ):
            for frame, value in enumerate([0, 1]):
                processor.process_frame(
                    np.full((4, 4, 3), value, np.uint8), frame, {"CDE": [0, 0, 4, 4]}
                )
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(
            [r["ocr_text"] for r in processor.multi_digit_groups["CDE"]],
            ["0.00", "0.01"],
        )

    def test_identical_rgb_reuses_only_inference(self):
        processor, runtime = self.processor(["0.00"])
        for i in range(3):
            processor.process_frame(
                np.zeros((4, 4, 3), np.uint8), i, {"CDE": [0, 0, 4, 4]}
            )
        self.assertEqual(runtime.calls, 1)
        group = processor.multi_digit_groups["CDE"][0]
        self.assertEqual(group["matched_frames"], [0])
        self.assertEqual(group["last_observed_frame"], 2)
        self.assertEqual(group["observed_frame_count"], 3)

    def test_blank_clears_previous_value_and_reappearance_starts_new_event(self):
        processor, _ = self.processor(["0", "", "0"])
        for i in range(3):
            processor.process_frame(
                np.full((4, 4, 3), i, np.uint8), i, {"ASP": [0, 0, 4, 4]}
            )
        self.assertEqual(
            [r["source_frame"] for r in processor.multi_digit_groups["ASP"]], [0, 2]
        )
        self.assertEqual(processor.single_digit_results["ASP"][0]["type"], "deactivate")
        self.assertEqual(processor.single_digit_results["ASP"][0]["frame"], 1)

    def test_out_of_frame_roi_fails_instead_of_partial_crop(self):
        processor, _ = self.processor(["0"])
        with self.assertRaisesRegex(ValueError, "outside"):
            processor.process_frame(
                np.zeros((4, 4, 3), np.uint8), 0, {"ASP": [0, 0, 5, 4]}
            )

    def test_exact_binary_cache_does_not_hide_dot_sized_change_or_recurrence(self):
        class FakeNumeric:
            def recognize_region(self, image, region, allowlist):
                return ("10" if image.any() else "0"), 0.9

        processor = OCRProcessor(
            {}, {}, {}, ocr_runtime=FakeNumeric(), exact_cache_regions=["region1"]
        )
        with (
            patch(
                "surgery_analysis_process.binarize", side_effect=lambda x: x[:, :, 0]
            ),
            patch("surgery_analysis_process.is_single_digit", return_value=False),
        ):
            images = [np.zeros((32, 32, 3), np.uint8) for _ in range(3)]
            images[1][0, 0] = 255
            for i, image in enumerate(images):
                processor.process_frame(image, i, {"region1": [0, 0, 32, 32]})
        self.assertEqual(
            [r["ocr_text"] for r in processor.multi_digit_groups["region1"]],
            ["0", "10", "0"],
        )
        self.assertEqual(
            [r["source_frame"] for r in processor.multi_digit_groups["region1"]],
            [0, 1, 2],
        )


class BundleTests(unittest.TestCase):
    def test_doctor_in_bare_python_reports_setup_command_without_traceback(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "tools" / "install_ocr_bundle.py"
        result = subprocess.run(
            [sys.executable, "-S", str(script), "--doctor"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup", result.stderr)
        self.assertIn("torch", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_shipped_default_requires_selected_weights_for_all_machines(self):
        from utils.ocr_runtime import default_ocr_profile

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_ocr_profile(), "enhanced")
            for machine in (1, 2, 3):
                profile, _, manifest = resolve_ocr_profile(
                    machine, default_ocr_profile()
                )
                self.assertEqual(profile, "enhanced")
                self.assertEqual(manifest["machine12_route"], "single_numeric")
            with tempfile.TemporaryDirectory() as missing:
                with self.assertRaises(FileNotFoundError):
                    resolve_ocr_profile(1, default_ocr_profile(), missing)

    def test_open_stream_without_decodable_frames_is_not_success(self):
        import run_pipeline as pipeline

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 1
        stage = SimpleNamespace(template_compares=0)
        ocr = SimpleNamespace(
            stats={}, provenance={}, ocr_iface=SimpleNamespace(), save_results=Mock()
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "MachineDetector"),
            patch.object(
                pipeline,
                "_initialize_machine_components",
                return_value=(stage, {}, ocr, {}),
            ),
            patch.object(pipeline.cv2, "VideoCapture", return_value=capture),
            patch.object(pipeline, "iter_video_frames", return_value=iter([])),
        ):
            with self.assertRaisesRegex(RuntimeError, "No decodable frames"):
                pipeline.run_pipeline(
                    Path("fixture.mp4"), Path(folder), mode="ram", machine_id_override=1
                )
            self.assertFalse((Path(folder) / "fixture/run_stats.json").exists())
            ocr.save_results.assert_not_called()
        capture.release.assert_called_once()

    def test_cli_missing_input_and_empty_folder_fail(self):
        import run_pipeline as pipeline
        import sys

        with tempfile.TemporaryDirectory() as folder:
            for target in [Path(folder) / "missing.mp4", Path(folder)]:
                with patch.object(
                    sys, "argv", ["run_pipeline.py", "--video", str(target)]
                ):
                    with self.assertRaises(SystemExit) as error:
                        pipeline.main()
                    self.assertNotEqual(error.exception.code, 0)

    def test_unopenable_video_fails_and_releases_capture(self):
        import run_pipeline as pipeline

        capture = Mock()
        capture.isOpened.return_value = False
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "MachineDetector"),
            patch.object(
                pipeline,
                "_initialize_machine_components",
                return_value=(Mock(), {}, Mock(), {}),
            ),
            patch.object(pipeline.cv2, "VideoCapture", return_value=capture),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot open video"):
                pipeline.run_pipeline(
                    Path("fixture.mp4"), Path(folder), mode="ram", machine_id_override=3
                )
        capture.release.assert_called_once()

    def test_recognizer_exception_does_not_write_success_artifacts(self):
        import run_pipeline as pipeline

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 1
        stage, ocr = Mock(), Mock()
        stage.process_frame.return_value = {"STAGE": {"pattern_id": 1}}
        ocr.process_frame.side_effect = RuntimeError("recognizer failed")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "MachineDetector"),
            patch.object(
                pipeline,
                "_initialize_machine_components",
                return_value=(stage, {}, ocr, {"region1": [0, 0, 4, 4]}),
            ),
            patch.object(pipeline.cv2, "VideoCapture", return_value=capture),
            patch.object(
                pipeline,
                "iter_video_frames",
                return_value=iter([(0, np.zeros((4, 4, 3), np.uint8), 0.0)]),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "recognizer failed"):
                pipeline.run_pipeline(
                    Path("fixture.mp4"), Path(folder), mode="ram", machine_id_override=1
                )
            self.assertFalse((Path(folder) / "fixture/run_stats.json").exists())
            ocr.save_results.assert_not_called()
        capture.release.assert_called_once()

    def test_model_initialization_error_stops_detail_writer(self):
        import run_pipeline as pipeline

        writer = Mock()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "MachineDetector"),
            patch.object(pipeline, "AsyncImageSaver", return_value=writer),
            patch.object(
                pipeline,
                "_initialize_machine_components",
                side_effect=FileNotFoundError("model missing"),
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "model missing"):
                pipeline.run_pipeline(
                    Path("fixture.mp4"),
                    Path(folder),
                    mode="detail",
                    machine_id_override=3,
                )
        writer.stop.assert_called_once()

    def test_explicit_machine_precedes_mapping_without_mutating_it(self):
        mapping = {"clip": 1}
        self.assertEqual(configured_machine_id(mapping, "clip", 3), 3)
        self.assertEqual(mapping, {"clip": 1})
        self.assertEqual(configured_machine_id(mapping, "clip"), 1)
        self.assertIsNone(configured_machine_id(mapping, "unknown"))
        with self.assertRaises(ValueError):
            configured_machine_id(mapping, "clip", 4)

    def test_known_machine_initialization_failure_keeps_original_error(self):
        import run_pipeline as pipeline

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "MachineDetector"),
            patch.object(
                pipeline,
                "_initialize_machine_components",
                side_effect=FileNotFoundError("verified model missing"),
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "verified model missing"):
                pipeline.run_pipeline(
                    Path("fixture.mp4"),
                    Path(folder),
                    mode="ram",
                    ocr_profile="enhanced",
                    machine_id_override=3,
                )

    def test_setup_declares_tested_recognizer_and_cpu_runtime_versions(self):
        root = Path(__file__).resolve().parents[1]
        full = (root / "requirements.txt").read_text(encoding="utf-16")
        release = (root / "requirements_release.txt").read_text(encoding="utf-8")
        self.assertIn("easyocr==1.7.2", full.splitlines())
        self.assertIn(
            "onnxruntime==1.25.1; python_version >= '3.11'", full.splitlines()
        )
        self.assertIn(
            "onnxruntime==1.25.1; python_version >= '3.11'", release.splitlines()
        )

    def test_artifact_keys_cannot_escape_installation_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.bundle(root)
            manifest["artifacts"]["../escape"] = manifest["artifacts"]["machine12"]
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact key"):
                load_bundle(root)

    def test_failed_candidate_does_not_poison_reader_for_next_region(self):
        runtime = ConservativeEasyOCR.__new__(ConservativeEasyOCR)
        runtime.lock = threading.RLock()
        runtime.baseline, runtime.candidate = object(), object()
        runtime.reader = SimpleNamespace(
            reader=SimpleNamespace(recognizer=runtime.baseline), skip_detector=False
        )
        runtime.machine_id = 1
        runtime._read = Mock(
            side_effect=[("10", 0.9), RuntimeError("candidate failed")]
        )
        with self.assertRaisesRegex(RuntimeError, "candidate failed"):
            runtime.recognize_region(np.zeros((4, 4), np.uint8), "region1")
        self.assertIs(runtime.reader.reader.recognizer, runtime.baseline)
        self.assertFalse(runtime.reader.skip_detector)

    def test_untrained_region7_never_uses_numeric_candidate(self):
        runtime = ConservativeEasyOCR.__new__(ConservativeEasyOCR)
        runtime.lock = threading.RLock()
        runtime.baseline, runtime.candidate = object(), object()
        runtime.reader = SimpleNamespace(
            reader=SimpleNamespace(recognizer=runtime.baseline), skip_detector=False
        )
        runtime.machine_id = 1
        runtime._read = Mock(return_value=("12", 0.8))
        self.assertEqual(
            runtime.recognize_region(np.zeros((4, 4), np.uint8), "region7"), ("12", 0.8)
        )
        self.assertEqual(runtime._read.call_count, 1)
        self.assertEqual(runtime.last_decision["source"], "original_untrained_region")

    def test_nonfinite_model_output_is_not_treated_as_a_confident_blank(self):
        runtime = SceneNumericOCR.__new__(SceneNumericOCR)
        runtime.model = None
        runtime.characters = "+-0123456789. "
        runtime.inference_batches = 0
        runtime.input_name = "image"
        runtime.session = SimpleNamespace(
            run=Mock(return_value=[np.full((1, 5, 15), np.nan)])
        )
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            runtime.recognize_regions({"ASP": np.zeros((4, 4, 3), np.uint8)})

    def test_single_numeric_uses_one_model_for_each_region_including_blank(self):
        runtime = SingleNumericEasyOCR.__new__(SingleNumericEasyOCR)
        runtime.lock = threading.RLock()
        runtime.baseline, runtime.candidate = object(), object()
        runtime.reader = SimpleNamespace(reader=SimpleNamespace(recognizer=None))
        paths = []

        def read(image, allowlist):
            paths.append(
                (runtime.reader.reader.recognizer, runtime.reader.skip_detector)
            )
            return "", 0.0

        runtime._read = read
        for region in ("region6", "region7", "region2"):
            self.assertEqual(runtime.recognize_region(None, region), ("", 0.0))
            self.assertEqual(runtime.last_decision, {})
        self.assertEqual(
            paths,
            [
                (runtime.candidate, True),
                (runtime.baseline, False),
                (runtime.candidate, True),
            ],
        )

    def bundle(self, root):
        (root / "weights.pth").write_bytes(b"fixture")
        manifest = {
            "schema_version": 1,
            "version": "fixture",
            "artifacts": {
                "machine12": {
                    "file": "weights.pth",
                    "sha256": file_sha256(root / "weights.pth"),
                }
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_auto_without_bundle_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                resolve_ocr_profile(1, "auto", folder), ("legacy", None, None)
            )

    def test_single_numeric_bundle_requires_explicit_versioned_route(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.bundle(root)
            manifest.update(schema_version=2, machine12_route="single_numeric")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                resolve_ocr_profile(1, "auto", root)[2]["machine12_route"],
                "single_numeric",
            )
            for version, route in ((1, "single_numeric"), (2, None), (2, "unknown")):
                manifest.update(schema_version=version, machine12_route=route)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "machine12_route"):
                    load_bundle(root)

    def test_explicit_enhanced_never_silently_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "install_ocr_bundle"):
                resolve_ocr_profile(1, "enhanced", folder)

    def test_corrupt_auto_bundle_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.bundle(root)
            (root / "weights.pth").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                resolve_ocr_profile(1, "auto", root)

    def test_manifest_cannot_escape_bundle_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.bundle(root)
            manifest["artifacts"]["machine12"]["file"] = "../outside.pth"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_bundle(root)

    def test_install_is_idempotent_and_rechecks_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            self.bundle(source)
            first = install(source, root / "installed")
            second = install(source, root / "installed")
            self.assertEqual(first, second)
            self.assertEqual(len(list((root / "installed").glob("*.pth"))), 1)


class ScalarAndTrainingTests(unittest.TestCase):
    def test_stage_suppression_is_reported_without_changing_visible_value_labels(self):
        stages = {
            "regions": {"STAGE": [{"pattern": 1, "start_frame": 0, "end_frame": 4}]}
        }
        rules = {"region4": [1, 2, 15]}
        self.assertTrue(stage_rule_allows(stages, rules, "region4", 4))
        self.assertFalse(stage_rule_allows(stages, rules, "region4", 5))
        self.assertTrue(stage_rule_allows(stages, rules, "region6", 5))

    def test_comparison_rejects_duplicate_stems_and_path_shaped_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            rows = [
                {"video_id": "a", "machine_id": 1, "source_path": "/one/clip.mp4"},
                {"video_id": "b", "machine_id": 2, "source_path": "/two/clip.mp4"},
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_plan(path)
            rows = [
                {
                    "video_id": "../escape",
                    "machine_id": 1,
                    "source_path": "/one/clip.mp4",
                }
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filename-safe"):
                load_plan(path)

    def test_unknown_scene_field_is_rejected_even_when_blank(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            scalar_valid("", "unconfigured")

    def test_saved_timeline_expands_cache_matches_and_blank_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roi.jsonl"
            rows = [
                {
                    "type": "multi_digit_group",
                    "source_frame": 0,
                    "matched_frames": [0, 4],
                    "ocr_text": "0",
                },
                {"type": "single_digit", "frame": 2, "ocr_text": "10"},
                {"type": "deactivate", "frame": 5, "ocr_text": "(Deactivated)"},
            ]
            path.write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
            timeline = value_timeline(path)
            self.assertEqual(
                [value_at(timeline, f) for f in [0, 1, 2, 3, 4, 5, 6]],
                ["0", "0", "10", "10", "0", "", ""],
            )

    def test_pipeline_comparison_reports_gains_losses_and_blank_hallucinations(self):
        rows = [
            {"id": "gain", "label": "-10", "legacy": "10", "enhanced": "-10"},
            {"id": "loss", "label": "0", "legacy": "0", "enhanced": ""},
            {"id": "blank", "label": "", "legacy": "1", "enhanced": ""},
        ]
        result = metrics(rows)
        self.assertEqual(result["improved"], ["gain", "blank"])
        self.assertEqual(result["regressed"], ["loss"])
        self.assertEqual(result["legacy"]["blank_false_positives"], 1)

    def test_formats_do_not_construct_missing_sign_or_decimal(self):
        for text, region in [
            ("--10", "PEL"),
            ("220", "INCISION"),
            ("120", "CDE"),
            ("1.5", "IOP"),
        ]:
            self.assertFalse(scalar_valid(text, region))
        for text, region in [
            ("", "ASP"),
            ("0", "ASP"),
            ("-0.4", "INFLOW"),
            ("31", "INFLOW"),
            ("0.00", "CDE"),
        ]:
            self.assertTrue(scalar_valid(text, region))

    def test_scene_preprocessing_keeps_bgr_and_padding(self):
        image = np.zeros((20, 40, 3), np.uint8)
        image[:, :, 2] = 255
        tensor = scene_input(image)
        self.assertEqual(tensor.shape, (3, 48, 320))
        self.assertEqual(tensor[0, 0, 0], -1)
        self.assertEqual(tensor[2, 0, 0], 1)
        self.assertEqual(tensor[:, :, 96:].sum(), 0)

    def test_ctc_duplicate_collapse_preserves_repeated_digits_separated_by_blank(self):
        scores = np.eye(3, dtype=np.float32)[[1, 1, 0, 1, 2]]
        self.assertEqual(decode_scene(scores, "12")[0], "112")

    def test_combined_training_has_no_claimed_independent_test(self):
        rows = [
            {"id": str(i), "time_partition": part, "image_sha256": str(i)}
            for i, part in enumerate(["train", "train", "val", "purged_gap"])
        ]
        train, val, test = scene_partitions(rows, "combined")
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 0))


if __name__ == "__main__":
    unittest.main()
