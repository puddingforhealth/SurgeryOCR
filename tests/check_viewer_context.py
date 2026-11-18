import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.viewer_context import (
    analysis_machine,
    analysis_rois,
    machine_config,
    numeric_summary_regions,
    scene_stage_changes,
)
from utils.get_configs import load_roi_config, load_roi_header_config


class ViewerContextTests(unittest.TestCase):
    def test_saved_layout_machine_wins_over_stale_source_bookkeeping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "_source.json").write_text(
                json.dumps({"machine_id": 1}), encoding="utf8"
            )
            (root / "analysis_layout.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "machine_id": 3,
                        "rois": {"CDE": [1, 2, 3, 4]},
                    }
                ),
                encoding="utf8",
            )
            self.assertEqual(analysis_machine(root, "clip", self.rois), (3, "分析結果"))

    def test_machine3_stage_summary_uses_named_numeric_fields(self):
        self.assertEqual(numeric_summary_regions(3), ("IOP", "ASP", "VAC"))
        self.assertEqual(numeric_summary_regions(1), ("region1", "region2", "region3"))
        self.assertEqual(numeric_summary_regions(2), ("region1", "region2", "region3"))

    def test_constant_scene_value_is_carried_into_stage_with_original_event_frame(self):
        rows = [
            {
                "type": "multi_digit_group",
                "source_frame": 0,
                "matched_frames": [0],
                "last_observed_frame": 20,
                "ocr_text": "0",
                "confidence": 0.99,
            }
        ]
        result = scene_stage_changes(rows, 5, 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            (result[0]["frame"], result[0]["source_frame"], result[0]["text"]),
            (5, 0, "0"),
        )
        self.assertTrue(result[0]["carried"])

    def test_scene_blank_boundary_stops_carry_and_keeps_same_value_reappearance(self):
        rows = [
            {
                "type": "multi_digit_group",
                "source_frame": 0,
                "matched_frames": [0],
                "ocr_text": "0",
            },
            {"type": "deactivate", "frame": 4},
            {
                "type": "multi_digit_group",
                "source_frame": 8,
                "matched_frames": [8],
                "ocr_text": "0",
            },
        ]
        self.assertEqual(scene_stage_changes(rows, 5, 7), [])
        self.assertEqual([r["frame"] for r in scene_stage_changes(rows, 0, 10)], [0, 8])

    def test_scene_summary_expands_legacy_noncontiguous_cache_matches(self):
        rows = [
            {
                "type": "multi_digit_group",
                "source_frame": 0,
                "matched_frames": [0, 8],
                "ocr_text": "0",
            },
            {
                "type": "multi_digit_group",
                "source_frame": 4,
                "matched_frames": [4],
                "ocr_text": "10",
            },
        ]
        self.assertEqual(
            [(r["frame"], r["text"]) for r in scene_stage_changes(rows, 3, 9)],
            [(3, "0"), (4, "10"), (8, "0")],
        )

    def test_real_preloader_loads_configured_machine3_fields_including_cde(self):
        module = ast.parse(
            (Path(__file__).resolve().parents[1] / "video_annotator_gui.py").read_text(
                encoding="utf8"
            )
        )
        cls = next(
            n
            for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == "VideoAnnotator"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_preload_ocr_data"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for region, value in [("CDE", "0.00"), ("IOP", "30")]:
                (root / f"{region}_ocr_testing.jsonl").write_text(
                    json.dumps(
                        {
                            "type": "multi_digit_group",
                            "source_frame": 0,
                            "matched_frames": [0],
                            "ocr_text": value,
                        }
                    )
                    + "\n",
                    encoding="utf8",
                )
            ns = {"resolve_video_analysis_dir": lambda path: root, "json": json}
            exec(
                compile(
                    ast.Module(body=[method], type_ignores=[]),
                    "<real-preloader>",
                    "exec",
                ),
                ns,
            )
            view = SimpleNamespace(
                video_file_path=Path("clip.mp4"),
                roi_dict={"CDE": [], "IOP": []},
                ocr_cache={},
                ocr_data_cache={},
            )
            ns["_preload_ocr_data"](view)
            self.assertEqual(set(view.ocr_data_cache), {"CDE", "IOP"})
            self.assertEqual(view.ocr_cache["CDE"][0], "0.00")

    def test_saved_pipeline_layout_overrides_current_global_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = {"CDE": [10, 20, 30, 40]}
            (root / "analysis_layout.json").write_text(
                json.dumps({"schema_version": 1, "machine_id": 3, "rois": saved}),
                encoding="utf8",
            )
            self.assertEqual(analysis_rois(root, {"CDE": [1, 2, 3, 4]}, 3), saved)
            self.assertEqual(analysis_machine(root, "unknown", self.rois)[0], 3)

    def test_old_machine3_uses_preserved_legacy_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profiles = root / "profiles.json"
            profiles.write_text(
                json.dumps({"legacy_machine3_rois": {"CDE": [10, 20, 30, 40]}}),
                encoding="utf8",
            )
            self.assertEqual(
                analysis_rois(root, {"CDE": [1, 2, 3, 4]}, 3, profiles),
                {"CDE": [10, 20, 30, 40]},
            )
            (root / "run_stats.json").write_text(
                json.dumps({"ocr": {"profile": "enhanced"}}), encoding="utf8"
            )
            self.assertEqual(
                analysis_rois(root, {"CDE": [1, 2, 3, 4]}, 3, profiles),
                {"CDE": [1, 2, 3, 4]},
            )

    def test_layout_machine_mismatch_never_draws_plausible_wrong_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "analysis_layout.json").write_text(
                json.dumps({"schema_version": 1, "machine_id": 2, "rois": {}}),
                encoding="utf8",
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                analysis_rois(root, {}, 3)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rois = self.root / "rois.json"
        self.rois.write_text(
            json.dumps(
                {
                    "video_machine_mapping": {"clip": 1},
                    "regions": {
                        "region1": {
                            "1": [[1640, 325, 1836, 397], [1710, 287, 1767, 313]],
                            "2": [[1640, 225, 1836, 297], [1710, 187, 1767, 213]],
                        },
                        "region4": {"1": [[1, 2, 3, 4], []]},
                    },
                }
            ),
            encoding="utf8",
        )

    def test_cloud_machine_overrides_local_mapping_without_editing_it(self):
        before = self.rois.read_bytes()
        (self.root / "pipeline.log").write_text(
            "✅ 偵測機型: Type 2 (Frame 1043)", encoding="utf8"
        )
        machine, source = analysis_machine(self.root, "clip", self.rois)
        self.assertEqual((machine, source), (2, "分析日誌"))
        self.assertEqual(
            load_roi_config(self.rois, "clip", machine_id=machine),
            {"region1": [1640, 225, 1836, 297]},
        )
        self.assertEqual(
            load_roi_header_config(self.rois, "clip", machine_id=machine),
            {"region1": [1710, 187, 1767, 213]},
        )
        self.assertEqual(self.rois.read_bytes(), before)

    def test_metadata_has_precedence_over_legacy_log(self):
        (self.root / "_source.json").write_text('{"machine_id": 3}')
        (self.root / "pipeline.log").write_text("偵測機型: Type 2", encoding="utf8")
        self.assertEqual(analysis_machine(self.root, "clip", self.rois)[0], 3)

    def test_known_and_fallback_machine_messages(self):
        for message in [
            "ℹ️ 檢測到已知設定: clip (Type 2)，跳過機型偵測。",
            "⚠️ 影片結束仍未偵測到機型，使用預設 Type 2 回溯處理 123 幀",
        ]:
            (self.root / "pipeline.log").write_text(message, encoding="utf8")
            self.assertEqual(analysis_machine(self.root, "clip", self.rois)[0], 2)

    def test_conflicting_log_is_not_silently_accepted(self):
        (self.root / "pipeline.log").write_text(
            "偵測機型: Type 1\n偵測機型: Type 2", encoding="utf8"
        )
        with self.assertRaises(ValueError):
            analysis_machine(self.root, "clip", self.rois)

    def test_no_analysis_uses_existing_mapping_then_default(self):
        self.assertEqual(
            analysis_machine(self.root, "clip", self.rois), (1, "本機設定")
        )
        self.assertEqual(
            analysis_machine(self.root, "unknown", self.rois), (1, "預設（無機型紀錄）")
        )
        self.assertEqual(
            load_roi_config(self.rois, "clip")["region1"], [1640, 325, 1836, 397]
        )

    def test_switching_video_does_not_retain_previous_machine(self):
        other = self.root / "other"
        other.mkdir()
        (self.root / "pipeline.log").write_text("偵測機型: Type 2", encoding="utf8")
        self.assertEqual(analysis_machine(self.root, "clip", self.rois)[0], 2)
        self.assertEqual(analysis_machine(other, "clip", self.rois)[0], 1)

    def test_machine_specific_stage_config_and_shared_fallback(self):
        override = self.root / "machine_3" / "surgery_stage_rois.json"
        override.parent.mkdir()
        override.write_text("{}")
        self.assertEqual(machine_config(self.root, override.name, 3), override)
        self.assertEqual(
            machine_config(self.root, override.name, 2), self.root / override.name
        )

    def test_show_frame_preserves_exact_index_at_fractional_fps(self):
        # Execute the real GUI handler with a thin fake display; no GUI/GPU initialization.
        module = ast.parse(
            (Path(__file__).resolve().parents[1] / "video_annotator_gui.py").read_text(
                encoding="utf8"
            )
        )
        cls = next(
            n
            for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == "VideoAnnotator"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_show_frame"
        )
        ns = {}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "<gui-handler>", "exec"
            ),
            ns,
        )
        view = SimpleNamespace(
            video_total_frames=10000,
            video_fps=30.00098054646733,
            _get_frame_by_index=Mock(return_value=None),
            _get_frame_at_timestamp=Mock(
                side_effect=AssertionError("Frame number converted through time")
            ),
            lbl_video=Mock(),
        )
        for frame in (1, 6964):
            view._get_frame_by_index.reset_mock()
            ns["_show_frame"](view, frame)
            view._get_frame_by_index.assert_called_once_with(frame)


if __name__ == "__main__":
    unittest.main()
