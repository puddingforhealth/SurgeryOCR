import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.viewer_context import analysis_machine, machine_config
from utils.get_configs import load_roi_config, load_roi_header_config


class ViewerContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rois = self.root / 'rois.json'
        self.rois.write_text(json.dumps({'video_machine_mapping': {'clip': 1}, 'regions': {
            'region1': {'1': [[1640, 325, 1836, 397], [1710, 287, 1767, 313]],
                        '2': [[1640, 225, 1836, 297], [1710, 187, 1767, 213]]},
            'region4': {'1': [[1, 2, 3, 4], []]}}}), encoding='utf8')

    def test_cloud_machine_overrides_local_mapping_without_editing_it(self):
        before = self.rois.read_bytes()
        (self.root/'pipeline.log').write_text('✅ 偵測機型: Type 2 (Frame 1043)', encoding='utf8')
        machine, source = analysis_machine(self.root, 'clip', self.rois)
        self.assertEqual((machine, source), (2, '分析日誌'))
        self.assertEqual(load_roi_config(self.rois, 'clip', machine_id=machine),
                         {'region1': [1640, 225, 1836, 297]})
        self.assertEqual(load_roi_header_config(self.rois, 'clip', machine_id=machine),
                         {'region1': [1710, 187, 1767, 213]})
        self.assertEqual(self.rois.read_bytes(), before)

    def test_metadata_has_precedence_over_legacy_log(self):
        (self.root/'_source.json').write_text('{"machine_id": 3}')
        (self.root/'pipeline.log').write_text('偵測機型: Type 2', encoding='utf8')
        self.assertEqual(analysis_machine(self.root, 'clip', self.rois)[0], 3)

    def test_known_and_fallback_machine_messages(self):
        for message in ['ℹ️ 檢測到已知設定: clip (Type 2)，跳過機型偵測。',
                        '⚠️ 影片結束仍未偵測到機型，使用預設 Type 2 回溯處理 123 幀']:
            (self.root/'pipeline.log').write_text(message, encoding='utf8')
            self.assertEqual(analysis_machine(self.root, 'clip', self.rois)[0], 2)

    def test_conflicting_log_is_not_silently_accepted(self):
        (self.root/'pipeline.log').write_text('偵測機型: Type 1\n偵測機型: Type 2', encoding='utf8')
        with self.assertRaises(ValueError):
            analysis_machine(self.root, 'clip', self.rois)

    def test_no_analysis_uses_existing_mapping_then_default(self):
        self.assertEqual(analysis_machine(self.root, 'clip', self.rois), (1, '本機設定'))
        self.assertEqual(analysis_machine(self.root, 'unknown', self.rois), (1, '預設（無機型紀錄）'))
        self.assertEqual(load_roi_config(self.rois, 'clip')['region1'], [1640, 325, 1836, 397])

    def test_switching_video_does_not_retain_previous_machine(self):
        other = self.root/'other'; other.mkdir()
        (self.root/'pipeline.log').write_text('偵測機型: Type 2', encoding='utf8')
        self.assertEqual(analysis_machine(self.root, 'clip', self.rois)[0], 2)
        self.assertEqual(analysis_machine(other, 'clip', self.rois)[0], 1)

    def test_machine_specific_stage_config_and_shared_fallback(self):
        override = self.root/'machine_3'/'surgery_stage_rois.json'
        override.parent.mkdir(); override.write_text('{}')
        self.assertEqual(machine_config(self.root, override.name, 3), override)
        self.assertEqual(machine_config(self.root, override.name, 2), self.root/override.name)

    def test_show_frame_preserves_exact_index_at_fractional_fps(self):
        # Execute the real GUI handler with a thin fake display; no GUI/GPU initialization.
        module = ast.parse((Path(__file__).resolve().parents[1]/'video_annotator_gui.py').read_text(encoding='utf8'))
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'VideoAnnotator')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_show_frame')
        ns = {}; exec(compile(ast.Module(body=[method], type_ignores=[]), '<gui-handler>', 'exec'), ns)
        view = SimpleNamespace(video_total_frames=10000, video_fps=30.00098054646733,
            _get_frame_by_index=Mock(return_value=None),
            _get_frame_at_timestamp=Mock(side_effect=AssertionError('Frame number converted through time')),
            lbl_video=Mock())
        for frame in (1, 6964):
            view._get_frame_by_index.reset_mock()
            ns['_show_frame'](view, frame)
            view._get_frame_by_index.assert_called_once_with(frame)


if __name__ == '__main__':
    unittest.main()
