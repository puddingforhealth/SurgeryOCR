import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.video_display import fit_video_size


def gui_methods(*names):
    tree = ast.parse((Path(__file__).resolve().parents[1]/'video_annotator_gui.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VideoAnnotator')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'Image': Image, 'ImageDraw': ImageDraw, 'ImageTk': SimpleNamespace(PhotoImage=lambda img: img),
          'fit_video_size': fit_video_size}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<gui-display>', 'exec'), ns)
    return ns


class VideoDisplayTests(unittest.TestCase):
    def test_wide_viewport_touches_height(self):
        self.assertEqual(fit_video_size((1920, 1080), (1580, 710)), (1262, 710))

    def test_tall_viewport_touches_width(self):
        self.assertEqual(fit_video_size((1920, 1080), (800, 900)), (800, 450))

    def test_portrait_and_square_video_keep_aspect(self):
        self.assertEqual(fit_video_size((1080, 1920), (1200, 600)), (337, 600))
        self.assertEqual(fit_video_size((500, 500), (1200, 600)), (600, 600))

    def test_small_video_can_grow(self):
        self.assertEqual(fit_video_size((320, 180), (1280, 720)), (1280, 720))

    def test_tiny_and_invalid_dimensions(self):
        self.assertEqual(fit_video_size((1920, 1080), (1, 1)), (1, 1))
        with self.assertRaises(ValueError):
            fit_video_size((0, 1080), (800, 600))

    def test_real_renderer_scales_picture_and_roi_together(self):
        ns = gui_methods('_display_frame_from_image')
        view = SimpleNamespace(lbl_video=Mock(), original_vid_w=200, original_vid_h=100,
            surgery_stage_mode=False, roi_coords=(50, 20, 100, 50), region_name='', master=Mock())
        view.lbl_video.winfo_width.return_value = 800
        view.lbl_video.winfo_height.return_value = 300
        raw = Image.new('RGB', (200, 100), 'black')
        ns['_display_frame_from_image'](view, raw, focus=False)
        output = view.current_display_image
        self.assertEqual(output.size, (600, 300))
        self.assertEqual(output.getpixel((150, 60)), (255, 0, 0))
        self.assertEqual(raw.getpixel((50, 20)), (0, 0, 0))
        view.master.focus_set.assert_not_called()

    def test_paused_resize_reuses_frame_without_advancing_or_seeking(self):
        ns = gui_methods('_redraw_video_viewport')
        frame = Image.new('RGB', (20, 10))
        view = SimpleNamespace(_video_resize_after='timer', _last_video_frame=frame,
            _display_frame_from_image=Mock(), current_frame_idx=123, video_playing=False, video_cap=Mock())
        ns['_redraw_video_viewport'](view)
        view._display_frame_from_image.assert_called_once_with(frame, focus=False)
        view.video_cap.set.assert_not_called()
        view.video_cap.read.assert_not_called()
        self.assertEqual(view.current_frame_idx, 123)
        self.assertFalse(view.video_playing)

    def test_resize_debounces_and_ignores_duplicate_dimensions(self):
        ns = gui_methods('_on_video_resize')
        view = SimpleNamespace(_video_viewport_size=None, _video_resize_after=None,
            after=Mock(return_value='timer'), after_cancel=Mock(), _redraw_video_viewport=Mock())
        ns['_on_video_resize'](view, SimpleNamespace(width=800, height=600))
        ns['_on_video_resize'](view, SimpleNamespace(width=800, height=600))
        self.assertEqual(view.after.call_count, 1)
        ns['_on_video_resize'](view, SimpleNamespace(width=1200, height=600))
        view.after_cancel.assert_called_once_with('timer')


if __name__ == '__main__':
    unittest.main()
