"""Exercise the styled workstation against disposable video/result fixtures."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.get_paths import get_analysis_root, set_analysis_root
from utils.workstation_theme import COLORS
from check_indexed_video import make_vfr_video


class WorkstationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import video_annotator_gui as gui

        cls.gui = gui

    def setUp(self):
        self.logs = io.StringIO()
        self.output = contextlib.redirect_stdout(self.logs)
        self.output.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.old_root, self.overridden = get_analysis_root()
        set_analysis_root(self.folder)
        self.release = patch.object(self.gui, "IS_RELEASE", True)
        self.release.start()
        self.dialogs = patch.multiple(
            self.gui.messagebox,
            showinfo=lambda *a, **k: None,
            showerror=self.fail_dialog,
        )
        self.dialogs.start()
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.dialogs.stop()
            self.release.stop()
            set_analysis_root(self.old_root if self.overridden else None)
            self.temp.cleanup()
            self.output.__exit__(None, None, None)
            raise unittest.SkipTest(f"Tk display unavailable: {error}")
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(
            args
        )
        self.app = self.gui.VideoAnnotator(self.root)
        self.root.geometry("1440x900")
        self.root.update()

    def fail_dialog(self, *args, **kwargs):
        raise AssertionError(f"Unexpected GUI error: {args}")

    def tearDown(self):
        try:
            self.app._stop_playback()
            if self.app.video_cap is not None:
                self.app.video_cap.release()
            for timer in self.root.tk.call("after", "info"):
                # Cancel the Tcl timer only; its owning widget deletes the Python
                # callback during destroy. Root.after_cancel would delete it twice
                # when the callback was registered by an app child widget.
                self.root.tk.call("after", "cancel", timer)
            self.root.destroy()
            self.assertEqual(self.callback_errors, [], self.logs.getvalue())
        finally:
            self.dialogs.stop()
            self.release.stop()
            set_analysis_root(self.old_root if self.overridden else None)
            self.temp.cleanup()
            self.output.__exit__(None, None, None)

    def load_fixture(self):
        video = self.folder / "review.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 64)
        )
        self.assertTrue(writer.isOpened())
        for frame in range(30):
            writer.write(np.full((64, 96, 3), frame * 5, dtype=np.uint8))
        writer.release()
        results = self.folder / "review"
        results.mkdir()
        (results / "_source.json").write_text('{"machine_id": 2}', encoding="utf8")
        (results / "region1_ocr_testing.jsonl").write_text(
            '{"frame": 0, "ocr_text": "25"}\n{"frame": 15, "ocr_text": "30"}\n',
            encoding="utf8",
        )
        segment = {"start_frame": 0, "end_frame": 29, "pattern": 0, "avg_rmse": 0}
        (results / "stage_analysis.json").write_text(
            json.dumps({"regions": {"STAGE": [segment], "PEDAL": [segment]}}),
            encoding="utf8",
        )
        with patch.object(
            self.gui.filedialog, "askopenfilename", return_value=str(video)
        ):
            self.app.btn_load.invoke()
        self.root.update()
        return results

    def test_load_button_populates_ocr_table_machine_and_video(self):
        self.load_fixture()
        self.assertEqual(self.app.analysis_machine_id, 2)
        self.assertEqual(self.app.roi_coords, (1640, 225, 1836, 297))
        self.assertEqual(self.app.video_total_frames, 30)
        self.assertEqual(self.app.tree.set("F15", "content"), "30")
        self.assertGreater(self.app.current_display_image.width(), 1)
        self.assertIn("review.avi", self.app.lbl_video_path.cget("text"))

    def test_transport_buttons_goto_and_keyboard_step_preserve_frame_navigation(self):
        self.load_fixture()
        self.app.goto_var.set(12)
        self.root.focus_force()
        self.app.goto_entry.focus_set()
        self.root.update()
        self.app.goto_entry.event_generate("<Return>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 12)
        self.root.event_generate("<Right>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 13)
        self.app.btn_play.invoke()
        self.assertTrue(self.app.video_playing)
        self.app.btn_play.invoke()
        self.assertFalse(self.app.video_playing)

    def test_mode_tabs_keep_stage_columns_tracks_and_save_visibility(self):
        self.load_fixture()
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        self.app.surgery_stage_combobox.set("STAGE")
        self.app.surgery_stage_combobox.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertTrue(self.app.surgery_stage_mode)
        self.assertEqual(
            tuple(self.app.tree["columns"]),
            ("frame", "content", "end_frame", "iop", "asp", "vac"),
        )
        self.assertFalse(self.app.btn_save.winfo_ismapped())
        self.assertEqual(set(self.app.timeline_tracks), {"PEDAL", "STAGE"})
        self.assertTrue(self.app.timeline_tracks["PEDAL"]["meta"])
        self.app.mode_notebook.select(self.app.ocr_mode_frame)
        self.root.update()
        self.assertFalse(self.app.surgery_stage_mode)
        self.assertTrue(self.app.btn_save.winfo_ismapped())
        self.assertEqual(self.app.tree.set("F15", "content"), "30")

    def test_edit_dialog_and_save_button_write_same_adjacent_result_format(self):
        results = self.load_fixture()
        self.app.tree.selection_set("F15")
        self.root.update()
        self.app._on_edit_annotation()
        self.root.update()
        dialog = next(
            w for w in self.app.winfo_children() if isinstance(w, tk.Toplevel)
        )
        entry = next(w for w in dialog.winfo_children() if isinstance(w, ttk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "31")
        button_frame = next(
            w for w in dialog.winfo_children() if isinstance(w, tk.Frame)
        )
        save = next(
            w for w in button_frame.winfo_children() if w.cget("text") == "保存"
        )
        save.invoke()
        self.assertEqual(self.app.annotations[15], "31")
        self.app.btn_save.invoke()
        records = [
            json.loads(line)
            for line in (results / "region1.jsonl")
            .read_text(encoding="utf8")
            .splitlines()
        ]
        self.assertEqual(
            records, [{"frame": 0, "ocr_text": "25"}, {"frame": 15, "ocr_text": "31"}]
        )

    def test_small_and_large_windows_keep_primary_controls_and_video_accessible(self):
        self.load_fixture()
        self.app._show_frame(12)
        for size in ("1050x720", "1440x900", "1850x1000"):
            self.root.geometry(size)
            self.root.update()
            self.app._redraw_video_viewport()
            for widget in (
                self.app.btn_load,
                self.app.btn_save,
                self.app.btn_play,
                self.app.goto_entry,
                self.app.tree,
                self.app.lbl_video,
            ):
                with self.subTest(size=size, widget=str(widget)):
                    self.assertTrue(widget.winfo_ismapped())
                    self.assertGreaterEqual(
                        widget.winfo_rootx(), self.root.winfo_rootx()
                    )
                    self.assertLessEqual(
                        widget.winfo_rootx() + widget.winfo_width(),
                        self.root.winfo_rootx() + self.root.winfo_width(),
                    )
                    self.assertLessEqual(
                        widget.winfo_rooty() + widget.winfo_height(),
                        self.root.winfo_rooty() + self.root.winfo_height(),
                    )
            self.assertEqual(self.app.current_frame_idx, 12)
            self.assertEqual(self.app.annotation_frame.winfo_width(), 320)
            self.assertGreater(self.app.lbl_video.winfo_height(), 200)

    def test_text_input_retains_arrows_and_space_without_seeking(self):
        self.load_fixture()
        self.app._on_shortcut("goto")
        self.root.focus_force()
        self.app.goto_entry.focus_set()
        self.root.update()
        self.app.goto_entry.delete(0, "end")
        self.app.goto_entry.insert(0, "12")
        self.app.goto_entry.icursor("end")
        frame = self.app.current_frame_idx
        self.app.goto_entry.event_generate("<Left>")
        self.root.update()
        self.assertEqual(self.app.goto_entry.index("insert"), 1)
        self.assertEqual(self.app.current_frame_idx, frame)
        self.app.goto_entry.event_generate("<space>")
        self.root.update()
        self.assertFalse(self.app.video_playing)
        self.assertEqual(self.app.current_frame_idx, frame)
        self.app.goto_entry.event_generate("<Escape>")
        self.root.update()
        self.assertEqual(self.root.focus_get(), self.root)

    def test_tree_arrow_moves_exactly_one_event_and_modifier_shortcuts_work(self):
        self.load_fixture()
        self.app.tree.selection_set("F0")
        self.root.focus_force()
        self.app.tree.focus_set()
        self.root.update()
        self.app.tree.event_generate("<Down>")
        self.root.update()
        self.assertEqual(self.app.tree.selection(), ("F15",))
        self.assertEqual(self.app.current_frame_idx, 15)
        self.app.tree.focus_set()
        self.app.tree.event_generate("<Alt-Key-2>")
        self.root.update()
        self.assertTrue(self.app.surgery_stage_mode)
        region = self.app.surgery_stage_combobox.get()
        self.app.tree.focus_set()
        self.app.tree.event_generate("<Alt-Down>")
        self.root.update()
        self.assertNotEqual(self.app.surgery_stage_combobox.get(), region)

    def test_readout_highlights_manual_changes_but_not_playback_or_disabled_motion(
        self,
    ):
        self.load_fixture()
        self.app._show_frame(12)
        self.assertIsNotNone(self.app._readout_feedback.timer)
        self.assertEqual(self.app.lbl_frame_num.cget("text"), "FRAME  12")
        self.assertEqual(
            self.app.lbl_timecode.cget("text"), "00:00:00.40 / 00:00:01.00"
        )
        self.app._update_frame_readout(13, 13 / 30, animate=False)
        self.assertIsNone(self.app._readout_feedback.timer)
        self.app.motion_enabled.set(False)
        self.app._show_frame(14)
        self.assertIsNone(self.app._readout_feedback.timer)
        self.assertEqual(self.app.lbl_frame_num.cget("bg"), COLORS["panel"])

    def test_shortcut_help_is_reused_and_escape_returns_to_review(self):
        self.app._on_shortcut("help")
        self.root.update()
        dialog = self.app._shortcut_window
        self.app._on_shortcut("help")
        self.assertIs(self.app._shortcut_window, dialog)
        dialog.event_generate("<Escape>")
        self.root.update()
        self.assertFalse(dialog.winfo_exists())
        self.assertEqual(self.root.focus_get(), self.root)

    def test_bound_jump_save_and_boundary_shortcuts(self):
        results = self.load_fixture()
        self.root.focus_force()
        self.root.event_generate("<Control-l>")
        self.root.update()
        self.assertEqual(self.root.focus_get(), self.app.goto_entry)
        self.app.goto_entry.delete(0, "end")
        self.app.goto_entry.insert(0, "12")
        self.app.goto_entry.event_generate("<Return>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 12)
        self.root.event_generate("<Shift-Right>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 29)
        self.root.event_generate("<Home>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 0)
        self.root.event_generate("<End>")
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 29)
        self.root.event_generate("<Control-s>")
        self.root.update()
        self.assertTrue((results / "region1.jsonl").is_file())

    def test_programmatic_stage_selection_does_not_seek_back_to_event_start(self):
        self.load_fixture()
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        self.app._show_frame(20)
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 20)
        self.assertTrue(self.app.tree.selection())
        self.app._show_frame(24)
        self.app._sync_treeview_selection_to_frame(24, force=True)
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 24)

    def test_loading_video_with_late_first_event_still_starts_at_frame_zero(self):
        results = self.load_fixture()
        (results / "region1_ocr_testing.jsonl").write_text(
            '{"frame": 15, "ocr_text": "30"}\n', encoding="utf8"
        )
        self.app._open_video(str(self.folder / "review.avi"))
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 0)
        self.assertEqual(self.app.tree.selection(), ("F15",))

    def test_seek_to_exact_duration_shows_last_valid_frame(self):
        self.load_fixture()
        self.app._jump_to_timestamp(self.app.video_duration)
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 29)
        self.assertEqual(self.app.lbl_frame_num.cget("text"), "FRAME  29")

    def test_playback_image_matches_next_frame_after_roi_preview(self):
        self.load_fixture()
        self.app._show_frame(12)
        self.app.btn_play.invoke()
        self.assertEqual(self.app.current_frame_idx, 13)
        pixel = self.app._last_video_frame.getpixel((48, 32))
        self.assertTrue(all(abs(value - 65) <= 1 for value in pixel), pixel)

    def test_invalid_jump_gives_inline_feedback_and_valid_jump_clears_it(self):
        self.load_fixture()
        self.app._show_frame(12)
        for value in ("abc", "1.5", "-1", "30"):
            self.app.goto_entry.delete(0, "end")
            self.app.goto_entry.insert(0, value)
            self.app._on_goto_frame()
            self.assertTrue(self.app.goto_entry.instate(["invalid"]), value)
            self.assertEqual(self.app.current_frame_idx, 12)
        self.app.goto_entry.delete(0, "end")
        self.app.goto_entry.insert(0, "20")
        self.app._on_goto_frame()
        self.assertFalse(self.app.goto_entry.instate(["invalid"]))
        self.assertEqual(self.app.current_frame_idx, 20)

    def test_focused_button_enter_activates_once_and_save_shortcut_is_nonmodal(self):
        results = self.load_fixture()
        self.root.focus_force()
        self.app.btn_analysis_root.focus_set()
        self.root.update()
        with patch.object(
            self.gui.filedialog, "askdirectory", return_value=""
        ) as chooser:
            self.app.btn_analysis_root.event_generate("<Return>")
            self.root.update()
            chooser.assert_called_once()
        with patch.object(self.gui.messagebox, "showinfo") as notice:
            self.app._on_shortcut("save")
            notice.assert_not_called()
        self.assertTrue((results / "region1.jsonl").exists())
        self.assertIn("已儲存", self.app.status_var.get())

    def test_editor_preserves_time_and_other_columns_and_escape_cancels(self):
        self.load_fixture()
        self.app._configure_treeview_columns("surgery_stage", "STAGE")
        values = ("0.17s(5)", "Phase", "0.97s(29)", "25", "100", "400")
        self.app.tree.insert("", "end", iid="extra", values=values)
        self.app.tree.selection_set("extra")
        self.root.update()
        self.app._on_edit_annotation()
        self.root.update()
        dialog = next(
            w for w in self.app.winfo_children() if isinstance(w, tk.Toplevel)
        )
        entry = next(w for w in dialog.winfo_children() if isinstance(w, ttk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "Edited")
        save_frame = next(w for w in dialog.winfo_children() if isinstance(w, tk.Frame))
        next(
            w for w in save_frame.winfo_children() if w.cget("text") == "保存"
        ).invoke()
        self.assertEqual(
            self.app.tree.item("extra", "values"), (values[0], "Edited", *values[2:])
        )
        self.app.tree.selection_set("extra")
        self.root.update()
        self.app._on_edit_annotation()
        self.root.update()
        dialog = next(
            w for w in self.app.winfo_children() if isinstance(w, tk.Toplevel)
        )
        entry = next(w for w in dialog.winfo_children() if isinstance(w, ttk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "Cancelled")
        dialog.focus_force()
        dialog.event_generate("<Escape>")
        self.root.update()
        self.assertFalse(dialog.winfo_exists())
        self.assertEqual(self.app.tree.set("extra", "content"), "Edited")

    def test_path_hint_uses_full_current_path_stays_onscreen_and_does_not_take_focus(
        self,
    ):
        self.load_fixture()
        hint = next(
            h for h in self.app._ui_hints if h.widget is self.app.lbl_video_path
        )
        self.root.focus_force()
        self.root.update()
        focus = self.root.focus_get()
        hint.show()
        self.root.update()
        label = hint.popup.winfo_children()[0]
        self.assertEqual(label.cget("text"), str(self.folder / "review.avi"))
        self.assertEqual(self.root.focus_get(), focus)
        self.assertGreaterEqual(hint.popup.winfo_rootx(), 0)
        self.assertLessEqual(
            hint.popup.winfo_rootx() + hint.popup.winfo_width(),
            hint.popup.winfo_screenwidth(),
        )
        self.assertLessEqual(
            hint.popup.winfo_rooty() + hint.popup.winfo_height(),
            hint.popup.winfo_screenheight(),
        )
        hint.hide()
        self.assertIsNone(hint.popup)

    def test_event_count_stripes_empty_state_and_source_metadata_stay_current(self):
        self.load_fixture()
        self.assertEqual(self.app.lbl_event_count.cget("text"), "2 筆")
        self.assertIn("review_alternate", self.app.tree.item("F15", "tags"))
        self.assertIn("96×64", self.app.lbl_video_meta.cget("text"))
        self.app.tree.selection_set("F15")
        self.root.update()
        with patch.object(self.gui.messagebox, "askyesno", return_value=True):
            self.app._on_delete_annotation()
        self.assertEqual(self.app.lbl_event_count.cget("text"), "1 筆")
        self.app._clear_previous_video_data()
        self.root.update()
        self.assertEqual(self.app.lbl_event_count.cget("text"), "0 筆")
        self.assertTrue(self.app.empty_events_label.winfo_ismapped())

    def load_vfr_fixture(self):
        path = self.folder / "vfr.mp4"
        make_vfr_video(path)
        sequential = cv2.VideoCapture(str(path))
        self.vfr_expected_frames = []
        while True:
            ok, frame = sequential.read()
            if not ok:
                break
            self.vfr_expected_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sequential.release()
        results = self.folder / "vfr"
        results.mkdir()
        (results / "_source.json").write_text('{"machine_id":2}', encoding="utf8")
        (results / "stage_analysis.json").write_text(
            json.dumps(
                {
                    "video": "vfr",
                    "total_frames": 18,
                    "regions": {
                        "STAGE": [{"start_frame": 6, "end_frame": 12, "pattern": 6}],
                        "PEDAL": [{"start_frame": 0, "end_frame": 17, "pattern": 1}],
                    },
                }
            ),
            encoding="utf8",
        )
        self.app._open_video(str(path))
        self.root.update()

    def test_loading_video_while_in_stage_mode_populates_current_table(self):
        self.load_fixture()
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        self.load_vfr_fixture()
        self.assertEqual(self.app.current_surgery_stage_region, "STAGE")
        self.assertTrue(self.app.tree.get_children())
        self.assertEqual(
            self.app._parse_frame_from_display(
                self.app.tree.item(self.app.tree.get_children()[0], "values")[0]
            ),
            6,
        )
        self.assertIn("STAGE", self.app.timeline_tracks)

    def test_vfr_table_slider_and_time_readout_use_same_exact_frame(self):
        self.load_vfr_fixture()
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        self.app.surgery_stage_combobox.set("STAGE")
        self.app.surgery_stage_combobox.event_generate("<<ComboboxSelected>>")
        self.root.update()
        item = self.app.tree.get_children()[0]
        self.app.tree.selection_set(item)
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 6)
        self.assertTrue(self.app.tree.set(item, "frame").startswith("0.90s"))
        self.assertAlmostEqual(self.app.slider_var.get(), self.app._frame_time(6))
        self.app.slider.set(self.app._frame_time(12))
        self.app._on_slider_release()
        self.root.update()
        self.assertEqual(self.app.current_frame_idx, 12)
        np.testing.assert_array_equal(
            np.asarray(self.app._last_video_frame), self.vfr_expected_frames[12]
        )
        self.assertIn("00:00:01.10", self.app.lbl_timecode.cget("text"))

    def test_vfr_timeline_click_switches_region_and_seeks_exact_segment_start(self):
        from types import SimpleNamespace

        self.load_vfr_fixture()
        track = self.app.timeline_tracks["STAGE"]
        item = next(item for item, meta in track["meta"].items() if meta["start"] == 6)
        x1, y1, x2, y2 = track["canvas"].coords(item)
        self.app._on_track_click(
            SimpleNamespace(x=(x1 + x2) / 2, y=(y1 + y2) / 2), "STAGE"
        )
        self.root.update()
        self.assertTrue(self.app.surgery_stage_mode)
        self.assertEqual(self.app.current_surgery_stage_region, "STAGE")
        self.assertEqual(self.app.current_frame_idx, 6)
        np.testing.assert_array_equal(
            np.asarray(self.app._last_video_frame), self.vfr_expected_frames[6]
        )

    def test_vfr_timeline_geometry_uses_real_seconds_including_timestamp_gap(self):
        self.load_vfr_fixture()
        track = self.app.timeline_tracks["STAGE"]
        item = next(item for item, meta in track["meta"].items() if meta["start"] == 6)
        x1, _, _, _ = track["canvas"].coords(item)
        expected = (
            self.app._frame_time(6)
            / self.app.video_duration
            * (track["canvas"].winfo_width() - 1)
            + 0.5
        )
        self.assertAlmostEqual(x1, expected, places=4)
        self.assertGreater(x1 / (track["canvas"].winfo_width() - 1), 0.6)

    def test_drag_preview_keeps_committed_image_and_readout_until_release(self):
        self.load_vfr_fixture()
        self.app._show_frame(6)
        before = self.app.lbl_timecode.cget("text")
        image = np.asarray(self.app._last_video_frame).copy()
        self.app.slider.set(self.app._frame_time(12))
        self.app._on_slider_move(str(self.app._frame_time(12)))
        self.assertEqual(self.app.current_frame_idx, 6)
        self.assertEqual(self.app.lbl_timecode.cget("text"), before)
        np.testing.assert_array_equal(np.asarray(self.app._last_video_frame), image)
        self.assertIn("定位預覽", self.app.status_var.get())
        self.app._on_slider_release()
        self.assertEqual(self.app.current_frame_idx, 12)
        self.assertIn("已定位", self.app.status_var.get())
        self.assertNotIn("定位預覽", self.app.status_var.get())

    def test_playing_refreshes_roi_pair_and_first_frame_clears_previous_preview(self):
        from PIL import ImageTk

        self.load_fixture()
        self.app.roi_coords = (0, 0, 40, 30)
        self.app._show_frame(12)
        self.app.btn_play.invoke()
        self.assertEqual(self.app.current_frame_idx, 13)
        current = ImageTk.getimage(self.app.current_roi_preview_label.image)
        previous = ImageTk.getimage(self.app.stage_roi_preview_label.image)
        self.assertEqual(current.getpixel((10, 10))[:3], (65, 65, 65))
        self.assertEqual(previous.getpixel((10, 10))[:3], (60, 60, 60))
        self.assertEqual(self.app.video_cap.get(cv2.CAP_PROP_POS_FRAMES), 14)
        self.app._stop_playback()
        self.app._show_frame(0)
        self.assertEqual(self.app.stage_roi_preview_label.cget("image"), "")

    def test_stage_status_checks_recorded_pattern_without_false_latest_cache_alarm(
        self,
    ):
        from unittest.mock import Mock

        self.load_fixture()
        self.app.stage_analysis["regions"]["STAGE"] = [
            {
                "start_frame": 0,
                "end_frame": 29,
                "pattern": 6,
                "pattern_name": "Capsulorhexis",
            }
        ]
        self.app.surgery_stage_mode = True
        self.app.current_surgery_stage_region = "STAGE"
        checker = Mock()
        checker.inspect.return_value = {"state": "match", "rmse": 0}
        self.app._reference_checker = checker
        self.app._show_frame(12)
        self.assertEqual(checker.inspect.call_args.args[1:], (2, "STAGE", 6))
        self.assertIn("Capsulorhexis", self.app.status_var.get())
        self.assertIn("參照一致", self.app.status_var.get())
        self.assertNotIn("Cache MISS", self.app.status_var.get())

    def test_stage_table_accepts_legacy_patterns_dictionary(self):
        self.load_fixture()
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        for region in ("STAGE", "PEDAL"):
            self.app.stage_analysis["regions"][region] = {
                "patterns": self.app.stage_analysis["regions"][region]
            }
        self.app._refresh_treeview()
        self.assertTrue(self.app.tree.get_children())

    def test_clicked_frame_axis_seeks_and_arrows_then_step_single_frames(self):
        self.load_status_fixture()
        ruler = self.app.slider
        x = ruler.x_for(self.app._frame_time(12))
        self.root.focus_force()
        ruler.event_generate("<Button-1>", x=int(round(x)), y=5)
        ruler.event_generate("<ButtonRelease-1>", x=int(round(x)), y=5)
        self.root.update()
        self.assertEqual(
            self.app.current_frame_idx,
            self.app._frame_at_time(self.app.slider_var.get()),
        )
        self.assertIn(self.app.current_frame_idx, (11, 12, 13))
        self.assertIn("已定位", self.app.status_var.get())
        self.assertEqual(
            (self.app._keyboard_panel, self.app._active_track), ("timeline", "FRAME")
        )
        self.assertEqual(self.app.frame_axis_label.cget("text"), "▸ 幀軸 FRAME")
        self.assertEqual(self.app.footer_hint.current[0], "幀軸")
        start = self.app.current_frame_idx
        self.press("<Right>")
        self.assertEqual(self.app.current_frame_idx, start + 1)
        self.press("<Left>")
        self.press("<Left>")
        self.assertEqual(self.app.current_frame_idx, start - 1)
        self.press("<Up>")
        self.assertEqual(self.app._active_track, "FRAME")
        self.press("<Down>")
        self.assertEqual(self.app._active_track, "PEDAL")
        self.press("<Up>")
        self.assertEqual(self.app._active_track, "FRAME")

    def test_frame_axis_and_tracks_share_one_x_scale(self):
        self.load_status_fixture()
        for size in ("1050x720", "1440x900"):
            self.root.geometry(size)
            self.root.update()
            ruler = self.app.slider
            for region, track in self.app.timeline_tracks.items():
                canvas = track["canvas"]
                with self.subTest(size=size, region=region):
                    self.assertEqual(canvas.winfo_rootx(), ruler.winfo_rootx())
                    self.assertEqual(canvas.winfo_width(), ruler.winfo_width())
            label = self.app.frame_axis_label
            self.assertEqual(
                label.winfo_rootx(),
                self.app.timeline_track_labels["PEDAL"].winfo_rootx(),
            )
            self.assertEqual(
                label.winfo_width(),
                self.app.timeline_track_labels["PEDAL"].winfo_width(),
            )

    def test_roi_previews_fit_their_reserved_block(self):
        from PIL import ImageTk

        self.load_status_fixture()
        for size in ("1050x720", "1440x900"):
            self.root.geometry(size)
            self.root.update()
            self.app._show_frame(6)
            self.root.update()
            for label in (
                self.app.stage_roi_preview_label,
                self.app.current_roi_preview_label,
            ):
                image = ImageTk.getimage(label.image)
                with self.subTest(size=size, label=str(label)):
                    self.assertLessEqual(image.width, label.winfo_width())
                    self.assertLessEqual(image.height, label.winfo_height())
            # ROI block: top right, exactly above and as wide as the event list.
            compare, events, video = (
                self.app.roi_compare_frame,
                self.app.annotation_frame,
                self.app.video_card,
            )
            self.assertEqual(compare.winfo_rootx(), events.winfo_rootx())
            self.assertEqual(compare.winfo_width(), events.winfo_width())
            self.assertLess(
                compare.winfo_rooty() + compare.winfo_height(), events.winfo_rooty()
            )
            self.assertGreaterEqual(compare.winfo_height(), 120)
            # The event list is exactly as tall as the video card beside it.
            self.assertEqual(events.winfo_rooty(), video.winfo_rooty())
            self.assertEqual(events.winfo_height(), video.winfo_height())

    def test_selected_tab_is_the_highlighted_one_and_editing_is_explained(self):
        style = ttk.Style(self.root)
        self.assertEqual(
            style.lookup("TNotebook.Tab", "background", ["selected"]),
            COLORS["selection"],
        )
        self.assertEqual(
            style.lookup("TNotebook.Tab", "background"), COLORS["background"]
        )
        self.assertIn("修正標註", self.app.lbl_edit_hint.cget("text"))
        badge, hints = self.app.footer_hint.current
        self.assertEqual(badge, "事件紀錄")
        self.assertIn(("雙擊 / Enter", "修正標註"), hints)

    def test_timestamp_verification_failure_clears_stale_frame_instead_of_showing_wrong_image(
        self,
    ):
        self.load_fixture()
        for navigate in (
            lambda: self.app._show_frame(13),
            lambda: self.app._jump_to_timestamp(self.app._frame_time(13)),
        ):
            self.app._show_frame(12)
            with patch.object(
                self.app.video_cap,
                "read",
                side_effect=self.gui.VideoIndexError("時間戳不一致"),
            ):
                navigate()
            self.assertEqual(self.app.current_frame_idx, 12)
            self.assertAlmostEqual(self.app.slider_var.get(), self.app._frame_time(12))
            self.assertEqual(self.app.lbl_video.cget("image"), "")
            self.assertIsNone(self.app._last_video_frame)
            self.assertIn("時間戳不一致", self.app.status_var.get())

    def load_status_fixture(self, stage=None):
        """30-frame clip with STAGE / PEDAL / AS / STATUS tracks, opened in the status page."""
        results = self.load_fixture()
        stage = stage or [
            {"start_frame": s, "end_frame": s + 9, "pattern": p}
            for s, p in ((0, 1), (10, 2), (20, 3))
        ]
        (results / "stage_analysis.json").write_text(
            json.dumps(
                {
                    "machine_id": 2,
                    "regions": {
                        "STAGE": stage,
                        "PEDAL": [{"start_frame": 0, "end_frame": 29, "pattern": 1}],
                        "AS": [
                            {
                                "start_frame": 3,
                                "end_frame": 25,
                                "pattern": 1,
                                "pattern_name": "AS",
                            }
                        ],
                        "STATUS": [
                            {
                                "start_frame": f,
                                "end_frame": f + 1,
                                "pattern": 1,
                                "pattern_name": "Occlusion",
                            }
                            for f in (5, 15, 24)
                        ],
                    },
                }
            ),
            encoding="utf8",
        )
        self.app.mode_notebook.select(self.app.surgery_mode_frame)
        self.root.update()
        self.app._open_video(str(self.folder / "review.avi"))
        self.root.update()

    def click_track(self, region, frame):
        canvas = self.app.timeline_tracks[region]["canvas"]
        x = (
            self.app._frame_time(frame)
            / self.app.video_duration
            * (canvas.winfo_width() - 1)
            + 2
        )
        self.root.focus_force()
        canvas.event_generate("<Button-1>", x=int(x), y=canvas.winfo_height() // 2)
        self.root.update()

    def press(self, key):
        self.root.focus_force()
        self.root.event_generate(key)
        self.root.update()

    def test_timeline_shows_named_labels_for_every_track(self):
        self.load_status_fixture()
        self.assertEqual(
            list(self.app.timeline_tracks), ["PEDAL", "STAGE", "AS", "STATUS"]
        )
        for region, name in (
            ("PEDAL", "踏板 PEDAL"),
            ("STAGE", "階段 STAGE"),
            ("AS", "AS 圖示"),
            ("STATUS", "Occlusion"),
        ):
            label = self.app.timeline_track_labels[region]
            self.assertEqual(label.cget("text"), name)
            self.assertTrue(label.winfo_ismapped(), region)
            self.assertGreater(label.winfo_width(), 40, region)

    def test_clicked_timeline_takes_arrow_keys_until_escape(self):
        self.load_status_fixture()
        self.click_track("STATUS", 5)
        self.assertEqual(self.app._keyboard_panel, "timeline")
        self.assertEqual(self.app.current_frame_idx, 5)
        self.assertEqual(
            self.app.timeline_track_labels["STATUS"].cget("text"), "▸ Occlusion"
        )
        self.assertIn("方向鍵操作：Occlusion", self.app.timeline_heading.cget("text"))
        self.assertEqual(self.app.footer_hint.current[0], "時間軸")
        self.assertTrue(
            self.app.timeline_tracks["STATUS"]["canvas"].find_withtag("segment_cursor")
        )

        self.press("<Right>")
        self.assertEqual(self.app.current_frame_idx, 15)
        self.assertIn("第 2/3 段", self.app.status_var.get())
        self.press("<Right>")
        self.press("<Right>")
        self.assertEqual(self.app.current_frame_idx, 24)
        self.assertIn("最後一段", self.app.status_var.get())
        self.press("<Left>")
        self.assertEqual(self.app.current_frame_idx, 15)

        self.press("<Up>")
        self.assertEqual(self.app._active_track, "AS")
        self.assertEqual(self.app.surgery_stage_combobox.get(), "AS")
        self.press("<Up>")
        self.assertEqual(self.app._active_track, "STAGE")
        self.press("<Right>")
        self.assertEqual(self.app.current_frame_idx, 20)
        self.press("<Down>")
        self.press("<Down>")
        self.press("<Down>")
        self.assertEqual(self.app._active_track, "STATUS")

        self.press("<Escape>")
        self.assertEqual(self.app._keyboard_panel, "events")
        self.assertFalse(
            self.app.timeline_tracks["STATUS"]["canvas"].find_withtag("segment_cursor")
        )
        self.press("<Right>")
        self.assertEqual(self.app.current_frame_idx, 21)

    def test_clicking_event_list_returns_arrow_keys_to_it(self):
        self.load_status_fixture()
        self.click_track("STAGE", 10)
        self.assertEqual(self.app._keyboard_panel, "timeline")
        self.app.tree.event_generate("<ButtonPress-1>", x=5, y=5)
        self.root.update()
        self.assertEqual(self.app._keyboard_panel, "events")

    def test_stepping_events_scrolls_only_when_the_row_leaves_the_list(self):
        self.load_status_fixture(
            stage=[
                {"start_frame": f, "end_frame": f, "pattern": 1 + f % 3}
                for f in range(30)
            ]
        )
        self.root.geometry("1440x900")
        self.root.update()
        rows = self.app.tree.get_children()
        self.assertEqual(len(rows), 30)
        tree = self.app.tree

        def fully_visible(row):
            # Tk also reports a bbox for the row just below the visible area.
            box = tree.bbox(row)
            return bool(box) and box[1] >= 0 and box[1] + box[3] <= tree.winfo_height()

        tree.selection_set(rows[0])
        self.root.update()
        self.root.focus_force()
        tree.focus_set()
        self.root.update()
        scrolled = 0
        for step in range(1, 29):
            top_before = tree.yview()[0]
            was_visible = fully_visible(rows[step])
            tree.event_generate("<Down>")
            self.root.update()
            self.assertEqual(tree.selection(), (rows[step],))
            self.assertTrue(
                fully_visible(rows[step]), f"row {step} left the visible list"
            )
            if was_visible:
                self.assertEqual(
                    tree.yview()[0], top_before, f"list jumped at row {step}"
                )
            else:
                # The list advances by about one row, never a page or a jump.
                self.assertTrue(
                    0 < tree.yview()[0] - top_before < 1.5 / len(rows),
                    f"list moved {tree.yview()[0] - top_before:.3f} at row {step}",
                )
                scrolled += 1
        self.assertGreater(scrolled, 0)

    def test_page_tabs_name_their_alt_shortcut(self):
        tabs = [
            self.app.mode_notebook.tab(tab, "text")
            for tab in self.app.mode_notebook.tabs()
        ]
        self.assertEqual(tabs, ["數值追蹤  Alt+1", "狀態追蹤  Alt+2"])


class WorkstationContrastTests(unittest.TestCase):
    def test_small_text_and_primary_button_have_readable_contrast(self):
        def luminance(color):
            rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [
                v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb
            ]
            return sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722)))

        for fg, bg in (
            ("text", "panel"),
            ("muted", "panel"),
            ("muted", "background"),
            ("ink", "accent"),
            ("text", "selection"),
        ):
            a, b = sorted((luminance(COLORS[fg]), luminance(COLORS[bg])))
            self.assertGreaterEqual((b + 0.05) / (a + 0.05), 4.5, (fg, bg))


if __name__ == "__main__":
    unittest.main()
