import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.workstation_interactions import (
    EVENTS,
    FRAME_TRACK,
    PANEL_HINTS,
    SHORTCUTS,
    TIMELINE,
    hint_context,
    ReadoutFeedback,
    adjacent_segment,
    blend_color,
    format_timecode,
    navigation_owned_by_widget,
    route_action,
    segment_at,
    step_track,
)
from utils.workstation_theme import COLORS, short_time, tick_step


class Scheduler:
    def __init__(self):
        self.pending = {}
        self.index = 0
        self.elapsed = 0

    def after(self, delay, callback):
        self.index += 1
        self.pending[self.index] = (delay, callback)
        return self.index

    def after_cancel(self, token):
        del self.pending[token]

    def drain(self):
        while self.pending:
            self.assert_bounded()
            token = next(iter(self.pending))
            delay, callback = self.pending.pop(token)
            self.elapsed += delay
            callback()

    def assert_bounded(self):
        if len(self.pending) != 1 or self.elapsed > 320:
            raise AssertionError("Feedback scheduled unbounded animation")


class WorkstationInteractionTests(unittest.TestCase):
    def test_native_inputs_keep_cursor_keys_and_spaces(self):
        for widget in (
            "Entry",
            "TEntry",
            "Text",
            "Spinbox",
            "TSpinbox",
            "TCombobox",
            "TScale",
        ):
            self.assertTrue(navigation_owned_by_widget(widget))
            self.assertTrue(navigation_owned_by_widget(widget, space=True))
        self.assertFalse(navigation_owned_by_widget("Treeview"))

    def test_button_space_does_not_trigger_video_shortcut(self):
        self.assertTrue(navigation_owned_by_widget("Button", space=True))
        self.assertFalse(navigation_owned_by_widget("Button"))

    def test_timecode_handles_hours_rounding_and_negative_values(self):
        self.assertEqual(format_timecode(232.133), "00:03:52.13")
        self.assertEqual(format_timecode(3599.999), "01:00:00.00")
        self.assertEqual(format_timecode(-1), "00:00:00.00")

    def test_fade_returns_to_exact_rest_colors_and_stops_after_320ms(self):
        owner = Scheduler()
        frame = Mock()
        timecode = Mock()
        feedback = ReadoutFeedback(owner, frame, timecode)
        feedback.pulse()
        frame.configure.assert_called_with(bg=COLORS["selection"], fg=COLORS["accent"])
        owner.drain()
        self.assertEqual(owner.elapsed, 320)
        self.assertIsNone(feedback.timer)
        self.assertFalse(owner.pending)
        for label in (frame, timecode):
            label.configure.assert_called_with(bg=COLORS["panel"], fg=COLORS["text"])

    def test_rapid_steps_replace_previous_fade_and_stop_cancels_it(self):
        owner = Scheduler()
        feedback = ReadoutFeedback(owner, Mock())
        feedback.pulse()
        old = feedback.timer
        feedback.pulse()
        self.assertNotIn(old, owner.pending)
        self.assertEqual(len(owner.pending), 1)
        feedback.stop()
        self.assertFalse(owner.pending)

    def test_shortcuts_have_unique_bindings_and_a_visible_reference(self):
        bindings = [s[0] for s in SHORTCUTS]
        self.assertEqual(len(bindings), len(set(bindings)))
        actions = {s[3] for s in SHORTCUTS}
        self.assertTrue({"open", "save", "goto", "ocr", "stage", "help"} <= actions)
        self.assertEqual(blend_color("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(blend_color("#000000", "#ffffff", 1), "#ffffff")


SEGMENTS = [
    {"start_frame": 20, "end_frame": 25},
    {"start_frame": 0, "end_frame": 4},
    {"start_frame": 10, "end_frame": 10},
]


class TimelineNavigationTests(unittest.TestCase):
    def test_arrows_mean_tracks_and_segments_only_on_the_timeline(self):
        self.assertEqual(route_action(TIMELINE, "previous_event"), "previous_track")
        self.assertEqual(route_action(TIMELINE, "next_frame"), "next_segment")
        self.assertEqual(route_action(TIMELINE, "next_second"), "next_second")
        self.assertEqual(route_action(TIMELINE, "play"), "play")
        for action in ("previous_event", "next_frame", "play"):
            self.assertEqual(route_action(EVENTS, action), action)
        # On the frame axis left / right stay single-frame steps.
        self.assertEqual(
            route_action(TIMELINE, "next_frame", FRAME_TRACK), "next_frame"
        )
        self.assertEqual(
            route_action(TIMELINE, "next_event", FRAME_TRACK), "next_track"
        )
        self.assertEqual(hint_context(TIMELINE, FRAME_TRACK), FRAME_TRACK)
        self.assertEqual(hint_context(TIMELINE, "STAGE"), TIMELINE)
        self.assertEqual(hint_context(EVENTS, FRAME_TRACK), EVENTS)
        for context in (EVENTS, TIMELINE, FRAME_TRACK):
            badge, hints = PANEL_HINTS[context]
            self.assertTrue(badge and all(keys and text for keys, text in hints))

    def test_track_steps_stop_at_the_ends(self):
        tracks = ["PEDAL", "STAGE", "AS", "STATUS"]
        self.assertEqual(step_track(tracks, "STAGE", 1), "AS")
        self.assertEqual(step_track(tracks, "PEDAL", -1), "PEDAL")
        self.assertEqual(step_track(tracks, "STATUS", 1), "STATUS")
        self.assertEqual(step_track(tracks, None, 1), "PEDAL")
        self.assertIsNone(step_track([], "STAGE", 1))

    def test_next_and_previous_segment_follow_start_frames_in_unsorted_input(self):
        self.assertEqual(adjacent_segment(SEGMENTS, 0, 1)["start_frame"], 10)
        self.assertEqual(adjacent_segment(SEGMENTS, 12, 1)["start_frame"], 20)
        self.assertIsNone(adjacent_segment(SEGMENTS, 20, 1))
        self.assertEqual(adjacent_segment(SEGMENTS, 20, -1)["start_frame"], 10)
        # Inside a segment, "previous" first returns to that segment's start.
        self.assertEqual(adjacent_segment(SEGMENTS, 23, -1)["start_frame"], 20)
        self.assertIsNone(adjacent_segment(SEGMENTS, 0, -1))
        self.assertIsNone(adjacent_segment([], 5, 1))

    def test_segment_at_includes_both_ends_and_gaps_have_none(self):
        self.assertEqual(segment_at(SEGMENTS, 4)["start_frame"], 0)
        self.assertEqual(segment_at(SEGMENTS, 10)["start_frame"], 10)
        self.assertIsNone(segment_at(SEGMENTS, 7))


class TimeRulerTickTests(unittest.TestCase):
    def test_tick_labels_stay_apart_on_long_and_short_videos(self):
        self.assertEqual(tick_step(2699, 1290), 300)  # 45 min on a 1920 window
        self.assertEqual(tick_step(1, 900), 1)
        for duration, width in ((2699, 900), (60, 1290), (7200, 1770)):
            step = tick_step(duration, width)
            self.assertGreaterEqual(step / duration * (width - 1), 80)

    def test_short_time_labels(self):
        self.assertEqual(short_time(0), "0:00")
        self.assertEqual(short_time(305.9), "5:05")
        self.assertEqual(short_time(3725), "1:02:05")


if __name__ == "__main__":
    unittest.main()
