import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.workstation_interactions import (
    SHORTCUTS, ReadoutFeedback, blend_color, format_timecode, navigation_owned_by_widget,
)
from utils.workstation_theme import COLORS


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
            raise AssertionError('Feedback scheduled unbounded animation')


class WorkstationInteractionTests(unittest.TestCase):
    def test_native_inputs_keep_cursor_keys_and_spaces(self):
        for widget in ('Entry', 'TEntry', 'Text', 'Spinbox', 'TSpinbox', 'TCombobox', 'TScale'):
            self.assertTrue(navigation_owned_by_widget(widget))
            self.assertTrue(navigation_owned_by_widget(widget, space=True))
        self.assertFalse(navigation_owned_by_widget('Treeview'))

    def test_button_space_does_not_trigger_video_shortcut(self):
        self.assertTrue(navigation_owned_by_widget('Button', space=True))
        self.assertFalse(navigation_owned_by_widget('Button'))

    def test_timecode_handles_hours_rounding_and_negative_values(self):
        self.assertEqual(format_timecode(232.133), '00:03:52.13')
        self.assertEqual(format_timecode(3599.999), '01:00:00.00')
        self.assertEqual(format_timecode(-1), '00:00:00.00')

    def test_fade_returns_to_exact_rest_colors_and_stops_after_320ms(self):
        owner = Scheduler(); frame = Mock(); timecode = Mock()
        feedback = ReadoutFeedback(owner, frame, timecode)
        feedback.pulse()
        frame.configure.assert_called_with(bg=COLORS['selection'], fg=COLORS['accent'])
        owner.drain()
        self.assertEqual(owner.elapsed, 320)
        self.assertIsNone(feedback.timer)
        self.assertFalse(owner.pending)
        for label in (frame, timecode):
            label.configure.assert_called_with(bg=COLORS['panel'], fg=COLORS['text'])

    def test_rapid_steps_replace_previous_fade_and_stop_cancels_it(self):
        owner = Scheduler(); feedback = ReadoutFeedback(owner, Mock())
        feedback.pulse(); old = feedback.timer
        feedback.pulse()
        self.assertNotIn(old, owner.pending)
        self.assertEqual(len(owner.pending), 1)
        feedback.stop()
        self.assertFalse(owner.pending)

    def test_shortcuts_have_unique_bindings_and_a_visible_reference(self):
        bindings = [s[0] for s in SHORTCUTS]
        self.assertEqual(len(bindings), len(set(bindings)))
        actions = {s[3] for s in SHORTCUTS}
        self.assertTrue({'open', 'save', 'goto', 'ocr', 'stage', 'help'} <= actions)
        self.assertEqual(blend_color('#000000', '#ffffff', 0), '#000000')
        self.assertEqual(blend_color('#000000', '#ffffff', 1), '#ffffff')


if __name__ == '__main__':
    unittest.main()
