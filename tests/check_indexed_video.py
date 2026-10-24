from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.indexed_video import IndexedVideoCapture, video_index, _index_video


def make_vfr_video(path):
    times = [i*33+(700 if i >= 6 else 0) for i in range(18)]
    with av.open(str(path), 'w') as destination:
        stream = destination.add_stream('libx264', rate=30)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {'crf': '0', 'preset': 'ultrafast', 'g': '5', 'bf': '2'}
        for index, time_ms in enumerate(times):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), index*10, np.uint8), format='bgr24')
            frame.pts = time_ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                destination.mux(packet)
        for packet in stream.encode():
            destination.mux(packet)
    return times


class IndexedVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'vfr.mp4'
        self.times = make_vfr_video(self.path)
        source = cv2.VideoCapture(str(self.path))
        self.frames, self.timestamps = [], []
        while True:
            ok, frame = source.read()
            if not ok:
                break
            self.frames.append(frame)
            self.timestamps.append(source.get(cv2.CAP_PROP_POS_MSEC)/1000)
        source.release()
        self.reader = IndexedVideoCapture(self.path, cache_dir=self.root/'indices', frame_cache_size=3)
        self.addCleanup(self.reader.release)

    def test_random_frame_seeks_match_sequential_decoder_pixels_despite_pts_gap(self):
        for index in (12, 1, 6, 17, 5, 11, 0, 7):
            with self.subTest(frame=index):
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, actual = self.reader.read()
                self.assertTrue(ok)
                np.testing.assert_array_equal(actual, self.frames[index])

    def test_time_mapping_uses_real_pts_and_handles_gap_and_end(self):
        for index, expected in enumerate(self.timestamps):
            self.assertAlmostEqual(self.reader.frame_time(index), expected, places=5)
            self.assertEqual(self.reader.frame_at_time(expected), index)
        self.assertEqual(self.reader.frame_at_time(.5), 5)
        self.assertEqual(self.reader.frame_at_time(999), 17)
        self.assertGreater(self.reader.duration, self.timestamps[-1])

    def test_preview_does_not_change_requested_playback_cursor(self):
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 12)
        self.reader.read()
        self.reader.read_frame(11)
        ok, actual = self.reader.read()
        self.assertTrue(ok)
        np.testing.assert_array_equal(actual, self.frames[13])

    def test_eof_release_and_cache_bounds(self):
        for index in range(18):
            self.reader.read_frame(index)
        self.assertLessEqual(len(self.reader._cache), 3)
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 18)
        self.assertEqual(self.reader.read(), (False, None))
        self.reader.release()
        self.assertFalse(self.reader.isOpened())

    def test_index_cache_is_reused_and_invalidated_when_source_changes(self):
        cache = self.root/'separate-index-cache'
        with patch('utils.indexed_video._index_video', wraps=_index_video) as build:
            video_index(self.path, cache)
            video_index(self.path, cache)
            self.assertEqual(build.call_count, 1)
            self.path.touch()
            video_index(self.path, cache)
            self.assertEqual(build.call_count, 2)

    def test_corrupt_index_cache_is_rebuilt_without_changing_decoded_frame(self):
        self.reader.release()
        next((self.root/'indices').glob('*.gz')).write_bytes(b'broken-cache')
        restored = IndexedVideoCapture(self.path, cache_dir=self.root/'indices')
        self.addCleanup(restored.release)
        np.testing.assert_array_equal(restored.read_frame(12), self.frames[12])


if __name__ == '__main__':
    unittest.main()
