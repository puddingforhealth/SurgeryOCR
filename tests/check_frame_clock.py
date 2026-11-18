import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import frame_clock
from utils.pipeline_utils import iter_video_frames
from check_indexed_video import make_vfr_video

spec = importlib.util.spec_from_file_location(
    "frame_clock_tool", Path(__file__).parents[1] / "tools/frame_clock.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

STEP = 1 / 30


def regular(count, start_frame=0, start_time=0.0):
    return [(start_frame + i, start_time + i * STEP) for i in range(count)]


class SegmentTests(unittest.TestCase):
    def test_regular_video_is_one_segment_matching_index_over_fps(self):
        clock = frame_clock.build_clock("a.mp4", regular(900), container_frames=900)
        self.assertEqual(clock["segments"], [{"frame": 0, "time": 0.0}])
        self.assertAlmostEqual(clock["nominal_fps"], 30.0, places=4)
        self.assertEqual(clock["decoded_frames"], 900)
        self.assertEqual(clock["dropped_frames"], 0)
        self.assertAlmostEqual(frame_clock.frame_time(clock, 450), 15.0, places=6)

    def test_start_stall_shifts_every_later_frame(self):
        # Recorder stall seen in real recordings: frame 46 held 2.5 s, then
        # undecodable pictures; the rest of the video runs late by a fixed amount.
        frames = regular(47) + regular(
            200, start_frame=47, start_time=47 * STEP + 2.956
        )
        clock = frame_clock.build_clock("stall.mp4", frames, container_frames=262)
        self.assertEqual([s["frame"] for s in clock["segments"]], [0, 47])
        self.assertEqual(clock["dropped_frames"], 15)
        for index, time in frames:
            self.assertAlmostEqual(frame_clock.frame_time(clock, index), time, places=5)
        self.assertAlmostEqual(
            frame_clock.frame_time(clock, 100) - 100 / 30, 2.956, places=5
        )

    def test_mid_video_stall_starts_offset_only_there(self):
        frames = regular(1000) + regular(
            500, start_frame=1000, start_time=1000 * STEP + 0.5
        )
        clock = frame_clock.build_clock("mid.mp4", frames)
        self.assertEqual([s["frame"] for s in clock["segments"]], [0, 1000])
        self.assertAlmostEqual(frame_clock.frame_time(clock, 999), 999 / 30, places=6)
        self.assertAlmostEqual(
            frame_clock.frame_time(clock, 1200), 1200 / 30 + 0.5, places=6
        )
        self.assertIsNone(clock["dropped_frames"])

    def test_slow_drift_is_caught_from_segment_start(self):
        # After frame 1000 every gap is only 0.2 % long, so no single gap looks wrong.
        frames = regular(1000) + [
            (1000 + i, 1000 * STEP + i * STEP * 1.002) for i in range(1000)
        ]
        clock = frame_clock.build_clock("drift.mp4", frames)
        self.assertGreater(len(clock["segments"]), 1)
        for index, time in frames:
            self.assertLessEqual(
                abs(frame_clock.frame_time(clock, index) - time), STEP / 2 + 1e-9
            )

    def test_millisecond_timestamps_stay_one_segment_at_true_rate(self):
        # A 1 ms timescale stores 30 fps as 33/33/34 ms gaps.
        frames = [(i, round(i * 1000 / 30) / 1000) for i in range(54000)]
        clock = frame_clock.build_clock("ms.mp4", frames)
        self.assertEqual(len(clock["segments"]), 1)
        self.assertAlmostEqual(clock["nominal_fps"], 30.0, places=2)
        self.assertAlmostEqual(
            frame_clock.frame_time(clock, 53999), frames[-1][1], delta=STEP / 2
        )

    def test_small_jitter_does_not_create_segments(self):
        frames = [(i, i * STEP + (0.002 if i % 2 else -0.002)) for i in range(1, 300)]
        clock = frame_clock.build_clock("jitter.mp4", frames)
        self.assertEqual(len(clock["segments"]), 1)

    def test_odd_short_gaps_do_not_bias_the_rate(self):
        # Seen in real recordings: exact 1/30 s steps with a few 10 ms gaps. Each
        # odd gap is a real shift (one new segment), but it must not skew the rate
        # so that the clock re-anchors every few minutes on its own.
        frames, time = [], 0.0
        for i in range(50000):
            frames.append((i, time))
            time += 0.010 if i in (4000, 30000) else STEP
        clock = frame_clock.build_clock("odd.mp4", frames)
        self.assertAlmostEqual(clock["nominal_fps"], 30.0, places=6)
        self.assertEqual([s["frame"] for s in clock["segments"]], [0, 4001, 30001])

    def test_skipped_index_keeps_formula(self):
        frames = [(i, i * STEP) for i in range(100) if i != 40]
        clock = frame_clock.build_clock("skip.mp4", frames)
        self.assertEqual(len(clock["segments"]), 1)
        self.assertEqual(clock["decoded_frames"], 100)

    def test_nominal_step_ignores_long_gaps(self):
        times = [i * STEP for i in range(10)] + [
            10 * STEP + 3 + i * STEP for i in range(10)
        ]
        self.assertAlmostEqual(frame_clock.nominal_step(times), STEP, places=9)

    def test_too_few_frames_is_an_error(self):
        with self.assertRaises(ValueError):
            frame_clock.build_clock("x.mp4", [])
        with self.assertRaises(ValueError):
            frame_clock.build_clock("x.mp4", [(0, 0.0)])


class ValidationTests(unittest.TestCase):
    def test_round_trip_and_validation(self):
        clock = frame_clock.build_clock("a.mp4", regular(10))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / frame_clock.FILE_NAME
            frame_clock.write_clock(path, clock)
            self.assertEqual(frame_clock.read_clock(path), clock)
            path.write_text(
                json.dumps(
                    {
                        **clock,
                        "segments": [{"frame": 5, "time": 0}, {"frame": 5, "time": 1}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must increase"):
                frame_clock.read_clock(path)

    def test_validate_reports_each_problem(self):
        problems = frame_clock.validate_clock(
            {"version": 2, "nominal_fps": 0, "decoded_frames": -1, "segments": []}
        )
        self.assertEqual(len(problems), 4)
        self.assertEqual(
            frame_clock.validate_clock([]), ["frame_clock must be a JSON object"]
        )


class VideoTests(unittest.TestCase):
    """Real H.264 with a 700 ms timestamp gap and B-frames."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "stall.mp4"
        cls.times_ms = make_vfr_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pipeline_iterator_reports_time_of_returned_frame(self):
        # Regression: the timestamp used to be sampled before read(), one frame late.
        cap = cv2.VideoCapture(str(self.video))
        try:
            seen = [(i, round(ms)) for i, _, ms in iter_video_frames(cap)]
        finally:
            cap.release()
        self.assertEqual(seen, list(enumerate(self.times_ms)))

    def test_scan_video_matches_presentation_times(self):
        frames, container_frames = frame_clock.scan_video(self.video)
        clock = frame_clock.build_clock(self.video.name, frames, container_frames)
        self.assertEqual(clock["decoded_frames"], len(self.times_ms))
        self.assertEqual([s["frame"] for s in clock["segments"]], [0, 6])
        for index, ms in enumerate(self.times_ms):
            self.assertAlmostEqual(
                frame_clock.frame_time(clock, index), ms / 1000, delta=0.002
            )

    def test_tool_writes_into_existing_results_folder_only(self):
        output = self.root / "out"
        output.mkdir(exist_ok=True)
        status, message = tool.process(self.video, output, force=False)
        self.assertEqual(status, "skip", message)
        self.assertFalse((output / "stall").exists())

        (output / "stall").mkdir()
        status, message = tool.process(self.video, output, force=False)
        self.assertEqual(status, "ok", message)
        written = frame_clock.read_clock(output / "stall" / frame_clock.FILE_NAME)
        self.assertEqual(written["source"], "stall.mp4")
        self.assertEqual(len(written["segments"]), 2)

        status, _ = tool.process(self.video, output, force=False)
        self.assertEqual(status, "skip")
        status, _ = tool.process(self.video, output, force=True)
        self.assertEqual(status, "ok")

    def test_tool_refuses_clock_that_disagrees_with_ocr_run(self):
        folder = self.root / "mismatch" / "stall"
        folder.mkdir(parents=True)
        (folder / "run_stats.json").write_text(
            json.dumps({"frames": len(self.times_ms) + 3}), encoding="utf-8"
        )
        status, message = tool.process(self.video, folder.parent, force=False)
        self.assertEqual(status, "fail")
        self.assertIn("run_stats.json reports", message)
        self.assertFalse((folder / frame_clock.FILE_NAME).exists())

        (folder / "run_stats.json").write_text(
            json.dumps({"frames": len(self.times_ms)}), encoding="utf-8"
        )
        status, message = tool.process(self.video, folder.parent, force=False)
        self.assertEqual(status, "ok", message)

    def test_tool_main_accepts_results_folder_named_after_video(self):
        folder = self.root / "named" / "stall"
        folder.mkdir(parents=True)
        self.assertEqual(
            tool.main(["--video", str(self.video), "--output-dir", str(folder)]), 0
        )
        self.assertTrue((folder / frame_clock.FILE_NAME).exists())
        self.assertEqual(tool.main(["--video", str(self.root / "missing.mp4")]), 2)


class ResultsDirTests(unittest.TestCase):
    def test_default_is_folder_beside_video(self):
        video = Path("batch") / "202509" / "a+OD.MP4"
        self.assertEqual(
            tool.results_dir(video, None), Path("batch") / "202509" / "a+OD"
        )

    def test_output_dir_follows_run_pipeline_rule(self):
        video = Path("data") / "0816" / "clip.MP4"
        self.assertEqual(tool.results_dir(video, Path("data")), Path("data") / "clip")
        self.assertEqual(
            tool.results_dir(video, Path("runs") / "clip"), Path("runs") / "clip"
        )


if __name__ == "__main__":
    unittest.main()
