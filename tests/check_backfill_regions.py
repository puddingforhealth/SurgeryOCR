import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import av
import cv2
import numpy as np

from utils.cv_processing import hsv_mask
from utils.idle_gate import START, STOP, IdleGate, others_percent
from utils import frame_clock

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "backfill_regions", REPO / "tools/backfill_regions.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

AS_BOX = (1738, 542, 1838, 640)  # machine 2
STATUS_BOX = {1: (1628, 682, 1836, 718), 2: (1628, 664, 1836, 700)}
CACHE = REPO / "data/roi_img_caches/machine_1_2"


def paint_template(frame, box, template, ink, panel):
    """Draw a pattern template into a frame: template pixels in `ink` on `panel`."""
    x1, y1, x2, y2 = box
    patch = np.empty((y2 - y1, x2 - x1, 3), np.uint8)
    patch[:] = panel
    patch[template > 0] = ink
    frame[y1:y2, x1:x2] = patch


def make_clip(
    path,
    machine,
    as_frames=(),
    occlusion_frames=(),
    vent_frames=(),
    frames=12,
    vent_panel=None,
):
    """1080p clip: dark background, AS icon / Occlusion panel / Vent panel on chosen frames."""
    as_icon = np.load(CACHE / "AS/1.npy")
    occlusion = np.load(CACHE / "STATUS/1.npy")
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv444p"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i in range(frames):
            frame = np.full((1080, 1920, 3), (20, 30, 60), np.uint8)
            if i in as_frames:
                paint_template(
                    frame, AS_BOX, as_icon, ink=(235, 235, 235), panel=(110, 60, 20)
                )
            if i in occlusion_frames:
                paint_template(
                    frame,
                    STATUS_BOX[machine],
                    occlusion,
                    ink=(30, 30, 220),
                    panel=(190, 190, 190),
                )
            if i in vent_frames:
                x1, y1, x2, y2 = STATUS_BOX[machine]
                frame[y1:y2, x1:x2] = vent_panel
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


class IdleGateTests(unittest.TestCase):
    def test_starts_only_after_settled_quiet(self):
        gate = IdleGate(start_below=25, stop_above=40, settle_samples=3, stop_samples=2)
        self.assertEqual(
            [gate.observe(p, running=False) for p in (10, 10, 50, 10, 10, 10)],
            [None, None, None, None, None, START],
        )

    def test_stops_on_sustained_load_not_a_spike(self):
        gate = IdleGate(start_below=25, stop_above=40, settle_samples=1, stop_samples=2)
        self.assertEqual(
            [gate.observe(p, running=True) for p in (80, 10, 80, 80)],
            [None, None, None, STOP],
        )

    def test_others_percent_removes_the_job(self):
        # 8 cores, system 60 %, the job uses 4 full cores (400 % of one core) = 50 %.
        self.assertAlmostEqual(others_percent(60, 400, 8), 10)
        self.assertEqual(others_percent(10, 400, 8), 0)

    def test_rejects_inverted_thresholds(self):
        with self.assertRaises(ValueError):
            IdleGate(start_below=50, stop_above=40)


class HsvMaskTests(unittest.TestCase):
    def test_selects_red_and_dark_text_but_not_green_or_gray(self):
        image = np.zeros((1, 4, 3), np.uint8)
        image[0] = [(30, 30, 220), (20, 20, 20), (60, 220, 60), (190, 190, 190)]  # BGR
        ranges = json.loads(
            (REPO / "config/diff_rule.json").read_text(encoding="utf-8")
        )["STATUS"]["hsv_ranges"]
        self.assertEqual(hsv_mask(image, ranges)[0].tolist(), [255, 255, 0, 0])

    def test_needs_a_range(self):
        with self.assertRaises(ValueError):
            hsv_mask(np.zeros((2, 2, 3), np.uint8), [])


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def results(self, name, machine=None, frames=12):
        folder = self.root / name
        folder.mkdir()
        stage = {
            "video": name,
            "total_frames": frames,
            "regions": {"STAGE": [{"pattern": 2, "start_frame": 0, "end_frame": 11}]},
        }
        if machine is not None:
            stage["machine_id"] = machine
        (folder / "stage_analysis.json").write_text(json.dumps(stage), encoding="utf-8")
        (folder / "run_stats.json").write_text(
            json.dumps({"frames": frames}), encoding="utf-8"
        )
        return folder

    def test_machine_2_gets_as_and_status_segments(self):
        video = self.root / "m2.mp4"
        make_clip(video, 2, as_frames=range(3, 7), occlusion_frames=range(8, 10))
        folder = self.results("m2", machine=2)

        status, message = tool.process(video, None, ["AS", "STATUS"])
        self.assertEqual(status, "ok", message)
        stage = json.loads((folder / "stage_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(stage["machine_id"], 2)
        self.assertEqual(
            stage["regions"]["STAGE"],
            [{"pattern": 2, "start_frame": 0, "end_frame": 11}],
        )
        self.assertEqual(
            [
                (s["pattern_name"], s["start_frame"], s["end_frame"])
                for s in stage["regions"]["AS"]
            ],
            [("AS", 3, 6)],
        )
        self.assertEqual(
            [
                (s["pattern_name"], s["start_frame"], s["end_frame"])
                for s in stage["regions"]["STATUS"]
            ],
            [("Occlusion", 8, 9)],
        )
        self.assertEqual(
            frame_clock.read_clock(folder / "frame_clock.json")["decoded_frames"], 12
        )

        self.assertTrue(tool.is_done(folder, ["AS", "STATUS"]))
        self.assertEqual(tool.process(video, None, ["AS", "STATUS"])[0], "skip")

    def test_vent_is_not_tracked_only_occlusion(self):
        panel = np.full((36, 208, 3), 190, dtype=np.uint8)
        cv2.putText(
            panel,
            "Vent",
            (66, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        self.assert_vent_excluded("synthetic", panel)

        fixture = REPO / "tests/fixtures/status_vent_panel.png"
        if fixture.is_file():
            self.assert_vent_excluded("recorded", cv2.imread(str(fixture)))

    def assert_vent_excluded(self, name, panel):
        video = self.root / f"{name}.mp4"
        make_clip(
            video,
            2,
            occlusion_frames=range(2, 4),
            vent_frames=range(4, 9),
            vent_panel=panel,
        )
        folder = self.results(name, machine=2)

        status, message = tool.process(video, None, ["AS", "STATUS"])
        self.assertEqual(status, "ok", message)
        stage = json.loads((folder / "stage_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [
                (s["pattern_name"], s["start_frame"], s["end_frame"])
                for s in stage["regions"]["STATUS"]
            ],
            [("Occlusion", 2, 3)],
        )

    def test_machine_1_has_no_as_region(self):
        video = self.root / "m1.mp4"
        make_clip(video, 1, as_frames=range(3, 7), occlusion_frames=range(2, 5))
        folder = self.results("m1", machine=1)

        status, message = tool.process(video, None, ["AS", "STATUS"])
        self.assertEqual(status, "ok", message)
        stage = json.loads((folder / "stage_analysis.json").read_text(encoding="utf-8"))
        self.assertNotIn("AS", stage["regions"])
        self.assertEqual(
            [(s["start_frame"], s["end_frame"]) for s in stage["regions"]["STATUS"]],
            [(2, 4)],
        )
        self.assertTrue(tool.is_done(folder, ["AS", "STATUS"]))

    def test_refuses_when_decode_differs_from_ocr_run(self):
        video = self.root / "bad.mp4"
        make_clip(video, 2)
        folder = self.results("bad", machine=2)
        (folder / "run_stats.json").write_text(
            json.dumps({"frames": 99}), encoding="utf-8"
        )
        before = (folder / "stage_analysis.json").read_text(encoding="utf-8")

        status, message = tool.process(video, None, ["AS", "STATUS"])
        self.assertEqual(status, "fail")
        self.assertIn("nothing written", message)
        self.assertEqual(
            (folder / "stage_analysis.json").read_text(encoding="utf-8"), before
        )
        self.assertFalse((folder / "frame_clock.json").exists())

    def test_unknown_machine_without_anchor_fails_instead_of_guessing(self):
        video = self.root / "unknown.mp4"
        make_clip(video, 2, frames=6)
        self.results("unknown", machine=None, frames=6)
        status, message = tool.process(video, None, ["AS", "STATUS"])
        self.assertEqual(status, "fail")
        self.assertIn("machine type not confirmed", message)

    def test_skips_video_without_results(self):
        video = self.root / "none.mp4"
        make_clip(video, 2, frames=3)
        self.assertEqual(tool.process(video, None, ["AS", "STATUS"])[0], "skip")


if __name__ == "__main__":
    unittest.main()
