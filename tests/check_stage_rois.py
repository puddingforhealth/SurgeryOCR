from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.get_configs import (
    read_surgery_stage_rois,
    resolve_stage_rois,
    update_stage_roi,
)

RAW = {
    "STAGE": [610, 55, 885, 90],
    "AS": {"by_machine": {"2": [1738, 542, 1838, 640]}},
    "STATUS": {
        "by_machine": {"1": [1628, 682, 1836, 718], "2": [1628, 664, 1836, 700]}
    },
}


class ResolveTests(unittest.TestCase):
    def test_plain_boxes_apply_to_every_machine(self):
        for machine in (1, 2, "2", None):
            self.assertEqual(
                resolve_stage_rois(RAW, machine)["STAGE"], [610, 55, 885, 90]
            )

    def test_per_machine_boxes_pick_the_machine(self):
        self.assertEqual(resolve_stage_rois(RAW, 1)["STATUS"], [1628, 682, 1836, 718])
        self.assertEqual(resolve_stage_rois(RAW, "2")["STATUS"], [1628, 664, 1836, 700])

    def test_region_absent_on_machine_without_entry(self):
        self.assertNotIn("AS", resolve_stage_rois(RAW, 1))
        self.assertIn("AS", resolve_stage_rois(RAW, 2))

    def test_unknown_machine_keeps_only_plain_boxes(self):
        self.assertEqual(set(resolve_stage_rois(RAW, None)), {"STAGE"})

    def test_read_from_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "surgery_stage_rois.json"
            path.write_text(json.dumps(RAW), encoding="utf-8")
            self.assertEqual(
                set(read_surgery_stage_rois(path, 2)), {"STAGE", "AS", "STATUS"}
            )
            self.assertEqual(set(read_surgery_stage_rois(path)), {"STAGE"})


class UpdateTests(unittest.TestCase):
    def test_editing_per_machine_box_keeps_other_machines(self):
        raw = json.loads(json.dumps(RAW))
        update_stage_roi(raw, "STATUS", [1, 2, 3, 4], 2)
        self.assertEqual(
            raw["STATUS"]["by_machine"],
            {"1": [1628, 682, 1836, 718], "2": [1, 2, 3, 4]},
        )

    def test_editing_plain_box_and_adding_region(self):
        raw = json.loads(json.dumps(RAW))
        update_stage_roi(raw, "STAGE", [1, 1, 9, 9], 1)
        update_stage_roi(raw, "NEW", [5, 5, 6, 6])
        self.assertEqual(raw["STAGE"], [1, 1, 9, 9])
        self.assertEqual(raw["NEW"], [5, 5, 6, 6])

    def test_round_trip_through_resolve_is_lossless(self):
        # What the GUI does: resolve for its machine, save every shown box back.
        raw = json.loads(json.dumps(RAW))
        for region, box in resolve_stage_rois(raw, 1).items():
            update_stage_roi(raw, region, box, 1)
        self.assertEqual(raw, RAW)

    def test_per_machine_edit_needs_machine(self):
        with self.assertRaises(ValueError):
            update_stage_roi(json.loads(json.dumps(RAW)), "AS", [1, 2, 3, 4], None)


class TimelineTrackTests(unittest.TestCase):
    """The annotator draws a track for AS / STATUS only when the analysis has them."""

    def setUp(self):
        import video_annotator_gui as gui

        self.timeline_regions = gui.timeline_regions

    def test_machine_2_results_show_as_and_status_tracks(self):
        regions = {"STAGE": [], "PEDAL": [], "AS": [], "STATUS": []}
        self.assertEqual(
            self.timeline_regions(regions), ["PEDAL", "STAGE", "AS", "STATUS"]
        )

    def test_old_results_and_machine_1_keep_their_tracks(self):
        self.assertEqual(
            self.timeline_regions({"STAGE": [], "PEDAL": []}), ["PEDAL", "STAGE"]
        )
        self.assertEqual(
            self.timeline_regions({"STAGE": [], "STATUS": []}),
            ["PEDAL", "STAGE", "STATUS"],
        )
        self.assertEqual(self.timeline_regions(None), ["PEDAL", "STAGE"])


if __name__ == "__main__":
    unittest.main()
