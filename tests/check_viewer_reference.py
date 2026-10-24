from pathlib import Path
import sys
import json
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.viewer_reference import ReferenceChecker, recorded_segment, region_segments


class ViewerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'config'; self.config.mkdir()
        (self.config/'diff_rule.json').write_text(json.dumps({'STAGE':{'diff_threshold':15},
            'PEDAL':{'analysis_mode':'sub_roi','sub_roi_coords':[0,0,1,1],'diff_threshold':2}}))
        self.cache = self.root/'templates'
        self.checker = ReferenceChecker(self.config,self.cache)

    def reference(self, route, region, pattern, array):
        folder=self.cache/route/region;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/f'{pattern}.npy',array)

    def test_recorded_pattern_is_checked_instead_of_largest_template_number(self):
        array=np.array([[0,255],[255,0]],dtype=np.uint8)
        self.reference('machine_1_2','STAGE',6,array)
        self.reference('machine_1_2','STAGE',25,255-array)
        result=self.checker.inspect(Image.fromarray(array).convert('RGB'),2,'STAGE',6)
        self.assertEqual(result['state'],'match')
        self.assertEqual(result['rmse'],0)
        self.assertEqual(result['threshold'],15)

    def test_machine_three_uses_its_templates_and_rule_binarization(self):
        override=self.config/'machine_3';override.mkdir()
        (override/'diff_rule.json').write_text('{"STAGE":{"binarize":"rule","diff_threshold":60}}')
        self.reference('machine_3','STAGE',1,np.zeros((2,2),np.uint8))
        self.reference('machine_1_2','STAGE',1,np.zeros((2,2),np.uint8))
        image=Image.fromarray(np.array([[80,120],[120,80]],np.uint8)).convert('RGB')
        self.assertEqual(self.checker.inspect(image,3,'STAGE',1)['state'],'match')
        self.assertEqual(self.checker.inspect(image,2,'STAGE',1)['state'],'different')

    def test_pedal_compares_configured_rgb_subregion(self):
        reference=np.zeros((2,2,3),np.uint8)
        self.reference('machine_1_2','PEDAL',1,reference)
        current=np.full((2,2,3),255,np.uint8);current[0,0]=0
        self.assertEqual(self.checker.inspect(Image.fromarray(current),2,'PEDAL',1)['state'],'match')

    def test_missing_reference_is_unavailable_rather_than_failed_analysis(self):
        self.assertEqual(self.checker.inspect(Image.new('RGB',(2,2)),2,'STAGE',6)['state'],'unavailable')

    def test_incompatible_reference_does_not_report_false_match(self):
        self.reference('machine_1_2','STAGE',6,np.zeros((4,4),np.uint8))
        self.assertEqual(self.checker.inspect(Image.new('RGB',(2,2)),2,'STAGE',6)['state'],'unavailable')

    def test_legacy_and_current_segment_formats_select_inclusive_frame_boundaries(self):
        segment={'start_frame':6,'end_frame':12,'pattern':6}
        for value in ([segment],{'patterns':[segment]}):
            analysis={'regions':{'STAGE':value}}
            self.assertEqual(region_segments(analysis,'STAGE'),[segment])
            for index in (6,12):self.assertEqual(recorded_segment(analysis,'STAGE',index),segment)
            self.assertIsNone(recorded_segment(analysis,'STAGE',13))
        self.assertIsNone(recorded_segment(None,'STAGE',0))


if __name__ == '__main__':
    unittest.main()
