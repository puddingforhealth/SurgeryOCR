from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_paths
from utils.get_paths import get_analysis_root, resolve_video_analysis_dir, set_analysis_root


class AnalysisRootTest(unittest.TestCase):
    def setUp(self):
        self.saved = (get_paths._ENV_ANALYSIS_ROOT, get_paths.DEFAULT_ANALYSIS_ROOT)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video = root / 'data/0816/clip.MP4'
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'')
        self.local = root / 'data/0816/clip'           # results next to the video
        self.cloud = root / 'runs/cloud/clip'          # results pulled from the cloud
        for d in (self.local, self.cloud):
            d.mkdir(parents=True)
            (d / 'stage_analysis.json').write_text('{}', encoding='utf-8')
        self.cloud_root = root / 'runs/cloud'

    def tearDown(self):
        get_paths._ENV_ANALYSIS_ROOT, get_paths.DEFAULT_ANALYSIS_ROOT = self.saved
        self.tmp.cleanup()

    def test_default_uses_results_next_to_the_video(self):
        set_analysis_root(None)
        self.assertEqual(resolve_video_analysis_dir(self.video), self.local)
        self.assertEqual(get_analysis_root(), (Path('data'), False))

    def test_switching_root_prefers_its_stem_folder(self):
        set_analysis_root(self.cloud_root)
        self.assertEqual(resolve_video_analysis_dir(self.video), self.cloud)
        self.assertEqual(get_analysis_root(), (self.cloud_root, True))

    def test_root_without_this_video_falls_back_to_the_sibling_folder(self):
        set_analysis_root(self.cloud_root.parent / 'empty')
        self.assertEqual(resolve_video_analysis_dir(self.video), self.local)

    def test_switching_back_restores_the_default(self):
        set_analysis_root(self.cloud_root)
        set_analysis_root(None)
        self.assertEqual(resolve_video_analysis_dir(self.video), self.local)


if __name__ == '__main__':
    unittest.main()
