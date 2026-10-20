from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pipeline_utils import duplicate_stems, find_videos


def touch(root: Path, rel: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return path


class FindVideosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_includes_subfolders(self):
        top = touch(self.root, 'a.MP4')
        nested = touch(self.root, '0816/b.mp4')
        deep = touch(self.root, '0816/x/c.mp4')
        self.assertEqual(find_videos(self.root), sorted([top, nested, deep]))

    def test_suffix_case_is_ignored_and_other_files_are_not(self):
        mixed = touch(self.root, 'v.Mp4')
        touch(self.root, 'notes.txt')
        touch(self.root, 'v/stage_analysis.json')
        (self.root / 'folder.mp4').mkdir()
        self.assertEqual(find_videos(self.root), [mixed])

    def test_file_is_returned_as_is(self):
        video = touch(self.root, 'one.mp4')
        self.assertEqual(find_videos(video), [video])

    def test_empty_directory(self):
        self.assertEqual(find_videos(self.root), [])

    def test_duplicate_stems_across_subfolders(self):
        a = touch(self.root, 'd1/same.mp4')
        b = touch(self.root, 'd2/same.MP4')
        c = touch(self.root, 'd2/other.mp4')
        self.assertEqual(duplicate_stems(find_videos(self.root)), {'same': sorted([a, b])})
        self.assertEqual(duplicate_stems([a, c]), {})


if __name__ == '__main__':
    unittest.main()
