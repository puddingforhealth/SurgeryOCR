import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cloud_videos', ROOT / 'tools/cloud_videos.py')
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


class OpenCvPinTest(unittest.TestCase):
    """OpenCV 5.0 decodes the videos to different pixels than 4.11 and changes OCR readings,
    so both wheels that provide cv2 must stay pinned to the same version."""

    def test_both_opencv_wheels_are_pinned_to_the_same_version(self):
        full = cv.requirement_pin('opencv-python')
        headless = cv.requirement_pin('opencv-python-headless')
        self.assertIsNotNone(full)
        self.assertEqual(full, headless)
        self.assertTrue(full.startswith('4.11.'), full)


class RequirementPinTest(unittest.TestCase):
    def write(self, text, encoding):
        tmp = tempfile.NamedTemporaryFile('wb', suffix='.txt', delete=False)
        tmp.write(text.encode(encoding))
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_reads_utf16_with_bom_and_crlf(self):
        path = self.write('# c\r\nopencv-python==4.11.0.86\r\nnumpy\r\n', 'utf-16')
        self.assertEqual(cv.requirement_pin('opencv-python', path), '4.11.0.86')

    def test_reads_utf8_and_ignores_comments_and_case(self):
        path = self.write('OpenCV-Python == 4.11.0.86  # pinned\nnumpy\n', 'utf-8')
        self.assertEqual(cv.requirement_pin('opencv-python', path), '4.11.0.86')

    def test_unpinned_or_missing_package_gives_none(self):
        path = self.write('opencv-python\nopencv-python-headless>=4\n', 'utf-8')
        self.assertIsNone(cv.requirement_pin('opencv-python', path))
        self.assertIsNone(cv.requirement_pin('opencv-python-headless', path))
        self.assertIsNone(cv.requirement_pin('torch', path))

    def test_headless_name_does_not_match_the_full_package(self):
        path = self.write('opencv-python-headless==4.11.0.86\n', 'utf-8')
        self.assertIsNone(cv.requirement_pin('opencv-python', path))


if __name__ == '__main__':
    unittest.main()
