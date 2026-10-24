import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import drive_local as m


class LocalDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = {'id': 'video', 'path': '202507/clip.MP4', 'size': 3,
                    'md5': '900150983cd24fb0d6963f7d28e17f72'}
        m.save(self.root / 'manifest.json', {'folder_id': 'folder', 'videos': [self.row]})
        self.config = self.root / 'config.json'
        m.save(self.config, {'manifest': str(self.root / 'manifest.json'),
                            'output': str(self.root / 'out'), 'rclone': 'rclone', 'bandwidth_mib': 8})
        self.video = self.root / 'out' / self.row['path']

    def state(self):
        return json.loads((self.root / 'out/_video_download_status.json').read_text())

    def transfer(self, cmd, **kwargs):
        self.assertEqual(cmd[cmd.index('--bwlimit') + 1], '8M')
        self.assertEqual(cmd[cmd.index('--transfers') + 1], '1')
        self.assertEqual(cmd[cmd.index('--multi-thread-streams') + 1], '0')
        self.assertFalse(self.video.exists(), 'partial video exposed before verification')
        Path(cmd[cmd.index('copyto') + 2]).write_bytes(b'abc')
        return SimpleNamespace(returncode=0)

    def test_verified_video_published_beside_results_and_rerun_skips(self):
        result = self.video.with_suffix('') / 'stage_analysis.json'
        m.save(result, {'total_frames': 10})
        with patch.object(m.subprocess, 'run', side_effect=self.transfer), redirect_stdout(io.StringIO()):
            self.assertEqual(m.download(self.config), 0)
        self.assertEqual(self.video.read_bytes(), b'abc')
        self.assertEqual(json.loads(result.read_text()), {'total_frames': 10})
        self.assertFalse(self.video.with_name('clip.MP4.part').exists())
        with patch.object(m.subprocess, 'run') as run:
            self.assertEqual(m.download(self.config), 0)
            run.assert_not_called()

    def test_existing_different_video_not_overwritten(self):
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'original')
        with patch.object(m.subprocess, 'run') as run:
            self.assertEqual(m.download(self.config), 1)
            run.assert_not_called()
        self.assertEqual(self.video.read_bytes(), b'original')
        self.assertEqual(self.state()['files']['video']['status'], 'failed')

    def test_bad_checksum_retries_without_publishing_video(self):
        def bad(cmd, **kwargs):
            Path(cmd[cmd.index('copyto') + 2]).write_bytes(b'bad')
            return SimpleNamespace(returncode=0)
        with patch.object(m.subprocess, 'run', side_effect=bad) as run, patch.object(m.time, 'sleep'):
            self.assertEqual(m.download(self.config), 1)
            self.assertEqual(run.call_count, 3)
        self.assertFalse(self.video.exists())
        self.assertIn('MD5 mismatch', self.state()['files']['video']['error'])

    def test_space_guard_waits_before_transfer(self):
        with patch.object(m.shutil, 'disk_usage', side_effect=[SimpleNamespace(free=0), SimpleNamespace(free=2**40)]), patch.object(m.time, 'sleep') as sleep, patch.object(m.subprocess, 'run', side_effect=self.transfer), redirect_stdout(io.StringIO()):
            self.assertEqual(m.download(self.config), 0)
        sleep.assert_called_once_with(30)

    def test_transfer_failure_retries_then_success(self):
        responses = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]
        def transfer(cmd, **kwargs):
            self.transfer(cmd, **kwargs)
            return responses.pop(0)
        with patch.object(m.subprocess, 'run', side_effect=transfer) as run, patch.object(m.time, 'sleep'), redirect_stdout(io.StringIO()):
            self.assertEqual(m.download(self.config), 0)
            self.assertEqual(run.call_count, 2)
        self.assertIsNone(self.state()['files']['video']['error'])

    def test_status_reports_incomplete_bytes_separately_from_completion(self):
        self.video.parent.mkdir(parents=True)
        self.video.with_name('clip.MP4.part').write_bytes(b'a')
        m.save(self.root / 'out/_video_download_status.json', {'status': 'running', 'files': {'video': {'status': 'downloading'}}})
        with redirect_stdout(io.StringIO()) as stream:
            m.status(self.config)
        data = json.loads(stream.getvalue())
        self.assertEqual(data['active'][0]['bytes'], 1)
        self.assertEqual(data['counts'], {'downloading': 1})

    def test_path_traversal_rejected_before_writing(self):
        self.row['path'] = '../outside.mp4'
        m.save(self.root / 'manifest.json', {'folder_id': 'folder', 'videos': [self.row]})
        with self.assertRaises(ValueError):
            m.download(self.config)
        self.assertFalse((self.root / 'out').exists())

    def test_status_reads_rclone_atomic_partial_filename(self):
        self.video.parent.mkdir(parents=True)
        self.video.with_name('clip.MP4.part.7828ae77.partial').write_bytes(b'ab')
        m.save(self.root / 'out/_video_download_status.json', {'status': 'running', 'files': {'video': {'status': 'downloading'}}})
        with redirect_stdout(io.StringIO()) as stream:
            m.status(self.config)
        self.assertEqual(json.loads(stream.getvalue())['active'][0]['bytes'], 2)


if __name__ == '__main__':
    unittest.main()
