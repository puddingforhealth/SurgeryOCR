import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import threading
import io
from contextlib import redirect_stdout

spec = importlib.util.spec_from_file_location('drive_batch', Path(__file__).parents[1] / 'tools/drive_batch.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(path='202511/test.mp4', identity='abc'):
    return {'Path': path, 'Name': path.split('/')[-1], 'ID': identity,
            'Size': 3, 'Hashes': {'md5': '900150983cd24fb0d6963f7d28e17f72'}}


def output(folder):
    folder.mkdir(parents=True, exist_ok=True)
    m.save(folder / 'stage_analysis.json', {'total_frames': 2})
    m.save(folder / 'run_stats.json', {'frames': 2, 'total_seconds': 1})
    (folder / 'test_ocr_testing.jsonl').write_text('{"frame":1}\n', encoding='utf-8')


class DriveBatchTests(unittest.TestCase):
    def test_live_monitor_shows_counts_and_latest_frame_progress(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            rows = m.prepare([row('202507/a.mp4', 'a'), row('202507/b.mp4', 'b'), row('202507/c.mp4', 'c')])
            m.save(p / 'manifest.json', {'videos': rows})
            m.save(p / 'state.json', {'files': {rows[0]['id']: {'status': 'collected'},
                                               rows[1]['id']: {'status': 'analyzing'}}, 'updated_at': 100})
            log = p / 'results' / rows[1]['output'] / 'pipeline.log'
            log.parent.mkdir(parents=True)
            log.write_bytes(b'10/100 [1s]\r\x1b[32m60/100 [6s, 10fps]\x1b[0m\r')
            before = (p / 'state.json').read_bytes()
            view = m.progress_text(p, now=112)
            self.assertIn('Returned to local: 1/3', view)
            self.assertIn('pending: 1', view)
            self.assertIn('[analyzing] 202507/b.mp4', view)
            self.assertIn('60/100 [6s, 10fps]', view)
            self.assertIn('12s ago', view)
            self.assertNotIn('\x1b', view)
            self.assertEqual((p / 'state.json').read_bytes(), before)

    def test_live_monitor_handles_download_ready_and_failure(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            rows = m.prepare([row('a.mp4', 'a'), row('b.mp4', 'b'), row('c.mp4', 'c')])
            m.save(p / 'manifest.json', {'videos': rows})
            m.save(p / 'state.json', {'files': {
                rows[0]['id']: {'status': 'downloading', 'download_attempt': 2},
                rows[1]['id']: {'status': 'ready'},
                rows[2]['id']: {'status': 'analysis_failed', 'error': 'Pipeline exit 1'}},
                'producer_error': 'connection interrupted'})
            view = m.progress_text(p)
            self.assertIn('attempt 2/3', view)
            self.assertIn('waiting for verified local collection', view)
            self.assertIn('Pipeline exit 1', view)
            self.assertIn('connection interrupted', view)

    def test_watch_once_and_interrupt_do_not_stop_worker(self):
        with patch.object(m, 'progress_text', return_value='progress'), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(m.watch(Path('.'), once=True), 0)
            self.assertEqual(stream.getvalue().strip(), 'progress')
        with patch.object(m, 'progress_text', return_value='progress'), patch.object(m.time, 'sleep', side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            self.assertEqual(m.watch(Path('.')), 0)

    def test_log_tail_is_bounded_and_missing_log_is_ok(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'pipeline.log'
            self.assertEqual(m.log_tail(p), [])
            p.write_bytes(b'x' * 10000 + b'\rLatest progress\r')
            self.assertEqual(m.log_tail(p, limit=16), ['Latest progress'])

    def test_status_distinguishes_unobserved_from_complete(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            m.save(p / 'manifest.json', {'videos': m.prepare([row()])})
            m.save(p / 'config.json', {'manifest': str(p / 'manifest.json'), 'output': str(p / 'out')})
            stream = io.StringIO()
            with redirect_stdout(stream):
                m.status(p / 'config.json')
            info = json.loads(stream.getvalue())
            self.assertEqual(info['batch_status'], 'not_observed')
            self.assertEqual(info['counts']['pending'], 1)

    def test_status_reports_failed_source_path(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); rows = m.prepare([row()])
            m.save(p / 'manifest.json', {'videos': rows})
            m.save(p / 'config.json', {'manifest': str(p / 'manifest.json'), 'output': str(p / 'out')})
            m.save(p / 'out/_batch_status.json', {'files': {rows[0]['id']: {'status': 'download_failed'}}})
            stream = io.StringIO()
            with redirect_stdout(stream):
                m.status(p / 'config.json')
            self.assertEqual(json.loads(stream.getvalue())['failures'][0]['path'], '202511/test.mp4')

    def test_preserves_months_and_names(self):
        self.assertEqual(m.prepare([row('202511/子目錄/影片.MP4')])[0]['output'], '202511/子目錄/影片')

    def test_filters_images_and_sorts(self):
        rows = [row('202511/b.mp4', 'b'), row('x.jpg', 'j'), row('202507/a.mp4', 'a')]
        self.assertEqual([r['path'] for r in m.prepare(rows)], ['202507/a.mp4', '202511/b.mp4'])

    def test_different_month_same_name_allowed(self):
        self.assertEqual(len(m.prepare([row('202507/a.mp4'), row('202511/a.mp4', 'b')])), 2)

    def test_collision_rejected(self):
        with self.assertRaises(ValueError):
            m.prepare([row('a.mp4'), row('A.MP4', 'b')])

    def test_duplicate_drive_id_rejected(self):
        with self.assertRaises(ValueError):
            m.prepare([row('a.mp4'), row('b.mp4')])

    def test_missing_checksum_rejected(self):
        r = row(); r['Hashes'] = {}
        with self.assertRaises(ValueError):
            m.prepare([r])

    def test_unsafe_paths_rejected(self):
        for path in ('../a.mp4', '/a.mp4', 'a/../b.mp4', 'C:/a.mp4', 'a\\b.mp4'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                m.prepare([row(path)])

    def test_empty_manifest_rejected(self):
        with self.assertRaises(ValueError):
            m.prepare([])

    def test_missing_ocr_not_success(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); output(p)
            (p / 'test_ocr_testing.jsonl').unlink()
            with self.assertRaises(ValueError):
                m.validate_outputs(p)

    def test_frame_counts_must_match(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); output(p)
            m.save(p / 'run_stats.json', {'frames': 1})
            with self.assertRaises(ValueError):
                m.validate_outputs(p)

    def test_result_archive_roundtrip_and_repeat(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); output(p / 'source')
            checksum = m.package(p / 'source', p / 'result.tgz')
            for _ in range(2):
                m.unpack_verified(p / 'result.tgz', p / 'month/clip', checksum)
            self.assertEqual(m.validate_outputs(p / 'month/clip')['frames'], 2)

    def test_archive_wrong_hash_not_published(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); output(p / 'source')
            m.package(p / 'source', p / 'result.tgz')
            with self.assertRaises(ValueError):
                m.unpack_verified(p / 'result.tgz', p / 'dest', '0' * 64)
            self.assertFalse((p / 'dest').exists())

    def test_local_damage_not_acknowledged(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); output(p / 'source')
            checksum = m.package(p / 'source', p / 'result.tgz')
            m.unpack_verified(p / 'result.tgz', p / 'dest', checksum)
            (p / 'dest/run_stats.json').write_text('{}')
            with self.assertRaises(ValueError):
                m.unpack_verified(p / 'result.tgz', p / 'dest', checksum)

    def test_traversal_archive_not_extracted(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            with tarfile.open(p / 'bad.tgz', 'w:gz') as tar:
                tar.addfile(tarfile.TarInfo('../escape'))
            with self.assertRaises(ValueError):
                m.unpack_verified(p / 'bad.tgz', p / 'dest', m.digest(p / 'bad.tgz'))
            self.assertFalse((p / 'escape').exists())

    def test_download_overlaps_analysis_and_cleanup_requires_ack(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); (p / 'code').mkdir()
            rows = m.prepare([row('202507/a.mp4', 'a'), row('202511/b.mp4', 'b')])
            m.save(p / 'manifest.json', {'videos': rows, 'folder_id': 'folder', 'python': 'python', 'code_commit': 'test'})
            second_download = threading.Event()
            calls = []
            def fake(cmd, **kwargs):
                calls.append(cmd)
                if 'copyto' in cmd:
                    target = Path(cmd[cmd.index('copyto') + 2]); target.write_bytes(b'abc')
                    if target.name == 'b.mp4':
                        second_download.set()
                else:
                    self.assertTrue(second_download.wait(2), 'download waited for analysis')
                    self.assertEqual(cmd[-1], 'ram')
                    output(Path(cmd[cmd.index('--output-dir') + 1]))
                return 0
            original = m.package
            def pack(folder, archive):
                checksum = original(folder, archive)
                # Raw inputs must remain until verified collection is acknowledged.
                self.assertTrue((p / 'inputs' / next(r['path'] for r in rows if r['id'] == archive.stem)).exists())
                m.save(p / 'acks' / (archive.stem + '.json'), {'sha256': checksum})
                return checksum
            with patch.object(m.subprocess, 'call', side_effect=fake), patch.object(m, 'package', side_effect=pack), patch.object(m.time, 'sleep'):
                m.worker(p, reserve_gib=1)
            state = json.loads((p / 'state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(len(list((p / 'inputs').rglob('*.mp4'))), 0)
            downloads = [c for c in calls if 'copyto' in c]
            self.assertEqual(len(downloads), 2)
            self.assertEqual(downloads[0][downloads[0].index('--multi-thread-streams') + 1], '0')
            with patch.object(m.subprocess, 'call') as call:
                m.worker(p, reserve_gib=1)
                call.assert_not_called()

    def test_failed_download_retries_and_never_analyzes_partial(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); (p / 'code').mkdir()
            rows = m.prepare([row()])
            m.save(p / 'manifest.json', {'videos': rows, 'folder_id': 'folder', 'python': 'python', 'code_commit': 'test'})
            def fail(cmd, **kwargs):
                self.assertIn('copyto', cmd)
                Path(cmd[cmd.index('copyto') + 2]).write_bytes(b'bad')
                return 0  # Even exit 0 isn't success without correct content.
            with patch.object(m.subprocess, 'call', side_effect=fail) as call, patch.object(m.threading.Event, 'wait', return_value=False):
                m.worker(p, reserve_gib=1)
            self.assertEqual(call.call_count, 3)
            state = json.loads((p / 'state.json').read_text())
            self.assertEqual(state['files'][rows[0]['id']]['status'], 'download_failed')
            self.assertEqual(state['status'], 'failed')
            self.assertFalse((p / 'inputs' / rows[0]['path']).exists())

    def test_space_guard_waits_before_downloading(self):
        with tempfile.TemporaryDirectory() as t:
            from collections import namedtuple
            usage = namedtuple('Usage', 'free')
            p = Path(t); (p / 'code').mkdir()
            rows = m.prepare([row()])
            m.save(p / 'manifest.json', {'videos': rows, 'folder_id': 'folder', 'python': 'python', 'code_commit': 'test'})
            # Space initially insufficient, then enough. Download fails without data.
            with patch.object(m.shutil, 'disk_usage', side_effect=[usage(0), usage(2**40)]), patch.object(m.threading.Event, 'wait', return_value=False) as wait, patch.object(m.subprocess, 'call', return_value=1):
                m.worker(p, reserve_gib=1)
            self.assertIn(unittest.mock.call(10), wait.call_args_list)

    def test_resume_clears_old_terminal_state_before_new_download(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); (p / 'code').mkdir()
            rows = m.prepare([row()])
            m.save(p / 'manifest.json', {'videos': rows, 'folder_id': 'folder', 'python': 'python', 'code_commit': 'test'})
            m.save(p / 'state.json', {'files': {rows[0]['id']: {'status': 'download_failed'}},
                                    'finished_at': 123, 'status': 'failed', 'producer_error': 'old failure'})
            def fail(cmd, **kwargs):
                state = json.loads((p / 'state.json').read_text())
                for field in ('finished_at', 'status', 'producer_error'):
                    self.assertNotIn(field, state, 'Collector could mistake retry for finished batch')
                return 1
            with patch.object(m.subprocess, 'call', side_effect=fail), patch.object(m.threading.Event, 'wait', return_value=False):
                m.worker(p, reserve_gib=1)


if __name__ == '__main__':
    unittest.main()
