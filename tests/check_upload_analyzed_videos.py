import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('uploader', Path(__file__).resolve().parents[1] / 'tools/upload_analyzed_videos.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


class FakeRcs:
    identity = 'endpoint-one'

    def __init__(self, free=1000):
        self.free = free
        self.files = {}
        self.pushes = []
        self.mkdir_calls = []

    def probe(self, path, mkdir=False):
        if mkdir:
            self.mkdir_calls.append(path)
        data = self.files.get(path)
        return {'exists': data is not None, 'free': self.free,
                **({'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} if data is not None else {})}

    def push(self, source, destination):
        data = Path(source).read_bytes()
        self.files[destination] = data
        self.free -= len(data)
        self.pushes.append(destination)


class BatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / 'manifest.json'
        self.state = self.root / 'state.json'
        self.rows = []
        for key, data in [('a/中文 space.MP4', b'abc'), ('b/next.mp4', b'defgh')]:
            path = self.root / 'data' / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.rows.append({'local_path': 'data/' + key, 'bytes': len(data),
                              'sha256': hashlib.sha256(data).hexdigest(),
                              'analysis_evidence': [{'path': 'fixture/stage_analysis.json'}]})
        u.save_json(self.manifest, {'videos': self.rows})
        self.videos = u.load_videos(self.manifest, self.root)
        self.client = FakeRcs()

    def run_upload(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return u.upload(self.videos, self.client, '/remote/data', self.state, reserve=10, **kwargs)

    def test_sequential_upload_preserves_subdirectories_and_records_success(self):
        self.assertEqual(self.run_upload(), 0)
        self.assertEqual(self.client.pushes, ['/remote/data/a/中文 space.MP4', '/remote/data/b/next.mp4'])
        state = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual({row['status'] for row in state['files'].values()}, {'verified'})
        self.assertEqual(len(self.client.mkdir_calls), 2)

    def test_rerun_hashes_remote_files_and_skips_completed_uploads(self):
        self.run_upload()
        self.client.pushes.clear()
        self.assertEqual(self.run_upload(), 0)
        self.assertEqual(self.client.pushes, [])
        # Journal alone never suffices: changed remote content is a conflict.
        self.client.files['/remote/data/a/中文 space.MP4'] = b'bad'
        self.assertEqual(self.run_upload(), 1)
        self.assertEqual(self.client.files['/remote/data/a/中文 space.MP4'], b'bad')

    def test_deleted_remote_video_is_uploaded_again_despite_journal(self):
        self.run_upload()
        self.client.files.clear()
        self.client.pushes.clear()
        self.assertEqual(self.run_upload(), 0)
        self.assertEqual(len(self.client.pushes), 2)

    def test_insufficient_space_stops_before_next_upload_and_retains_reserve(self):
        self.client.free = 15
        self.assertEqual(self.run_upload(), 3)
        self.assertEqual(len(self.client.pushes), 1)
        self.assertEqual(self.client.free, 12)
        state = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual(state['files']['b/next.mp4']['status'], 'blocked_space')
        self.client.free = 20
        self.assertEqual(self.run_upload(), 0)
        self.assertEqual(len(self.client.pushes), 2)

    def test_matching_remote_file_is_skipped_even_when_disk_is_full(self):
        self.run_upload()
        self.client.free = 0
        self.client.pushes.clear()
        self.assertEqual(self.run_upload(), 0)
        self.assertFalse(self.client.pushes)

    def test_transfer_failure_stops_batch_and_does_not_claim_verified(self):
        self.client.push = Mock(side_effect=RuntimeError('transfer failed'))
        self.assertEqual(self.run_upload(), 1)
        self.client.push.assert_called_once()
        state = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual(state['files']['a/中文 space.MP4']['status'], 'failed')
        self.assertNotIn('b/next.mp4', state['files'])

    def test_keyboard_interrupt_saves_state_and_returns_130(self):
        self.client.push = Mock(side_effect=KeyboardInterrupt())
        self.assertEqual(self.run_upload(), 130)
        state = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual(state['files']['a/中文 space.MP4']['status'], 'interrupted')

    def test_local_content_change_same_size_stops_before_remote_access(self):
        Path(self.videos[0]['local']).write_bytes(b'xyz')
        self.client.probe = Mock(side_effect=AssertionError('remote access'))
        self.assertEqual(self.run_upload(), 1)
        self.client.probe.assert_not_called()

    def test_post_upload_checksum_must_match_manifest(self):
        def corrupt(source, destination):
            self.client.files[destination] = b'bad'
        self.client.push = corrupt
        self.assertEqual(self.run_upload(), 1)
        state = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual(state['files']['a/中文 space.MP4']['status'], 'failed')

    def test_space_drop_between_probe_and_upload_stops(self):
        original = self.client.probe
        def probe(path, mkdir=False):
            if mkdir:
                self.client.free = 0
            return original(path, mkdir)
        self.client.probe = probe
        self.assertEqual(self.run_upload(), 1)
        self.assertFalse(self.client.pushes)

    def test_journal_cannot_be_reused_for_another_endpoint(self):
        self.run_upload()
        self.client.identity = 'endpoint-two'
        with self.assertRaisesRegex(ValueError, 'different manifest or destination'):
            self.run_upload()

    def test_limit_does_not_upload_more_files_and_resume_can_change_order(self):
        self.assertEqual(self.run_upload(limit=1), 0)
        self.assertEqual(len(self.client.pushes), 1)
        self.videos.reverse()
        self.assertEqual(self.run_upload(), 0)
        self.assertEqual(len(self.client.pushes), 2)

    def test_os_lock_excludes_second_writer_and_releases_after_failure(self):
        with u.batch_lock(self.state):
            with self.assertRaisesRegex(ValueError, 'Another upload'):
                with u.batch_lock(self.state):
                    self.fail('second writer acquired lock')
        with u.batch_lock(self.state):
            pass

    def test_manifest_rejects_traversal_duplicates_size_mismatch_and_missing_hash(self):
        mutations = [lambda rows: rows[0].update(local_path='../outside.mp4'),
                     lambda rows: rows[0].update(local_path='data/../outside.mp4'),
                     lambda rows: rows.append(rows[0]),
                     lambda rows: rows[0].update(bytes=999),
                     lambda rows: rows[0].update(sha256='invalid'),
                     lambda rows: rows[0].update(analysis_evidence=[])]
        for mutate in mutations:
            rows = json.loads(json.dumps(self.rows))
            mutate(rows)
            u.save_json(self.manifest, {'videos': rows})
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                u.load_videos(self.manifest, self.root)

    def test_plan_never_connects_and_help_lists_options(self):
        with patch.object(u, 'ROOT', self.root), patch.object(u, 'Rcs') as client, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(u.main(['plan', '--manifest', str(self.manifest)]), 0)
        client.assert_not_called()
        self.assertIn('nothing uploaded', out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as result:
            u.main(['-h'])
        self.assertEqual(result.exception.code, 0)
        for flag in ('--manifest', '--reserve-gib', '--smallest-first', '--state', '--limit'):
            self.assertIn(flag, out.getvalue())


class PreparationTests(unittest.TestCase):
    def test_nested_video_can_use_the_pipeline_default_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / 'data/nested').mkdir(parents=True)
            (root / 'data/nested/selected.MP4').write_bytes(b'video')
            output = root / 'data/selected'
            output.mkdir()
            (output / 'stage_analysis.json').write_text('{"total_frames": 10}')
            (output / 'region1_ocr_testing.jsonl').write_text('{}\n')
            result = u.discover(root)
            self.assertEqual(result['video_count'], 1)
            self.assertEqual(result['videos'][0]['local_path'], 'data/nested/selected.MP4')

    def test_discovery_requires_analysis_and_ocr_evidence_and_hashes_selected(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / 'data').mkdir()
            (root / 'data/selected.MP4').write_bytes(b'video')
            (root / 'data/unprocessed.mp4').write_bytes(b'other')
            output = root / 'runs/baseline/selected'
            output.mkdir(parents=True)
            (output / 'stage_analysis.json').write_text('{"total_frames": 10}')
            (output / 'region1_ocr_testing.jsonl').write_text('{}\n')
            result = u.discover(root)
            self.assertEqual(result['video_count'], 1)
            self.assertEqual(result['videos'][0]['local_path'], 'data/selected.MP4')
            self.assertEqual(result['videos'][0]['sha256'], hashlib.sha256(b'video').hexdigest())

    def test_ambiguous_stems_do_not_inherit_another_videos_golden_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ('data/one', 'data/two', 'runs/baseline/same'):
                (root / directory).mkdir(parents=True)
            for directory in ('one', 'two'):
                (root / 'data' / directory / 'same.mp4').write_bytes(b'video')
            (root / 'runs/baseline/same/stage_analysis.json').write_text('{"total_frames": 10}')
            (root / 'runs/baseline/same/region1_ocr_testing.jsonl').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'No videos'):
                u.discover(root)


class RcsProtocolTests(unittest.TestCase):
    def test_probe_ignores_banners_and_rejects_missing_duplicate_and_invalid_records(self):
        record = '{"exists":false,"free":123}'
        self.assertEqual(u.parse_probe('banner\r\nMARK:' + record + '\r\n', 'MARK:'), {'exists': False, 'free': 123})
        for output in ('banner', 'MARK:' + record + '\nMARK:' + record, 'MARK:{"exists":false,"free":-1}'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                u.parse_probe(output, 'MARK:')

    def test_remote_helper_executes_file_hash_and_space_check_with_literal_paths(self):
        # Execute the exact shipped helper against temp files; no network/mock filesystem.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / '中文 a;$(not-a-command).mp4'
            path.write_bytes(bytes(range(256)))
            payload = base64.b64encode(json.dumps({'path': str(path), 'mkdir': False, 'marker': 'MARK:'}).encode()).decode()
            proc = subprocess.run([sys.executable, '-X', 'utf8', '-c', u.REMOTE_PROBE, payload],
                                  capture_output=True, encoding='utf-8')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            record = u.parse_probe(proc.stdout, 'MARK:')
            self.assertEqual(record['sha256'], hashlib.sha256(bytes(range(256))).hexdigest())
            self.assertEqual(record['bytes'], 256)
            self.assertGreater(record['free'], 0)

    def test_target_url_is_pinned_and_push_never_requests_overwrite(self):
        responses = [Mock(returncode=0, stdout='rcs 0.3.0\n'),
                     Mock(returncode=0, stdout=json.dumps({'type': 'webide', 'url': 'https://example.com/ide/'})),
                     Mock(returncode=0)]
        with patch.object(u.subprocess, 'run', side_effect=responses) as run:
            client = u.Rcs('rcs.exe', 'zettabyte', 'python3')
            client.push('C:/中文 video.mp4', '/work/中文 video.mp4')
        self.assertEqual(run.call_args.args[0], ['rcs.exe', '--url', 'https://example.com/ide/',
                                               'push', 'C:/中文 video.mp4', '/work/中文 video.mp4'])

    def test_old_rcs_version_fails_before_target_lookup_or_remote_commands(self):
        with patch.object(u.subprocess, 'run', return_value=Mock(returncode=0, stdout='rcs 0.2.0\n')) as run:
            with self.assertRaisesRegex(ValueError, '0.3.0'):
                u.Rcs('rcs.exe', 'gpu', 'python3')
        run.assert_called_once()

    def test_remote_directory_rejects_relative_and_traversal(self):
        self.assertEqual(u.normalize_remote_dir('/work/a b/'), '/work/a b')
        for path in ('relative', '/work/../other', '/work\nother', 'C:\\work'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                u.normalize_remote_dir(path)


if __name__ == '__main__':
    unittest.main()
