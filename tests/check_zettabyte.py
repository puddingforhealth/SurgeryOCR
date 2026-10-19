"""Offline CLI contract and byte-for-byte roundtrip tests; no real credentials."""
import ast
import contextlib
import time
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock, mock_open

from zettabyte_transfer import cli
import zettabyte_analyze

SAS = 'https://example.blob.core.windows.net/tenant?sr=c&sp=rwl&sig=TEST_SECRET'
FAKE = '''#!/usr/bin/env python3
import os, sys, shutil
from pathlib import Path
from urllib.parse import urlsplit, unquote
args = sys.argv[1:]
print('diagnostic: ' + ' '.join(args))
if os.environ.get('FAKE_EXIT'):
    sys.exit(int(os.environ['FAKE_EXIT']))
if args[0] != 'copy':
    sys.exit(0)
def local(s):
    if s.startswith('https://'):
        return Path(os.environ['FAKE_BLOB']) / unquote(urlsplit(s).path.lstrip('/'))
    return Path(s)
src, dst = map(local, args[1:3])
if src.is_dir():
    for f in src.rglob('*'):
        if f.is_file():
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or '--overwrite=true' in args:
                shutil.copyfile(f, target)
else:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or '--overwrite=true' in args:
        shutil.copyfile(src, dst)
'''


class Transfers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fake = self.root / 'azcopy'
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        self.env = patch.dict(os.environ, {'ZETTABYTE_SAS_URL': SAS, 'FAKE_BLOB': str(self.root / 'blob')})
        self.env.start()
        self.output = io.StringIO()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def call(self, *args):
        with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output):
            return cli.main(['--azcopy', str(self.fake), *map(str, args)])

    def test_directory_roundtrip_models_and_video(self):
        source = self.root / '來源 with spaces'
        (source / 'weights').mkdir(parents=True)
        (source / 'weights' / '模型.bin').write_bytes(bytes(range(256)) * 31)
        (source / 'case.mp4').write_bytes(b'video\0payload')
        self.assertEqual(self.call('upload', source, 'jobs/a'), 0)
        pvc = self.root / 'blob/tenant/jobs/a'  # stand-in for provider's sync; not testing that service
        self.assertEqual((pvc / 'weights/模型.bin').read_bytes(), (source / 'weights/模型.bin').read_bytes())
        self.assertEqual(self.call('publish', pvc, 'results/a'), 0)
        destination = self.root / 'downloads'
        self.assertEqual(self.call('download', 'results/a', destination), 0)
        self.assertEqual((destination / 'case.mp4').read_bytes(), b'video\0payload')
        self.assertEqual((destination / 'weights/模型.bin').read_bytes(), (source / 'weights/模型.bin').read_bytes())
        self.assertIn('--as-subdir=false', self.output.getvalue())
        self.assertNotIn('TEST_SECRET', self.output.getvalue())

    def test_no_overwrite_then_explicit_overwrite(self):
        source = self.root / 'model.bin'
        source.write_bytes(b'old')
        self.assertEqual(self.call('upload', source, 'models/model.bin'), 0)
        source.write_bytes(b'new')
        self.assertEqual(self.call('upload', source, 'models/model.bin'), 0)
        remote = self.root / 'blob/tenant/models/model.bin'
        self.assertEqual(remote.read_bytes(), b'old')
        self.assertEqual(self.call('upload', source, 'models/model.bin', '--overwrite'), 0)
        self.assertEqual(remote.read_bytes(), b'new')
        destination = self.root / 'restored.bin'
        self.assertEqual(self.call('download', 'models/model.bin', destination, '--container'), 0)
        self.assertEqual(destination.read_bytes(), b'new')

    def test_rejects_bad_paths_and_outbound_upload(self):
        source = self.root / 'a'
        source.write_text('a')
        for remote in ['../a', '/a', 'a//b', 'a/./b', 'a\\b', 'a?b', 'a*b', 'outbound/a', 'OUTBOUND/a']:
            with self.subTest(remote=remote):
                self.assertEqual(self.call('upload', source, remote), 2)
        self.assertEqual(self.call('publish', source, 'outbound/a'), 2)
        self.assertEqual(self.call('download', 'outbound/a', self.root / 'd'), 2)

    def test_dry_run_does_not_execute_or_expose_secret(self):
        self.assertEqual(self.call('--dry-run', 'list'), 0)
        self.assertFalse((self.root / 'blob').exists())
        self.assertNotIn('TEST_SECRET', self.output.getvalue())

    def test_nonzero_azcopy_exit_propagates(self):
        with patch.dict(os.environ, {'FAKE_EXIT': '17'}):
            self.assertEqual(self.call('list'), 17)
        self.assertNotIn('TEST_SECRET', self.output.getvalue())

    def test_jobs_and_redaction(self):
        job = '12345678-1234-1234-1234-123456789abc'
        self.assertEqual(self.call('status', job), 0)
        self.assertEqual(self.call('resume', job), 0)
        self.assertNotIn('TEST_SECRET', self.output.getvalue())
        self.assertEqual(self.call('resume', 'bad'), 2)

    def test_configure_and_file_fallback(self):
        config = self.root / 'config.json'
        with patch('getpass.getpass', return_value=SAS):
            self.assertEqual(self.call('--config', config, 'configure'), 0)
        self.assertEqual(config.stat().st_mode & 0o777, 0o600)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.call('--config', config, 'list'), 0)
        self.assertNotIn('TEST_SECRET', self.output.getvalue())

    def test_invalid_url_and_missing_config(self):
        for value in ['http://example.blob.core.windows.net/c?sr=c&sig=x',
                      'https://example.com/c?sr=c&sig=x',
                      SAS.replace('/tenant?', '/tenant/sub?'), SAS.replace('sr=c', 'sr=b')]:
            with self.assertRaises(ValueError):
                cli.validate_url(value)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.call('--config', self.root / 'absent', 'list'), 2)

    def test_configure_dry_run_never_prompts_or_writes(self):
        with patch('getpass.getpass') as prompt:
            self.assertEqual(self.call('--dry-run', 'configure'), 2)
            prompt.assert_not_called()

    def test_missing_source_or_azcopy(self):
        self.assertEqual(self.call('upload', self.root / 'absent', 'a'), 2)
        with contextlib.redirect_stderr(self.output), contextlib.redirect_stdout(self.output):
            self.assertEqual(cli.main(['--azcopy', str(self.root / 'absent'), 'list']), 2)

    def test_module_entry_point(self):
        result = subprocess.run([sys.executable, '-m', 'zettabyte_transfer', '--dry-run', 'list'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn('TEST_SECRET', result.stdout + result.stderr)


class AnalysisAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / 'case.mp4'
        self.video.write_bytes(b'fake')
        self.args = ['--video', str(self.video), '--job-id', 'job001',
                     '--output-root', str(self.root / 'out'), '--publish']

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_publishes_only_valid_results(self):
        def runner(command, **kwargs):
            output = Path(command[command.index('--output-dir') + 1]) / 'case'
            output.mkdir()
            (output / 'stage_analysis.json').write_text('{"total_frames": 5}')
            (output / 'region_ocr_testing.jsonl').write_text('{}\n')
            return subprocess.CompletedProcess(command, 0)
        with patch('zettabyte_analyze.subprocess.run', side_effect=runner), patch('zettabyte_analyze.transfer', return_value=0) as publish:
            self.assertEqual(zettabyte_analyze.main(self.args), 0)
            self.assertEqual(publish.call_args.args[0][-1], 'results/job001')
        metadata = json.loads((self.root / 'out/job001/job.json').read_text())
        self.assertEqual(metadata['status'], 'complete')

    def test_failed_pipeline_never_publishes(self):
        with patch('zettabyte_analyze.subprocess.run', return_value=subprocess.CompletedProcess([], 1)), patch('zettabyte_analyze.transfer') as publish:
            self.assertEqual(zettabyte_analyze.main(self.args), 1)
            publish.assert_not_called()
        self.assertEqual(json.loads((self.root / 'out/job001/job.json').read_text())['status'], 'failed')

    def test_zero_exit_without_outputs_never_publishes(self):
        with patch('zettabyte_analyze.subprocess.run', return_value=subprocess.CompletedProcess([], 0)), patch('zettabyte_analyze.transfer') as publish:
            self.assertEqual(zettabyte_analyze.main(self.args), 1)
            publish.assert_not_called()

    def test_existing_job_is_rejected(self):
        (self.root / 'out/job001').mkdir(parents=True)
        with patch('zettabyte_analyze.subprocess.run') as runner:
            self.assertEqual(zettabyte_analyze.main(self.args), 1)
            runner.assert_not_called()


class PipelineFailures(unittest.TestCase):
    """Exercise the real pipeline function with video/GPU boundaries stubbed."""
    def test_frame_errors_and_empty_video_do_not_produce_success_results(self):
        tree = ast.parse((zettabyte_analyze.ROOT / 'run_pipeline.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_pipeline')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        ast.fix_missing_locations(module)
        for error in (RuntimeError('decoder failed'), KeyboardInterrupt(), None):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                cap = Mock()
                cap.isOpened.return_value = error is not None
                cap.get.return_value = 5
                cap.read.side_effect = error
                cv2 = Mock()
                cv2.VideoCapture.return_value = cap
                analyzer = Mock()
                analyzer.roi_dict = {}
                write_results = Mock()
                scope = dict(Path=Path, time=time, json=json, cv2=cv2, StageAnalyzer=Mock(return_value=analyzer),
                             MachineDetector=Mock(), load_stage_config=Mock(return_value={}),
                             load_ocr_char_sets_config=Mock(return_value={}),
                             load_pattern_name_mapping=Mock(return_value={}),
                             load_roi_config=Mock(return_value={}), tqdm=Mock(),
                             _write_stage_analysis=write_results)
                exec(compile(module, 'run_pipeline.py', 'exec'), scope)
                with patch('builtins.open', mock_open(read_data='{"video_machine_mapping": {}}')):
                    with self.assertRaises(KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else RuntimeError):
                        scope['run_pipeline'](Path('case.mp4'), Path(tmp), mode='ram')
                cap.release.assert_called_once()
                write_results.assert_not_called()


if __name__ == '__main__':
    unittest.main()
