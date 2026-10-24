import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import drive_watch as m


class WatchTests(unittest.TestCase):
    def test_missing_rows_are_pending_not_complete(self):
        report = m.summary({'files': {'a': {'status': 'collected'}}}, [{'id': 'a'}, {'id': 'b'}], 'worker')
        self.assertFalse(report['complete'])
        self.assertEqual(report['counts']['pending'], 1)

    def test_ready_is_not_received(self):
        self.assertFalse(m.summary({'files': {'a': {'status': 'ready'}}}, [{'id': 'a'}], 'collect')['complete'])

    def test_local_completion_uses_verified_download_status(self):
        self.assertTrue(m.summary({'files': {'a': {'status': 'complete'}}}, [{'id': 'a'}], 'download')['complete'])

    def test_failure_is_visible_and_retried_after_backoff(self):
        report = m.summary({'files': {'a': {'status': 'download_failed'}}}, [{'id': 'a'}], 'worker')
        self.assertEqual(report['failed_ids'], ['a'])
        self.assertEqual(m.action(report, False, 299, 300), 'waiting_retry')
        self.assertEqual(m.action(report, False, 300, 300), 'restart')

    def test_active_worker_never_restarted_even_after_deadline(self):
        self.assertEqual(m.action({'complete': False}, True, 1000, 0), 'monitoring')

    def test_finished_batch_never_restarted(self):
        self.assertEqual(m.action({'complete': True}, False, 1000, 0), 'complete')

    def test_backoff_caps_at_thirty_minutes(self):
        self.assertEqual([m.delay(n) for n in (1, 2, 3, 4, 100)], [300, 600, 1200, 1800, 1800])

    def test_live_lock_detected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'worker.lock'
            self.assertFalse(m.held(p))
            with m.lock(p):
                self.assertTrue(m.held(p))

    def test_restart_then_completion_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            m.save(root / 'manifest.json', {'videos': [{'id': 'a'}]})
            m.save(root / 'state.json', {'files': {'a': {'status': 'download_failed'}}})
            def finish(_):
                m.save(root / 'state.json', {'files': {'a': {'status': 'collected'}}})
            with patch.object(m.subprocess, 'Popen') as run, patch.object(m.time, 'sleep', side_effect=finish):
                m.watch('worker', root)
            run.assert_called_once()
            state = m.read(root / '_watch_worker.json')
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['finished'], 1)

    def test_stop_marker_does_not_kill_active_worker(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            m.save(root / 'manifest.json', {'videos': [{'id': 'a'}]})
            (root / '_watch_worker.stop').touch()
            with patch.object(m.subprocess, 'Popen') as run:
                m.watch('worker', root)
            run.assert_not_called()
            self.assertEqual(m.read(root / '_watch_worker.json')['status'], 'stopped')
