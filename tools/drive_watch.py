"""Watch an existing batch; restart exited workers with backoff, never kill active work."""
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from drive_batch import lock, save


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'files': {}}


def summary(state, rows, role):
    success = 'complete' if role == 'download' else 'collected'
    counts = Counter(state.get('files', {}).get(r['id'], {}).get('status', 'pending') for r in rows)
    failures = [r['id'] for r in rows
                if state.get('files', {}).get(r['id'], {}).get('status', '').endswith('failed')]
    return {'total': len(rows), 'finished': counts[success], 'counts': dict(counts),
            'failed_ids': failures, 'complete': bool(rows) and counts[success] == len(rows)}


def delay(attempt):
    return min(1800, 300 * 2 ** min(max(attempt - 1, 0), 3))


def action(report, busy, now, retry_at):
    if report['complete']:
        return 'complete'
    if busy:
        return 'monitoring'
    return 'restart' if now >= retry_at else 'waiting_retry'


def held(path):
    try:
        with lock(path):
            return False
    except (BlockingIOError, PermissionError):
        return True


def settings(role, target):
    target = Path(target).resolve()
    script = Path(__file__).resolve().parent
    if role == 'worker':
        root = target
        manifest = root / 'manifest.json'
        state = root / 'state.json'
        worker_lock = root / 'worker.lock'
        command = [sys.executable, '-u', str(root / 'drive_batch.py'), 'worker', str(root)]
    else:
        cfg = read(target)
        root = Path(cfg['output'])
        manifest = Path(cfg['manifest'])
        state = root / ('_video_download_status.json' if role == 'download' else '_batch_status.json')
        worker_lock = root / '_video_download.lock' if role == 'download' else Path(cfg['cache']) / 'collector.lock'
        command = [sys.executable, '-u', str(script / ('drive_local.py' if role == 'download' else 'drive_batch.py')),
                   role, str(target)]
    rows = read(manifest)['videos']
    if not rows:
        raise ValueError('Empty manifest')
    return root, state, worker_lock, command, rows


def watch(role, target, interval=30):
    root, state_path, worker_lock, command, rows = settings(role, target)
    root.mkdir(parents=True, exist_ok=True)
    worker_lock.parent.mkdir(parents=True, exist_ok=True)
    prefix = root / ('_watch_' + role)
    previous = read(prefix.with_suffix('.json'))
    attempts = previous.get('attempts', 0)
    retry_at = previous.get('next_retry_at', 0)
    finished = previous.get('finished', 0)
    child = None
    with lock(prefix.with_suffix('.lock')):
        while True:
            now = time.time()
            try:
                report = summary(read(state_path), rows, role)
                if report['finished'] > finished:
                    attempts, retry_at = 0, 0
                finished = report['finished']
                busy = (child is not None and child.poll() is None) or held(worker_lock)
                step = action(report, busy, now, retry_at)
                if prefix.with_suffix('.stop').exists():
                    step = 'stopped'
                if step == 'restart':
                    with prefix.with_suffix('.log').open('ab') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
                    attempts += 1
                    retry_at = now + delay(attempts)
                    print(f'{role}: started worker pid={child.pid}, attempt={attempts}', flush=True)
                    step = 'monitoring'
                save(prefix.with_suffix('.json'), dict(report, status=step, updated_at=now,
                     attempts=attempts, next_retry_at=retry_at, worker_active=busy or (child is not None and child.poll() is None)))
                if step in ('complete', 'stopped'):
                    return
            except Exception as e:
                save(prefix.with_suffix('.json'), {'status': 'monitor_error', 'updated_at': now,
                     'error': str(e), 'attempts': attempts, 'next_retry_at': retry_at})
                print(f'{role}: observation error: {type(e).__name__}', flush=True)
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['worker', 'download', 'collect'])
    parser.add_argument('target', help='Batch root (worker), or existing config JSON (download/collect)')
    args = parser.parse_args()
    watch(args.role, args.target)
