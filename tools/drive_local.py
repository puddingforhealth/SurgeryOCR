"""Download Drive videos serially beside their analysis results, with a bandwidth cap."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from drive_batch import digest, lock, relative, save


def load(config):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    manifest = json.loads(Path(cfg['manifest']).read_text(encoding='utf-8'))
    for row in manifest['videos']:
        relative(row['path'])
    if cfg.get('reserve_gib', 20) < 1 or cfg.get('bandwidth_mib', 8) <= 0:
        raise ValueError('Positive bandwidth and at least 1 GiB reserve required')
    return cfg, manifest


def valid(path, row):
    return path.is_file() and path.stat().st_size == row['size'] and digest(path, 'md5') == row['md5']


def download(config):
    cfg, manifest = load(config)
    root = Path(cfg['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / '_download_logs'
    logs.mkdir(exist_ok=True)
    state_path = root / '_video_download_status.json'
    state = {'files': {}, 'status': 'running'}
    def update(row, **changes):
        state['files'].setdefault(row['id'], {}).update(changes, updated_at=time.time())
        state['updated_at'] = time.time()
        save(state_path, state)
    with lock(root / '_video_download.lock'):
        save(state_path, state)
        for row in manifest['videos']:
            dest = root / row['path']
            part = dest.with_name(dest.name + '.part')
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Never replace a pre-existing, different video with a remote copy.
            if dest.exists():
                if valid(dest, row):
                    update(row, status='complete')
                else:
                    update(row, status='failed', error='Existing video differs; left untouched')
                continue
            while shutil.disk_usage(root).free < row['size'] + cfg.get('reserve_gib', 20) * 2**30:
                update(row, status='waiting_space')
                time.sleep(30)
            for attempt in range(1, 4):
                update(row, status='downloading', attempt=attempt)
                cmd = [cfg['rclone'], 'copyto', 'gdrive:' + row['path'], str(part),
                       '--drive-root-folder-id', manifest['folder_id'],
                       '--transfers', '1', '--checkers', '1', '--multi-thread-streams', '0',
                       '--bwlimit', str(cfg.get('bandwidth_mib', 8)) + 'M',
                       '--checksum', '--retries', '2', '--low-level-retries', '5',
                       '--contimeout', '30s', '--timeout', '2m', '--stats', '30s', '--stats-one-line']
                try:
                    with (logs / (row['id'] + '.log')).open('ab') as log:
                        result = subprocess.run(cmd, stdout=log, stderr=log, timeout=7200)
                    if result.returncode:
                        raise RuntimeError('rclone exit ' + str(result.returncode))
                    update(row, status='verifying', attempt=attempt)
                    if not valid(part, row):
                        raise ValueError('Size or MD5 mismatch against manifest')
                    if dest.exists():
                        raise FileExistsError('Destination appeared during download; left untouched')
                    os.rename(part, dest)
                    update(row, status='complete', error=None)
                    print('Downloaded and verified ' + row['id'], flush=True)
                    break
                except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as e:
                    update(row, status='retrying' if attempt < 3 else 'failed', error=str(e))
                    if attempt < 3:
                        time.sleep(10)
            # Incomplete copies use .part and are never exposed as playable .mp4.
        state['status'] = 'complete' if all(s['status'] == 'complete' for s in state['files'].values()) else 'failed'
        state['finished_at'] = time.time()
        save(state_path, state)
        return 0 if state['status'] == 'complete' else 1


def status(config):
    cfg, manifest = load(config)
    root = Path(cfg['output'])
    path = root / '_video_download_status.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'files': {}}
    counts = Counter(state['files'].get(row['id'], {}).get('status', 'pending') for row in manifest['videos'])
    active, failures = [], []
    for row in manifest['videos']:
        s = state['files'].get(row['id'], {})
        if s.get('status') in ('downloading', 'verifying', 'retrying', 'waiting_space'):
            part = root / (row['path'] + '.part')
            # rclone atomically renames its own .<hash>.partial into our .part.
            candidates = [part] if part.exists() else [p for p in part.parent.iterdir()
                if p.is_file() and p.name.startswith(part.name + '.') and p.name.endswith('.partial')]
            current = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
            active.append({'path': row['path'], **s, 'bytes': current.stat().st_size if current else 0,
                           'total_bytes': row['size']})
        elif s.get('status') == 'failed':
            failures.append({'path': row['path'], **s})
    print(json.dumps({'total': len(manifest['videos']), 'counts': dict(counts),
                      'batch_status': state.get('status', 'not_observed'), 'active': active,
                      'failures': failures, 'bandwidth_mib': cfg.get('bandwidth_mib', 8),
                      'last_state_update': state.get('updated_at'), 'output': str(root)},
                     ensure_ascii=False, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['download', 'status'])
    p.add_argument('config')
    args = p.parse_args()
    return download(args.config) if args.command == 'download' else status(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
