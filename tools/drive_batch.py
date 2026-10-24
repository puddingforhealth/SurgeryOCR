"""Bounded Drive download / RAM analysis / verified result collection.

Run worker on Linux; collect on the workstation. Credentials are supplied separately.
Successful inputs/results are cleaned after workstation acknowledgement. Failed input
copies are discarded after retries; originals remain in Drive and failure logs remain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('Unsafe relative path')
    if any(c in value for c in '\\:\0') or any(ord(c) < 32 for c in value):
        raise ValueError('Nonportable path')
    return p


def prepare(rows):
    result, seen, ids = [], set(), set()
    for row in sorted(rows, key=lambda x: x['Path'].casefold()):
        if not row['Name'].lower().endswith('.mp4'):
            continue
        p = relative(row['Path'])
        key = str(p.with_suffix(''))
        if key.casefold() in seen or row['ID'] in ids:
            raise ValueError('Duplicate video identity/output path')
        md5 = row.get('Hashes', {}).get('md5', '')
        if not re.fullmatch('[0-9a-f]{32}', md5) or row['Size'] <= 0:
            raise ValueError('Video needs size and MD5')
        seen.add(key.casefold())
        ids.add(row['ID'])
        result.append({'id': hashlib.sha256(row['ID'].encode()).hexdigest()[:20],
                       'drive_id': row['ID'], 'path': str(p), 'output': key,
                       'size': row['Size'], 'md5': md5})
    if not result:
        raise ValueError('No MP4 files')
    return result


@contextmanager
def lock(path):
    with Path(path).open('a+b') as f:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not f.read(1):
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def validate_outputs(folder):
    stage = json.loads((folder / 'stage_analysis.json').read_text(encoding='utf-8'))
    stats = json.loads((folder / 'run_stats.json').read_text(encoding='utf-8'))
    if stage.get('total_frames', 0) <= 0 or stats.get('frames') != stage['total_frames']:
        raise ValueError('Missing/inconsistent processed frame counts')
    ocr = list(folder.glob('*ocr*.jsonl'))
    if not ocr:
        raise ValueError('OCR output missing')
    for f in ocr:
        with f.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    json.loads(line)
    return {'frames': stats['frames'], 'seconds': stats.get('total_seconds')}


def package(folder, archive):
    records = []
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():
            raise ValueError('Output symlink refused')
        if p.is_file() and p.name != '_files.json':
            records.append({'path': p.relative_to(folder).as_posix(),
                            'size': p.stat().st_size, 'sha256': digest(p)})
    save(folder / '_files.json', records)
    temp = archive.with_suffix('.part')
    with tarfile.open(temp, 'w:gz') as tar:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(folder).as_posix(), recursive=False)
    os.replace(temp, archive)
    return digest(archive)


def unpack_verified(archive, destination, expected):
    if digest(archive) != expected:
        raise ValueError('Archive checksum mismatch')
    destination = Path(destination)
    staging = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            seen = set()
            for member in tar.getmembers():
                name = str(relative(member.name))
                if not member.isfile() or name.casefold() in seen:
                    raise ValueError('Unsafe/duplicate archive member')
                seen.add(name.casefold())
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        records = json.loads((staging / '_files.json').read_text(encoding='utf-8'))
        expected_names = {'_files.json'}
        for row in records:
            name = str(relative(row['path']))
            expected_names.add(name)
            p = staging / name
            if p.stat().st_size != row['size'] or digest(p) != row['sha256']:
                raise ValueError('Result file checksum mismatch')
        if expected_names != {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file()}:
            raise ValueError('Archive inventory mismatch')
        validate_outputs(staging)
        save(staging / '_received.json', {'archive_sha256': expected, 'received_at': time.time()})
        if destination.exists():
            old = json.loads((destination / '_received.json').read_text(encoding='utf-8'))
            if old['archive_sha256'] != expected:
                raise ValueError('Existing result differs; refusing overwrite')
            for row in records:
                p = destination / row['path']
                if p.stat().st_size != row['size'] or digest(p) != row['sha256']:
                    raise ValueError('Existing result is damaged')
        else:
            os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def worker(root, reserve_gib=20, max_pending=3):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['videos']
    for name in ('inputs', 'results', 'archives', 'logs', 'acks'):
        (root / name).mkdir(exist_ok=True)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'files': {}}
    guard = threading.Lock()
    def update(row, **changes):
        with guard:
            state['files'].setdefault(row['id'], {}).update(changes, updated_at=time.time())
            state['updated_at'] = time.time()
            save(state_path, state)
    def cleanup():
        for row in rows:
            s = state['files'].get(row['id'], {})
            ack = root / 'acks' / (row['id'] + '.json')
            if s.get('status') != 'ready' or not ack.exists():
                continue
            if json.loads(ack.read_text()).get('sha256') != s.get('sha256'):
                continue
            # All paths derived from our validated manifest and fixed batch root.
            for p in (root / 'inputs' / row['path'], root / 'archives' / (row['id'] + '.tgz')):
                p.unlink(missing_ok=True)
            folder = root / 'results' / row['output']
            if folder.exists():
                shutil.rmtree(folder)
            update(row, status='collected')
    work = queue.Queue(maxsize=max_pending)
    stop = threading.Event()
    def producer():
        try:
            for row in rows:
                cleanup()
                if state['files'].get(row['id'], {}).get('status') in ('ready', 'collected'):
                    continue
                # Bound input retention, including analyses waiting for collection.
                while sum(s.get('status') in ('downloading', 'downloaded', 'analyzing', 'ready')
                          for s in list(state['files'].values())) >= max_pending:
                    if stop.wait(3):
                        return
                    cleanup()
                dest = root / 'inputs' / row['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                good = dest.exists() and dest.stat().st_size == row['size'] and digest(dest, 'md5') == row['md5']
                while not good and shutil.disk_usage(root).free < row['size'] + reserve_gib * 2**30:
                    update(row, status='waiting_space')
                    if stop.wait(10):
                        return
                    cleanup()
                if not good:
                    update(row, status='downloading')
                    ok = False
                    for attempt in range(1, 4):
                        update(row, download_attempt=attempt)
                        cmd = [str(root / 'bin/rclone'), '--config', str(root / 'private/rclone.conf'),
                               'copyto', 'gdrive:' + row['path'], str(dest),
                               '--drive-root-folder-id', manifest['folder_id'],
                               '--transfers', '1', '--checkers', '1', '--multi-thread-streams', '0',
                               '--checksum', '--retries', '2', '--low-level-retries', '5',
                               '--contimeout', '30s', '--timeout', '2m', '--stats', '30s']
                        with (root / 'logs' / (row['id'] + '-download.log')).open('ab') as log:
                            rc = subprocess.call(cmd, stdout=log, stderr=log)
                        if rc == 0 and dest.exists() and dest.stat().st_size == row['size'] and digest(dest, 'md5') == row['md5']:
                            ok = True
                            break
                        if stop.wait(5):
                            return
                    if not ok:
                        dest.unlink(missing_ok=True)
                        update(row, status='download_failed')
                        continue
                update(row, status='downloaded')
                work.put(row)
        except Exception as e:
            with guard:
                state['producer_error'] = str(e)
                save(state_path, state)
        finally:
            work.put(None)
    with lock(root / 'worker.lock'):
        # Interrupted in-progress records may be retried; completed results survive.
        for key in ('finished_at', 'status', 'producer_error'):
            state.pop(key, None)
        for s in state['files'].values():
            if s.get('status') not in ('ready', 'collected'):
                s['status'] = 'pending'
        save(state_path, state)
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            row = work.get()
            if row is None:
                break
            output = root / 'results' / row['output']
            output.mkdir(parents=True, exist_ok=True)
            update(row, status='analyzing')
            try:
                cmd = [manifest['python'], '-u', str(root / 'code/run_pipeline.py'),
                       '--video', str(root / 'inputs' / row['path']),
                       '--output-dir', str(output), '--mode', 'ram']
                env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                with (output / 'pipeline.log').open('wb') as log:
                    rc = subprocess.call(cmd, cwd=root / 'code', env=env, stdout=log, stderr=log)
                if rc:
                    raise RuntimeError('Pipeline exit ' + str(rc))
                details = validate_outputs(output)
                save(output / '_source.json', dict(row, folder_id=manifest['folder_id'], mode='ram',
                                                   code_commit=manifest['code_commit']))
                archive = root / 'archives' / (row['id'] + '.tgz')
                checksum = package(output, archive)
                update(row, status='ready', sha256=checksum, archive_bytes=archive.stat().st_size, **details)
            except Exception as e:
                # A failed input can be fetched again from Drive; don't accumulate
                # an unbounded number of failed videos. Retain all logs/results.
                (root / 'inputs' / row['path']).unlink(missing_ok=True)
                update(row, status='analysis_failed', error=str(e))
        thread.join()
        # Keep acknowledging/cleaning until every successful result has been retrieved.
        while any(s.get('status') == 'ready' for s in state['files'].values()):
            cleanup()
            time.sleep(3)
        state['finished_at'] = time.time()
        state['status'] = 'failed' if state.get('producer_error') or any(s.get('status') != 'collected' for s in state['files'].values()) else 'complete'
        save(state_path, state)


def collect(config):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    manifest = json.loads(Path(cfg['manifest']).read_text(encoding='utf-8'))
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save(output / '_batch_manifest.json', manifest)
    cache = Path(cfg['cache']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    base = [cfg['rcs'], '--url', cfg['url']]
    def call(args):
        p = subprocess.run([*base, *args], capture_output=True, encoding='utf-8', errors='replace', timeout=300)
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-1200:])
    with lock(cache / 'collector.lock'):
        while True:
            try:
                snapshot = cache / ('state-' + uuid.uuid4().hex + '.json')
                call(['pull', cfg['remote'] + '/state.json', str(snapshot)])
                state = json.loads(snapshot.read_text(encoding='utf-8'))
                snapshot.unlink()
                save(output / '_batch_status.json', state)
                for row in manifest['videos']:
                    s = state['files'].get(row['id'], {})
                    if s.get('status') != 'ready':
                        continue
                    target = output / row['output']
                    archive = cache / (row['id'] + '.tgz')
                    if not archive.exists() or digest(archive) != s['sha256']:
                        temp = cache / (row['id'] + '-' + uuid.uuid4().hex + '.tgz')
                        call(['pull', cfg['remote'] + '/archives/' + row['id'] + '.tgz', str(temp)])
                        os.replace(temp, archive)
                    unpack_verified(archive, target, s['sha256'])
                    ack = cache / (row['id'] + '.json')
                    save(ack, {'sha256': s['sha256']})
                    call(['push', str(ack), cfg['remote'] + '/acks/' + row['id'] + '.json', '--overwrite'])
                    archive.unlink()
                    print('Collected ' + row['id'], flush=True)
                if state.get('finished_at'):
                    print('Batch ' + state['status'], flush=True)
                    return 0 if state['status'] == 'complete' else 1
            except Exception as e:
                save(cache / 'collector-error.json', {'error': str(e), 'at': time.time()})
                print('Collection retry: ' + type(e).__name__, flush=True)
            time.sleep(15)


def status(config):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    manifest = json.loads(Path(cfg['manifest']).read_text(encoding='utf-8'))
    path = Path(cfg['output']) / '_batch_status.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'files': {}}
    counts = Counter(s.get('status', 'pending') for s in state['files'].values())
    counts['pending'] += len(manifest['videos']) - len(state['files'])
    failures = [{'path': row['path'], **state['files'][row['id']]}
                for row in manifest['videos']
                if state['files'].get(row['id'], {}).get('status', '').endswith('_failed')]
    print(json.dumps({'total': len(manifest['videos']), 'counts': dict(counts),
                      'last_remote_update': state.get('updated_at'),
                      'batch_status': state.get('status', 'running' if path.exists() else 'not_observed'),
                      'failures': failures, 'producer_error': state.get('producer_error'),
                      'output': cfg['output']}, ensure_ascii=False, indent=2))
    return 0


def log_tail(path, limit=8192):
    """Read only the end, including tqdm's carriage-return updates."""
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            text = stream.read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return []
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    return [re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', line).strip()
            for line in text.splitlines() if line.strip()][-3:]


def progress_text(root, now=None):
    """Read-only live view; monitoring never owns or stops the worker."""
    root = Path(root)
    now = time.time() if now is None else now
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'files': {}}
    rows = manifest['videos']
    counts = Counter(state['files'].get(row['id'], {}).get('status', 'pending') for row in rows)
    phase = state.get('status', 'running' if state_path.exists() else 'not_observed')
    lines = ['SurgeryOCR Drive batch | ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
             f"Returned to local: {counts['collected']}/{len(rows)} | Batch: {phase}",
             ' | '.join(f'{key}: {value}' for key, value in sorted(counts.items()) if value),
             f'/tmp free: {shutil.disk_usage(root).free / 2**30:.1f} GiB']
    if state.get('updated_at'):
        lines.append(f"Last state change: {max(0, now - state['updated_at']):.0f}s ago (logs below are live)")
    for row in rows:
        s = state['files'].get(row['id'], {})
        status = s.get('status', 'pending')
        if status in ('pending', 'collected'):
            continue
        lines.extend(['', f"[{status}] {row['path']}"])
        if status == 'analyzing':
            lines.extend('  ' + line for line in log_tail(root / 'results' / row['output'] / 'pipeline.log'))
        elif status == 'downloading':
            lines.append(f"  Source size: {row['size'] / 2**30:.2f} GiB | attempt {s.get('download_attempt', '?')}/3")
            lines.extend('  ' + line for line in log_tail(root / 'logs' / (row['id'] + '-download.log')))
        elif status == 'ready':
            lines.append('  Analysis complete; waiting for verified local collection')
        if s.get('error'):
            lines.append('  ' + s['error'])
    if state.get('producer_error'):
        lines.extend(['', 'Download worker error: ' + state['producer_error']])
    lines.extend(['', 'Read-only monitor. Ctrl+C stops this view only; tmux detach: Ctrl+B, then D.'])
    return '\n'.join(lines)


def watch(root, interval=2, once=False):
    try:
        while True:
            view = progress_text(root)
            if sys.stdout.isatty() and not once:
                print('\033[2J\033[H', end='')
            print(view, flush=True)
            if once:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker')
    w.add_argument('root', type=Path)
    w.add_argument('--reserve-gib', type=int, default=20)
    w.add_argument('--max-pending', type=int, default=3)
    c = sub.add_parser('collect')
    c.add_argument('config')
    s = sub.add_parser('status', help='show the last locally observed batch progress (no network)')
    s.add_argument('config')
    v = sub.add_parser('watch', help='live read-only progress on the remote worker machine')
    v.add_argument('root', type=Path)
    v.add_argument('--interval', type=float, default=2)
    v.add_argument('--once', action='store_true')
    args = p.parse_args()
    if args.command == 'watch':
        if args.interval < 0.5:
            p.error('Watch interval must be >= 0.5 seconds')
        return watch(args.root, args.interval, args.once)
    if args.command == 'worker':
        if args.reserve_gib < 1 or args.max_pending < 2:
            p.error('Need reserve >= 1 GiB and max-pending >= 2')
        worker(args.root, args.reserve_gib, args.max_pending)
        return 0
    return status(args.config) if args.command == 'status' else collect(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
