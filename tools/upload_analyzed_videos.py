#!/usr/bin/env python3
"""Prepare and sequentially upload analyzed videos through the installed rcs CLI.

Standard library only. No GPU work, automatic deletion, or GPU allocation changes.
The manifest and journal stay in gitignored build/; they can contain private names.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/rcs-video-upload'
GIB = 1024 ** 3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def batch_lock(path):
    """OS-owned lock: a crash releases it without deleting another run's lock."""
    path = Path(str(path) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('Another upload is using this state file; wait for it to finish.') from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def discover(root):
    videos = sorted(p for p in (root / 'data').rglob('*') if p.is_file() and p.suffix.lower() == '.mp4')
    stems = {}
    for video in videos:
        stems[video.stem] = stems.get(video.stem, 0) + 1
    selected = []
    for video in videos:
        evidence = []
        candidates = [video.with_suffix('') / 'stage_analysis.json']
        # A stem-only golden output cannot disambiguate two videos with the same stem.
        if stems[video.stem] == 1:
            candidates.append(root / 'data' / video.stem / 'stage_analysis.json')
            candidates += [root / 'runs' / run / video.stem / 'stage_analysis.json'
                           for run in ('baseline', 'candidate')]
        for path in dict.fromkeys(candidates):
            if not path.is_file():
                continue
            try:
                result = json.loads(path.read_text(encoding='utf-8'))
                count = result.get('total_frames', 0)
                ocr_files = list(path.parent.glob('*_ocr_testing.jsonl'))
                if isinstance(count, int) and count > 0 and ocr_files:
                    evidence.append({'path': path.relative_to(root).as_posix(),
                                     'total_frames': count, 'ocr_files': len(ocr_files)})
            except (ValueError, AttributeError):
                continue
        if evidence:
            if video.is_symlink() or not video.resolve().is_relative_to((root / 'data').resolve()):
                raise ValueError('Video must be a regular file within data/: ' + str(video))
            before = video.stat()
            print('Hashing: ' + video.relative_to(root).as_posix(), flush=True)
            digest = sha256(video)
            after = video.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Video changed while preparing manifest: ' + str(video))
            selected.append({'local_path': video.relative_to(root).as_posix(), 'bytes': before.st_size,
                             'sha256': digest, 'analysis_evidence': evidence})
    if not selected:
        raise ValueError('No videos with stage analysis and OCR results were found.')
    return {'version': 1, 'video_count': len(selected),
            'total_video_bytes': sum(row['bytes'] for row in selected), 'videos': selected}


def load_videos(path, root):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Manifest must be a JSON object.')
    rows = manifest.get('videos')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must contain a nonempty videos list. Run prepare first.')
    videos, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('local_path'), str):
            raise ValueError('Each manifest video must have a local_path string.')
        value = row.get('local_path', '')
        rel = PurePosixPath(value)
        if (not value or '\\' in value or any(ord(c) < 32 for c in value)
                or rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2 or rel.parts[0] != 'data'):
            raise ValueError('Manifest path must be a relative path inside data/.')
        key = PurePosixPath(*rel.parts[1:]).as_posix()
        if key.casefold() in seen:
            raise ValueError('Duplicate destination in manifest: ' + key)
        seen.add(key.casefold())
        local = root.joinpath(*rel.parts)
        if (local.is_symlink() or not local.is_file() or local.suffix.lower() != '.mp4'
                or not local.resolve().is_relative_to((root / 'data').resolve())):
            raise ValueError('Missing or invalid local video: ' + str(local))
        if type(row.get('bytes')) is not int or row['bytes'] <= 0 or local.stat().st_size != row['bytes']:
            raise ValueError('Local video size differs from manifest: ' + str(local))
        if not re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', '')) or not row.get('analysis_evidence'):
            raise ValueError('Each video needs SHA-256 and analysis evidence.')
        videos.append({'local_path': rel.as_posix(), 'key': key, 'local': str(local),
                       'bytes': row['bytes'], 'sha256': row['sha256']})
    return videos


def normalize_remote_dir(value):
    path = PurePosixPath(value)
    if not value.startswith('/') or '..' in path.parts or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('--remote-dir must be an absolute POSIX path without traversal or control characters.')
    return path.as_posix()


# A small Python 3 helper sent as argv through rcs exec; filenames stay JSON data.
# Filesystem/hash checks execute on the remote host, without copying video bytes back.
REMOTE_PROBE = r'''
import base64, hashlib, json, os, pathlib, shutil, stat, sys
p = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
path = pathlib.Path(p['path'])
if p['mkdir']:
    path.parent.mkdir(parents=True, exist_ok=True)
parent = path.parent
while not parent.exists():
    parent = parent.parent
result = {'exists': False, 'free': shutil.disk_usage(str(parent)).free}
try:
    before = path.lstat()
except FileNotFoundError:
    before = None
if before is not None:
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Remote destination is not a regular file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Remote file changed while verifying: ' + str(path))
    result.update(exists=True, bytes=before.st_size, sha256=digest.hexdigest())
print(p['marker'] + json.dumps(result, separators=(',', ':')), flush=True)
'''


def parse_probe(output, marker):
    matches = [line[len(marker):] for line in output.splitlines() if line.startswith(marker)]
    if len(matches) != 1:
        raise ValueError('Could not read exactly one remote filesystem response.')
    result = json.loads(matches[0])
    if type(result.get('free')) is not int or result['free'] < 0 or type(result.get('exists')) is not bool:
        raise ValueError('Invalid remote filesystem response.')
    if result['exists'] and (type(result.get('bytes')) is not int or result['bytes'] < 0
                             or not re.fullmatch(r'[0-9a-f]{64}', result.get('sha256', ''))):
        raise ValueError('Invalid remote file checksum response.')
    return result


class Rcs:
    def __init__(self, executable, target, remote_python):
        self.executable = executable
        self.remote_python = remote_python
        version = subprocess.run([executable, '--version'], capture_output=True,
                                 encoding='utf-8', errors='replace', timeout=30)
        match = re.fullmatch(r'rcs (\d+)\.(\d+)\.(\d+)', version.stdout.strip())
        if version.returncode or not match or tuple(map(int, match.groups())) < (0, 3, 0):
            raise ValueError('This uploader requires rcs 0.3.0 or newer. Update remote-console first.')
        proc = subprocess.run([executable, 'target', 'show', target], capture_output=True,
                              encoding='utf-8', errors='replace', timeout=30)
        if proc.returncode:
            raise RuntimeError('Could not resolve rcs target: ' + proc.stderr.strip())
        settings = json.loads(proc.stdout)
        if settings.get('type') != 'webide':
            raise ValueError('This uploader needs an rcs Web IDE target (verified push), not an SSH target.')
        # Pin the URL for the entire batch even if target use/add changes in another terminal.
        self.base = [executable, '--url', settings['url']]
        self.identity = hashlib.sha256(settings['url'].encode('utf-8')).hexdigest()

    def probe(self, path, mkdir=False):
        marker = 'RCS_UPLOAD_' + uuid.uuid4().hex + ':'
        payload = base64.b64encode(json.dumps({'path': path, 'mkdir': mkdir, 'marker': marker},
                                             ensure_ascii=False).encode('utf-8')).decode('ascii')
        proc = subprocess.run([*self.base, 'exec', self.remote_python, '-c', REMOTE_PROBE, payload],
                              capture_output=True, encoding='utf-8', errors='replace')
        if proc.returncode:
            raise RuntimeError('Remote file/space check failed: ' + (proc.stderr + proc.stdout)[-1500:].strip())
        return parse_probe(proc.stdout, marker)

    def push(self, local, destination):
        proc = subprocess.run([*self.base, 'push', local, destination])
        if proc.returncode:
            raise RuntimeError(f'rcs push failed (exit {proc.returncode}); no automatic retry.')


def manifest_id(videos):
    content = [{key: row[key] for key in ('key', 'bytes', 'sha256')} for row in sorted(videos, key=lambda row: row['key'])]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def upload(videos, client, remote_dir, state_path, reserve, limit=None):
    binding = {'version': 1, 'manifest': manifest_id(videos), 'endpoint': client.identity,
               'remote_dir': remote_dir}
    state_path = Path(state_path)
    with batch_lock(state_path):
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {**binding, 'files': {}}
        if any(state.get(key) != value for key, value in binding.items()) or not isinstance(state.get('files'), dict):
            raise ValueError('State belongs to a different manifest or destination. Choose a different --state file.')
        def record(row, status, **details):
            state['files'][row['key']] = {'status': status, 'bytes': row['bytes'], 'sha256': row['sha256'],
                                          'at': datetime.now(timezone.utc).isoformat(), **details}
            save_json(state_path, state)
        count = 0
        confirmed = 0
        for index, row in enumerate(videos, 1):
            print(f"[{index}/{len(videos)}] {row['key']} ({row['bytes']/GIB:.3f} GiB)", flush=True)
            destination = str(PurePosixPath(remote_dir) / row['key'])
            try:
                if sha256(row['local']) != row['sha256']:
                    raise ValueError('Local content changed since manifest preparation; prepare a new manifest.')
                info = client.probe(destination)
                if info['exists']:
                    if info['bytes'] != row['bytes'] or info['sha256'] != row['sha256']:
                        raise ValueError('Remote file already exists with different content; it was not overwritten.')
                    record(row, 'verified', action='skipped_matching_remote')
                    confirmed += 1
                    print('  Remote SHA-256 matches; skipped.', flush=True)
                    continue
                if limit is not None and count >= limit:
                    print(f'Upload limit reached; {confirmed}/{len(videos)} verified this run.', flush=True)
                    return 0
                required = row['bytes'] + reserve
                if info['free'] < required:
                    record(row, 'blocked_space', free=info['free'], required=required)
                    print(f"STOP: {info['free']/GIB:.2f} GiB free; need {required/GIB:.2f} GiB "
                          f"including {reserve/GIB:.2f} GiB reserve. {len(videos)-confirmed} videos remain.", flush=True)
                    return 3
                info = client.probe(destination, mkdir=True)
                if info['exists'] or info['free'] < required:
                    raise RuntimeError('Remote destination or free space changed; rerun to recheck safely.')
                record(row, 'uploading')
                client.push(row['local'], destination)
                info = client.probe(destination)
                if not info['exists'] or info['bytes'] != row['bytes'] or info['sha256'] != row['sha256']:
                    raise RuntimeError('Remote file does not match the manifest after upload.')
                record(row, 'verified', action='uploaded')
                confirmed += 1
                count += 1
            except KeyboardInterrupt:
                record(row, 'interrupted')
                print('Interrupted. Rerun to verify completed files and continue.', file=sys.stderr)
                return 130
            except Exception as exc:
                record(row, 'failed', error=str(exc))
                print(f'STOP: {exc}', file=sys.stderr)
                return 1
        print(f'Complete: {confirmed}/{len(videos)} videos verified; {count} newly uploaded.', flush=True)
        return 0


def find_rcs(value):
    candidate = value or shutil.which('rcs.exe') or shutil.which('rcs')
    if not candidate and os.environ.get('LOCALAPPDATA'):
        candidate = str(Path(os.environ['LOCALAPPDATA']) / 'remote-console/bin/rcs.exe')
    if not candidate or not Path(candidate).is_file():
        raise ValueError('rcs is not installed/on PATH. Install remote-console or pass --rcs PATH.')
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'plan', 'upload'),
                        help='prepare local manifest; plan without network; upload sequentially')
    parser.add_argument('--manifest', type=Path, default=WORK / 'manifest.json')
    parser.add_argument('--state', type=Path, default=WORK / 'state.json')
    parser.add_argument('--target', default='zettabyte')
    parser.add_argument('--remote-dir', default='/root/SurgeryOCR/data', help='absolute remote video root; data/ subfolders are preserved')
    parser.add_argument('--remote-python', default='python3', help='remote Python 3 for space/hash checks only')
    parser.add_argument('--reserve-gib', type=float, default=2, help='leave this much free space after each upload (default: 2)')
    parser.add_argument('--limit', type=int, help='maximum newly uploaded videos this run')
    parser.add_argument('--smallest-first', action='store_true', help='start with smaller videos; default preserves manifest order')
    parser.add_argument('--rcs', help='path to native rcs executable')
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0 or (args.limit is not None and args.limit < 1):
            raise ValueError('Reserve must be finite and nonnegative; limit must be positive.')
        remote_dir = normalize_remote_dir(args.remote_dir)
        if args.action == 'prepare':
            if args.manifest.exists():
                raise ValueError('Manifest already exists. Use a new --manifest path to prepare another snapshot.')
            with batch_lock(args.manifest):
                if args.manifest.exists():
                    raise ValueError('Manifest was created by another process.')
                save_json(args.manifest, discover(ROOT))
            print('Prepared: ' + str(args.manifest))
        videos = load_videos(args.manifest, ROOT)
        if args.smallest_first:
            videos.sort(key=lambda row: (row['bytes'], row['key']))
        print(f"{len(videos)} analyzed videos, {sum(row['bytes'] for row in videos)/GIB:.2f} GiB; "
              f"target={args.target}, destination={remote_dir}", flush=True)
        if args.action != 'upload':
            for index, row in enumerate(videos, 1):
                print(f"{index:2}. {row['bytes']/GIB:6.3f} GiB  {row['key']}")
            print('Local plan only; nothing uploaded. Manifest: ' + str(args.manifest))
            return 0
        client = Rcs(find_rcs(args.rcs), args.target, args.remote_python)
        return upload(videos, client, remote_dir, args.state, math.ceil(args.reserve_gib * GIB), args.limit)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print('upload: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
