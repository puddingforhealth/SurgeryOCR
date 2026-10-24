"""Resume a single Drive video using verified persistent byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def checksum(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download_ranges(size, md5, destination, cache, fetch, chunk_size=32*1024**2, workers=2):
    """fetch(offset, count, output) must write only that requested byte range."""
    destination, cache = Path(destination), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    spans = [(i, start, min(chunk_size, size-start)) for i, start in enumerate(range(0, size, chunk_size))]
    if not spans:
        raise ValueError('Empty source')
    def part_paths(i):
        return cache / f'{i:08}.bin', cache / f'{i:08}.sha256'
    def get(span):
        i, offset, count = span
        part, proof = part_paths(i)
        if part.exists() and proof.exists() and part.stat().st_size == count and checksum(part) == proof.read_text():
            return part
        temp = part.with_suffix('.partial')
        for attempt in range(4):
            try:
                fetch(offset, count, temp)
                if temp.stat().st_size != count:
                    raise ValueError('Incomplete byte range')
                value = checksum(temp)
                os.replace(temp, part)
                proof.write_text(value)
                print(f'Range {i+1}/{len(spans)} verified ({offset+count}/{size} bytes endpoint)', flush=True)
                return part
            except Exception:
                temp.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    pending = spans
    for cycle in range(3):
        errors, retry = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(get, span): span for span in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(e); retry.append(futures[future])
        if not retry:
            break
        pending = retry
        if cycle < 2:
            print(f'Waiting 30s before retrying {len(retry)} missing ranges', flush=True)
            time.sleep(30)
    if errors:
        raise RuntimeError(f'{len(errors)} ranges failed; verified ranges retained') from errors[0]
    assembled = destination.with_name(destination.name + '.assembling')
    try:
        with assembled.open('wb') as out:
            for i, _, _ in spans:
                with part_paths(i)[0].open('rb') as src:
                    for block in iter(lambda: src.read(4 * 1024**2), b''):
                        out.write(block)
        if assembled.stat().st_size != size or checksum(assembled, 'md5') != md5:
            # Range hashes prove storage integrity, but only the manifest proves source identity.
            for i, _, _ in spans:
                for p in part_paths(i): p.unlink(missing_ok=True)
            raise ValueError('Complete video MD5 mismatch')
        if destination.exists():
            if destination.stat().st_size != size or checksum(destination, 'md5') != md5:
                raise ValueError('Existing destination differs; refusing overwrite')
        else:
            os.rename(assembled, destination)
        for i, _, _ in spans:
            for p in part_paths(i): p.unlink(missing_ok=True)
    finally:
        assembled.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    args = sys.argv[1:]
    real = root / 'bin/rclone-real'
    if 'copyto' not in args:
        os.execv(real, [str(real), *args])
    pos = args.index('copyto')
    source, destination = args[pos+1:pos+3]
    manifest = json.loads((root/'manifest.json').read_text())
    row = next(r for r in manifest['videos'] if source == 'gdrive:' + r['path'])
    expected = (root/'inputs'/row['path']).resolve()
    if Path(destination).resolve() != expected:
        raise ValueError('Destination not in this batch')
    cache = root/'range-parts'/row['id']
    def fetch(offset, count, output):
        cmd = [str(real), '--config', str(root/'private/rclone.conf'), 'cat', source,
               '--drive-root-folder-id', manifest['folder_id'], '--offset', str(offset), '--count', str(count),
               '--disable-http2', '--bwlimit', '16M', '--contimeout', '15s', '--timeout', '20s',
               '--retries', '1', '--low-level-retries', '2', '--log-level', 'ERROR']
        with output.open('wb') as stream:
            subprocess.run(cmd, stdout=stream, check=True, timeout=90)
    started = time.monotonic()
    download_ranges(row['size'], row['md5'], expected, cache, fetch)
    print(f'Complete video verified; download invocation {time.monotonic()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
