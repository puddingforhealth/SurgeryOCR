import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import range_download as m


class RangeDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.payload = b'abcdefghij'
        self.dest = self.root/'video.mp4'; self.cache = self.root/'cache'
        self.md5 = hashlib.md5(self.payload).hexdigest()

    def fetch(self, start, count, out):
        self.assertFalse(self.dest.exists())
        out.write_bytes(self.payload[start:start+count])

    def run_download(self, fetch=None, md5=None):
        with redirect_stdout(io.StringIO()), patch.object(m.time, 'sleep'):
            m.download_ranges(len(self.payload), md5 or self.md5, self.dest, self.cache,
                              fetch or self.fetch, chunk_size=4, workers=2)

    def test_exact_ranges_assemble_and_cleanup_after_hash(self):
        self.run_download()
        self.assertEqual(self.dest.read_bytes(), self.payload)
        self.assertFalse(list(self.cache.glob('*.bin')))

    def test_eof_retry_reuses_successful_ranges_after_restart(self):
        def interrupted(start, count, out):
            if start == 4: raise EOFError('unexpected EOF')
            self.fetch(start, count, out)
        with self.assertRaises(RuntimeError): self.run_download(interrupted)
        self.assertFalse(self.dest.exists())
        calls = []
        def resume(start, count, out):
            calls.append(start); self.fetch(start, count, out)
        self.run_download(resume)
        self.assertEqual(calls, [4])
        self.assertEqual(self.dest.read_bytes(), self.payload)

    def test_short_successful_transfer_is_not_accepted(self):
        def short(start, count, out): out.write_bytes(b'x')
        with self.assertRaises(RuntimeError): self.run_download(short)
        self.assertFalse(self.dest.exists())

    def test_rate_limit_cooldown_retries_only_missing_range(self):
        attempts = {0: 0, 4: 0, 8: 0}
        def fetch(start, count, out):
            attempts[start] += 1
            if start == 4 and attempts[start] <= 4: raise RuntimeError('rateLimitExceeded')
            self.fetch(start, count, out)
        self.run_download(fetch)
        self.assertEqual(attempts, {0: 1, 4: 5, 8: 1})

    def test_changed_source_hash_never_publishes(self):
        with self.assertRaises(ValueError): self.run_download(md5='0'*32)
        self.assertFalse(self.dest.exists())
        self.assertFalse(list(self.cache.glob('*.bin')))

    def test_corrupt_saved_chunk_is_refetched(self):
        def interrupted(start, count, out):
            if start == 4: raise EOFError()
            self.fetch(start, count, out)
        with self.assertRaises(RuntimeError): self.run_download(interrupted)
        (self.cache/'00000000.bin').write_bytes(b'xxxx')
        calls = []
        def resume(start, count, out): calls.append(start); self.fetch(start, count, out)
        self.run_download(resume)
        self.assertEqual(sorted(calls), [0, 4])

    def test_existing_different_destination_is_preserved(self):
        self.dest.write_bytes(b'user-original')
        def fetch(start, count, out): out.write_bytes(self.payload[start:start+count])
        with self.assertRaises(ValueError): self.run_download(fetch)
        self.assertEqual(self.dest.read_bytes(), b'user-original')


if __name__ == '__main__': unittest.main()
