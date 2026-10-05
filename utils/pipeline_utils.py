import threading
import queue
import cv2
import time
from pathlib import Path
import numpy as np

class AsyncImageSaver:
    def __init__(self, max_queue_size=200):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.total_saved = 0
        self._created_dirs = set()

    def save(self, img: np.ndarray, path: Path, params=None):
        """非同步儲存圖片"""
        if img is None or img.size == 0:
            return
        self.queue.put((img.copy(), Path(path), params))

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                img, path, params = item
                
                # 確保目錄存在
                parent = path.parent
                parent_key = str(parent)
                if parent_key not in self._created_dirs:
                    try:
                        parent.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                    self._created_dirs.add(parent_key)
                
                try:
                    suffix = path.suffix.lower() or ".png"
                    # cv2.imencode 需要前導點，如 .jpg/.png
                    encode_params = params if params is not None else []
                    success, encoded_img = cv2.imencode(suffix, img, encode_params)
                    if not success:
                        raise RuntimeError("imencode failed")
                    with open(path, "wb") as f:
                        f.write(encoded_img.tobytes())
                    self.total_saved += 1
                except Exception as e:
                    print(f"[AsyncSaver] Error saving {path}: {e}")
                
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        """等待所有任務完成並停止"""
        self.queue.join()
        self.stop_event.set()
        self.worker_thread.join()
        # print(f"[AsyncSaver] Finished. Total saved: {self.total_saved}")


class PrefetchIterator:
    """Run an iterator on a background thread and hand items over a bounded queue.

    Video decoding is CPU-bound inside FFmpeg and releases the GIL, so decoding the
    next frames while the main thread runs pattern matching and OCR on the GPU
    overlaps two otherwise serial stages. Order is preserved exactly: a single
    producer thread puts items in the order the source yields them.

    The queue is bounded so a fast producer cannot grow unboundedly; at 1080p a
    BGR frame is ~6 MB, so the default depth caps the buffer at roughly 50 MB.
    """

    def __init__(self, iterable, maxsize: int = 8):
        self._queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._error = None
        self._sentinel = object()
        self._iterable = iterable
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for item in self._iterable:
                # Block only briefly so a close() during a full queue is noticed.
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    break
        except BaseException as exc:  # re-raised on the consumer thread
            self._error = exc
        finally:
            try:
                self._queue.put(self._sentinel, timeout=1.0)
            except queue.Full:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise self._error
                    raise StopIteration
                continue
            if item is self._sentinel:
                if self._error is not None:
                    raise self._error
                raise StopIteration
            return item

    def close(self):
        """Stop the producer and drain the queue so the thread can exit."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=2.0)
