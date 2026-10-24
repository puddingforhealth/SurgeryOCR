"""PTS-verified frame navigation, retaining the pipeline's OpenCV decoder pixels."""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import av
import cv2


class VideoIndexError(RuntimeError):
    pass


def _index_video(path):
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        time_base = stream.time_base
        declared = stream.frames
        packets = [(p.pts, p.duration, p.is_keyframe) for p in source.demux(stream) if p.size and p.pts is not None]
    packets.sort()
    pts = [p[0] for p in packets]
    kind = 'packets'
    # Otherwise index decoded frames; never assume one packet equals one frame.
    if not declared or len(pts) != declared or len(set(pts)) != len(pts):
        with av.open(str(path)) as source:
            stream = source.streams.video[0]
            time_base = stream.time_base
            pts = [frame.pts for frame in source.decode(stream)]
        kind = 'decoded'
    if not pts or any(p is None for p in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise VideoIndexError('影片時間戳缺漏或重複，無法建立可靠的逐幀索引。')
    deltas = sorted(b-a for a, b in zip(pts, pts[1:]) if b > a)
    last_duration = packets[-1][1] if kind == 'packets' else 0
    last_duration = last_duration or (deltas[len(deltas)//2] if deltas else 1)
    return {'pts': pts, 'time_base': [time_base.numerator, time_base.denominator],
            'last_duration': last_duration, 'kind': kind,
            'keyframes': [p[0] for p in packets if p[2]]}


def video_index(path, cache_dir=None):
    path = Path(path).resolve()
    stat = path.stat()
    fingerprint = [str(path), stat.st_size, stat.st_mtime_ns, av.__version__, 1]
    key = hashlib.sha256(json.dumps(fingerprint).encode('utf8')).hexdigest()
    cache = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())/'surgeryocr-frame-index'
    target = cache / (key+'.json.gz')
    try:
        with gzip.open(target, 'rt', encoding='utf8') as source:
            result = json.load(source)
        if result.get('fingerprint') == fingerprint:
            pts = result['pts']
            if pts and all(isinstance(p, int) for p in pts) and all(b > a for a, b in zip(pts, pts[1:])):
                return result
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        pass
    result = _index_video(path)
    result['fingerprint'] = fingerprint
    temporary = None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, prefix='index-', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
        with gzip.open(temporary, 'wt', encoding='utf8') as destination:
            json.dump(result, destination)
        temporary.replace(target)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


class IndexedVideoCapture:
    """VideoCapture-compatible viewer interface. The analysis pipeline is unchanged."""
    def __init__(self, path, *, cache_dir=None, frame_cache_size=8):
        index = video_index(path, cache_dir)
        self.pts = index['pts']
        self.time_base = Fraction(*index['time_base'])
        self.index_kind = index['kind']
        self.duration = float((self.pts[-1]-self.pts[0]+index['last_duration'])*self.time_base)
        self._cap = cv2.VideoCapture(str(path))
        self._cache = OrderedDict()
        self._cache_limit = max(2, frame_cache_size)
        self._position = 0
        self._decoder_next = 0
        if not self._cap.isOpened():
            raise VideoIndexError('無法開啟影片解碼器。')
        ok, image = self._cap.read()
        if not ok:
            self.release()
            raise VideoIndexError('無法解碼影片的第一幀。')
        self._clock_offset = float(self.pts[0]*self.time_base)-self._cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        self._remember(0, image)
        self._decoder_next = 1

    def frame_time(self, index):
        if index >= len(self.pts):
            return self.duration
        return float((self.pts[max(0, int(index))]-self.pts[0])*self.time_base)

    def frame_at_time(self, seconds):
        tick = self.pts[0]+round(max(0, seconds)/float(self.time_base))
        return max(0, min(len(self.pts)-1, bisect_right(self.pts, tick)-1))

    def _remember(self, index, image):
        self._cache[index] = image
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def _decode_until(self, target):
        while True:
            ok, image = self._cap.read()
            if not ok:
                return None
            seconds = self._cap.get(cv2.CAP_PROP_POS_MSEC)/1000+self._clock_offset
            tick = round(seconds/float(self.time_base))
            index = bisect_left(self.pts, tick)
            if index == len(self.pts) or self.pts[index] != tick:
                raise VideoIndexError('解碼時間戳與索引不一致，已停止顯示以避免錯誤跳幀。')
            self._decoder_next = index+1
            if index > target:
                return None
            if index >= target-self._cache_limit:
                self._remember(index, image)
            if index == target:
                return image

    def read_frame(self, index):
        index = int(index)
        if not self.isOpened() or not 0 <= index < len(self.pts):
            return None
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index].copy()
        if self._decoder_next == index:
            image = self._decode_until(index)
            if image is not None:
                return image.copy()
        margin = .05
        while True:
            start = max(0, self.frame_time(index)-margin)
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
            image = self._decode_until(index)
            if image is not None:
                return image.copy()
            if start == 0:
                raise VideoIndexError(f'無法確認第 {index} 幀的時間戳，已停止跳轉。')
            margin *= 2

    def read(self):
        image = self.read_frame(self._position)
        if image is None:
            return False, None
        self._position += 1
        return True, image

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.pts)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self._position
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.frame_time(max(0, self._position-1))*1000
        return self._cap.get(prop) if self.isOpened() else 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            if not 0 <= int(value) <= len(self.pts):
                return False
            self._position = int(value)
            return True
        if prop == cv2.CAP_PROP_POS_MSEC:
            self._position = self.frame_at_time(value/1000)
            return True
        return False

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cache.clear()
