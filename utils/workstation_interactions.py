"""Keyboard reference and bounded visual feedback for the review workstation."""

from utils.workstation_theme import COLORS


SHORTCUTS = (
    ("<Control-o>", "Ctrl + O", "載入影片", "open"),
    ("<Control-Shift-O>", "Ctrl + Shift + O", "選擇分析資料夾", "results"),
    ("<Control-s>", "Ctrl + S", "儲存 OCR 標註", "save"),
    ("<Control-l>", "Ctrl + L", "輸入要跳轉的幀號", "goto"),
    ("<space>", "Space", "播放／暫停", "play"),
    ("<Left>", "← / →", "前一幀／後一幀（時間軸：上一段／下一段）", "previous_frame"),
    ("<Right>", "", "", "next_frame"),
    ("<Shift-Left>", "Shift + ← / →", "向前／向後約一秒", "previous_second"),
    ("<Shift-Right>", "", "", "next_second"),
    ("<Up>", "↑ / ↓", "上一筆／下一筆事件（時間軸：切換軌道）", "previous_event"),
    ("<Down>", "", "", "next_event"),
    ("<Home>", "Home / End", "第一幀／最後一幀", "first"),
    ("<End>", "", "", "last"),
    ("<Alt-Key-1>", "Alt + 1 / 2", "數值追蹤／狀態追蹤頁面", "ocr"),
    ("<Alt-Key-2>", "", "", "stage"),
    ("<Alt-Up>", "Alt + ↑ / ↓", "切換目前模式的 ROI 區域", "previous_region"),
    ("<Alt-Down>", "", "", "next_region"),
    ("<Return>", "Enter", "編輯選取的事件；跳幀欄位則執行跳轉", "edit"),
    ("<Escape>", "Esc", "返回影片鍵盤操作；時間軸操作中則交回事件紀錄", "review"),
    ("<F1>", "F1", "快捷鍵與動態回饋設定", "help"),
)


# Arrow keys act on the panel the user last clicked: the event list (default)
# or the timeline. The timeline's first row is the whole-video frame axis
# (FRAME_TRACK), where left / right step single frames; on the analysis tracks
# they jump between segments.
EVENTS = "events"
TIMELINE = "timeline"
FRAME_TRACK = "FRAME"
_TIMELINE_ACTIONS = {
    "previous_event": "previous_track",
    "next_event": "next_track",
    "previous_frame": "previous_segment",
    "next_frame": "next_segment",
}
# Hint strip per context: (badge, ((keys, what they do), ...)).
PANEL_HINTS = {
    EVENTS: (
        "事件紀錄",
        (
            ("↑ ↓", "上／下一筆"),
            ("← →", "逐幀"),
            ("雙擊 / Enter", "修正標註"),
            ("右鍵", "刪除／詳情"),
            ("Space", "播放"),
            ("F1", "全部快捷鍵"),
        ),
    ),
    TIMELINE: (
        "時間軸",
        (
            ("↑ ↓", "切換軌道"),
            ("← →", "上／下一段"),
            ("Esc", "回事件紀錄"),
            ("Space", "播放"),
            ("F1", "全部快捷鍵"),
        ),
    ),
    FRAME_TRACK: (
        "幀軸",
        (
            ("↑ ↓", "切換軌道"),
            ("← →", "逐幀"),
            ("Shift + ← →", "約 ±1 秒"),
            ("Esc", "回事件紀錄"),
            ("Space", "播放"),
            ("F1", "全部快捷鍵"),
        ),
    ),
}


def hint_context(panel, track):
    """Which PANEL_HINTS entry describes the keys right now."""
    return FRAME_TRACK if panel == TIMELINE and track == FRAME_TRACK else panel


def route_action(panel, action, track=None):
    """Shortcut action after applying the focused panel's arrow-key meaning."""
    if panel != TIMELINE:
        return action
    if track == FRAME_TRACK and action in ("previous_frame", "next_frame"):
        return action
    return _TIMELINE_ACTIONS.get(action, action)


def step_track(tracks, current, step):
    """Neighbouring track name; stays on the first / last track instead of wrapping."""
    tracks = list(tracks)
    if not tracks:
        return None
    if current not in tracks:
        return tracks[0]
    return tracks[max(0, min(len(tracks) - 1, tracks.index(current) + step))]


def _bounds(segment):
    start = int(segment.get("start_frame", 0))
    return start, int(segment.get("end_frame", start))


def segment_at(segments, frame):
    """The segment containing frame, or None."""
    return next(
        (seg for seg in segments if _bounds(seg)[0] <= frame <= _bounds(seg)[1]), None
    )


def adjacent_segment(segments, frame, step):
    """Next segment starting after frame (step > 0), or the last one starting
    before it (step < 0), like a player's previous / next chapter."""
    ordered = sorted(segments, key=lambda seg: _bounds(seg)[0])
    if step > 0:
        return next((seg for seg in ordered if _bounds(seg)[0] > frame), None)
    return next((seg for seg in reversed(ordered) if _bounds(seg)[0] < frame), None)


def navigation_owned_by_widget(widget_class, *, space=False):
    """Native text/selection controls retain their cursor/navigation semantics."""
    owned = {
        "Entry",
        "TEntry",
        "Text",
        "Spinbox",
        "TSpinbox",
        "TCombobox",
        "Scale",
        "TScale",
        "TNotebook",
    }
    if space:
        owned |= {
            "Button",
            "TButton",
            "Checkbutton",
            "TCheckbutton",
            "Radiobutton",
            "TRadiobutton",
        }
    return widget_class in owned


def format_timecode(seconds):
    hundredths = max(0, round(float(seconds) * 100))
    hours, rest = divmod(hundredths, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, fraction = divmod(rest, 100)
    return f"{hours:02}:{minutes:02}:{secs:02}.{fraction:02}"


def blend_color(start, end, fraction):
    channels = [
        round(
            int(start[i : i + 2], 16) * (1 - fraction)
            + int(end[i : i + 2], 16) * fraction
        )
        for i in (1, 3, 5)
    ]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


class ReadoutFeedback:
    """A single, restartable 320 ms fade; never schedules a continuous animation."""

    def __init__(self, owner, *labels):
        self.owner = owner
        self.labels = labels
        self.timer = None

    def stop(self):
        if self.timer is not None:
            self.owner.after_cancel(self.timer)
            self.timer = None
        for label in self.labels:
            label.configure(bg=COLORS["panel"], fg=COLORS["text"])

    def pulse(self):
        self.stop()
        self._fade(0)

    def _fade(self, step):
        self.timer = None
        fraction = step / 8
        for label in self.labels:
            label.configure(
                bg=blend_color(COLORS["selection"], COLORS["panel"], fraction),
                fg=blend_color(COLORS["accent"], COLORS["text"], fraction),
            )
        if step < 8:
            self.timer = self.owner.after(40, lambda: self._fade(step + 1))
