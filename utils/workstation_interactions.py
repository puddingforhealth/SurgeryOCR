"""Keyboard reference and bounded visual feedback for the review workstation."""
from utils.workstation_theme import COLORS


SHORTCUTS = (
    ('<Control-o>', 'Ctrl + O', '載入影片', 'open'),
    ('<Control-Shift-O>', 'Ctrl + Shift + O', '選擇分析資料夾', 'results'),
    ('<Control-s>', 'Ctrl + S', '儲存 OCR 標註', 'save'),
    ('<Control-l>', 'Ctrl + L', '輸入要跳轉的幀號', 'goto'),
    ('<space>', 'Space', '播放／暫停', 'play'),
    ('<Left>', '← / →', '前一幀／後一幀', 'previous_frame'),
    ('<Right>', '', '', 'next_frame'),
    ('<Shift-Left>', 'Shift + ← / →', '向前／向後約一秒', 'previous_second'),
    ('<Shift-Right>', '', '', 'next_second'),
    ('<Up>', '↑ / ↓', '上一筆／下一筆事件', 'previous_event'),
    ('<Down>', '', '', 'next_event'),
    ('<Home>', 'Home / End', '第一幀／最後一幀', 'first'),
    ('<End>', '', '', 'last'),
    ('<Alt-Key-1>', 'Alt + 1 / 2', 'OCR／手術階段模式', 'ocr'),
    ('<Alt-Key-2>', '', '', 'stage'),
    ('<Alt-Up>', 'Alt + ↑ / ↓', '切換目前模式的 ROI 區域', 'previous_region'),
    ('<Alt-Down>', '', '', 'next_region'),
    ('<Return>', 'Enter', '編輯選取的事件；跳幀欄位則執行跳轉', 'edit'),
    ('<Escape>', 'Esc', '返回影片鍵盤操作', 'review'),
    ('<F1>', 'F1', '快捷鍵與動態回饋設定', 'help'),
)


def navigation_owned_by_widget(widget_class, *, space=False):
    """Native text/selection controls retain their cursor/navigation semantics."""
    owned = {'Entry', 'TEntry', 'Text', 'Spinbox', 'TSpinbox', 'TCombobox', 'Scale', 'TScale', 'TNotebook'}
    if space:
        owned |= {'Button', 'TButton', 'Checkbutton', 'TCheckbutton', 'Radiobutton', 'TRadiobutton'}
    return widget_class in owned


def format_timecode(seconds):
    hundredths = max(0, round(float(seconds) * 100))
    hours, rest = divmod(hundredths, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, fraction = divmod(rest, 100)
    return f'{hours:02}:{minutes:02}:{secs:02}.{fraction:02}'


def blend_color(start, end, fraction):
    channels = [round(int(start[i:i+2], 16)*(1-fraction) + int(end[i:i+2], 16)*fraction)
                for i in (1, 3, 5)]
    return '#' + ''.join(f'{channel:02x}' for channel in channels)


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
            label.configure(bg=COLORS['panel'], fg=COLORS['text'])

    def pulse(self):
        self.stop()
        self._fade(0)

    def _fade(self, step):
        self.timer = None
        fraction = step/8
        for label in self.labels:
            label.configure(bg=blend_color(COLORS['selection'], COLORS['panel'], fraction),
                            fg=blend_color(COLORS['accent'], COLORS['text'], fraction))
        if step < 8:
            self.timer = self.owner.after(40, lambda: self._fade(step+1))
