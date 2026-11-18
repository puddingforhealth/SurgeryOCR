"""Presentation-only styling for the surgical video review workstation."""

import tkinter as tk
from tkinter import ttk


COLORS = {
    "background": "#0b111b",
    "panel": "#121d2b",
    "raised": "#192839",
    "border": "#2a3e51",
    "text": "#e7eff7",
    "muted": "#9cafc3",
    "accent": "#49dcc6",
    "accent_hover": "#7ce7d7",
    "ink": "#08221f",
    "selection": "#244c60",
    "video": "#03080e",
    "track": "#142332",
}
FONT = ("Microsoft JhengHei UI", 10)
MONO = ("Consolas", 10)


def install_theme(root):
    """Style Tk/ttk without changing commands, variables, or widget bindings."""
    c = COLORS
    root.configure(bg=c["background"])
    for pattern, value in {
        "*Font": FONT,
        "*Background": c["panel"],
        "*Foreground": c["text"],
        "*insertBackground": c["accent"],
        "*selectBackground": c["selection"],
        "*selectForeground": c["text"],
        "*highlightBackground": c["border"],
        "*highlightColor": c["accent"],
        "*activeBackground": c["raised"],
        "*activeForeground": c["text"],
        "*disabledForeground": c["muted"],
        "*Button.relief": "flat",
        "*Button.borderWidth": 0,
        "*Button.padX": 12,
        "*Button.padY": 4,
        "*Button.cursor": "hand2",
        "*Label.padX": 0,
        "*Label.padY": 0,
        "*Menu.borderWidth": 0,
        "*Menu.activeBackground": c["selection"],
        "*Listbox.background": c["raised"],
        "*Listbox.foreground": c["text"],
    }.items():
        root.option_add(pattern, value)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        ".",
        background=c["panel"],
        foreground=c["text"],
        font=FONT,
        bordercolor=c["border"],
        lightcolor=c["border"],
        darkcolor=c["border"],
    )
    style.configure("TFrame", background=c["panel"])
    style.configure(
        "TNotebook", background=c["panel"], borderwidth=0, tabmargins=(0, 0, 0, 0)
    )
    # The selected tab is the lit one (selection fill, white bold text, accent top
    # edge); unselected tabs sit back in the page colour with muted text.
    style.configure(
        "TNotebook.Tab",
        background=c["background"],
        foreground=c["muted"],
        padding=(16, 5),
        borderwidth=1,
        bordercolor=c["border"],
        lightcolor=c["background"],
        darkcolor=c["background"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", c["selection"]), ("active", c["raised"])],
        foreground=[("selected", "#ffffff"), ("active", c["text"])],
        lightcolor=[("selected", c["accent"])],
        bordercolor=[("selected", c["accent"])],
        font=[("selected", ("Microsoft JhengHei UI", 10, "bold"))],
    )
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            name,
            fieldbackground=c["raised"],
            background=c["raised"],
            foreground=c["text"],
            arrowcolor=c["muted"],
            padding=5,
            insertcolor=c["accent"],
            selectbackground=c["selection"],
        )
        style.map(
            name,
            fieldbackground=[("readonly", c["raised"]), ("disabled", c["panel"])],
            foreground=[("disabled", c["muted"]), ("readonly", c["text"])],
            bordercolor=[("invalid", "#f4b37f"), ("focus", c["accent"])],
        )
    style.configure(
        "Treeview",
        background=c["panel"],
        fieldbackground=c["panel"],
        foreground=c["text"],
        rowheight=32,
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Treeview.Heading",
        background=c["raised"],
        foreground=c["muted"],
        font=("Microsoft JhengHei UI", 9, "bold"),
        padding=(8, 9),
        relief="flat",
    )
    style.map(
        "Treeview",
        background=[("selected", c["selection"])],
        foreground=[("selected", "#ffffff")],
    )
    style.map("Treeview.Heading", background=[("active", c["selection"])])
    style.configure(
        "Horizontal.TScale",
        background=c["panel"],
        troughcolor=c["background"],
        borderwidth=0,
        sliderlength=18,
        sliderthickness=12,
    )
    style.map("Horizontal.TScale", background=[("active", c["accent"])])
    for orientation in ("Vertical", "Horizontal"):
        style.configure(
            f"{orientation}.TScrollbar",
            background=c["border"],
            troughcolor=c["panel"],
            arrowcolor=c["muted"],
            borderwidth=0,
            gripcount=0,
            width=10,
        )
    return style


def button(parent, text, command, *, primary=False, **kwargs):
    c = COLORS
    background = c["accent"] if primary else c["raised"]
    foreground = c["ink"] if primary else c["text"]
    widget = tk.Button(
        parent,
        text=text,
        command=command,
        bg=background,
        fg=foreground,
        activebackground=c["accent_hover"] if primary else c["selection"],
        activeforeground=foreground,
        highlightthickness=1,
        highlightbackground=c["border"],
        highlightcolor=c["accent"],
        font=("Microsoft JhengHei UI", 10, "bold") if primary else FONT,
        **kwargs,
    )

    def activate(event=None):
        widget.invoke()
        return "break"

    def hover(event):
        if str(widget.cget("state")) != "disabled":
            widget.configure(bg=c["accent_hover"] if primary else c["selection"])

    widget.bind("<Return>", activate)
    widget.bind("<Enter>", hover)
    widget.bind("<Leave>", lambda event: widget.configure(bg=background))
    return widget


class WidgetHint:
    """A bounded, nonmodal hint that reads dynamic paths when it is displayed."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.timer = None
        self.popup = None
        for event in ("<Enter>", "<FocusIn>"):
            widget.bind(event, self.schedule, add="+")
        for event in ("<Leave>", "<FocusOut>", "<ButtonPress>", "<Unmap>", "<Destroy>"):
            widget.bind(event, self.hide, add="+")

    def schedule(self, event=None):
        self.hide()
        self.timer = self.widget.after(500, self.show)

    def show(self):
        if self.timer is not None:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.popup is not None:
            return
        if not self.widget.winfo_ismapped():
            return
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        self.popup = tk.Toplevel(self.widget.winfo_toplevel())
        self.popup.overrideredirect(True)
        tk.Label(
            self.popup,
            text=text,
            bg=COLORS["raised"],
            fg=COLORS["text"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            padx=10,
            pady=8,
            wraplength=520,
            justify="left",
        ).pack()
        self.popup.update_idletasks()
        width, height = self.popup.winfo_reqwidth(), self.popup.winfo_reqheight()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth() - width - 8)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        if y + height > self.widget.winfo_screenheight() - 8:
            y = self.widget.winfo_rooty() - height - 6
        self.popup.geometry(f"+{max(0, x)}+{max(0, y)}")

    def hide(self, event=None):
        if self.timer is not None:
            try:
                self.widget.after_cancel(self.timer)
            except tk.TclError:
                pass
            self.timer = None
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None


PLAYHEAD = "#ff3b3b"


def tick_step(duration, width, min_gap=80):
    """Seconds between labelled ticks so labels stay at least min_gap pixels apart."""
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if duration <= 0 or width <= 1 or step / duration * (width - 1) >= min_gap:
            return step
    return 3600


def short_time(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes}:{secs:02}"


class TimeRuler(tk.Canvas):
    """Seek bar for the whole video, drawn with the analysis tracks' mapping
    (x = seconds / duration * (width - 1)) so every timeline row lines up.

    Keeps the small part of the ttk.Scale interface the viewer uses: set(), get(),
    configure(to=..., state=...), a DoubleVar and a command called while dragging.
    """

    def __init__(self, parent, variable, command=None, height=24):
        super().__init__(
            parent,
            height=height,
            bg=COLORS["track"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            cursor="hand2",
            takefocus=1,
        )
        self.variable = variable
        self.command = command
        self.maximum = 0.0
        self.enabled = True
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<Button-1>", self._drag, add="+")
        self.bind("<B1-Motion>", self._drag, add="+")
        variable.trace_add("write", lambda *args: self.redraw())

    def configure(self, cnf=None, **kw):
        if "to" in kw:
            self.maximum = max(0.0, float(kw.pop("to")))
        kw.pop("from_", None)
        if "state" in kw:
            self.enabled = str(kw.pop("state")) != "disabled"
        result = super().configure(cnf, **kw) if (cnf or kw) else None
        self.redraw()
        return result

    config = configure

    def set(self, value):
        self.variable.set(min(self.maximum, max(0.0, float(value))))

    def get(self):
        return float(self.variable.get())

    def x_for(self, seconds):
        width = self.winfo_width()
        return (
            seconds / self.maximum * (width - 1)
            if self.maximum > 0 and width > 1
            else 0.0
        )

    def seconds_at(self, x):
        width = self.winfo_width()
        if self.maximum <= 0 or width <= 1:
            return 0.0
        return min(self.maximum, max(0.0, x / (width - 1) * self.maximum))

    def _drag(self, event):
        if not self.enabled or self.maximum <= 0:
            return
        self.set(self.seconds_at(event.x))
        if self.command:
            self.command(str(self.get()))

    def redraw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or self.maximum <= 0:
            return
        step = tick_step(self.maximum, width)
        tick = 0
        while tick <= self.maximum:
            x = self.x_for(tick)
            self.create_line(x, height - 7, x, height, fill=COLORS["muted"])
            if x < width - 30:
                self.create_text(
                    x + 3,
                    height - 4,
                    text=short_time(tick),
                    anchor="sw",
                    fill=COLORS["muted"],
                    font=("Consolas", 8),
                )
            tick += step
        x = self.x_for(self.get())
        self.create_line(x, 0, x, height, fill=PLAYHEAD, width=3, tags="playhead")


class KeyHintBar(tk.Frame):
    """Context hint strip: a badge naming the panel the keys drive, then keycaps
    with what each does. Reads better than one long line of text."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["background"])
        self.current = None

    def show(self, badge, hints):
        if self.current == (badge, tuple(hints)):
            return
        self.current = (badge, tuple(hints))
        for widget in self.winfo_children():
            widget.destroy()
        tk.Label(
            self,
            text=badge,
            bg=COLORS["selection"],
            fg=COLORS["accent"],
            padx=8,
            pady=1,
            font=("Microsoft JhengHei UI", 9, "bold"),
        ).pack(side="left", padx=(0, 12))
        for keys, text in hints:
            tk.Label(
                self,
                text=keys,
                bg=COLORS["raised"],
                fg=COLORS["text"],
                padx=5,
                pady=0,
                font=("Consolas", 9, "bold"),
                highlightthickness=1,
                highlightbackground=COLORS["border"],
            ).pack(side="left", padx=(0, 4))
            tk.Label(
                self,
                text=text,
                bg=COLORS["background"],
                fg=COLORS["muted"],
                font=("Microsoft JhengHei UI", 9),
            ).pack(side="left", padx=(0, 12))
