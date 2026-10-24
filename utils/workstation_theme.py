"""Presentation-only styling for the surgical video review workstation."""
import tkinter as tk
from tkinter import ttk


COLORS = {
    'background': '#0b111b', 'panel': '#121d2b', 'raised': '#192839',
    'border': '#2a3e51', 'text': '#e7eff7', 'muted': '#9cafc3',
    'accent': '#49dcc6', 'accent_hover': '#7ce7d7', 'ink': '#08221f',
    'selection': '#244c60', 'video': '#03080e', 'track': '#142332',
}
FONT = ('Microsoft JhengHei UI', 10)
MONO = ('Consolas', 10)


def install_theme(root):
    """Style Tk/ttk without changing commands, variables, or widget bindings."""
    c = COLORS
    root.configure(bg=c['background'])
    for pattern, value in {
        '*Font': FONT, '*Background': c['panel'], '*Foreground': c['text'],
        '*insertBackground': c['accent'], '*selectBackground': c['selection'],
        '*selectForeground': c['text'], '*highlightBackground': c['border'],
        '*highlightColor': c['accent'], '*activeBackground': c['raised'],
        '*activeForeground': c['text'], '*disabledForeground': c['muted'],
        '*Button.relief': 'flat', '*Button.borderWidth': 0,
        '*Button.padX': 12, '*Button.padY': 4, '*Button.cursor': 'hand2',
        '*Label.padX': 0, '*Label.padY': 0,
        '*Menu.borderWidth': 0, '*Menu.activeBackground': c['selection'],
        '*Listbox.background': c['raised'], '*Listbox.foreground': c['text'],
    }.items():
        root.option_add(pattern, value)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=c['panel'], foreground=c['text'], font=FONT,
                    bordercolor=c['border'], lightcolor=c['border'], darkcolor=c['border'])
    style.configure('TFrame', background=c['panel'])
    style.configure('TNotebook', background=c['panel'], borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure('TNotebook.Tab', background=c['panel'], foreground=c['muted'],
                    padding=(16, 5), borderwidth=0)
    style.map('TNotebook.Tab', background=[('selected', c['raised']), ('active', c['raised'])],
              foreground=[('selected', c['accent']), ('active', c['text'])])
    for name in ('TEntry', 'TCombobox', 'TSpinbox'):
        style.configure(name, fieldbackground=c['raised'], background=c['raised'],
                        foreground=c['text'], arrowcolor=c['muted'], padding=5,
                        insertcolor=c['accent'], selectbackground=c['selection'])
        style.map(name, fieldbackground=[('readonly', c['raised']), ('disabled', c['panel'])],
                  foreground=[('disabled', c['muted']), ('readonly', c['text'])],
                  bordercolor=[('invalid', '#f4b37f'), ('focus', c['accent'])])
    style.configure('Treeview', background=c['panel'], fieldbackground=c['panel'],
                    foreground=c['text'], rowheight=32, borderwidth=0, relief='flat')
    style.configure('Treeview.Heading', background=c['raised'], foreground=c['muted'],
                    font=('Microsoft JhengHei UI', 9, 'bold'), padding=(8, 9), relief='flat')
    style.map('Treeview', background=[('selected', c['selection'])],
              foreground=[('selected', '#ffffff')])
    style.map('Treeview.Heading', background=[('active', c['selection'])])
    style.configure('Horizontal.TScale', background=c['panel'], troughcolor=c['background'],
                    borderwidth=0, sliderlength=18, sliderthickness=12)
    style.map('Horizontal.TScale', background=[('active', c['accent'])])
    for orientation in ('Vertical', 'Horizontal'):
        style.configure(f'{orientation}.TScrollbar', background=c['border'],
                        troughcolor=c['panel'], arrowcolor=c['muted'], borderwidth=0,
                        gripcount=0, width=10)
    return style


def button(parent, text, command, *, primary=False, **kwargs):
    c = COLORS
    background = c['accent'] if primary else c['raised']
    foreground = c['ink'] if primary else c['text']
    widget = tk.Button(parent, text=text, command=command, bg=background, fg=foreground,
                     activebackground=c['accent_hover'] if primary else c['selection'],
                     activeforeground=foreground, highlightthickness=1,
                     highlightbackground=c['border'], highlightcolor=c['accent'],
                     font=('Microsoft JhengHei UI', 10, 'bold') if primary else FONT, **kwargs)
    def activate(event=None):
        widget.invoke()
        return 'break'
    def hover(event):
        if str(widget.cget('state')) != 'disabled':
            widget.configure(bg=c['accent_hover'] if primary else c['selection'])
    widget.bind('<Return>', activate)
    widget.bind('<Enter>', hover)
    widget.bind('<Leave>', lambda event: widget.configure(bg=background))
    return widget


class WidgetHint:
    """A bounded, nonmodal hint that reads dynamic paths when it is displayed."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.timer = None
        self.popup = None
        for event in ('<Enter>', '<FocusIn>'):
            widget.bind(event, self.schedule, add='+')
        for event in ('<Leave>', '<FocusOut>', '<ButtonPress>', '<Unmap>', '<Destroy>'):
            widget.bind(event, self.hide, add='+')

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
        tk.Label(self.popup, text=text, bg=COLORS['raised'], fg=COLORS['text'],
                 highlightthickness=1, highlightbackground=COLORS['border'],
                 padx=10, pady=8, wraplength=520, justify='left').pack()
        self.popup.update_idletasks()
        width, height = self.popup.winfo_reqwidth(), self.popup.winfo_reqheight()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth()-width-8)
        y = self.widget.winfo_rooty()+self.widget.winfo_height()+6
        if y+height > self.widget.winfo_screenheight()-8:
            y = self.widget.winfo_rooty()-height-6
        self.popup.geometry(f'+{max(0, x)}+{max(0, y)}')

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
