#!/usr/bin/env python3
"""Interactive ROI picker for annotating machine-screen regions.

A lightweight dev tool for grabbing pixel coordinates and binarized templates
from surgery videos (or screenshots). Typical uses:
  - locate UI text/labels on a new machine layout and read out [x1, y1, x2, y2]
  - preview binarization with adjustable thresholds before saving a template
  - save binarized (and raw) crops as pattern/anchor cache images

Usage:
    python roi_picker.py                       # open file via dialog
    python roi_picker.py --video data/xxx.mp4  # open a video
    python roi_picker.py --video shot.png      # open a single image

Controls:
    mouse drag          draw ROI rectangle
    Left / Right        prev / next frame  (Shift: +-10, Ctrl: +-100)
    W / A / S / D       nudge rectangle by 1 px (Shift: 5 px)
    coordinate fields   type exact values, then "套用座標"
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.cv_processing import binarize

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
PREVIEW_ZOOM = 4
PREVIEW_MAX_W = 360
PREVIEW_MAX_H = 160


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """cv2.imread replacement that survives non-ASCII Windows paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix if path.suffix else ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


class RoiPicker(tk.Frame):
    def __init__(self, master: tk.Tk, source: Optional[Path] = None):
        super().__init__(master)
        self.master = master
        master.title("ROI Picker - 圈選座標工具")
        self.pack(fill="both", expand=True)

        self.cap: Optional[cv2.VideoCapture] = None
        self.source_path: Optional[Path] = None
        self.is_image = False
        self.total_frames = 0
        self.frame_idx = 0
        self.frame_bgr: Optional[np.ndarray] = None
        self.scale = 1.0
        self.rect: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2 original px
        self._drag_start: Optional[Tuple[int, int]] = None
        self._photo = None
        self._preview_photos = []

        self._build_ui()
        self._bind_keys()

        if source:
            self._open_source(source)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill="x", padx=6, pady=4)
        tk.Button(top, text="開啟影片/圖片", command=self._open_dialog).pack(side="left")
        self.lbl_source = tk.Label(top, text="(未載入)", anchor="w")
        self.lbl_source.pack(side="left", padx=8)

        nav = tk.Frame(self)
        nav.pack(fill="x", padx=6)
        self.slider = tk.Scale(nav, from_=0, to=0, orient="horizontal",
                               showvalue=False, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True)
        self.ent_frame = tk.Entry(nav, width=8)
        self.ent_frame.pack(side="left", padx=4)
        self.ent_frame.bind("<Return>", self._on_goto_frame)
        self.lbl_total = tk.Label(nav, text="/ 0")
        self.lbl_total.pack(side="left")

        body = tk.Frame(self)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        self.canvas = tk.Canvas(body, bg="#222", cursor="tcross",
                                width=MAX_DISPLAY_W, height=MAX_DISPLAY_H)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)

        side = tk.Frame(body, width=380)
        side.pack(side="left", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        self.lbl_mouse = tk.Label(side, text="滑鼠: -", anchor="w")
        self.lbl_mouse.pack(fill="x")

        coord = tk.LabelFrame(side, text="ROI 座標 [x1, y1, x2, y2]")
        coord.pack(fill="x", pady=4)
        row = tk.Frame(coord)
        row.pack(fill="x", pady=2)
        self.coord_vars = [tk.StringVar() for _ in range(4)]
        for i, name in enumerate(["x1", "y1", "x2", "y2"]):
            tk.Label(row, text=name).pack(side="left")
            tk.Entry(row, width=6, textvariable=self.coord_vars[i]).pack(side="left", padx=(0, 6))
        btns = tk.Frame(coord)
        btns.pack(fill="x", pady=2)
        tk.Button(btns, text="套用座標", command=self._apply_coord_fields).pack(side="left", padx=2)
        tk.Button(btns, text="複製座標", command=self._copy_coords).pack(side="left", padx=2)
        self.lbl_size = tk.Label(coord, text="尺寸: -", anchor="w")
        self.lbl_size.pack(fill="x")

        binf = tk.LabelFrame(side, text="二值化參數 (binarize)")
        binf.pack(fill="x", pady=4)
        self.var_sat = tk.IntVar(value=30)
        self.var_gray = tk.IntVar(value=150)
        tk.Scale(binf, from_=0, to=100, orient="horizontal", label="HSV 飽和度上限 (S%)",
                 variable=self.var_sat, command=lambda _v: self._update_preview()).pack(fill="x")
        tk.Scale(binf, from_=0, to=255, orient="horizontal", label="灰階亮度下限",
                 variable=self.var_gray, command=lambda _v: self._update_preview()).pack(fill="x")

        prev = tk.LabelFrame(side, text="預覽 (原始 / 二值化)")
        prev.pack(fill="x", pady=4)
        self.preview_raw = tk.Label(prev, bg="#111")
        self.preview_raw.pack(fill="x", pady=2)
        self.preview_bin = tk.Label(prev, bg="#111")
        self.preview_bin.pack(fill="x", pady=2)
        self.lbl_white = tk.Label(prev, text="白色像素比例: -", anchor="w")
        self.lbl_white.pack(fill="x")

        quick = tk.LabelFrame(side, text="機型樣板快速存檔 (自動路徑+記座標)")
        quick.pack(fill="x", pady=4)
        qrow1 = tk.Frame(quick)
        qrow1.pack(fill="x", pady=1)
        tk.Label(qrow1, text="機型 ID").pack(side="left")
        self.var_machine = tk.StringVar(value="3")
        tk.Entry(qrow1, width=4, textvariable=self.var_machine).pack(side="left", padx=(2, 10))
        tk.Label(qrow1, text="區域名").pack(side="left")
        self.var_region = tk.StringVar()
        tk.Entry(qrow1, width=12, textvariable=self.var_region).pack(side="left", padx=2)
        qrow2 = tk.Frame(quick)
        qrow2.pack(fill="x", pady=1)
        self.var_kind = tk.StringVar(value="pattern")
        for text, val in [("Pattern樣板", "pattern"), ("Header樣板", "header"), ("數值區座標", "value")]:
            tk.Radiobutton(qrow2, text=text, variable=self.var_kind, value=val).pack(side="left")
        qrow3 = tk.Frame(quick)
        qrow3.pack(fill="x", pady=1)
        tk.Label(qrow3, text="樣板標籤").pack(side="left")
        self.var_label = tk.StringVar()
        tk.Entry(qrow3, width=20, textvariable=self.var_label).pack(side="left", padx=2, fill="x", expand=True)
        tk.Button(quick, text="快速儲存", command=self._quick_save).pack(fill="x", pady=2)

        save = tk.LabelFrame(side, text="手動存檔")
        save.pack(fill="x", pady=4)
        tk.Button(save, text="存二值化樣板 PNG", command=lambda: self._save_crop(binary=True)).pack(fill="x", pady=1)
        tk.Button(save, text="存原始裁切 PNG", command=lambda: self._save_crop(binary=False)).pack(fill="x", pady=1)
        tk.Button(save, text="存整張畫面 PNG", command=self._save_full_frame).pack(fill="x", pady=1)

        self.status = tk.Label(self, text="拖曳滑鼠圈選區域；方向鍵切換幀 (Shift ±10, Ctrl ±100)；WASD 微調位置",
                               anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")

    def _bind_keys(self):
        m = self.master
        m.bind("<Left>", lambda e: self._step(-self._step_size(e)))
        m.bind("<Right>", lambda e: self._step(self._step_size(e)))
        for key, (dx, dy) in {"a": (-1, 0), "d": (1, 0), "w": (0, -1), "s": (0, 1)}.items():
            m.bind(key, lambda e, dx=dx, dy=dy: self._nudge(dx, dy))
            m.bind(key.upper(), lambda e, dx=dx, dy=dy: self._nudge(dx * 5, dy * 5))

    def _typing_in_entry(self) -> bool:
        """True when keyboard focus is inside a text entry — global hotkeys must not fire."""
        return isinstance(self.master.focus_get(), tk.Entry)

    @staticmethod
    def _step_size(event) -> int:
        if event.state & 0x0004:  # Ctrl
            return 100
        if event.state & 0x0001:  # Shift
            return 10
        return 1

    # ------------------------------------------------------------- source IO
    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="選擇影片或圖片",
            filetypes=[("影片/圖片", "*.mp4 *.MP4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp"),
                       ("所有檔案", "*.*")])
        if path:
            self._open_source(Path(path))

    def _open_source(self, path: Path):
        if not path.exists():
            messagebox.showerror("錯誤", f"路徑不存在: {path}")
            return
        if self.cap:
            self.cap.release()
            self.cap = None

        self.source_path = path
        self.is_image = path.suffix.lower() in IMAGE_EXTS

        if self.is_image:
            frame = imread_unicode(path)
            if frame is None:
                messagebox.showerror("錯誤", f"無法讀取圖片: {path}")
                return
            self.total_frames = 1
            self.frame_idx = 0
            self.frame_bgr = frame
        else:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                messagebox.showerror("錯誤", f"無法開啟影片: {path}")
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx = 0
            self._read_frame(0)

        self.slider.configure(to=max(0, self.total_frames - 1))
        self.lbl_total.configure(text=f"/ {self.total_frames - 1}")
        self.lbl_source.configure(text=str(path))
        self._show_frame()
        self._set_status(f"已載入: {path.name} ({self.total_frames} 幀)")

    def _read_frame(self, idx: int) -> bool:
        if self.cap is None:
            return False
        idx = max(0, min(idx, self.total_frames - 1))
        # avoid a costly seek when stepping to the immediate next frame
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx != current_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False
        self.frame_idx = idx
        self.frame_bgr = frame
        return True

    # ------------------------------------------------------------ navigation
    def _step(self, delta: int):
        if self._typing_in_entry():
            return
        if self.is_image or self.cap is None:
            return
        if self._read_frame(self.frame_idx + delta):
            self.slider.set(self.frame_idx)
            self._show_frame()

    def _on_slider(self, value):
        if self.is_image or self.cap is None:
            return
        idx = int(float(value))
        if idx != self.frame_idx:
            if self._read_frame(idx):
                self._show_frame()

    def _on_goto_frame(self, _event=None):
        try:
            idx = int(self.ent_frame.get())
        except ValueError:
            return
        if self._read_frame(idx):
            self.slider.set(self.frame_idx)
            self._show_frame()

    # --------------------------------------------------------------- display
    def _show_frame(self):
        if self.frame_bgr is None:
            return
        h, w = self.frame_bgr.shape[:2]
        self.scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        if self.scale < 1.0:
            rgb = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.configure(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.ent_frame.delete(0, "end")
        self.ent_frame.insert(0, str(self.frame_idx))
        self._draw_rect()
        self._update_preview()

    def _draw_rect(self):
        self.canvas.delete("roi")
        if not self.rect:
            return
        x1, y1, x2, y2 = [v * self.scale for v in self.rect]
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff00", width=2, tags="roi")

    def _to_orig(self, cx: float, cy: float) -> Tuple[int, int]:
        if self.frame_bgr is None:
            return 0, 0
        h, w = self.frame_bgr.shape[:2]
        x = int(round(cx / self.scale))
        y = int(round(cy / self.scale))
        return max(0, min(x, w)), max(0, min(y, h))

    # ----------------------------------------------------------- mouse events
    def _on_motion(self, event):
        if self.frame_bgr is None:
            return
        x, y = self._to_orig(event.x, event.y)
        self.lbl_mouse.configure(text=f"滑鼠: ({x}, {y})")

    def _on_press(self, event):
        if self.frame_bgr is None:
            return
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        x1, y1 = self._to_orig(*self._drag_start)
        x2, y2 = self._to_orig(event.x, event.y)
        self.rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._draw_rect()
        self._sync_coord_fields()

    def _on_release(self, event):
        if self._drag_start is None:
            return
        self._on_drag(event)
        self._drag_start = None
        self._update_preview()

    def _nudge(self, dx: int, dy: int):
        if self._typing_in_entry():
            return
        if not self.rect or self.frame_bgr is None:
            return
        h, w = self.frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.rect
        if x1 + dx < 0 or x2 + dx > w or y1 + dy < 0 or y2 + dy > h:
            return
        self.rect = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self._draw_rect()
        self._sync_coord_fields()
        self._update_preview()

    # ------------------------------------------------------------ coordinates
    def _sync_coord_fields(self):
        if not self.rect:
            return
        for var, v in zip(self.coord_vars, self.rect):
            var.set(str(v))
        x1, y1, x2, y2 = self.rect
        self.lbl_size.configure(text=f"尺寸: {x2 - x1} x {y2 - y1}")

    def _apply_coord_fields(self):
        try:
            vals = [int(v.get()) for v in self.coord_vars]
        except ValueError:
            messagebox.showwarning("警告", "座標必須是整數")
            return
        x1, y1, x2, y2 = vals
        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("警告", "需要 x2 > x1 且 y2 > y1")
            return
        self.rect = (x1, y1, x2, y2)
        self._draw_rect()
        self._sync_coord_fields()
        self._update_preview()

    def _copy_coords(self):
        if not self.rect:
            return
        text = json.dumps(list(self.rect))
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        self._set_status(f"已複製到剪貼簿: {text}")

    # --------------------------------------------------------------- preview
    def _get_crop(self) -> Optional[np.ndarray]:
        if self.frame_bgr is None or not self.rect:
            return None
        x1, y1, x2, y2 = self.rect
        crop = self.frame_bgr[y1:y2, x1:x2]
        return crop if crop.size else None

    def _get_binary(self) -> Optional[np.ndarray]:
        crop = self._get_crop()
        if crop is None:
            return None
        return binarize(crop, hsv_s_thresh=self.var_sat.get(), gray_thresh=self.var_gray.get())

    def _update_preview(self):
        crop = self._get_crop()
        self._preview_photos = []
        if crop is None:
            self.preview_raw.configure(image="")
            self.preview_bin.configure(image="")
            self.lbl_white.configure(text="白色像素比例: -")
            return
        binary = self._get_binary()
        white_ratio = np.count_nonzero(binary) / binary.size if binary.size else 0.0
        self.lbl_white.configure(text=f"白色像素比例: {white_ratio:.4f}")

        for img, widget in ((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), self.preview_raw),
                            (binary, self.preview_bin)):
            h, w = img.shape[:2]
            zoom = min(PREVIEW_ZOOM, PREVIEW_MAX_W / w, PREVIEW_MAX_H / h)
            zoom = max(zoom, 0.1)
            disp = cv2.resize(img, (max(1, int(w * zoom)), max(1, int(h * zoom))),
                              interpolation=cv2.INTER_NEAREST)
            photo = ImageTk.PhotoImage(Image.fromarray(disp))
            self._preview_photos.append(photo)
            widget.configure(image=photo)

    # ------------------------------------------------- quick machine capture
    def _machine_dir(self) -> Optional[Path]:
        mid = self.var_machine.get().strip()
        if not mid:
            messagebox.showwarning("警告", "請輸入機型 ID")
            return None
        return Path("data/roi_img_caches") / f"machine_{mid}"

    @staticmethod
    def _load_capture_log(mdir: Path) -> dict:
        p = mdir / "capture_log.json"
        log = {"pattern_regions": {}, "regions": {}}
        if p.exists():
            try:
                # utf-8-sig: tolerate a BOM written by external editors/scripts
                with open(p, "r", encoding="utf-8-sig") as f:
                    log = json.load(f)
            except Exception:
                # never silently start from scratch over an existing file
                backup = p.with_suffix(".corrupt.bak")
                try:
                    p.replace(backup)
                except OSError:
                    pass
                messagebox.showwarning(
                    "capture_log 讀取失敗",
                    f"無法解析 {p}，已備份為 {backup.name} 並改用空白記錄。")
        log.setdefault("pattern_regions", {})
        log.setdefault("regions", {})
        # migrate legacy top-level STAGE entry into pattern_regions
        legacy = log.pop("STAGE", None)
        if legacy and (legacy.get("coords") or legacy.get("patterns")):
            log["pattern_regions"].setdefault("STAGE", legacy)
        return log

    def _write_capture_log(self, mdir: Path, log: dict):
        log["binarize"] = {"hsv_s_thresh": self.var_sat.get(), "gray_thresh": self.var_gray.get()}
        with open(mdir / "capture_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    def _quick_save(self):
        """Save the current selection to the conventional machine cache layout:
        STAGE patterns  -> machine_<id>/STAGE/<auto_id>.png + .npy (fixed shared box)
        header template -> machine_<id>/roi_headers/<region>.png
        value region    -> coordinates only
        All coordinates are recorded in machine_<id>/capture_log.json.
        """
        if not self.rect:
            messagebox.showwarning("警告", "請先圈選區域")
            return
        mdir = self._machine_dir()
        if mdir is None:
            return
        kind = self.var_kind.get()
        log = self._load_capture_log(mdir)
        src = self.source_path.stem if self.source_path else "unknown"

        name = self.var_region.get().strip()
        if not name:
            messagebox.showwarning("警告", "請填「區域名」(如 STAGE、PEDAL、ASP)")
            return

        if kind == "pattern":
            preg = log["pattern_regions"].setdefault(name, {"coords": None, "patterns": {}})
            logged = preg.get("coords")
            if logged and list(self.rect) != list(logged):
                use_logged = messagebox.askyesno(
                    "座標不一致",
                    f"已記錄的 {name} 框為 {logged}\n目前圈選為 {list(self.rect)}\n\n"
                    "是 = 改用已記錄座標（同區所有樣板必須同框）\n否 = 以目前圈選覆蓋記錄")
                if use_logged:
                    self.rect = tuple(logged)
                    self._draw_rect()
                    self._sync_coord_fields()
                    self._update_preview()
            # label may be left empty and auto-filled later (e.g. by reading the
            # saved template images); the compile step flags unlabeled patterns
            label = self.var_label.get().strip()
            binary = self._get_binary()
            if binary is None:
                return
            pat_dir = mdir / name
            pat_dir.mkdir(parents=True, exist_ok=True)
            existing = [int(p.stem) for p in pat_dir.glob("*.png") if p.stem.isdigit()]
            pid = max(existing, default=0) + 1
            out = pat_dir / f"{pid}.png"
            imwrite_unicode(out, binary)
            np.save(str(out.with_suffix(".npy")), binary)
            preg["coords"] = list(self.rect)
            entry = {"frame": self.frame_idx, "video": src}
            if label:
                entry["label"] = label
            preg.setdefault("patterns", {})[str(pid)] = entry
            self._write_capture_log(mdir, log)
            self._set_status(f"{name} 樣板 #{pid}『{label or '(未標記)'}』已存: {out}  座標={list(self.rect)}")
            self.var_label.set("")
            return
        region = log.setdefault("regions", {}).setdefault(name, {})
        if kind == "header":
            binary = self._get_binary()
            if binary is None:
                return
            hdir = mdir / "roi_headers"
            hdir.mkdir(parents=True, exist_ok=True)
            out = hdir / f"{name}.png"
            imwrite_unicode(out, binary)
            region["header_coords"] = list(self.rect)
            region["header_frame"] = self.frame_idx
            region["header_video"] = src
            self._write_capture_log(mdir, log)
            self._set_status(f"Header 樣板已存: {out}  座標={list(self.rect)}")
        else:  # value region: coordinates only, no image
            region["value_coords"] = list(self.rect)
            region["value_frame"] = self.frame_idx
            region["value_video"] = src
            self._write_capture_log(mdir, log)
            self._set_status(f"數值區座標已記錄: {name} = {list(self.rect)} → capture_log.json")

    # ----------------------------------------------------------------- saving
    def _save_crop(self, binary: bool):
        img = self._get_binary() if binary else self._get_crop()
        if img is None:
            messagebox.showwarning("警告", "請先圈選區域")
            return
        kind = "binary" if binary else "raw"
        default_name = f"{self.source_path.stem if self.source_path else 'crop'}_f{self.frame_idx}_{kind}.png"
        path = filedialog.asksaveasfilename(
            title="儲存裁切圖", defaultextension=".png", initialfile=default_name,
            initialdir=str(Path("data/roi_img_caches")),
            filetypes=[("PNG", "*.png")])
        if not path:
            return
        if imwrite_unicode(Path(path), img):
            # pattern caches are loaded from .npy by the pipeline; keep .png for humans
            if binary:
                np.save(str(Path(path).with_suffix(".npy")), img)
            self._set_status(f"已儲存 {kind} 樣板: {path}{' (+.npy)' if binary else ''}  座標={list(self.rect)}")
        else:
            messagebox.showerror("錯誤", f"儲存失敗: {path}")

    def _save_full_frame(self):
        if self.frame_bgr is None:
            return
        default_name = f"{self.source_path.stem if self.source_path else 'frame'}_f{self.frame_idx}.png"
        path = filedialog.asksaveasfilename(
            title="儲存整張畫面", defaultextension=".png", initialfile=default_name,
            filetypes=[("PNG", "*.png")])
        if not path:
            return
        if imwrite_unicode(Path(path), self.frame_bgr):
            self._set_status(f"已儲存畫面: {path}")

    def _set_status(self, msg: str):
        self.status.configure(text=msg)


def main():
    parser = argparse.ArgumentParser(description="Interactive ROI picker")
    parser.add_argument("--video", type=Path, default=None, help="影片或圖片路徑 (省略則開檔案對話框)")
    args = parser.parse_args()

    root = tk.Tk()
    app = RoiPicker(root, source=args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
