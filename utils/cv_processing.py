from __future__ import annotations
import math
import cv2
import numpy as np
from PIL import Image


def binarize(
    image_bgr: np.ndarray, hsv_s_thresh: int = 30, gray_thresh: int = 150
) -> np.ndarray:
    """Return a binary (uint8 0/255) image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    sat_limit = np.clip(hsv_s_thresh, 0, 100)
    sat_threshold = int(round((sat_limit / 100.0) * 255))

    _, low_sat_mask = cv2.threshold(sat, sat_threshold, 255, cv2.THRESH_BINARY_INV)
    _, bright_mask = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(low_sat_mask, bright_mask)
    return binary


def hsv_mask(image_bgr: np.ndarray, ranges) -> np.ndarray:
    """Return a binary (uint8 0/255) mask of pixels inside any of the HSV ranges.

    Each range is {"h": [lo, hi], "s": [lo, hi], "v": [lo, hi]} on OpenCV's scale
    (H 0-179, S and V 0-255), bounds inclusive. Used for overlay text drawn in a
    known colour on a translucent panel: selecting the text by colour ignores
    whatever shows through the panel.
    """
    if not ranges:
        raise ValueError("hsv_mask needs at least one range")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array([r["h"][0], r["s"][0], r["v"][0]], dtype=np.uint8)
        hi = np.array([r["h"][1], r["s"][1], r["v"][1]], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def binarize_pil(image: Image.Image) -> np.ndarray:
    """Return a binary (uint8 0/255) image."""
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    binary = binarize(image_bgr)
    return binary


def calculate_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        print(f"    [DEBUG] 🔴 錯誤: 形狀不匹配! 圖像A: {a.shape}, 圖像B: {b.shape}")
        return float("inf")
    diff = cv2.norm(a, b, cv2.NORM_L2)
    return float(diff / math.sqrt(a.size))


def calculate_ndarray_diff(
    a: np.ndarray, b: np.ndarray, sub_roi_coords: list[int]
) -> float:
    """計算兩張 ndarray 圖像在指定精細區域內的平均RGB顏色差異"""
    try:
        x1, y1, x2, y2 = sub_roi_coords

        # 從兩張圖像中裁剪出精細區域
        a_sub_roi = a[y1:y2, x1:x2]
        b_sub_roi = b[y1:y2, x1:x2]

        # 檢查尺寸是否一致
        if a_sub_roi.shape != b_sub_roi.shape:
            return 0.0

        rmse = calculate_rmse(a_sub_roi, b_sub_roi)
        return rmse

    except Exception as e:
        print(f"計算 ndarray 圖像差異時出錯: {e}")
        return 0.0


def calculate_roi_diff(
    prev_img: Image.Image, curr_img: Image.Image, coords: tuple[int, int, int, int]
) -> tuple[float, np.ndarray | None]:
    """Calculate the difference between two ROI images."""
    try:
        prev_sub_roi = prev_img.crop(coords)
        curr_sub_roi = curr_img.crop(coords)

        prev_arr = np.asarray(prev_sub_roi, dtype=np.float32)
        curr_arr = np.asarray(curr_sub_roi, dtype=np.float32)

        if prev_arr.shape != curr_arr.shape:
            return 0.0, None

        rmse = calculate_rmse(prev_arr, curr_arr)

        diff = cv2.absdiff(prev_arr, curr_arr)
        if diff.ndim == 3:
            coeff = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
            channel_avg = cv2.transform(diff, coeff).squeeze(axis=2)
            rmse_per_pixel = cv2.sqrt(channel_avg)
        else:
            rmse_per_pixel = diff

        return rmse, rmse_per_pixel

    except Exception as e:
        print(f"計算 roi_diff 時出錯: {e}")
        return 0.0, None


def calculate_average_binary_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两个二值化图像的差异比例。
    假设输入已经是二值化图像 (0/255, uint8)。
    """
    if img1.shape != img2.shape:
        return 0.0

    # 优化：假设输入已经是二值化图像，直接进行 XOR
    # 如果不确定输入是否为二值，调用者应负责 binarize
    diff = cv2.bitwise_xor(img1, img2)

    # countNonZero 是 OpenCV 中最快的非零计数方法
    different_pixels = cv2.countNonZero(diff)

    if diff.size == 0:
        return 0.0

    return float(different_pixels / diff.size)


SINGLE_DIGIT_BORDER = 40  # px 左右黑邊寬度門檻
SINGLE_DIGIT_THRESH = 0.03  # 左或右 >3% 白點 ⇒ 不是單一數字


def is_single_digit(bw: np.ndarray) -> bool:
    """Return True if ROI likely contains exactly *one* digit.

    Heuristic: check left & right 40‑pixel columns for white ratio.
    """
    h, w = bw.shape
    border = min(SINGLE_DIGIT_BORDER, w // 3)  # avoid over‑size for narrow ROI
    left_white = np.count_nonzero(bw[:, :border]) / (h * border)
    # right_white = np.count_nonzero(bw[:, -border:]) / (h * border)
    return left_white < SINGLE_DIGIT_THRESH  # and right_white < SINGLE_DIGIT_THRESH


def trim_black_borders(binary_img: np.ndarray, max_border: int = 1) -> np.ndarray:
    """
    去除二值化圖像四個方向的黑邊，保留中間的白色內容，最多保留指定像素的黑邊。

    Args:
        binary_img: 二值化圖像 (0=黑, 255=白)
        max_border: 最多保留的黑邊像素數 (預設1像素)

    Returns:
        裁切後的二值化圖像
    """
    if binary_img.size == 0:
        return binary_img

    h, w = binary_img.shape

    # 找到有白色像素的邊界
    white_mask = (binary_img > 127).astype(np.uint8)
    nonzero = cv2.findNonZero(white_mask)
    if nonzero is None:
        return binary_img

    x, y, w, h = cv2.boundingRect(nonzero)
    left = max(0, x - max_border)
    top = max(0, y - max_border)
    right = min(binary_img.shape[1], x + w + max_border)
    bottom = min(binary_img.shape[0], y + h + max_border)

    trimmed = binary_img[top:bottom, left:right]
    return trimmed


def resize_keep_aspect(
    image: Image.Image,
    max_size: tuple[int, int] = (200, 150),
    *,
    max_upscale: float = 3.0,
) -> Image.Image:
    """等比縮放 PIL 影像至指定最大尺寸，允許最多 max_upscale 倍放大。

    - 若原圖已小於 max_size，允許放大但不超過 max_upscale 倍。
    - 若原圖大於 max_size，按比例縮小以適配。
    """
    try:
        original_width, original_height = image.size
        max_width, max_height = max_size

        if original_width <= 0 or original_height <= 0:
            return image

        if original_width <= max_width and original_height <= max_height:
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y, float(max_upscale))
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            scale_x = max_width / original_width
            scale_y = max_height / original_height
            scale = min(scale_x, scale_y)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

        if new_width <= 0 or new_height <= 0:
            return image

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception:
        return image
