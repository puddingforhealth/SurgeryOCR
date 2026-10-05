#!/usr/bin/env python3
"""
自動檢測視頻機型並更新 rois.json 映射 (CLI & Library Mode)
"""

import argparse
import json
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
from PIL import Image

from utils.cv_processing import binarize
from utils.get_configs import (
    update_video_machine_mapping, 
    get_video_machine_id,
    load_roi_header_config,
)
from utils.get_paths import resolve_video_analysis_dir

# --------------------------------------------------------------------------
# MachineDetector: multi-anchor machine-type detection (config-driven)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_reference_header_image_cached(ref_path: str) -> Optional[np.ndarray]:
    path = Path(ref_path)
    if not path.exists():
        return None
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img
    except Exception as e:
        print(f"[MachineDetector] Error loading ref image: {e}")
        return None


class MachineDetector:
    """
    多錨點機型偵測器 (config/machine_identity.json)。

    每個機型定義一個錨點：畫面固定位置的二值化樣板。逐幀比對所有錨點，
    第一個 diff < 門檻的錨點產生一票；連續 consecutive_votes 幀投給同一
    機型即確認。所有錨點都不匹配時回傳 None（尚無法判定，呼叫端應繼續
    buffer 等待）。
    """

    def __init__(self, config_path: Path = Path("config/machine_identity.json")):
        self.anchors: List[Dict] = []
        self.consecutive_votes = 5
        self.fallback_machine_id = 1
        self.max_buffer_frames = 100000
        self._last_vote: Optional[int] = None
        self._streak = 0

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[MachineDetector] ⚠️ 無法載入 {config_path}: {e}")
            return

        self.consecutive_votes = int(config.get("consecutive_votes", 5))
        self.fallback_machine_id = int(config.get("fallback_machine_id", 1))
        self.max_buffer_frames = int(config.get("max_buffer_frames", 100000))

        for entry in config.get("anchors", []):
            template = _load_reference_header_image_cached(str(entry["template"]))
            if template is None:
                print(f"[MachineDetector] ⚠️ 無法載入錨點樣板: {entry['template']}")
                continue
            self.anchors.append({
                "machine_id": int(entry["machine_id"]),
                "template": template,
                "coords": entry["coords"],
                "diff_threshold": float(entry["diff_threshold"]),
            })

        if not self.anchors:
            print("[MachineDetector] ⚠️ 沒有可用的錨點，機型偵測停用")

    def detect_from_frame(self, frame_bgr: np.ndarray) -> Optional[int]:
        """單幀判斷：回傳第一個匹配錨點的機型，全部不匹配則回傳 None。"""
        for anchor in self.anchors:
            crop_binary = extract_and_binarize_header(frame_bgr, anchor["coords"])
            if crop_binary is None or crop_binary.shape != anchor["template"].shape:
                continue
            diff = calculate_pixel_difference_ratio(crop_binary, anchor["template"])
            if diff is not None and diff < anchor["diff_threshold"]:
                return anchor["machine_id"]
        return None

    def update(self, frame_bgr: np.ndarray) -> Optional[int]:
        """串流 API：連續 consecutive_votes 幀投給同一機型才確認並回傳。"""
        vote = self.detect_from_frame(frame_bgr)
        if vote is None:
            self._last_vote = None
            self._streak = 0
            return None
        if vote == self._last_vote:
            self._streak += 1
        else:
            self._last_vote = vote
            self._streak = 1
        if self._streak >= self.consecutive_votes:
            return vote
        return None

    def reset(self) -> None:
        """重置投票狀態（換影片時呼叫）。"""
        self._last_vote = None
        self._streak = 0

# --------------------------------------------------------------------------
# Original Utilities (Kept for backward compatibility)
# --------------------------------------------------------------------------

def load_stage_analysis(video_dir: Path) -> Optional[dict]:
    """載入視頻目錄下的 stage_analysis.json"""
    stage_file = video_dir / "stage_analysis.json"
    if not stage_file.exists():
        return None
    
    try:
        with open(stage_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  載入 stage_analysis.json 失敗: {e}")
        return None


def find_pattern2_detection_frame(stage_data: dict, offset: int = 5) -> Optional[int]:
    """
    從 stage_analysis.json 中找到 STAGE 區域 pattern 2 出現後的檢測幀
    
    Returns:
        檢測幀號（start_frame + 5），如果未找到則返回 None
    """
    if "regions" not in stage_data or "STAGE" not in stage_data["regions"]:
        return None
    
    stage_segments = stage_data["regions"]["STAGE"]
    
    # 尋找第一個 pattern 2 的段落
    for segment in stage_segments:
        if segment.get("pattern") == 2:
            start_frame = segment.get("start_frame")
            end_frame = segment.get("end_frame")
            
            if start_frame is not None and end_frame is not None:
                # 使用 start_frame + offset 作為檢測幀，但確保不超過 end_frame
                detection_frame = min(start_frame + max(0, int(offset)), end_frame)
                return detection_frame
    
    return None


def extract_frame_from_video(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    先從 frame_cache 讀取指定幀；若不存在則回退至解碼原影片擷取該幀。
    """
    # 決定分析資料夾與可能的影片檔位置
    if video_path.is_dir():
        video_dir = video_path
        # 嘗試從父層尋找同名影片
        video_file_candidates = []
        for ext in ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']:
            video_file_candidates.append(video_dir.parent / f"{video_dir.name}{ext}")
    elif video_path.is_file():
        video_dir = video_path.parent / video_path.stem
        video_file_candidates = [video_path]
    else:
        # 當作資料夾名處理
        video_dir = video_path
        video_file_candidates = []
        for ext in ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']:
            video_file_candidates.append(video_dir.parent / f"{video_dir.name}{ext}")

    # 1) 優先嘗試 frame_cache 快取
    frame_cache_dir = video_dir / "frame_cache"
    cache_candidates = [
        frame_cache_dir / f"frame_{frame_idx}.jpg",
        frame_cache_dir / f"frame_{frame_idx}.jpeg",
        frame_cache_dir / f"frame_{frame_idx}.png",
    ]
    for cache_file in cache_candidates:
        if cache_file.exists():
            try:
                img_rgb = Image.open(cache_file).convert("RGB")
                return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"  ⚠️  讀取快取影像失敗: {e}")
                break  # 快取讀取出錯，改走解碼路徑

    # 2) 回退至解碼原影片
    video_file: Optional[Path] = None
    for cand in video_file_candidates:
        if cand and cand.exists():
            video_file = cand
            break
    if video_file is None:
        return None

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            return frame
        return None
    finally:
        cap.release()


def save_frame_png(frame_bgr: np.ndarray, out_path: Path) -> bool:
    """將 BGR 幀保存為 PNG（以 RGB 色彩儲存）。"""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img.save(out_path, "PNG")
        return True
    except Exception as e:
        print(f"  ⚠️  儲存原始幀 PNG 失敗: {e}")
        return False


def extract_and_binarize_header(frame_bgr: np.ndarray, header_coords: List[int]) -> Optional[np.ndarray]:
    """
    提取並二值化 header 區域
    
    Args:
        frame_bgr: 完整幀（BGR格式）
        header_coords: header 座標 [x1, y1, x2, y2]
    
    Returns:
        二值化後的圖像（numpy array），失敗返回 None
    """
    if not header_coords or len(header_coords) != 4:
        return None
    
    x1, y1, x2, y2 = header_coords
    
    # 提取 header 區域
    try:
        header_bgr = frame_bgr[y1:y2, x1:x2]
        if header_bgr.size == 0:
            return None
    except Exception:
        return None
    
    # 二值化
    try:
        # 與 UI 相同：使用 rule 方法（S%<30 且 gray>150）
        binary = binarize(header_bgr, hsv_s_thresh=30, gray_thresh=150)
        return binary
    except Exception as e:
        print(f"  ⚠️  二值化失敗: {e}")
        return None


def load_reference_header_image(ref_path: Path) -> Optional[np.ndarray]:
    """載入參考 header 圖片（機型1的標準圖）"""
    if not ref_path.exists():
        print(f"  ⚠️  參考圖片不存在: {ref_path}")
        return None
    
    try:
        img = Image.open(ref_path)
        # 轉為灰度或二值化圖（假設參考圖是灰度圖）
        img_array = np.array(img)
        
        # 如果是 RGB，轉為灰度
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 確保是二值圖（0 或 255）
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        return img_array
    except Exception as e:
        print(f"  ⚠️  載入參考圖片失敗: {e}")
        return None


def calculate_pixel_difference_ratio(img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
    """
    計算兩張二值圖的像素差異比例
    
    Returns:
        差異比例（0.0-1.0），例如 0.03 表示 3% 的像素不同
    """
    if img1.shape != img2.shape:
        # print(f"  ⚠️  圖片尺寸不匹配: {img1.shape} vs {img2.shape}")
        return None
    
    # 計算不同像素的數量
    diff_pixels = np.sum(img1 != img2)
    total_pixels = img1.size
    
    diff_ratio = diff_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return diff_ratio


def detect_machine_type_for_video(
    video_name: str, 
    video_dir: Path, 
    data_root: Path = Path("data"),
    ref_image_path: Path = Path("data/roi_img_caches/machine_1_2/roi_headers/region1.png"),
    threshold: float = 0.03,
    verbose: bool = True,
    *,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    offset: int = 5,
    save_raw_header_png: bool = True
) -> Optional[int]:
    """
    檢測單個視頻的機型 (CLI 用, 依賴檔案)
    """
    if verbose:
        print(f"\n🔍 檢測視頻: {video_name}")
    
    # 1. 載入 stage_analysis.json
    stage_data = load_stage_analysis(video_dir)
    if not stage_data:
        if verbose:
            print(f"  ⚠️  未找到 stage_analysis.json，跳過")
        return None
    
    # 2. 找到 pattern 2 的檢測幀
    detection_frame = find_pattern2_detection_frame(stage_data, offset=offset)
    if detection_frame is None:
        # 視為機型2（此機型影片不具備對應的 STAGE 顯示）
        if verbose:
            print(f"  ⚠️  未找到 STAGE pattern 2，依規則直接判定為機型 2")
        return 2
    
    if verbose:
        print(f"  ℹ️  檢測幀: {detection_frame}")
    
    # 3. 找到視頻文件 (略...使用 helper)
    # 為了簡化，我們直接使用 detector class 來做核心比對，這裡只處理 IO
    
    # 4. 提取檢測幀
    frame_bgr = extract_frame_from_video(video_dir, detection_frame)
    if frame_bgr is None:
        if verbose:
            print(f"  ⚠️  無法從 frame_cache 取得幀 {detection_frame}")
        return None
    
    # 5. 使用多錨點偵測器比對（座標與門檻由 config/machine_identity.json 定義）
    detector = MachineDetector()
    machine_id = detector.detect_from_frame(frame_bgr)
    if machine_id is None:
        # 保留舊行為：偵測幀上所有錨點皆不匹配時視為機型 2
        machine_id = 2
    
    if verbose and machine_id:
        print(f"  ✓  判定機型: {machine_id}")
    
    return machine_id


def get_all_video_dirs(data_root: Path = Path("data")) -> List[Tuple[str, Path]]:
    """
    獲取所有視頻分析目錄（支持遞歸掃描子目錄）
    """
    if not data_root.exists():
        return []
    
    video_dirs = []
    
    # 遞歸掃描所有子目錄，查找包含分析結果的目錄
    for item in data_root.rglob("*"):
        if item.is_dir() and not item.name.startswith('.'):
            # 排除特殊目錄
            if item.name in ['roi_img_caches', 'pretrain', 'confusing', 'frame_cache']:
                continue
            
            # 檢查是否為分析目錄（包含 stage_analysis.json 或 frame_cache）
            if (item / "stage_analysis.json").exists() or \
               (item / "frame_cache").exists() or \
               any(item.glob("region*")):
                video_dirs.append((item.name, item))
    
    return sorted(video_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="自動檢測視頻機型並更新 rois.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--video",
        help="指定要檢測的對象：1) 視頻名稱（不含擴展名），或 2) 視頻文件路徑（.mp4 等），或 3) 目錄路徑（掃描該目錄下的已分析視頻目錄）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="預覽模式，只檢測不更新 rois.json"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新檢測已有映射的視頻"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="像素差異門檻（默認: 0.03，即 3%%）"
    )
    
    parser.add_argument(
        "--ref-image",
        type=Path,
        default=Path("data/roi_img_caches/machine_1_2/roi_headers/region1.png"),
        help="參考圖片路徑（機型1標準）"
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="data 根目錄"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="輸出除錯資訊：儲存二值化 header 與參考圖、列印逐像素相等矩陣"
    )

    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="指定除錯輸出資料夾（預設: data/roi_img_caches/debug）"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=5,
        help="選擇檢測幀時相對於 STAGE pattern 2 start_frame 的偏移（預設 5）"
    )

    parser.add_argument(
        "--save-raw-header",
        action="store_true",
        help="除錯時另外輸出檢測幀的原始 Header 區域 PNG"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("自動檢測視頻機型")
    print("=" * 70)
    
    # 檢查參考圖片是否存在
    if not args.ref_image.exists():
        print(f"\n❌ 錯誤: 參考圖片不存在: {args.ref_image}")
        print("請確保已運行過視頻標註並保存了 region1 的 header 圖片")
        return
    
    # 獲取要處理的視頻列表
    if args.video:
        candidate_path = Path(args.video)
        if candidate_path.exists():
            if candidate_path.is_dir():
                # 支援傳入目錄：掃描該目錄下的已分析視頻目錄
                scan_root = candidate_path
                print(f"將掃描目錄: {scan_root}")
                video_list = get_all_video_dirs(scan_root)
            elif candidate_path.is_file():
                # 支援傳入視頻文件：使用同級的 <stem>/ 作為分析目錄
                video_name = candidate_path.stem
                video_dir = candidate_path.parent / video_name
                if not video_dir.exists():
                    print(f"\n❌ 錯誤: 未找到對應的分析目錄: {video_dir}")
                    print("請先運行階段分析或確認目錄結構為 <data_root>/<視頻名稱>/stage_analysis.json")
                    return
                video_list = [(video_name, video_dir)]
            else:
                print(f"\n❌ 錯誤: 不支援的 --video 參數: {candidate_path}")
                return
        else:
            # 將其視為視頻名稱（不含副檔名）
            video_dir = args.data_root / args.video
            if not video_dir.exists():
                print(f"\n❌ 錯誤: 視頻目錄不存在: {video_dir}")
                return
            video_list = [(args.video, video_dir)]
    else:
        # 處理所有視頻：掃描 data_root 下的已分析視頻目錄
        video_list = get_all_video_dirs(args.data_root)
    
    if not video_list:
        print("\n⚠️  未找到任何視頻目錄")
        return
    
    print(f"\n找到 {len(video_list)} 個視頻目錄\n")
    
    # 統計
    total = 0
    detected = 0
    updated = 0
    skipped = 0
    failed = 0
    
    results = []
    
    for video_name, video_dir in video_list:
        total += 1
        
        try:
            # 檢查是否已有映射（除非使用 --force）
            if not args.force:
                existing_id = get_video_machine_id(video_name)
                if existing_id is not None:
                    print(f"⏩ 跳過 {video_name}（已有映射: 機型 {existing_id}）")
                    skipped += 1
                    continue
            
            # 檢測機型
            machine_id = detect_machine_type_for_video(
                video_name=video_name,
                video_dir=video_dir,
                data_root=args.data_root,
                ref_image_path=args.ref_image,
                threshold=args.threshold,
                verbose=True,
                debug=args.debug,
                debug_dir=args.debug_dir,
                offset=args.offset,
                save_raw_header_png=args.save_raw_header
            )
            
            if machine_id is None:
                failed += 1
                continue
            
            detected += 1
            results.append((video_name, machine_id))
            
            # 更新映射（除非是預覽模式）
            if not args.dry_run:
                success = update_video_machine_mapping(video_name, machine_id)
                if success:
                    updated += 1
                else:
                    print(f"  ⚠️  更新映射失敗")
        
        except Exception as e:
            print(f"\n❌ 處理 {video_name} 時發生錯誤:")
            print(f"   {e}")
            traceback.print_exc()
            failed += 1
    
    # 顯示摘要
    print("\n" + "=" * 70)
    print("檢測摘要")
    print("=" * 70)
    print(f"總視頻數: {total}")
    print(f"成功檢測: {detected}")
    print(f"已跳過: {skipped}")
    print(f"失敗: {failed}")
    
    if not args.dry_run:
        print(f"已更新映射: {updated}")
    else:
        print("\n⚠️  預覽模式：未實際更新 rois.json")
    
    if results:
        print("\n檢測結果:")
        for video_name, machine_id in results:
            print(f"  • {video_name}: 機型 {machine_id}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
