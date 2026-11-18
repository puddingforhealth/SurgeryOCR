import argparse
from pathlib import Path
import time
import cv2
import json
import traceback
from typing import List, Tuple, Any, Dict, Optional, Iterator

# 引入工具
from utils.pipeline_utils import (
    AsyncImageSaver,
    PrefetchIterator,
    FrameSpool,
    duplicate_stems,
    find_videos,
    iter_video_frames,
)
from utils.get_configs import (
    load_roi_config,
    load_stage_config,
    load_roi_header_config,
    load_ocr_char_sets_config,
    load_pattern_name_mapping,
    update_video_machine_mapping,
    get_machine_key,
    resolve_machine_config_path,
    get_pattern_cache_root,
    load_ocr_strategy,
)
from utils.cv_processing import binarize
from utils import frame_clock

# 引入各階段模組
from stage_pattern_analysis import StageAnalyzer, region_segments
from auto_detect_machine_type import MachineDetector
from surgery_analysis_process import OCRProcessor
from tqdm import tqdm
import numpy as np
import sys


def _initialize_machine_components(
    machine_id,
    video_name: str,
    roi_config_path: Path,
    stage_activation_dict: Dict,
    char_sets_dict: Dict,
    ocr_profile: str = "auto",
    ocr_bundle: Optional[Path] = None,
    ocr_device: str = "auto",
    ocr_sensitivity: str = "standard",
) -> Tuple[StageAnalyzer, Dict, OCRProcessor, Dict[str, Any]]:
    """
    機型確認後初始化所有機型相關組件。
    回傳: (stage_analyzer, pattern_name_map, ocr_processor, roi_dict)
    """
    machine_key = get_machine_key(machine_id)
    cache_root = get_pattern_cache_root(machine_key)

    # 1. 階段分析（樣板比對）：座標、樣板與差異規則皆依機型路線選擇
    stage_config_path = resolve_machine_config_path(
        "surgery_stage_rois.json", machine_key
    )
    diff_rules_path = resolve_machine_config_path("diff_rule.json", machine_key)
    stage_analyzer = StageAnalyzer(
        stage_config_path,
        cache_root,
        diff_rules_path=diff_rules_path,
        machine_id=machine_id,
    )
    pattern_name_map = load_pattern_name_mapping(
        resolve_machine_config_path("pattern_name_mapping.json", machine_key)
    )

    # 2. OCR：ROI 座標依 video->machine mapping 解析；策略依機型路線
    roi_dict = load_roi_config(
        roi_config_path, video_name=video_name, machine_id=machine_id
    )
    try:
        roi_header_dict = load_roi_header_config(
            roi_config_path, video_name=video_name, machine_id=machine_id
        )
    except Exception:
        roi_header_dict = {}

    strategy = load_ocr_strategy(machine_key)
    from utils.ocr_runtime import (
        resolve_ocr_profile,
        ConservativeEasyOCR,
        SceneNumericOCR,
        SingleNumericEasyOCR,
    )
    import torch

    selected_profile, bundle_root, bundle_manifest = resolve_ocr_profile(
        machine_id, ocr_profile, ocr_bundle
    )
    profiles = json.loads(Path("config/ocr_profiles.json").read_text(encoding="utf-8"))
    runtime = None
    if ocr_sensitivity not in {"standard", "exact"}:
        raise ValueError("OCR sensitivity must be standard or exact")
    exact_regions = None
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if ocr_device == "auto"
        else ocr_device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if selected_profile == "legacy":
        char_sets_dict = profiles["legacy_char_sets"]
        if str(machine_id) == "3":
            strategy = profiles["legacy_machine3_strategy"]
            roi_dict = profiles["legacy_machine3_rois"]
    else:
        if str(machine_id) == "3":
            runtime = SceneNumericOCR(bundle_root, bundle_manifest, device)
        else:
            reader = (
                SingleNumericEasyOCR
                if bundle_manifest.get("machine12_route") == "single_numeric"
                else ConservativeEasyOCR
            )
            runtime = reader(bundle_root, bundle_manifest, machine_id, device)
            exact_regions = [f"region{i}" for i in range(1, 7)]
    print(
        f"OCR profile: {selected_profile}; machine {machine_id}; backend {getattr(runtime, 'provenance', {}).get('backend', 'original-easyocr')}; bundle {bundle_manifest['version'] if bundle_manifest else 'none'}"
    )
    ocr_processor = OCRProcessor(
        stage_activation_dict,
        roi_header_dict,
        char_sets_dict,
        diff_threshold=float(strategy.get("diff_threshold", 0.01)),
        header_template_dir=cache_root / "roi_headers",
        ocr_input=strategy.get("ocr_input", "binary"),
        stable_frames=int(strategy.get("stable_frames", 0)),
        min_ocr_height=int(strategy.get("min_ocr_height", 0)),
        leading_minus_regions=strategy.get("leading_minus_regions"),
        value_formats=strategy.get("value_formats"),
        # "whole_roi" 跳過 CRAFT 偵測器（ROI 已是緊裁切）；預設 "craft" 維持原行為
        skip_ocr_detector=(strategy.get("ocr_detector", "craft") == "whole_roi"),
        ocr_runtime=runtime,
        exact_cache_regions=exact_regions,
        exact_change_regions=exact_regions if ocr_sensitivity == "exact" else [],
        gpu=(device == "cuda"),
        scene_frame_size=strategy.get("source_dimensions")
        if runtime is not None and str(machine_id) == "3"
        else None,
    )
    import hashlib

    ocr_processor.provenance = {
        "profile": selected_profile,
        "requested_profile": ocr_profile,
        "machine_id": machine_id,
        "roi_sha256": hashlib.sha256(
            json.dumps(roi_dict, sort_keys=True).encode()
        ).hexdigest(),
        **getattr(runtime, "provenance", {}),
    }
    ocr_processor.provenance["sensitivity"] = (
        "rgb_every_frame"
        if selected_profile == "enhanced" and str(machine_id) == "3"
        else ocr_sensitivity
        if selected_profile == "enhanced"
        else "legacy"
    )
    return stage_analyzer, pattern_name_map, ocr_processor, roi_dict


def _flush_buffer(
    stage_analyzer: StageAnalyzer,
    region_matches: Dict[str, List[Tuple[int, Any, Any]]],
    ocr_processor: OCRProcessor,
    roi_dict: Dict[str, Tuple[int, int, int, int]],
    frame_buffer: List[Tuple[int, np.ndarray]],
    async_saver: AsyncImageSaver,
    analysis_dir: Path,
    mode: str,
    force: bool,
) -> None:
    """
    回溯處理 Buffer 中的幀：階段分析 + OCR。
    （機型確認前不知道該用哪套 stage 樣板，因此階段分析也延後到這裡補跑）
    """
    if frame_buffer:
        print(f"⏪ 回溯處理 {len(frame_buffer)} 幀（階段分析 + OCR）...")
    for buf_idx, buf_encoded_img in frame_buffer:
        # Enhanced discovery buffers use lossless PNG; legacy retains JPEG.
        buf_frame = cv2.imdecode(buf_encoded_img, cv2.IMREAD_COLOR)
        if buf_frame is None:
            continue

        # 階段分析
        buf_stage = stage_analyzer.process_frame(buf_frame, buf_idx)
        _collect_stage_matches(region_matches, buf_stage, buf_idx)

        # 執行 OCR
        ocr_processor.process_frame(buf_frame, buf_idx, roi_dict, buf_stage)

        # 非同步存 ROI 小圖
        if mode == "detail":
            _save_roi_images(
                async_saver, buf_frame, roi_dict, buf_idx, analysis_dir, force
            )
    print("⏩ 回溯完成，進入即時模式")
    return


def _iter_cache_frames(
    cache_files: List[Path],
) -> Iterator[Tuple[int, np.ndarray, float]]:
    """Yield (frame_idx, frame_bgr, 0.0) from cached JPEG frames.

    The index comes from the filename rather than a counter. cv2.imread fails on
    non-ASCII paths on Windows, so the bytes are read with np.fromfile and decoded
    in memory. Cached frames carry no timestamp.
    """
    for img_path in cache_files:
        frame_idx = int(img_path.stem.split("_")[1])
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"⚠️ [Read Mode] Failed to read image: {img_path}")
            continue
        yield frame_idx, frame_bgr, 0.0


def _collect_stage_matches(
    region_matches: Dict[str, List[Tuple[int, Any, Any]]],
    stage_res: Dict[str, Any],
    frame_idx: int,
) -> None:
    for region_name, res in stage_res.items():
        pid = None
        rmse = None
        if isinstance(res, dict):
            pid = res.get("pattern_id")
            rmse = res.get("rmse")
        region_matches.setdefault(region_name, []).append((frame_idx, pid, rmse))


def run_pipeline(
    video_path: Path,
    base_output_dir: Path,
    mode: str = "detail",
    force: bool = False,
    ocr_profile: Optional[str] = None,
    ocr_bundle: Optional[Path] = None,
    ocr_device: str = "auto",
    ocr_sensitivity: str = "standard",
    machine_id_override: Optional[int] = None,
):
    """
    單次讀取 (Single-Pass) 分析管線
    """
    from utils.ocr_runtime import default_ocr_profile

    if ocr_profile is None:
        ocr_profile = default_ocr_profile()
    video_name = video_path.stem

    # ... (省略部分路徑設定代碼) ...
    if base_output_dir.name == video_name:
        analysis_dir = base_output_dir
    else:
        analysis_dir = base_output_dir / video_name

    analysis_dir.mkdir(parents=True, exist_ok=True)

    # RAM mode 不需要 frame_cache 目錄；read mode 需要路徑但不建立目錄
    frame_cache_dir = None
    if mode in ["detail", "frame", "read"]:
        frame_cache_dir = analysis_dir / "frame_cache"
        if mode != "read":
            frame_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始串流分析: {video_name} (Mode: {mode})")
    print(f"📂 輸出目錄: {analysis_dir}")

    # --- 1. 初始化各個組件 ---
    # RAM mode 不需要 AsyncImageSaver
    async_saver = AsyncImageSaver() if mode in ["detail", "frame"] else None

    # Configs（機型無關的部分）
    stage_activation_path = Path("config/ocr_activation_stages.json")
    roi_config_path = Path("config/rois.json")
    char_config_path = Path("config/ocr_char_sets.json")

    machine_detector = MachineDetector()

    stage_activation_dict = load_stage_config(stage_activation_path)
    char_sets_dict = load_ocr_char_sets_config(char_config_path)

    # 機型相關組件：延後到機型確認後才初始化
    # （stage 樣板/座標、pattern 名稱、OCR 策略皆依機型路線選擇）
    machine_detected = False
    machine_id = None
    stage_analyzer: Optional[StageAnalyzer] = None
    pattern_name_map: Dict = {}
    region_matches: Dict[str, List[Tuple[int, Any, Any]]] = {}
    ocr_processor: Optional[OCRProcessor] = None
    roi_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None

    from utils.ocr_runtime import configured_machine_id

    with open(roi_config_path, "r", encoding="utf-8") as f:
        full_roi_config = json.load(f)
    machine_id = configured_machine_id(
        full_roi_config.get("video_machine_mapping", {}),
        video_name,
        machine_id_override,
    )
    if machine_id is not None:
        if machine_id_override is None:
            print(
                f"ℹ️  檢測到已知設定: {video_name} (Type {machine_id})，跳過機型偵測。"
            )
        else:
            print(f"Explicit machine override: Type {machine_id}")
        try:
            stage_analyzer, pattern_name_map, ocr_processor, roi_dict = (
                _initialize_machine_components(
                    machine_id,
                    video_name,
                    roi_config_path,
                    stage_activation_dict,
                    char_sets_dict,
                    ocr_profile,
                    ocr_bundle,
                    ocr_device,
                    ocr_sensitivity,
                )
            )
        except BaseException:
            if async_saver is not None:
                async_saver.stop()
            raise
        machine_detected = True

    # --- 2. 狀態變數 ---
    t0 = time.time()
    processed_frames = 0

    # 緩衝區：儲存 (frame_idx, frame_bgr, stage_result)
    # 用於在機型確認前暫存畫面，以便回溯 OCR
    frame_buffer = FrameSpool()

    # Frame Source Setup
    cap = None
    cache_files = []
    total_frames = 0

    if mode == "read":
        if not frame_cache_dir.exists():
            raise FileNotFoundError(
                f"Frame cache directory not found: {frame_cache_dir}"
            )

        # 讀取所有 cache 檔案並按 frame index 排序
        # 假設檔名格式為 frame_{idx}.jpg
        try:
            cache_files = sorted(
                frame_cache_dir.glob("frame_*.jpg"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
        except Exception as e:
            raise ValueError(
                f"Cannot parse frame-cache filenames: {frame_cache_dir}"
            ) from e

        if not cache_files:
            raise ValueError(f"No cached frames found in: {frame_cache_dir}")

        total_frames = len(cache_files)
        print(f"📂 [Read Mode] Found {total_frames} cached frames.")

    else:
        # 改用直接控制 VideoCapture 以獲取時間戳並支援不同模式
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            if async_saver is not None:
                async_saver.stop()
            raise RuntimeError(f"Cannot open video for decoding: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # (frame_idx, presentation seconds) per decoded frame -> frame_clock.json
    frame_times: List[Tuple[int, float]] = []
    frame_source: Optional[PrefetchIterator] = None
    # 分段計時，寫入 run_stats.json。用來判斷瓶頸在解碼、樣板比對還是 OCR，
    # 以及驗證「輸出相同但工作量下降」—— 只有兩者同時成立才是真的優化。
    stage_seconds = 0.0
    ocr_seconds = 0.0

    try:
        # 初始化 tqdm
        pbar = tqdm(
            total=total_frames,
            desc=f"Processing frames ({video_name})",
            mininterval=30.0,
            maxinterval=60.0,
        )

        # 解碼在背景執行緒進行，與主執行緒的階段分析 + OCR 重疊。
        # 解碼佔總時間近四成且與分析完全序列，重疊後這段大多可被隱藏。
        if mode == "read":
            frame_source = PrefetchIterator(_iter_cache_frames(cache_files))
        else:
            frame_source = PrefetchIterator(iter_video_frames(cap))

        for frame_idx, frame_bgr, ts in frame_source:
            if mode != "read":
                frame_times.append((frame_idx, ts / 1000.0))

            processed_frames += 1
            pbar.update(1)

            # [Step A] 根據 mode 決定是否存 Frame Cache (大圖)
            # mode='detail' or 'frame' -> 存大圖
            # mode='ram' -> 不存
            # mode='read' -> 已經從 cache 讀了，不需要再存
            if mode in ["detail", "frame"]:
                cache_path = frame_cache_dir / f"frame_{frame_idx}.jpg"
                if force or not cache_path.exists():
                    async_saver.save(
                        frame_bgr, cache_path, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )

            # [Step B/C] 機型偵測與分析分支
            # 機型未確認前：只做偵測與 buffer（stage 樣板依機型而異，無法先分析）
            # 機型確認後：即時做階段分析 + OCR
            if not machine_detected:
                # 1. 記憶體優化：將 Frame 壓縮為 JPEG Bytes 存入 Buffer
                if ocr_profile == "legacy":
                    success, encoded_img = cv2.imencode(
                        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )
                else:
                    success, encoded_img = cv2.imencode(
                        ".png", frame_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
                    )
                if success:
                    frame_buffer.append((frame_idx, encoded_img))

                # 2. 逐幀偵測：多錨點投票，連續同票數達標才確認
                detected_id = machine_detector.update(frame_bgr)

                # Buffer 超過安全上限時，強制使用 fallback 機型
                force_resolve = len(frame_buffer) > machine_detector.max_buffer_frames

                if detected_id is not None or force_resolve:
                    if detected_id is not None:
                        machine_id = detected_id
                        update_video_machine_mapping(video_name, machine_id)
                        print(f"\n✅ 偵測機型: Type {machine_id} (Frame {frame_idx})")
                    else:
                        machine_id = machine_detector.fallback_machine_id
                        print(
                            f"\n⚠️ 警告：Buffer 超過 {machine_detector.max_buffer_frames} 幀仍未偵測到機型，強制使用預設 Type {machine_id}"
                        )

                    if "pbar" in locals():
                        pbar.clear()
                    machine_detected = True

                    stage_analyzer, pattern_name_map, ocr_processor, roi_dict = (
                        _initialize_machine_components(
                            machine_id,
                            video_name,
                            roi_config_path,
                            stage_activation_dict,
                            char_sets_dict,
                            ocr_profile,
                            ocr_bundle,
                            ocr_device,
                            ocr_sensitivity,
                        )
                    )
                    _flush_buffer(
                        stage_analyzer,
                        region_matches,
                        ocr_processor,
                        roi_dict,
                        frame_buffer,
                        async_saver,
                        analysis_dir,
                        mode,
                        force,
                    )
                    frame_buffer.clear()

            else:
                # 機型已確認：即時處理模式
                _t = time.perf_counter()
                stage_res = stage_analyzer.process_frame(frame_bgr, frame_idx)
                stage_seconds += time.perf_counter() - _t
                _collect_stage_matches(region_matches, stage_res, frame_idx)

                if ocr_processor and roi_dict:
                    _t = time.perf_counter()
                    ocr_processor.process_frame(
                        frame_bgr, frame_idx, roi_dict, stage_res
                    )
                    ocr_seconds += time.perf_counter() - _t

                    # 非同步存 ROI 小圖
                    if mode == "detail":
                        _save_roi_images(
                            async_saver,
                            frame_bgr,
                            roi_dict,
                            frame_idx,
                            analysis_dir,
                            force,
                        )

    except KeyboardInterrupt:
        print("\n⚠️ 使用者中斷")
        raise
    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤: {e}")
        raise
    finally:
        if frame_source is not None:
            frame_source.close()
        if cap is not None:
            cap.release()
        if sys.exc_info()[0] is not None:
            frame_buffer.clear()
            if async_saver is not None:
                async_saver.stop()

    if processed_frames == 0:
        if async_saver is not None:
            async_saver.stop()
        raise RuntimeError(f"No decodable frames found: {video_path}")

    # [Step C-2] 影片結束仍未確認機型：以 fallback 機型回溯處理 Buffer，
    # 避免整段 OCR 結果被靜默丟棄（不寫入 mapping，因為機型並未實際確認）
    if not machine_detected and frame_buffer:
        machine_id = machine_detector.fallback_machine_id
        print(
            f"\n⚠️ 影片結束仍未偵測到機型，使用預設 Type {machine_id} 回溯處理 {len(frame_buffer)} 幀"
        )
        stage_analyzer, pattern_name_map, ocr_processor, roi_dict = (
            _initialize_machine_components(
                machine_id,
                video_name,
                roi_config_path,
                stage_activation_dict,
                char_sets_dict,
                ocr_profile,
                ocr_bundle,
                ocr_device,
                ocr_sensitivity,
            )
        )
        _flush_buffer(
            stage_analyzer,
            region_matches,
            ocr_processor,
            roi_dict,
            frame_buffer,
            async_saver,
            analysis_dir,
            mode,
            force,
        )
        frame_buffer.clear()

    # [Step D] 結束處理：儲存結果
    if ocr_processor:
        print(f"\n💾 儲存 OCR 結果至: {analysis_dir}")
        ocr_processor.save_results(analysis_dir)

    stage_total = total_frames if total_frames is not None else processed_frames
    _write_stage_analysis(
        video_name,
        analysis_dir,
        region_matches,
        stage_total,
        pattern_name_map,
        machine_id,
    )

    _write_frame_clock(video_path, analysis_dir, frame_times, total_frames)
    if ocr_processor is not None:
        ocr_processor.provenance["machine_id_override"] = machine_id_override

    if roi_dict is not None:
        (analysis_dir / "analysis_layout.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "machine_id": machine_id,
                    "rois": roi_dict,
                    "ocr_profile": getattr(ocr_processor, "provenance", {}).get(
                        "profile", "legacy"
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    _write_run_stats(
        analysis_dir,
        processed_frames,
        time.time() - t0,
        frame_source,
        stage_analyzer,
        ocr_processor,
        stage_seconds,
        ocr_seconds,
    )

    # 等待 IO (只有在有 async_saver 時才需要)
    if async_saver:
        print("⏳ 等待背景儲存完成...")
        async_saver.stop()
    print(
        f"\n✅ 分析完成，總耗時: {time.time() - t0:.2f}s (Frames: {processed_frames})"
    )


def _save_roi_images(
    saver: AsyncImageSaver,
    frame: np.ndarray,
    roi_config: Dict[str, Tuple[int, int, int, int]],
    frame_idx: int,
    base_dir: Path,
    force: bool,
) -> None:
    """儲存 ROI 原圖與二值圖，與舊流程一致。"""
    for region_name, (x1, y1, x2, y2) in roi_config.items():
        region_dir = base_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        try:
            roi_bgr = frame[y1:y2, x1:x2]
            if roi_bgr.size == 0:
                continue
        except Exception:
            continue

        orig_path = region_dir / f"frame_{frame_idx}.png"
        bin_path = region_dir / f"frame_{frame_idx}_binary.png"

        if force or not orig_path.exists():
            saver.save(roi_bgr, orig_path)

        if force or not bin_path.exists():
            try:
                roi_binary = binarize(roi_bgr, method="rule")
                saver.save(roi_binary, bin_path)
            except Exception:
                continue


def _write_frame_clock(
    video_path: Path,
    analysis_dir: Path,
    frame_times: List[Tuple[int, float]],
    container_frames: Optional[int],
) -> None:
    """Write frame_clock.json: decoded frame index -> presentation time.

    Read mode has no timestamps (frames come from a JPEG cache), so an existing
    clock from the decoding run is left untouched.
    """
    if len(frame_times) < 2:
        return
    try:
        clock = frame_clock.build_clock(video_path.name, frame_times, container_frames)
    except ValueError as e:
        print(f"⚠️ 無法建立 frame_clock.json: {e}")
        return
    frame_clock.write_clock(analysis_dir / frame_clock.FILE_NAME, clock)
    if len(clock["segments"]) > 1:
        print(
            f"⏱️ 影片時間軸有 {len(clock['segments']) - 1} 處中斷，已寫入 {frame_clock.FILE_NAME}"
        )


def _write_run_stats(
    analysis_dir: Path,
    processed_frames: int,
    total_seconds: float,
    frame_source: Optional[PrefetchIterator],
    stage_analyzer: Optional[StageAnalyzer],
    ocr_processor: Optional[OCRProcessor],
    stage_seconds: float,
    ocr_seconds: float,
) -> None:
    """Write run_stats.json: where the time went and how much work was done.

    Two separate questions a performance change has to answer. Timings say whether
    it got faster; the work counters (OCR calls, template comparisons, detected
    changes) say whether it got faster by doing the same work more efficiently or
    by quietly skipping some of it. An optimisation is only valid when the outputs
    match the baseline *and* the counters move the way the change predicts.
    """
    per_frame_ms = (
        (total_seconds / processed_frames * 1000) if processed_frames else 0.0
    )
    stats: Dict[str, Any] = {
        "frames": processed_frames,
        "total_seconds": round(total_seconds, 2),
        "ms_per_frame": round(per_frame_ms, 3),
        "timings": {
            # Decode runs on a background thread; this is only the part of it that
            # analysis could not hide, so it is the decode cost actually paid.
            "decode_wait_seconds": round(frame_source.wait_seconds, 2)
            if frame_source
            else None,
            "stage_seconds": round(stage_seconds, 2),
            "ocr_seconds": round(ocr_seconds, 2),
        },
        "work": {
            "template_compares": stage_analyzer.template_compares
            if stage_analyzer
            else None,
        },
    }

    if ocr_processor is not None:
        stats["ocr"] = getattr(ocr_processor, "provenance", {"profile": "legacy"})
        stats["ocr"]["review_events"] = len(
            getattr(ocr_processor, "review_results", [])
        )
        stats["ocr"]["exact_prediction_cache_hits"] = getattr(
            ocr_processor, "exact_prediction_cache_hits", 0
        )
        stats["ocr"]["inference_batches"] = getattr(
            ocr_processor.ocr_iface, "inference_batches", None
        )
        per_region = {}
        for region, s in ocr_processor.stats.items():
            per_region[region] = {
                "change_count": s.get("change_count", 0),
                "ocr_count": s.get("ocr_count", 0),
                "ocr_seconds": round(s.get("ocr_time_total", 0.0), 2),
            }
        stats["work"]["ocr_calls"] = sum(v["ocr_count"] for v in per_region.values())
        stats["work"]["changes_detected"] = sum(
            v["change_count"] for v in per_region.values()
        )
        stats["per_region"] = per_region

    with open(analysis_dir / "run_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _write_stage_analysis(
    video_name: str,
    analysis_dir: Path,
    region_matches: Dict[str, List[Tuple[int, Optional[int], Optional[float]]]],
    total_frames: int,
    pattern_name_map: Dict[str, Dict[str, str]],
    machine_id=None,
) -> None:
    regions_output: Dict[str, List[Dict[str, Any]]] = {
        region_name: region_segments(matches, pattern_name_map.get(region_name, {}))
        for region_name, matches in region_matches.items()
    }

    payload = {
        "video": video_name,
        "total_frames": total_frames,
        # Which screen layout the regions come from. A region the machine does not
        # display (AS on machine 1) is absent, not empty.
        "machine_id": int(machine_id)
        if machine_id is not None and str(machine_id).isdigit()
        else machine_id,
        "regions": regions_output,
    }

    out_path = analysis_dir / "stage_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass surgery pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="影片檔或包含影片的資料夾（含子資料夾）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="輸出根目錄"
    )
    parser.add_argument(
        "--force", action="store_true", help="覆蓋既有 frame cache 與 ROI 圖片"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ram",
        choices=["detail", "frame", "ram", "read"],
        help="存檔模式: detail (存大圖+ROI), frame (只存大圖), ram (不存圖, 最快), read (讀取快取)",
    )
    from utils.ocr_runtime import default_ocr_profile, default_ocr_sensitivity

    parser.add_argument(
        "--ocr-profile",
        choices=["auto", "legacy", "enhanced"],
        default=default_ocr_profile(),
        help="default: enhanced (bundled single numeric OCR for machines 1/2, RGB OCR for 3); auto permits legacy fallback",
    )
    parser.add_argument(
        "--ocr-bundle",
        type=Path,
        help="Local versioned OCR model bundle (or SURGERYOCR_OCR_BUNDLE)",
    )
    parser.add_argument(
        "--ocr-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="OCR device; scene ONNX uses CPU unless a CUDA TorchScript artifact is installed",
    )
    parser.add_argument(
        "--ocr-sensitivity",
        choices=["standard", "exact"],
        default=default_ocr_sensitivity(),
        help="Machine 1/2 trigger policy: original threshold or any binary pixel change; machine 3 always checks RGB every frame",
    )
    parser.add_argument(
        "--machine-id",
        type=int,
        choices=[1, 2, 3],
        help="Explicit known layout; omit to use mapping/automatic detection",
    )
    args = parser.parse_args()

    target_path = args.video
    if not target_path.exists():
        print(f"❌ 路徑不存在: {target_path}")
        sys.exit(1)

    video_files = find_videos(target_path)
    if not target_path.is_file():
        if not video_files:
            print(f"⚠️ 在目錄 {target_path}（含子資料夾）中未找到 .mp4 檔案")
            sys.exit(1)
        duplicates = duplicate_stems(video_files)
        if duplicates:
            print(
                "❌ 下列影片檔名相同，結果會寫進同一個輸出目錄而互相覆蓋；請改名或分批指定："
            )
            for stem, paths in duplicates.items():
                print(f"   {stem}: " + ", ".join(str(p) for p in paths))
            sys.exit(1)
        print(f"📂 找到 {len(video_files)} 個影片檔案，準備開始批次分析...")

    execution_failed = False
    for idx, vf in enumerate(video_files, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(video_files)}] 🎬 處理影片: {vf.name} (Mode: {args.mode})")
        print(f"{'=' * 60}")
        try:
            run_pipeline(
                vf,
                args.output_dir,
                args.mode,
                args.force,
                args.ocr_profile,
                args.ocr_bundle,
                args.ocr_device,
                args.ocr_sensitivity,
                args.machine_id,
            )
        except Exception as e:
            print(f"❌ 處理 {vf.name} 時發生錯誤: {e}")
            traceback.print_exc()
            execution_failed = True

    print("\n✅ 所有任務完成")
    if execution_failed:
        sys.exit(1)  # 失敗
    else:
        sys.exit(0)  # 成功


if __name__ == "__main__":
    main()
