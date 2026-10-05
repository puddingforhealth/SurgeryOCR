#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyOCR模型實現
"""
import easyocr
import numpy as np
from .base import BaseOCRModel
from PIL import Image
import cv2
import time
import traceback
class EasyOCRModel(BaseOCRModel):
    """EasyOCR引擎包裝類"""
    
    def __init__(self, gpu=True, lang_list=None, model_storage_directory=None,
                 confidence_threshold=0.3, allowlist="0123456789-+",
                 # 以下是 easyocr.Reader 的其他可選參數，如果需要從外部配置的話
                 download_enabled=True, user_network_directory=None,
                 recog_network='standard', detector=True, recognizer=True, verbose=True,
                 quantize=True, cudnn_benchmark=False, debug_output=True, **kwargs_extra_ignored): # 新增debug_output參數

        if lang_list is None:
            lang_list = ['en']

        self.gpu = gpu
        self.lang_list = lang_list
        self.model_storage_directory = model_storage_directory
        self.confidence_threshold = confidence_threshold # 在此存儲，用於 recognize 方法
        self.allowlist = allowlist # 在此存儲，用於 recognize 方法
        self.debug_output = debug_output # 控制是否輸出詳細調試信息

        # 準備僅easyocr.Reader構造函數支持的參數
        reader_params = {
            'lang_list': self.lang_list,
            'gpu': self.gpu,
            'model_storage_directory': self.model_storage_directory,
            'download_enabled': download_enabled,
            'user_network_directory': user_network_directory,
            'recog_network': recog_network,
            'detector': detector,
            'recognizer': recognizer,
            'verbose': verbose,
            'quantize': quantize,
            'cudnn_benchmark': cudnn_benchmark
        }
        
        try:
            self.reader = easyocr.Reader(**reader_params)
            self._unwrap_data_parallel()
            if self.debug_output:
                print(f"✓ EasyOCR Reader 初始化成功")
                print(f"  - GPU加速: {'啟用' if self.gpu else '停用'}")
                print(f"  - 語言: {self.lang_list}")
                print(f"  - 字符白名單: {self.allowlist}")
                print(f"  - 置信度閾值: {self.confidence_threshold}")
        except TypeError as e:
            print(f"初始化 EasyOCR Reader 時發生 TypeError: {e}")
            print(f"使用的參數: {reader_params}")
            print("請檢查 EasyOCR 版本與所傳參數是否兼容。嘗試使用最少參數初始化...")
            try:
                self.reader = easyocr.Reader(lang_list=self.lang_list, gpu=self.gpu, verbose=verbose)
            except Exception as fallback_e:
                print(f"使用最少參數初始化 EasyOCR Reader 失敗: {fallback_e}")
                raise e # 重新拋出原始錯誤
        
    def _unwrap_data_parallel(self):
        """Unwrap EasyOCR's nn.DataParallel wrappers around the detector/recognizer.

        EasyOCR always wraps both nets in DataParallel. On a single device that adds
        no parallelism, but every forward call still walks the whole parameter and
        buffer list to verify device placement — with one OCR call per changed ROI
        this dominated the per-call overhead. EasyOCR moves its inputs to the device
        itself, so calling the inner module directly produces identical outputs.
        """
        try:
            import torch.nn as nn
        except Exception:
            return
        for attr in ("detector", "recognizer"):
            net = getattr(self.reader, attr, None)
            if isinstance(net, nn.DataParallel) and len(net.device_ids) <= 1:
                setattr(self.reader, attr, net.module)
                if self.debug_output:
                    print(f"  - 已拆除 {attr} 的 DataParallel 包裝（單一裝置）")

    def recognize(self, image, allowlist=None):
        """
        使用EasyOCR識別圖像中的文字。
        此方法包含置信度過濾、按x座標排序和文本拼接。
        
        參數:
            image: 輸入圖像，numpy數組 (BGR) 或 PIL.Image
            allowlist: (可選) 覆蓋本次識別的字符白名單
            
        返回:
            result: 字符串，識別結果
            confidence: 浮點數，平均置信度 (0-1)
        """
        start_time = time.time()
        
        # 確保輸入是numpy數組
        if isinstance(image, Image.Image):
            # 如果是PIL Image，轉換為numpy array
            # EasyOCR 通常期望 BGR 格式的 numpy array
            if image.mode == 'RGB':
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError("輸入圖像必須是 PIL.Image 或 numpy.ndarray 類型")

        # 決定本次調用使用的allowlist
        current_allowlist = allowlist if allowlist is not None else self.allowlist

        if self.debug_output:
            print(f"\n{'='*60}")
            print(f"🔍 EasyOCR 模型調用開始")
            print(f"{'='*60}")
            print(f"📷 輸入圖像信息:")
            print(f"   - 尺寸: {image.shape}")
            print(f"   - 數據類型: {image.dtype}")
            print(f"   - 像素值範圍: [{image.min()}, {image.max()}]")
            print(f"   - 使用 allowlist: '{current_allowlist}'")

        # 執行OCR - 獲取最原始的模型輸出
        try:
            ocr_results = self.reader.readtext(
                image,
                allowlist=current_allowlist, # 在 readtext 中使用存儲的 allowlist
                detail=1,
                paragraph=False,
                width_ths=0.7,  # 添加更多參數以獲得更詳細的控制
                height_ths=0.7,
                decoder='greedy',  # 可以嘗試 'beamsearch' 獲得更好結果
                beamWidth=5,
                batch_size=1
            )
            
            processing_time = time.time() - start_time
            
            if self.debug_output:
                print(f"\n⚡ OCR 處理完成，耗時: {processing_time:.3f}秒")
                print(f"🔬 原始模型輸出 (共{len(ocr_results)}個檢測結果):")
                
                if not ocr_results:
                    print("   ❌ 沒有檢測到任何文字")
                else:
                    for i, (bbox, text, confidence) in enumerate(ocr_results):
                        # 計算邊界框信息
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        box_area = box_width * box_height
                        
                        # 判斷是否會通過置信度過濾
                        passed_filter = confidence >= self.confidence_threshold
                        status = "✅ 通過" if passed_filter else "❌ 過濾"
                        
                        print(f"\n   [{i+1}] {status}")
                        print(f"       📝 文字: '{text}'")
                        print(f"       🎯 置信度: {confidence:.4f} (閾值: {self.confidence_threshold})")
                        print(f"       📍 位置: ({x_min:.1f}, {y_min:.1f}) - ({x_max:.1f}, {y_max:.1f})")
                        print(f"       📏 尺寸: {box_width:.1f} x {box_height:.1f} (面積: {box_area:.1f})")
                        print(f"       🗂️  bbox: {bbox}")
                        
                        # 如果字符在白名單之外，也顯示警告
                        invalid_chars = [c for c in text if c not in current_allowlist and c != ' ']
                        if invalid_chars:
                            print(f"       ⚠️  包含白名單外字符: {invalid_chars}")
            
        except Exception as e:
            if self.debug_output:
                print(f"❌ OCR處理時發生錯誤: {e}")
                traceback.print_exc()
            return "", 0.0
            
        # 如果沒有結果返回空字符串
        if not ocr_results:
            if self.debug_output:
                print(f"\n📊 最終結果: 空字符串 (沒有檢測到文字)")
                print(f"{'='*60}\n")
            return "", 0.0
            
        # 根據置信度過濾結果，並記錄bbox的左上角x座標
        filtered_with_pos_conf = [
            (txt, bbox[0][0], conf) 
            for (bbox, txt, conf) in ocr_results
            if conf >= self.confidence_threshold # 在 recognize 中使用存儲的 confidence_threshold
        ]
        
        if self.debug_output:
            print(f"\n🔽 置信度過濾結果:")
            print(f"   - 原始檢測數: {len(ocr_results)}")
            print(f"   - 通過過濾數: {len(filtered_with_pos_conf)}")
            if len(filtered_with_pos_conf) != len(ocr_results):
                filtered_out = len(ocr_results) - len(filtered_with_pos_conf)
                print(f"   - 被過濾掉: {filtered_out} 個 (置信度 < {self.confidence_threshold})")
        
        # 按 x 座標 (左→右) 排序
        filtered_with_pos_conf.sort(key=lambda x: x[1])
        
        if self.debug_output and len(filtered_with_pos_conf) > 1:
            print(f"\n↔️  按X坐標排序後:")
            for i, (txt, x_pos, conf) in enumerate(filtered_with_pos_conf):
                print(f"   [{i+1}] x={x_pos:.1f}: '{txt}' (conf={conf:.3f})")
        
        # 只取文字部分拼接
        text = " ".join(item[0] for item in filtered_with_pos_conf)
        
        # 計算平均置信度
        if filtered_with_pos_conf:
            avg_confidence = sum(item[2] for item in filtered_with_pos_conf) / len(filtered_with_pos_conf)
        else:
            # 如果過濾後沒有結果，則文本為空，置信度為0
            text = ""
            avg_confidence = 0.0
        
        if self.debug_output:
            print(f"\n📊 最終輸出:")
            print(f"   - 拼接文字: '{text}'")
            print(f"   - 平均置信度: {avg_confidence:.4f}")
            print(f"   - 處理總時間: {time.time() - start_time:.3f}秒")
            
            # 如果有GPU，嘗試顯示GPU使用情況
            if self.gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                        memory_cached = torch.cuda.memory_reserved() / 1024**2      # MB
                        print(f"   - GPU記憶體: {memory_allocated:.1f}MB 已分配, {memory_cached:.1f}MB 已緩存")
                except ImportError:
                    pass
            
            print(f"{'='*60}\n")
            
        return text, avg_confidence
    
    def recognize_with_raw_output(self, image, allowlist=None):
        """
        擴展版的識別方法，返回完整的原始輸出
        
        返回:
            tuple: (final_text, avg_confidence, raw_results)
        """
        # 先執行標準識別
        final_text, avg_confidence = self.recognize(image, allowlist=allowlist)
        
        # 再次執行OCR以獲取原始結果（如果需要避免重複處理，可以修改recognize方法來緩存結果）
        if isinstance(image, Image.Image):
            if image.mode == 'RGB':
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = np.array(image)
        
        current_allowlist = allowlist if allowlist is not None else self.allowlist
        raw_results = self.reader.readtext(
            image,
            allowlist=current_allowlist,
            detail=1,
            paragraph=False
        )
        
        return final_text, avg_confidence, raw_results
    
    def set_debug_mode(self, debug=True):
        """設置調試模式"""
        self.debug_output = debug
        if debug:
            print("🔧 EasyOCR 調試模式已啟用")
        else:
            print("🔧 EasyOCR 調試模式已關閉")
    
    def get_model_info(self):
        """
        獲取模型信息
        
        返回:
            dict: 包含模型名稱、版本等信息
        """
        return {
            "name": "EasyOCR",
            "languages": self.lang_list,
            "gpu_enabled": self.gpu,
            "allowlist": self.allowlist, # 在模型信息中包含 allowlist
            "confidence_threshold": self.confidence_threshold, # 在模型信息中包含閾值
            "debug_output": self.debug_output
        } 