"""YOLO 物件追蹤與分析模組 (GPU 最佳化版本)"""

import cv2
import time
import queue
import threading
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── 全域常數 ──────────────────────────────────────────────
INFER_SIZE = 320   # YOLO 推論解析度 (降低此值可加速，320 是速度/精度最佳平衡)
INFER_EVERY = 3    # 每幾幀才跑一次 YOLO（中間幀用 Kalman 預測填補）
BATCH_SIZE  = 16   # 每批次一起送進 YOLO 的幀數 (太大可能爆 VRAM，RTX 2070 8GB 可用 16~32)


# ─── TensorRT 自動匯出與載入 ─────────────────────────────────
def _get_model(pt_path: str, device: int | str, use_half: bool) -> YOLO:
    """
    嘗試載入 TensorRT Engine (.engine)；若不存在則自動從 .pt 匯出（只需一次）。
    匯出失敗時（如未安裝 TensorRT）優雅降級，直接回傳原始 PyTorch 模型。
    """
    engine_path = Path(pt_path).with_suffix("") \
        .with_name(Path(pt_path).stem + f"_imgsz{INFER_SIZE}_fp16.engine")
    
    # 優先使用已存在的 TensorRT Engine
    if engine_path.exists():
        logger.info(f"✅ 載入 TensorRT Engine: {engine_path}")
        return YOLO(str(engine_path))
    
    # 嘗試自動匯出 (只在有 GPU 時才做)
    if device != "cpu":
        try:
            logger.info(f"🔧 首次執行：正在將模型轉換成 TensorRT Engine，這大約需要 2~5 分鐘，之後每次啟動都無需重複…")
            print("🔧 首次執行：正在將模型轉換成 TensorRT Engine，這大約需要 2~5 分鐘，之後每次啟動都無需重複…")
            base_model = YOLO(pt_path)
            base_model.export(
                format="engine",
                imgsz=INFER_SIZE,
                half=use_half,
                device=device,
                dynamic=True,   # 允許彈性批次大小，批次推論必須設定
            )
            # ultralytics 預設會在同目錄下產生同名 .engine
            default_engine = Path(pt_path).with_suffix(".engine")
            if default_engine.exists():
                default_engine.rename(engine_path)
            if engine_path.exists():
                logger.info(f"✅ TensorRT Engine 匯出成功，已儲存到: {engine_path}")
                print(f"✅ TensorRT Engine 匯出成功！之後都會直接使用加速版本。")
                return YOLO(str(engine_path))
        except Exception as e:
            logger.warning(f"⚠️ TensorRT 匯出失敗 ({e})，退回使用 PyTorch 模型。")
            print(f"⚠️ TensorRT 匯出失敗 ({e})，退回使用 PyTorch 模型。")
    
    # 降級：直接回傳 PyTorch 模型
    model = YOLO(pt_path)
    model.to(device)
    return model


# ─── 球追蹤器 ────────────────────────────────────────────────
class BallTracker:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.2, high_conf_thresh: float = 0.5):
        self.device   = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()
        self.conf_thresh      = conf_thresh
        self.high_conf_thresh = high_conf_thresh
        
        self.model = _get_model(model_path, self.device, self.use_half)
        self.is_trt = str(model_path).endswith(".engine") or \
                      any(Path(model_path).with_suffix(".engine") == p
                          for p in [Path(model_path).with_name(
                              Path(model_path).stem + f"_imgsz{INFER_SIZE}_fp16.engine")])
        
        device_label = f"GPU (CUDA) - {'TRT ✅' if self.is_trt else 'PyTorch'}" \
                        if self.device == 0 else "CPU"
        logger.info(f"BallTracker 使用裝置: {device_label}")
        print(f"🚀 BallTracker 使用裝置: {device_label}")
        
        self.reset_tracker()

    def reset_tracker(self):
        """重置 Kalman Filter 狀態"""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.P *= 1000.0
        self.kf.R = np.array([[10, 0], [0, 10]])
        self.kf.Q *= 0.1
        self.is_tracking   = False
        self.missed_frames = 0
        self.max_missed_frames = 15


# ─── 輔助：裁切 ROI 區域 ──────────────────────────────────────
def _crop_roi(frame: np.ndarray, roi: dict | None):
    """回傳 (裁切區域, rx, ry)"""
    if roi:
        rx, ry = roi["x"], roi["y"]
        rw, rh = roi["w"], roi["h"]
        area = frame[ry:ry+rh, rx:rx+rw].copy()
        if "points" in roi and len(roi["points"]) >= 3:
            mask = np.zeros(area.shape[:2], dtype=np.uint8)
            pts = np.array([[[p["x"] - rx, p["y"] - ry] for p in roi["points"]]], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
            area = cv2.bitwise_and(area, area, mask=mask)
        return area, rx, ry
    return frame, 0, 0


def _extract_best_box(result, rx: int, ry: int):
    """從 YOLO 結果取出信心最高的 bounding box (附加 ROI 偏移)"""
    best_box, best_conf = None, 0.0
    if len(result.boxes) > 0:
        for box in result.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                bx, by, bw, bh = box.xywh[0].cpu().numpy()
                best_box = {
                    "x": int(bx - bw / 2) + rx,
                    "y": int(by - bh / 2) + ry,
                    "w": int(bw),
                    "h": int(bh),
                }
    return best_box, best_conf


def _kalman_step(tracker: BallTracker, best_box, best_conf):
    """執行一步 Kalman 更新，回傳 (box_out, conf_out, status)"""
    tracker.kf.predict()

    if best_box and best_conf >= tracker.conf_thresh:
        cx = best_box["x"] + best_box["w"] / 2
        cy = best_box["y"] + best_box["h"] / 2
        if not tracker.is_tracking:
            tracker.kf.x = np.array([cx, cy, 0, 0]).reshape(4, 1)
            tracker.is_tracking = True
        else:
            tracker.kf.update(np.array([cx, cy]).reshape(2, 1))
        tracker.missed_frames = 0
        return best_box, best_conf, "DETECTED"
    else:
        tracker.missed_frames += 1
        if tracker.is_tracking and tracker.missed_frames <= tracker.max_missed_frames:
            px = int(tracker.kf.x[0, 0])
            py = int(tracker.kf.x[1, 0])
            return {"x": px - 10, "y": py - 10, "w": 20, "h": 20}, best_conf, "PREDICTED"
        else:
            tracker.is_tracking = False
            return None, 0.0, "LOST"


# ─── 多執行緒影像讀取 (消除硬碟 IO 瓶頸) ──────────────────
class ThreadedVideo:
    def __init__(self, path, queue_size=256):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        
        # 啟動背景執行緒專門負責讀檔
        self.t = threading.Thread(target=self._update)
        self.t.daemon = True
        self.t.start()

    def _update(self):
        while True:
            if self.stopped:
                return
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    self.q.put((False, None)) # 放入結束標記
                    return
                self.q.put((True, frame))
            else:
                time.sleep(0.005) # 倉庫滿了就歇會兒

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        return self.q.get()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.stopped = True
        self.cap.release()

# ─── 主分析函式 ───────────────────────────────────────────────
def analyze_video_with_yolo(
    video_path,
    roi,
    conf_thresh=0.2,
    high_conf_thresh=0.5,
    progress_callback=None,
    batch_size: int = BATCH_SIZE,
    infer_every: int = INFER_EVERY,
):
    """
    批次 + 跳幀推論主流程：
      - 每 infer_every 幀才送一次 YOLO，中間幀用 Kalman 填補。
      - 每批次 batch_size 幀同時送進 GPU 做批次加速。
      - 若有 TensorRT Engine 則自動使用。
    """
    model_name = "yolov8n.pt"
    custom_model_backup = Path("models/pickleball_best.pt")
    custom_model_local  = Path("dataset/runs/train/weights/best.pt")
    if custom_model_backup.exists():
        model_name = str(custom_model_backup)
    elif custom_model_local.exists():
        model_name = str(custom_model_local)

    tracker = BallTracker(model_path=model_name, conf_thresh=conf_thresh, high_conf_thresh=high_conf_thresh)

    # TRT 引擎有 ultralytics batch 推論 bug，改為逐幀送入（TRT 本身速度仍遠快於 PyTorch）
    effective_batch = 1 if tracker.is_trt else batch_size

    # 使用多執行緒讀取，不再讓 main thread 等待影片解碼
    cap = ThreadedVideo(video_path)
    if not cap.isOpened():
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    tracking_data, review_frames = [], []
    absolute_frame_idx = 0

    while True:
        # ── 1. 讀進一批幀，同時建立送 YOLO 的子集 ──────────────
        batch_frames       = []   # 所有幀
        infer_areas        = []   # 只送 YOLO 的 ROI 裁切幀
        infer_local_idx    = []   # 這些幀在 batch 內的位置

        for i in range(effective_batch):
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                print(f"警告：讀取第 {absolute_frame_idx + i} 幀時發生錯誤 ({e})")
                break

            batch_frames.append(frame)
            # 每隔 infer_every 幀才做 YOLO 推論
            if (absolute_frame_idx + i) % infer_every == 0:
                area, _, _ = _crop_roi(frame, roi)
                infer_areas.append(area)
                infer_local_idx.append(i)

        if not batch_frames:
            break

        # ── 2. YOLO 推論（逐幀，幀跳躍已提供主要加速）──────────────
        yolo_result_map = {}
        if infer_areas:
            _, rx, ry = _crop_roi(batch_frames[0], roi)
            for local_i, area in zip(infer_local_idx, infer_areas):
                res = tracker.model.predict(
                    area,
                    imgsz=INFER_SIZE,
                    classes=[32],
                    verbose=False,
                    device=tracker.device,
                    half=tracker.use_half,
                )
                best_box, best_conf = _extract_best_box(res[0], rx, ry)
                yolo_result_map[local_i] = (best_box, best_conf)


        # ── 3. 逐幀套 Kalman，記錄結果 ───────────────────────────
        for i, frame in enumerate(batch_frames):
            cur_idx = absolute_frame_idx + i

            if i in yolo_result_map:
                # 這幀有跑 YOLO
                best_box, best_conf = yolo_result_map[i]
            else:
                # 跳過的幀：不餵新觀測，讓 Kalman 自行預測
                best_box, best_conf = None, 0.0

            box_out, conf_out, status = _kalman_step(tracker, best_box, best_conf)

            save_frame = frame.copy() \
                if status == "PREDICTED" or (0.01 <= conf_out < high_conf_thresh) \
                else None

            tracking_data.append({
                "frame_idx": cur_idx,
                "time":  cur_idx / fps if fps > 0 else 0,
                "box":   box_out,
                "conf":  conf_out,
                "status": status,
                "frame": save_frame,
            })

            if status == "PREDICTED" or (0.01 <= conf_out < high_conf_thresh):
                review_frames.append(cur_idx)

        absolute_frame_idx += len(batch_frames)

        # 降低 Streamlit 進度條更新頻率 (每 90 幀更新一次)，減少網頁更新造成的延遲
        if progress_callback and absolute_frame_idx % 90 < effective_batch:
            progress_callback(
                absolute_frame_idx / total_frames,
                f"YOLO 追蹤中 ({absolute_frame_idx}/{total_frames})，推論幀率 1/{infer_every}"
            )

        if len(batch_frames) < effective_batch:
            break

    cap.release()
    return tracking_data, review_frames
