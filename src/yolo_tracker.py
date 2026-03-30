"""YOLO 物件追蹤與分析模組"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BallTracker:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.2, high_conf_thresh: float = 0.5):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"BallTracker 使用裝置: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        self.conf_thresh = conf_thresh
        self.high_conf_thresh = high_conf_thresh
        
        self.reset_tracker()
        
    def reset_tracker(self):
        """重置追蹤器狀態 (Kalman Filter)"""
        # 狀態矩陣 [x, y, dx, dy] -> 中心座標與速度
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # 狀態轉移矩陣 (恆定速度模型)
        dt = 1.0 # 時間步長
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 測量矩陣 (我們只測量 x, y)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 不確定性矩陣
        self.kf.P *= 1000.0
        # 測量雜訊 (假設視覺測量有時候會飄移)
        self.kf.R = np.array([
            [10, 0],
            [0, 10]
        ])
        # 過程雜訊 (球受空氣阻力、重力改變速度)
        self.kf.Q *= 0.1
        
        self.is_tracking = False
        self.missed_frames = 0
        self.max_missed_frames = 15 # 若速度快，0.5秒差不多 15幀
        
    def track(self, frame, roi=None):
        """
        在給定的畫面中尋找球並更新軌跡狀態。
        
        Returns:
            ball_pos: {x, y, w, h} 或是 None
            conf: 預測信心度
            status: "DETECTED" | "PREDICTED" | "LOST"
        """
        # YOLO 預測 (只抓 sports ball, class 32 in COCO)
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            search_area = frame[y:y+h, x:x+w].copy()

            # 若有提供多邊形頂點，且頂點數量大於 2，則繪製遮罩
            if "points" in roi and len(roi["points"]) >= 3:
                mask = np.zeros(search_area.shape[:2], dtype=np.uint8)
                
                # 計算相對座標點
                pts = np.array([[[p["x"] - x, p["y"] - y] for p in roi["points"]]], dtype=np.int32)
                cv2.fillPoly(mask, pts, 255)
                
                # 保留多邊形內的像素，多邊形外塗黑
                search_area = cv2.bitwise_and(search_area, search_area, mask=mask)

        else:
            search_area = frame
            x, y = 0, 0
            
        results = self.model.predict(
            search_area, 
            classes=[32], 
            verbose=False, 
            device=self.device,
            half=self.use_half
        )
        
        best_box = None
        best_conf = 0.0
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    bx, by, bw, bh = box.xywh[0].cpu().numpy()
                    best_box = {
                        "x": int(bx - bw/2) + x,
                        "y": int(by - bh/2) + y,
                        "w": int(bw),
                        "h": int(bh)
                    }
                    
        # Kalman Filter 邏輯
        self.kf.predict()
        
        if best_box and best_conf >= self.conf_thresh:
            # 觀測更新
            cx = best_box["x"] + best_box["w"]/2
            cy = best_box["y"] + best_box["h"]/2
            
            if not self.is_tracking:
                self.kf.x = np.array([cx, cy, 0, 0]).reshape(4, 1)
                self.is_tracking = True
            else:
                self.kf.update(np.array([cx, cy]).reshape(2, 1))
                
            self.missed_frames = 0
            return best_box, best_conf, "DETECTED"
            
        else:
            self.missed_frames += 1
            if self.is_tracking and self.missed_frames <= self.max_missed_frames:
                # 盲測階段
                pred_x = int(self.kf.x[0, 0])
                pred_y = int(self.kf.x[1, 0])
                pred_w = 20 # 假設固定大小
                pred_h = 20
                
                pred_box = {
                    "x": pred_x - pred_w//2,
                    "y": pred_y - pred_h//2,
                    "w": pred_w,
                    "h": pred_h
                }
                return pred_box, best_conf, "PREDICTED"
            else:
                self.is_tracking = False
                return None, 0.0, "LOST"

def analyze_video_with_yolo(video_path, roi, conf_thresh=0.2, high_conf_thresh=0.5, progress_callback=None, batch_size=16):
    model_name = "yolov8n.pt"
    
    # 若有客製化的備份模型則優先使用
    custom_model_backup = Path("models/pickleball_best.pt")
    custom_model_local = Path("dataset/runs/train/weights/best.pt")
    
    if custom_model_backup.exists():
        model_name = str(custom_model_backup)
    elif custom_model_local.exists():
        model_name = str(custom_model_local)
        
    tracker = BallTracker(model_path=model_name, conf_thresh=conf_thresh, high_conf_thresh=high_conf_thresh)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 預先裁切 ROI 的偏移量
    if roi:
        rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    else:
        rx, ry = 0, 0
    
    tracking_data = []
    review_frames = []
    frame_idx = 0
    
    while True:
        # ── 讀進一批幀 ──────────────────────────
        batch_frames = []   # 原始 BGR 完整幀
        batch_areas = []    # 裁切後送 YOLO 的區塊
        
        for _ in range(batch_size):
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                print(f"警告：讀取第 {frame_idx + len(batch_frames)} 幀時發生錯誤 ({e})，停止讀取後續影格。")
                break
            
            batch_frames.append(frame)
            
            if roi:
                area = frame[ry:ry+rh, rx:rx+rw].copy()
                # 多邊形遮罩
                if "points" in roi and len(roi["points"]) >= 3:
                    mask = np.zeros(area.shape[:2], dtype=np.uint8)
                    pts = np.array([[[p["x"] - rx, p["y"] - ry] for p in roi["points"]]], dtype=np.int32)
                    cv2.fillPoly(mask, pts, 255)
                    area = cv2.bitwise_and(area, area, mask=mask)
            else:
                area = frame
            batch_areas.append(area)
        
        if not batch_frames:
            break
        
        # ── 一次性批次送進 YOLO ──────────────────
        batch_results = tracker.model.predict(
            batch_areas,
            classes=[32],
            verbose=False,
            device=tracker.device,
            half=tracker.use_half,
        )
        
        # ── 逐幀套用 Kalman Filter 並記錄結果 ────
        for i, (frame, result) in enumerate(zip(batch_frames, batch_results)):
            cur_idx = frame_idx + i
            
            # 找這幀最高信心的框
            best_box = None
            best_conf = 0.0
            if len(result.boxes) > 0:
                for box in result.boxes:
                    c = float(box.conf[0])
                    if c > best_conf:
                        best_conf = c
                        bx, by, bw, bh = box.xywh[0].cpu().numpy()
                        best_box = {
                            "x": int(bx - bw/2) + rx,
                            "y": int(by - bh/2) + ry,
                            "w": int(bw),
                            "h": int(bh),
                        }
            
            # Kalman Filter 更新
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
                status = "DETECTED"
                box_out = best_box
                conf_out = best_conf
            else:
                tracker.missed_frames += 1
                if tracker.is_tracking and tracker.missed_frames <= tracker.max_missed_frames:
                    pred_x = int(tracker.kf.x[0, 0])
                    pred_y = int(tracker.kf.x[1, 0])
                    box_out = {"x": pred_x - 10, "y": pred_y - 10, "w": 20, "h": 20}
                    conf_out = best_conf
                    status = "PREDICTED"
                else:
                    tracker.is_tracking = False
                    box_out = None
                    conf_out = 0.0
                    status = "LOST"
            
            save_frame = frame.copy() if status == "PREDICTED" or (0.01 <= conf_out < high_conf_thresh) else None
            tracking_data.append({
                "frame_idx": cur_idx,
                "time": cur_idx / fps if fps > 0 else 0,
                "box": box_out,
                "conf": conf_out,
                "status": status,
                "frame": save_frame,
            })
            
            if status == "PREDICTED" or (0.01 <= conf_out < high_conf_thresh):
                review_frames.append(cur_idx)
        
        frame_idx += len(batch_frames)
        
        if progress_callback and frame_idx % (batch_size * 2) == 0:
            progress_callback(frame_idx / total_frames, f"YOLO 批次追蹤中 ({frame_idx}/{total_frames})")
        
        # 如果這批幀數不足 batch_size，代表影片已結束
        if len(batch_frames) < batch_size:
            break
            
    cap.release()
    return tracking_data, review_frames
