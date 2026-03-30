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
        self.model = YOLO(model_path)
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
                device=0 if torch.cuda.is_available() else "cpu", 
                half=True if torch.cuda.is_available() else False
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

def analyze_video_with_yolo(video_path, roi, conf_thresh=0.2, high_conf_thresh=0.5, progress_callback=None):
    model_name = "yolov8n.pt"
    
    # 若有客製化的模型則優先使用
    custom_model = Path("dataset/runs/train/weights/best.pt")
    if custom_model.exists():
        model_name = str(custom_model)
        
    tracker = BallTracker(model_path=model_name, conf_thresh=conf_thresh, high_conf_thresh=high_conf_thresh)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    tracking_data = [] # stores movement history
    review_frames = [] # frame indices for active learning
    
    frame_idx = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(f"警告：讀取第 {frame_idx} 幀時發生錯誤 ({e})，停止讀取後續影格。")
            break
            
        # 追蹤
        box, conf, status = tracker.track(frame, roi=roi)
        
        tracking_data.append({
            "frame_idx": frame_idx,
            "time": frame_idx / fps if fps > 0 else 0,
            "box": box,
            "conf": conf,
            "status": status,
            "frame": frame.copy() if status == "PREDICTED" or (0.01 <= conf < high_conf_thresh) else None
        })
        
        # 判斷是否需要人工審核 (有球但信心不夠，或者是本來有球卻 LOST 的前後)
        if status == "PREDICTED" or (0.01 <= conf < high_conf_thresh):
            review_frames.append(frame_idx)
            
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / total_frames, f"YOLO 追蹤分析中 ({frame_idx}/{total_frames})")
            
    cap.release()
    return tracking_data, review_frames
