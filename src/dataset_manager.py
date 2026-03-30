"""資料集管理模組 — 處理主動學習的標註資料收集"""

import os
from pathlib import Path
import cv2
import numpy as np

class DatasetManager:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        
        # 建立目錄
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def save_annotation(self, frame: np.ndarray, frame_id: str, box: dict | None, class_id: int = 0):
        """
        儲存影像與標註供 YOLO 訓練使用。
        
        Args:
            frame: 影像 BGR 陣列
            frame_id: 唯一識別碼 (例如: "video_name_frame_123")
            box: 標註框網要 {"x": pixel_x, "y": pixel_y, "w": pixel_w, "h": pixel_h}，若為 None 則代表此圖無球(背景圖)
            class_id: 類別 ID，我們預設球是 0
        """
        img_path = self.images_dir / f"{frame_id}.jpg"
        txt_path = self.labels_dir / f"{frame_id}.txt"
        
        # 儲存圖片
        if not img_path.exists():
            cv2.imwrite(str(img_path), frame)
            
        # 儲存標註 TXT
        # YOLO 格式: <class_index> <x_center> <y_center> <width> <height> (皆為 0~1 的相對比例)
        h, w = frame.shape[:2]
        with open(txt_path, "w") as f:
            if box is not None:
                # 轉換為 YOLO 格式
                cx = (box["x"] + box["w"] / 2) / w
                cy = (box["y"] + box["h"] / 2) / h
                nw = box["w"] / w
                nh = box["h"] / h
                
                # 寫入 (限制範圍避免超出圖外)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                # 無框，寫入空檔（對應背景學習）
                pass
                
        return str(img_path), str(txt_path)
