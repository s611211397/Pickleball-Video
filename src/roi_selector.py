"""ROI 選取模組 — 讓使用者框選球場範圍"""

import json
from pathlib import Path

import cv2


def select_roi(video_path: str, output_path: str = "roi.json") -> dict:
    """從影片第一幀讓使用者框選 ROI 並儲存。

    Args:
        video_path: 輸入影片路徑
        output_path: ROI 設定檔輸出路徑

    Returns:
        ROI 座標字典 {"x", "y", "w", "h"}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"無法讀取影片幀: {video_path}")

    window_name = "Select your court area, then press ENTER or SPACE"
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    if w == 0 or h == 0:
        raise ValueError("未選取有效的 ROI 區域")

    roi_data = {"x": x, "y": y, "w": w, "h": h}

    with open(output_path, "w") as f:
        json.dump(roi_data, f, indent=2)

    print(f"ROI 已儲存至 {output_path}: {roi_data}")
    return roi_data


def load_roi(config_path: str = "roi.json") -> dict:
    """載入已儲存的 ROI 設定。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI 設定檔不存在: {config_path}")

    with open(path) as f:
        roi_data = json.load(f)

    required_keys = {"x", "y", "w", "h"}
    if not required_keys.issubset(roi_data.keys()):
        raise ValueError(f"ROI 設定檔格式錯誤，需要: {required_keys}")

    return roi_data
