"""ROI 選取模組 — 讓使用者框選球場範圍"""

import json
from pathlib import Path

import cv2
import numpy as np


def select_roi(video_path: str, output_path: str = "roi.json") -> dict:
    """從影片第一幀讓使用者框選 ROI 並儲存。

    會自動縮放高解析度影片的顯示視窗，選取後顯示預覽讓使用者確認。

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
    frame_h, frame_w = frame.shape[:2]
    cap.release()

    if not ret:
        raise ValueError(f"無法讀取影片幀: {video_path}")

    # 高解析度影片縮放顯示（超過 1280 寬就等比縮小）
    max_display_w = 1280
    scale = 1.0
    display_frame = frame
    if frame_w > max_display_w:
        scale = max_display_w / frame_w
        display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    window_name = "Select your court area, then press ENTER/SPACE. Press C to cancel."
    roi_raw = cv2.selectROI(window_name, display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # 將縮放後的座標還原為原始座標
    x = int(roi_raw[0] / scale)
    y = int(roi_raw[1] / scale)
    w = int(roi_raw[2] / scale)
    h = int(roi_raw[3] / scale)

    if w == 0 or h == 0:
        raise ValueError("未選取有效的 ROI 區域")

    # 確保 ROI 不超出影片範圍
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    roi_data = {"x": x, "y": y, "w": w, "h": h}

    # 顯示預覽讓使用者確認
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    label = f"ROI: ({x},{y}) {w}x{h}"
    cv2.putText(preview, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    preview_display = preview
    if frame_w > max_display_w:
        preview_display = cv2.resize(preview, None, fx=scale, fy=scale)

    cv2.imshow("ROI Preview - Press any key to confirm", preview_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 儲存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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


def validate_roi(roi: dict, frame_width: int, frame_height: int) -> dict:
    """驗證並修正 ROI 座標，確保在影片範圍內。

    Args:
        roi: ROI 座標 {"x", "y", "w", "h"}
        frame_width: 影片寬度
        frame_height: 影片高度

    Returns:
        修正後的 ROI 座標
    """
    # 先檢查原始 ROI 是否完全在影片範圍外
    if roi["x"] >= frame_width or roi["y"] >= frame_height:
        raise ValueError(
            f"ROI 起點 ({roi['x']},{roi['y']}) 超出影片範圍 ({frame_width}x{frame_height})"
        )

    x = max(0, min(roi["x"], frame_width - 1))
    y = max(0, min(roi["y"], frame_height - 1))
    w = min(roi["w"], frame_width - x)
    h = min(roi["h"], frame_height - y)

    if w <= 0 or h <= 0:
        raise ValueError(
            f"ROI ({x},{y},{w},{h}) 超出影片範圍 ({frame_width}x{frame_height})"
        )

    return {"x": x, "y": y, "w": w, "h": h}
