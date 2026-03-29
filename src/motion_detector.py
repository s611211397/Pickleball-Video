"""視覺動態分析模組 — 偵測球場內的運動狀態"""

import cv2
import numpy as np
from tqdm import tqdm


def compute_motion_score(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    roi: dict,
    gaussian_kernel: int = 21,
) -> float:
    """計算 ROI 區域內的動態分數。

    Args:
        frame_prev: 前一幀
        frame_curr: 當前幀
        roi: ROI 座標 {"x", "y", "w", "h"}
        gaussian_kernel: 高斯模糊核大小

    Returns:
        動態分數 (0.0 ~ 1.0)，越高代表動態越大
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # 裁切 ROI 區域
    crop_prev = frame_prev[y : y + h, x : x + w]
    crop_curr = frame_curr[y : y + h, x : x + w]

    # 轉灰階
    gray_prev = cv2.cvtColor(crop_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(crop_curr, cv2.COLOR_BGR2GRAY)

    # 高斯模糊降噪
    k = (gaussian_kernel, gaussian_kernel)
    gray_prev = cv2.GaussianBlur(gray_prev, k, 0)
    gray_curr = cv2.GaussianBlur(gray_curr, k, 0)

    # 幀差取絕對值
    diff = cv2.absdiff(gray_prev, gray_curr)

    # 二值化 + 計算動態像素佔比
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = float(np.sum(thresh > 0)) / thresh.size

    return motion_score


def analyze_video_motion(
    video_path: str,
    roi: dict,
    frame_skip: int = 2,
    gaussian_kernel: int = 21,
) -> list[dict]:
    """分析整段影片的動態時序資料。

    Args:
        video_path: 影片路徑
        roi: ROI 座標
        frame_skip: 每 N 幀分析一次
        gaussian_kernel: 高斯模糊核大小

    Returns:
        動態時序列表 [{"time": float, "score": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("無法讀取第一幀")

    timeline = []
    frame_idx = 1

    with tqdm(total=total_frames, desc="分析動態", unit="frame") as pbar:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                score = compute_motion_score(
                    prev_frame, curr_frame, roi, gaussian_kernel
                )
                timeline.append({
                    "time": frame_idx / fps,
                    "score": score,
                })
                prev_frame = curr_frame

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return timeline
