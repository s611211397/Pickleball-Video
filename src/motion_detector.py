"""視覺動態分析模組 — 偵測球場內的運動狀態"""

from collections import deque

import cv2
import numpy as np
from tqdm import tqdm


def compute_motion_score(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    gaussian_kernel: int = 11,
) -> float:
    """計算動態分數（輸入為已裁切+縮放的灰階幀）。

    Args:
        gray_prev: 前一幀（灰階）
        gray_curr: 當前幀（灰階）
        gaussian_kernel: 高斯模糊核大小

    Returns:
        動態分數 (0.0 ~ 1.0)
    """
    k = (gaussian_kernel, gaussian_kernel)
    blur_prev = cv2.GaussianBlur(gray_prev, k, 0)
    blur_curr = cv2.GaussianBlur(gray_curr, k, 0)

    diff = cv2.absdiff(blur_prev, blur_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(thresh)) / thresh.size


def smooth_timeline(
    timeline: list[dict],
    window_size: int = 5,
) -> list[dict]:
    """對動態時序做滑動平均平滑化，降低瞬間雜訊。"""
    if len(timeline) <= window_size:
        return timeline

    scores = [p["score"] for p in timeline]
    smoothed_scores = []
    buf = deque(maxlen=window_size)

    for s in scores:
        buf.append(s)
        smoothed_scores.append(sum(buf) / len(buf))

    return [
        {"time": timeline[i]["time"], "score": smoothed_scores[i]}
        for i in range(len(timeline))
    ]


def _auto_frame_skip(fps: float, total_frames: int) -> int:
    """根據影片長度自動決定 frame_skip。

    短片（<5 分鐘）: 每 3 幀
    中片（5-30 分鐘）: 每 5 幀
    長片（>30 分鐘）: 每 8 幀

    動態偵測不需要每幀都看，每 0.2-0.3 秒看一次就夠了。
    """
    duration = total_frames / fps if fps > 0 else 0
    if duration < 300:
        return 3
    elif duration < 1800:
        return 5
    else:
        return 8


def _extract_roi_gray(frame: np.ndarray, roi: dict, scale: float) -> np.ndarray:
    """從幀中裁切 ROI 區域，轉灰階，並縮放。"""
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def analyze_video_motion(
    video_path: str,
    roi: dict,
    frame_skip: int = 0,
    gaussian_kernel: int = 11,
    smooth_window: int = 5,
    max_roi_dim: int = 320,
    progress_callback=None,
) -> list[dict]:
    """分析整段影片的動態時序資料。

    主要優化：
    - 跳過的幀用 grab() 而非 read()，避免無謂的完整解碼（快 3-5 倍）
    - ROI 區域縮放到 max_roi_dim 以內，大幅降低像素運算量
    - 自動根據影片長度決定 frame_skip

    Args:
        video_path: 影片路徑
        roi: ROI 座標
        frame_skip: 每 N 幀分析一次（0 = 自動決定）
        gaussian_kernel: 高斯模糊核大小
        smooth_window: 平滑窗口大小（0 = 不平滑）
        max_roi_dim: ROI 縮放後的最大邊長（像素）
        progress_callback: 進度回呼 fn(pct: float, msg: str)

    Returns:
        動態時序列表 [{"time": float, "score": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (roi["x"] + roi["w"] > frame_w) or (roi["y"] + roi["h"] > frame_h):
        cap.release()
        raise ValueError(
            f"ROI ({roi['x']},{roi['y']},{roi['w']},{roi['h']}) "
            f"超出影片範圍 ({frame_w}x{frame_h})"
        )

    # 自動決定 frame_skip
    if frame_skip <= 0:
        frame_skip = _auto_frame_skip(fps, total_frames)

    # 計算 ROI 縮放比例（動態偵測不需要全解析度）
    roi_max_side = max(roi["w"], roi["h"])
    roi_scale = min(1.0, max_roi_dim / roi_max_side) if roi_max_side > max_roi_dim else 1.0

    # 讀第一幀
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("無法讀取第一幀")

    prev_gray = _extract_roi_gray(first_frame, roi, roi_scale)

    timeline = []
    frame_idx = 1
    analyze_count = 0

    duration_sec = total_frames / fps if fps > 0 else 0
    estimated_points = total_frames // frame_skip

    with tqdm(total=total_frames, desc="分析動態", unit="frame") as pbar:
        while True:
            if frame_idx % frame_skip == 0:
                try:
                    ret = cap.grab()
                    if not ret:
                        break
                    ret, curr_frame = cap.retrieve()
                    if not ret:
                        break
                except Exception as e:
                    print(f"警告：讀取第 {frame_idx} 幀時發生錯誤 ({e})，停止讀取後續影格。")
                    break

                curr_gray = _extract_roi_gray(curr_frame, roi, roi_scale)
                score = compute_motion_score(prev_gray, curr_gray, gaussian_kernel)

                timeline.append({
                    "time": frame_idx / fps,
                    "score": score,
                })
                prev_gray = curr_gray
                analyze_count += 1

                # 進度回呼（給 Streamlit UI 用）
                if progress_callback and analyze_count % 50 == 0:
                    pct = frame_idx / total_frames
                    progress_callback(pct, f"已分析 {analyze_count} 個時間點...")
            else:
                # 跳過的幀：只 grab 不 decode，快很多
                try:
                    if not cap.grab():
                        break
                except Exception as e:
                    print(f"警告：掠過第 {frame_idx} 幀時發生錯誤 ({e})，停止讀取後續影格。")
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if smooth_window > 0 and len(timeline) > smooth_window:
        timeline = smooth_timeline(timeline, smooth_window)

    return timeline
