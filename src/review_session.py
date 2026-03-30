"""審核進度持久化模組 — 儲存/載入 YOLO 軌跡審核的中途進度，讓使用者可以跨 session 續審。"""

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np


# 審核資料統一存放目錄
SESSIONS_DIR = Path("review_sessions")


def _video_id(video_name: str) -> str:
    """從影片檔名產生安全的資料夾名稱。"""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(video_name).stem)
    return safe


def save_review_session(
    video_name: str,
    review_frames: list[int],
    current_review_idx: int,
    tracking_data: list[dict],
    segments: list,
    motion_timeline: list[dict],
    hit_times: list[float] | None,
    roi: dict | None,
):
    """將審核進度寫入磁碟。

    儲存結構：
        review_sessions/<video_id>/
            session.json     — 元資料 (review_frames, idx, segments, motion, hits, roi)
            frames/          — 審核用的影像 (只存 review_frames 指向的幀)
            tracking_meta/   — 每幀的追蹤資訊 (不含 numpy frame)
    """
    vid = _video_id(video_name)
    session_dir = SESSIONS_DIR / vid
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. 儲存每張 review frame 的影像 + tracking 元資料
    tracking_meta = {}
    for fi in review_frames:
        if fi >= len(tracking_data):
            continue
        td = tracking_data[fi]

        # 儲存元資料 (排除 numpy frame)
        tracking_meta[str(fi)] = {
            "frame_idx": td.get("frame_idx", fi),
            "time": td.get("time", 0),
            "box": td.get("box"),
            "conf": td.get("conf", 0),
            "status": td.get("status", "LOST"),
        }

        # 儲存影像 (只存有 frame 資料的)
        frame = td.get("frame")
        if frame is not None:
            img_path = frames_dir / f"{fi}.jpg"
            if not img_path.exists():
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 2. 序列化 segments
    seg_list = []
    for s in segments:
        seg_list.append({"start": s.start, "end": s.end})

    # 3. 寫入 session.json
    session_data = {
        "video_name": video_name,
        "review_frames": review_frames,
        "current_review_idx": current_review_idx,
        "total_review_frames": len(review_frames),
        "segments": seg_list,
        "motion_timeline": motion_timeline,
        "hit_times": hit_times,
        "roi": roi,
        "tracking_meta": tracking_meta,
    }

    with open(session_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return session_dir


def load_review_session(video_name: str) -> dict | None:
    """載入先前儲存的審核進度。

    Returns:
        dict with keys: review_frames, current_review_idx, tracking_data,
                        segments, motion_timeline, hit_times, roi
        None if no saved session exists.
    """
    vid = _video_id(video_name)
    session_dir = SESSIONS_DIR / vid
    session_file = session_dir / "session.json"

    if not session_file.exists():
        return None

    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 重建 tracking_data (只重建 review_frames 涉及的幀)
    # 先建立一個稀疏的 tracking_data list
    tracking_meta = data.get("tracking_meta", {})
    review_frames = data["review_frames"]

    if not review_frames:
        return None

    max_fi = max(review_frames)
    tracking_data = [
        {"frame_idx": i, "time": 0, "box": None, "conf": 0, "status": "LOST", "frame": None}
        for i in range(max_fi + 1)
    ]

    # 填入已存的元資料
    frames_dir = session_dir / "frames"
    for fi_str, meta in tracking_meta.items():
        fi = int(fi_str)
        if fi > max_fi:
            continue
        tracking_data[fi].update(meta)
        tracking_data[fi]["frame"] = None  # 先清空

        # 嘗試載入影像
        img_path = frames_dir / f"{fi}.jpg"
        if img_path.exists():
            frame = cv2.imread(str(img_path))
            if frame is not None:
                tracking_data[fi]["frame"] = frame

    # 重建 segments
    from src.rally_detector import Segment
    segments = [Segment(start=s["start"], end=s["end"]) for s in data.get("segments", [])]

    return {
        "review_frames": review_frames,
        "current_review_idx": data.get("current_review_idx", 0),
        "tracking_data": tracking_data,
        "segments": segments,
        "motion_timeline": data.get("motion_timeline", []),
        "hit_times": data.get("hit_times"),
        "roi": data.get("roi"),
    }


def has_saved_session(video_name: str) -> dict | None:
    """檢查是否有未完成的審核進度，回傳摘要資訊或 None。"""
    vid = _video_id(video_name)
    session_file = SESSIONS_DIR / vid / "session.json"

    if not session_file.exists():
        return None

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = data.get("total_review_frames", 0)
        current = data.get("current_review_idx", 0)

        if total == 0 or current >= total:
            return None  # 已完成，不需恢復

        return {
            "total": total,
            "current": current,
            "remaining": total - current,
            "segments_count": len(data.get("segments", [])),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def delete_session(video_name: str):
    """刪除已儲存的審核進度。"""
    vid = _video_id(video_name)
    session_dir = SESSIONS_DIR / vid
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)
