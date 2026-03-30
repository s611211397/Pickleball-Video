"""融合判斷引擎 — 結合視覺與音訊偵測 rally 起止點"""

import bisect
from dataclasses import dataclass


@dataclass
class Segment:
    """一段 rally 的時間區間。"""
    start: float  # 開始時間（秒）
    end: float    # 結束時間（秒）

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_rallies(
    motion_timeline: list[dict],
    hit_times: list[float] | None = None,
    tracking_data: list[dict] | None = None,
    gap_threshold: float = 4.0,
    min_duration: float = 3.0,
    activity_threshold: float = 0.3,
    motion_weight: float = 0.7,
    audio_weight: float = 0.3,
    motion_threshold: float = 0.08,
) -> list[Segment]:
    """偵測所有 rally 區段。

    Args:
        motion_timeline: 視覺動態時序 [{"time", "score"}, ...]
        hit_times: 擊球時間點列表（可為 None，純視覺模式）
        gap_threshold: 無活動間隔門檻 (秒)
        min_duration: 最短 rally 時長 (秒)
        activity_threshold: 綜合活動分數門檻
        motion_weight: 視覺權重
        audio_weight: 音訊權重
        motion_threshold: 視覺動態門檻

    Returns:
        rally 區段列表
    """
    if not motion_timeline:
        return []

    # 如果沒有音訊資料，改為純視覺模式
    if hit_times is None:
        hit_times = []
        motion_weight = 1.0
        audio_weight = 0.0

    sorted_hits = sorted(hit_times)

    # 計算每個時間點的綜合活動分數
    active_points = []
    for idx, point in enumerate(motion_timeline):
        t = point["time"]
        motion_score = point["score"]

        # 正規化動態分數：以 motion_threshold 為基準，超過則 clamp 到 1.0
        normalized_motion = min(motion_score / motion_threshold, 1.0) if motion_threshold > 0 else 0.0

        # 檢查附近是否有擊球聲（±0.5 秒內）
        has_hit = _has_nearby_hit(t, sorted_hits, window=0.5)
        audio_score = 1.0 if has_hit else 0.0

        # YOLO 追蹤判斷：如果狀態是偵測到，加強動態分數
        yolo_boost = 0.0
        if tracking_data and idx < len(tracking_data):
            td = tracking_data[idx]
            if td["status"] in ["DETECTED", "PREDICTED"]:
                yolo_boost = 1.0  # YOLO 確認球在畫面內

        # 融合加總 (給 YOLO 極高的權重)
        # 如果 YOLO 有抓到，等同於畫面有大量動態
        if tracking_data:
            combined = (0.5 * normalized_motion + 0.5 * yolo_boost) * motion_weight + audio_weight * audio_score
        else:
            combined = motion_weight * normalized_motion + audio_weight * audio_score

        if combined >= activity_threshold:
            active_points.append(t)

    if not active_points:
        return []

    # 根據間隔分群成各段 rally
    segments = _cluster_into_segments(active_points, gap_threshold)

    # 過濾太短的 rally（撿球、走動）
    segments = [seg for seg in segments if seg.duration >= min_duration]

    # 合併相鄰過近的 segments（間隔 < gap_threshold 的一半）
    segments = _merge_close_segments(segments, min_gap=gap_threshold / 2)

    return segments


def _cluster_into_segments(active_points: list[float], gap_threshold: float) -> list[Segment]:
    """將活動時間點根據間隔分群成段落。"""
    segments = []
    seg_start = active_points[0]
    prev_time = active_points[0]

    for t in active_points[1:]:
        if t - prev_time > gap_threshold:
            segments.append(Segment(start=seg_start, end=prev_time))
            seg_start = t
        prev_time = t

    # 最後一段
    segments.append(Segment(start=seg_start, end=prev_time))
    return segments


def _merge_close_segments(segments: list[Segment], min_gap: float) -> list[Segment]:
    """合併間隔太近的相鄰段落，避免碎片化。"""
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start - prev.end < min_gap:
            # 合併：擴展前一段的結束時間
            merged[-1] = Segment(start=prev.start, end=seg.end)
        else:
            merged.append(seg)

    return merged


def _has_nearby_hit(time: float, sorted_hits: list[float], window: float = 0.5) -> bool:
    """檢查指定時間點附近（±window 秒）是否有擊球聲。

    使用二分搜尋在已排序的擊球時間列表中查找。
    """
    if not sorted_hits:
        return False

    # 找到第一個 >= time - window 的位置
    idx = bisect.bisect_left(sorted_hits, time - window)

    # 檢查該位置的值是否 <= time + window
    if idx < len(sorted_hits) and sorted_hits[idx] <= time + window:
        return True

    return False
