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
    for point in motion_timeline:
        t = point["time"]
        motion_score = point["score"]

        # 檢查附近是否有擊球聲（±0.5 秒內）
        has_hit = _has_nearby_hit(t, sorted_hits, window=0.5)
        audio_score = 1.0 if has_hit else 0.0

        combined = motion_weight * min(motion_score / motion_threshold, 1.0) + audio_weight * audio_score

        if combined >= activity_threshold:
            active_points.append(t)

    if not active_points:
        return []

    # 根據間隔分群成各段 rally
    segments = []
    seg_start = active_points[0]
    prev_time = active_points[0]

    for t in active_points[1:]:
        if t - prev_time > gap_threshold:
            # 間隔超過門檻，前一段結束
            segments.append(Segment(start=seg_start, end=prev_time))
            seg_start = t
        prev_time = t

    # 最後一段
    segments.append(Segment(start=seg_start, end=prev_time))

    # 過濾太短的 rally（撿球、走動）
    segments = [seg for seg in segments if seg.duration >= min_duration]

    return segments


def _has_nearby_hit(time: float, sorted_hits: list[float], window: float = 0.5) -> bool:
    """檢查指定時間點附近是否有擊球聲。"""
    if not sorted_hits:
        return False

    idx = bisect.bisect_left(sorted_hits, time - window)
    while idx < len(sorted_hits) and sorted_hits[idx] <= time + window:
        return True

    return False
