"""融合判斷引擎 2.0 — 物理軌跡與影音交軌語意裁判系統"""

import bisect
import numpy as np
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
    motion_threshold: float = 0.08,
    **kwargs
) -> list[Segment]:
    """融合判斷引擎：結合 YOLO 追蹤、動態分析、音訊擊球聲三重訊號。

    策略：
    - YOLO 追蹤到球 (DETECTED/PREDICTED) → 活躍
    - 動態分數超過門檻 → 活躍
    - 前後 1 秒內有擊球聲 → 活躍
    只要任一訊號為活躍，該時間點即視為比賽進行中。
    連續活躍點以 gap_threshold 為斷點分段，過短的段落過濾掉。
    """
    if hit_times is None:
        hit_times = []
    sorted_hits = sorted(hit_times)

    # 建立動態分數的時間查找表 (用於快速查詢某時間點的動態分數)
    motion_lookup = {}
    if motion_timeline:
        for entry in motion_timeline:
            # 以 0.1 秒為精度建立查找表
            key = round(entry["time"], 1)
            motion_lookup[key] = entry["score"]

    # 決定主要時間軸來源
    has_tracking = tracking_data and len(tracking_data) > 0
    has_motion = motion_timeline and len(motion_timeline) > 0

    if not has_tracking and not has_motion:
        return []

    active_points = []

    if has_tracking:
        for td in tracking_data:
            t = td["time"]

            # 1. 視覺判定：YOLO 追蹤到球
            has_vision = (td["box"] is not None and td["status"] in ["DETECTED", "PREDICTED"])

            # 2. 聲學判定：前後 1.0 秒內有擊球聲
            has_audio = False
            if len(sorted_hits) > 0:
                idx = bisect.bisect_left(sorted_hits, t - 1.0)
                if idx < len(sorted_hits) and sorted_hits[idx] <= t + 1.0:
                    has_audio = True

            # 3. 動態判定：該時間點的動態分數超過門檻
            has_motion_signal = False
            t_key = round(t, 1)
            if t_key in motion_lookup and motion_lookup[t_key] >= motion_threshold:
                has_motion_signal = True

            if has_vision or has_audio or has_motion_signal:
                active_points.append(t)
    else:
        # 沒有 YOLO 追蹤資料時，純用動態 + 音訊
        for entry in motion_timeline:
            t = entry["time"]

            has_motion_signal = entry["score"] >= motion_threshold

            has_audio = False
            if len(sorted_hits) > 0:
                idx = bisect.bisect_left(sorted_hits, t - 1.0)
                if idx < len(sorted_hits) and sorted_hits[idx] <= t + 1.0:
                    has_audio = True

            if has_motion_signal or has_audio:
                active_points.append(t)

    if not active_points:
        return []

    # 把零星活躍點以 gap_threshold 為斷點黏成段落
    segments = []
    current_start = active_points[0]
    last_active = active_points[0]

    for t in active_points[1:]:
        if t - last_active > gap_threshold:
            segments.append(Segment(
                start=max(0.0, current_start - 1.5),
                end=last_active + 1.5
            ))
            current_start = t
        last_active = t

    # 最後一段收尾
    segments.append(Segment(
        start=max(0.0, current_start - 1.5),
        end=last_active + 1.5
    ))

    # 過濾過短片段
    final_segments = [s for s in segments if s.duration >= min_duration]

    # 合併過近的相鄰段落 (避免因短暫 gap 產生碎片)
    if len(final_segments) > 1:
        merged = [final_segments[0]]
        for seg in final_segments[1:]:
            prev = merged[-1]
            if seg.start - prev.end < gap_threshold / 2:
                merged[-1] = Segment(start=prev.start, end=seg.end)
            else:
                merged.append(seg)
        final_segments = merged

    print(f"🧐 [裁判 2.0] 報告：三重訊號融合分析，切出 {len(final_segments)} 段比賽回合。")
    return final_segments
