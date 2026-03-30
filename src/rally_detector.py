"""融合判斷引擎 2.0 — 物理軌跡與影音交軌語意裁判系統"""

import bisect
import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter


@dataclass
class Segment:
    """一段 rally 的時間區間。"""
    start: float  # 開始時間（秒）
    end: float    # 結束時間（秒）

    @property
    def duration(self) -> float:
        return self.end - self.start


"""融合判斷引擎 1.5 — 高容錯寬鬆分段演算法 (適合模型初期訓練階段)"""

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
    gap_threshold: float = 4.0,       # 左側參數：最大容忍斷軌時間
    min_duration: float = 3.0,        # 左側參數：最短回合長度
    **kwargs
) -> list[Segment]:
    """1.5 版：寬鬆的高容錯分群法。
    在 AI 視覺模型仍不穩定（常常漏抓球）的初期階段，
    採用最原始粗暴的方案：「只要有聲音」或「只有短暫看到一點球的影子」，皆視為比賽進行中。
    """
    if not tracking_data or len(tracking_data) == 0:
        return []

    if hit_times is None:
        hit_times = []
    sorted_hits = sorted(hit_times)

    active_points = []
    
    # 掃描每一格畫面，給予「生存判定」
    for td in tracking_data:
        t = td["time"]
        
        # 1. 視覺判定：這格到底有沒有抓到球？
        has_vision = (td["box"] is not None and td["status"] in ["DETECTED", "PREDICTED"])
        
        # 2. 聲學判定：這格的前後 1.0 秒內，有沒有聽到擊球聲？
        has_audio = False
        if len(sorted_hits) > 0:
            idx = bisect.bisect_left(sorted_hits, t - 1.0)
            if idx < len(sorted_hits) and sorted_hits[idx] <= t + 1.0:
                has_audio = True
                
        # 👑 寬容判定核心：只要「有聽到聲音」或「有看到一瞬間的球」，這個時間點就算「活著 (Active)」
        if has_vision or has_audio:
            active_points.append(t)
            
    if not active_points:
        return []

    # === 把零星生存點黏起來 ===
    segments = []
    current_start = active_points[0]
    last_active = active_points[0]
    
    for t in active_points[1:]:
        # 如果經過了 gap_threshold (預設 4 秒) 完全沒動靜也沒聲音
        if t - last_active > gap_threshold:
            # 宣告前一段回合結束！(往前/往後多包 1.5 秒做緩衝)
            segments.append(Segment(start=max(0.0, current_start - 1.5), end=last_active + 1.5))
            current_start = t
            
        last_active = t
        
    # 最後一段收尾
    segments.append(Segment(start=max(0.0, current_start - 1.5), end=last_active + 1.5))
    
    # 過濾：把走路、純脆按了一聲喇叭之類的極短片段濾掉
    final_segments = [s for s in segments if s.duration >= min_duration]
    
    print(f"🧐 [裁判 1.5] 報告：依據高容錯演算法，幫您成功切出 {len(final_segments)} 局比賽。")
    return final_segments
