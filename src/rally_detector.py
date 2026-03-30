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


def detect_rallies(
    motion_timeline: list[dict],
    hit_times: list[float] | None = None,
    tracking_data: list[dict] | None = None,
    gap_threshold: float = 4.0,       # 仍作為無資料的斷點兜底
    min_duration: float = 3.0,
    **kwargs
) -> list[Segment]:
    """2.0 版：透過追蹤軌跡與音效交集，判斷真實回合。"""
    if not tracking_data or len(tracking_data) < 10:
        return []
    
    if hit_times is None:
        hit_times = []

    # 1. 提取物理軌跡 (取 Bounding Box 底部作為落地基準)
    times = []
    xs = []
    ys = []
    for td in tracking_data:
        times.append(td["time"])
        if td["box"] is not None and td["status"] in ["DETECTED", "PREDICTED"]:
            xs.append(td["box"]["x"] + td["box"]["w"] / 2.0) # 中心 X
            ys.append(td["box"]["y"] + td["box"]["h"])       # 底部 Y，適合算彈地
        else:
            xs.append(np.nan)
            ys.append(np.nan)
            
    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)
    
    # 2. 內插補齊與平滑處理
    valid_mask = ~np.isnan(ys)
    if not np.any(valid_mask):
        return []
    
    indices = np.arange(len(ys))
    xs_interp = np.interp(indices, indices[valid_mask], xs[valid_mask])
    ys_interp = np.interp(indices, indices[valid_mask], ys[valid_mask])
    
    # 使用 Savitzky-Golay 濾波器平滑軌跡，去除雜訊抖動，窗格大約 15 幀(0.5秒)
    window = min(15, len(ys_interp) if len(ys_interp) % 2 == 1 else len(ys_interp) - 1)
    window = max(3, window)
    xs_smooth = savgol_filter(xs_interp, window_length=window, polyorder=2)
    ys_smooth = savgol_filter(ys_interp, window_length=window, polyorder=2)
    
    # 3. 尋找「落地點 (Bounce)」
    bounce_idxs, _ = find_peaks(ys_smooth, prominence=10, distance=5)
    bounce_times = times[bounce_idxs]

    # 4. 尋找「真實擊球點 (True Hits)」
    true_hits = []
    dx = np.abs(np.diff(xs_smooth, prepend=xs_smooth[0]))
    dy = np.abs(np.diff(ys_smooth, prepend=ys_smooth[0]))
    speed = np.sqrt(dx**2 + dy**2) # 綜合 2D 移動距離
    
    if len(hit_times) > 0:
        # 動態影音融合模式
        for ht in sorted(hit_times):
            idx = bisect.bisect_left(times, ht)
            if 0 <= idx < len(times):
                search_start = max(0, idx - 6)
                search_end = min(len(times), idx + 6)
                
                lost_ratio = np.sum(np.isnan(ys[search_start:search_end])) / (search_end - search_start + 1e-5)
                if lost_ratio > 0.8:
                    continue
                    
                avg_speed = np.mean(speed[search_start:search_end])
                if avg_speed > 1.5: # 調低一點寬容度
                    true_hits.append(ht)
    else:
        # 【純視覺備用模式】如果影片靜音或沒麥克風聲音，改用物理「速度暴增點」作為擊球點
        hit_idxs, _ = find_peaks(speed, prominence=5, distance=15)
        for idx in hit_idxs:
            if not np.isnan(ys[idx]): # 確認當下球不是遺失狀態
                true_hits.append(times[idx])

    print(f"🧐 [裁判 2.0] 偵測報告：找到 {len(bounce_times)} 次落地, {len(hit_times)} 次聲學峰值, 過濾後得 {len(true_hits)} 次場內有效擊球 (True Hits)")
    
    # 5. 語意裁判引擎 (State Machine)
    # 規則：以第一個 True Hit 為發球，之後若發生「連續兩次 bounce 中間無 hit」則死球結束
    segments = []
    
    if not true_hits:
        return []
        
    current_start = true_hits[0]
    last_hit_time = true_hits[0]
    last_bounce_time = 0.0
    consecutive_bounces = 0
    
    # 將事件合併排序 (時間, 種類)
    events = [(t, 'HIT') for t in true_hits] + [(t, 'BOUNCE') for t in bounce_times]
    events.sort(key=lambda x: x[0])
    
    for t, e_type in events:
        if t < current_start:
            continue
            
        if e_type == 'HIT':
            last_hit_time = t
            consecutive_bounces = 0 # 重置彈地計數器
            
        elif e_type == 'BOUNCE':
            if t > last_hit_time:
                consecutive_bounces += 1
                last_bounce_time = t
            
            # 死球規則 1：連續兩次彈地
            if consecutive_bounces >= 2:
                # 回合結束在第二次彈地後 1.5 秒
                segments.append(Segment(start=current_start - 1.0, end=t + 1.5))
                # 尋找下一次開局
                next_hit_idx = bisect.bisect_right(true_hits, t)
                if next_hit_idx < len(true_hits):
                    current_start = true_hits[next_hit_idx]
                    last_hit_time = current_start
                    consecutive_bounces = 0
                else:
                    current_start = None
                    break
        
        # 死球規則 2：太久沒動作 (兜底，球可能飛出場外沒落地)
        if t - last_hit_time > gap_threshold:
            segments.append(Segment(start=current_start - 1.0, end=last_hit_time + 1.5))
            
            next_hit_idx = bisect.bisect_right(true_hits, t)
            if next_hit_idx < len(true_hits):
                current_start = true_hits[next_hit_idx]
                last_hit_time = current_start
                consecutive_bounces = 0
            else:
                current_start = None
                break
                
    # 處理最後未閉合的局
    if current_start is not None and len(true_hits) > 0 and current_start <= true_hits[-1]:
        segments.append(Segment(start=current_start - 1.0, end=last_hit_time + 2.0))
        
    # 6. 修剪重疊與太短的片段
    merged = []
    for s in segments:
        if s.duration < min_duration:
            continue
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            if s.start < prev.end:
                merged[-1] = Segment(start=prev.start, end=max(prev.end, s.end))
            else:
                merged.append(s)

    return merged
