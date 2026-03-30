"""Rally 偵測引擎的單元測試"""

import pytest

from src.rally_detector import Segment, detect_rallies


class TestSegment:
    def test_duration(self):
        seg = Segment(start=10.0, end=25.0)
        assert seg.duration == 15.0

    def test_zero_duration(self):
        seg = Segment(start=5.0, end=5.0)
        assert seg.duration == 0.0


class TestDetectRallies:
    def _make_timeline(self, ranges, interval=0.1, active_score=0.2, idle_score=0.001):
        """建立模擬的動態時序。

        ranges: 活動區間列表 [(start, end), ...]
        """
        total = max(end for _, end in ranges) + 5
        timeline = []
        t = 0.0
        while t < total:
            score = idle_score
            for s, e in ranges:
                if s <= t <= e:
                    score = active_score
                    break
            timeline.append({"time": t, "score": score})
            t += interval
        return timeline

    def _make_tracking_data(self, ranges, interval=0.1, total=None):
        """建立模擬的 YOLO tracking data。"""
        if total is None:
            total = max(end for _, end in ranges) + 5
        data = []
        t = 0.0
        idx = 0
        while t < total:
            in_range = any(s <= t <= e for s, e in ranges)
            data.append({
                "frame_idx": idx,
                "time": t,
                "box": {"x": 100, "y": 100, "w": 20, "h": 20} if in_range else None,
                "conf": 0.9 if in_range else 0.0,
                "status": "DETECTED" if in_range else "LOST",
            })
            t += interval
            idx += 1
        return data

    def test_single_rally_motion_only(self):
        timeline = self._make_timeline([(5.0, 15.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0,
                                  motion_threshold=0.08)
        assert len(segments) == 1
        assert segments[0].start >= 3.0
        assert segments[0].end <= 17.0

    def test_two_rallies_with_gap(self):
        timeline = self._make_timeline([(5.0, 15.0), (25.0, 35.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0,
                                  motion_threshold=0.08)
        assert len(segments) == 2

    def test_short_activity_filtered(self):
        # 0.5 秒活動 + 1.5s 緩衝*2 = 3.5s → 剛好超過 min_duration=3.0
        # 用更短的活動 (0.2s) 來確保被過濾
        timeline = self._make_timeline([(5.0, 5.2)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=5.0,
                                  motion_threshold=0.08)
        assert len(segments) == 0

    def test_empty_timeline(self):
        assert detect_rallies([]) == []

    def test_no_activity(self):
        timeline = [{"time": i * 0.1, "score": 0.001} for i in range(100)]
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0,
                                  motion_threshold=0.08)
        assert len(segments) == 0

    def test_with_hit_times_boosts_detection(self):
        """低動態但有擊球聲 → 應該被偵測到"""
        timeline = [{"time": i * 0.1, "score": 0.05} for i in range(200)]
        hit_times = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
        segments = detect_rallies(
            timeline,
            hit_times=hit_times,
            gap_threshold=4.0,
            min_duration=3.0,
            motion_threshold=0.08,
        )
        assert len(segments) >= 1

    def test_pure_visual_mode_when_no_hits(self):
        timeline = self._make_timeline([(5.0, 15.0)])
        segments = detect_rallies(timeline, hit_times=None, gap_threshold=4.0,
                                  min_duration=3.0, motion_threshold=0.08)
        assert len(segments) == 1

    def test_tracking_data_detection(self):
        """YOLO 追蹤資料應能用於偵測 rally"""
        tracking = self._make_tracking_data([(5.0, 15.0)])
        timeline = [{"time": i * 0.1, "score": 0.001} for i in range(200)]
        segments = detect_rallies(
            timeline,
            tracking_data=tracking,
            gap_threshold=4.0,
            min_duration=3.0,
            motion_threshold=0.08,
        )
        assert len(segments) == 1

    def test_three_signals_combined(self):
        """三重訊號融合：motion + audio + tracking 都有部分覆蓋"""
        timeline = self._make_timeline([(5.0, 8.0)])
        hit_times = [9.0, 9.5, 10.0]
        tracking = self._make_tracking_data([(10.5, 14.0)], total=20)
        segments = detect_rallies(
            timeline,
            hit_times=hit_times,
            tracking_data=tracking,
            gap_threshold=4.0,
            min_duration=3.0,
            motion_threshold=0.08,
        )
        # All three signals are within 4s gap, should merge into 1 segment
        assert len(segments) == 1

    def test_nearby_segments_merged(self):
        """相近的段落應被合併"""
        timeline = self._make_timeline([(5.0, 10.0), (12.0, 17.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0,
                                  motion_threshold=0.08)
        # gap of 2.0s < gap_threshold/2=2.0, should merge
        assert len(segments) == 1
