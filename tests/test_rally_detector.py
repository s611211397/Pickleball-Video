"""Rally 偵測引擎的單元測試"""

import pytest

from src.rally_detector import Segment, _has_nearby_hit, _merge_close_segments, detect_rallies


class TestSegment:
    def test_duration(self):
        seg = Segment(start=10.0, end=25.0)
        assert seg.duration == 15.0

    def test_zero_duration(self):
        seg = Segment(start=5.0, end=5.0)
        assert seg.duration == 0.0


class TestHasNearbyHit:
    def test_hit_within_window(self):
        hits = [1.0, 5.0, 10.0]
        assert _has_nearby_hit(5.2, hits, window=0.5) is True

    def test_no_hit_nearby(self):
        hits = [1.0, 5.0, 10.0]
        assert _has_nearby_hit(7.0, hits, window=0.5) is False

    def test_empty_hits(self):
        assert _has_nearby_hit(5.0, [], window=0.5) is False

    def test_hit_at_boundary(self):
        hits = [5.0]
        assert _has_nearby_hit(5.5, hits, window=0.5) is True
        assert _has_nearby_hit(5.6, hits, window=0.5) is False

    def test_hit_before_target(self):
        hits = [4.6]
        assert _has_nearby_hit(5.0, hits, window=0.5) is True

    def test_hit_far_before(self):
        hits = [3.0]
        assert _has_nearby_hit(5.0, hits, window=0.5) is False


class TestMergeCloseSegments:
    def test_no_merge_needed(self):
        segs = [Segment(0, 5), Segment(15, 20)]
        result = _merge_close_segments(segs, min_gap=3.0)
        assert len(result) == 2

    def test_merge_close_segments(self):
        segs = [Segment(0, 5), Segment(6, 10)]
        result = _merge_close_segments(segs, min_gap=3.0)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_empty_list(self):
        assert _merge_close_segments([], min_gap=3.0) == []

    def test_single_segment(self):
        segs = [Segment(0, 5)]
        result = _merge_close_segments(segs, min_gap=3.0)
        assert len(result) == 1

    def test_chain_merge(self):
        segs = [Segment(0, 5), Segment(6, 10), Segment(11, 15)]
        result = _merge_close_segments(segs, min_gap=3.0)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 15


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

    def test_single_rally(self):
        timeline = self._make_timeline([(5.0, 15.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0)
        assert len(segments) == 1
        assert segments[0].start >= 4.0
        assert segments[0].end <= 16.0

    def test_two_rallies_with_gap(self):
        timeline = self._make_timeline([(5.0, 15.0), (25.0, 35.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0)
        assert len(segments) == 2

    def test_short_activity_filtered(self):
        # 只有 1 秒的活動，低於 min_duration
        timeline = self._make_timeline([(5.0, 6.0)])
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0)
        assert len(segments) == 0

    def test_empty_timeline(self):
        assert detect_rallies([]) == []

    def test_no_activity(self):
        timeline = [{"time": i * 0.1, "score": 0.001} for i in range(100)]
        segments = detect_rallies(timeline, gap_threshold=4.0, min_duration=3.0)
        assert len(segments) == 0

    def test_with_hit_times(self):
        # 低動態但有擊球聲 → 應該被偵測到
        timeline = [{"time": i * 0.1, "score": 0.05} for i in range(200)]
        hit_times = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
        segments = detect_rallies(
            timeline,
            hit_times=hit_times,
            gap_threshold=4.0,
            min_duration=3.0,
            motion_weight=0.3,
            audio_weight=0.7,
        )
        assert len(segments) >= 1

    def test_pure_visual_mode_when_no_hits(self):
        timeline = self._make_timeline([(5.0, 15.0)])
        segments = detect_rallies(timeline, hit_times=None, gap_threshold=4.0, min_duration=3.0)
        assert len(segments) == 1
