"""視覺動態分析模組的單元測試"""

import numpy as np
import pytest

from src.motion_detector import compute_motion_score, smooth_timeline


class TestComputeMotionScore:
    def _make_frame(self, w, h, color=(0, 0, 0)):
        """建立指定顏色的 BGR 幀。"""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_identical_frames_zero_motion(self):
        frame = self._make_frame(200, 200, (128, 128, 128))
        roi = {"x": 0, "y": 0, "w": 200, "h": 200}
        score = compute_motion_score(frame, frame.copy(), roi)
        assert score == 0.0

    def test_completely_different_frames_high_motion(self):
        frame1 = self._make_frame(200, 200, (0, 0, 0))
        frame2 = self._make_frame(200, 200, (255, 255, 255))
        roi = {"x": 0, "y": 0, "w": 200, "h": 200}
        score = compute_motion_score(frame1, frame2, roi)
        assert score > 0.5

    def test_partial_motion(self):
        frame1 = self._make_frame(200, 200, (100, 100, 100))
        frame2 = frame1.copy()
        # 只改右半邊
        frame2[:, 100:] = (200, 200, 200)
        roi = {"x": 0, "y": 0, "w": 200, "h": 200}
        score = compute_motion_score(frame1, frame2, roi)
        assert 0.1 < score < 0.9

    def test_roi_crops_correctly(self):
        frame1 = self._make_frame(400, 400, (100, 100, 100))
        frame2 = frame1.copy()
        # 只在左上角 100x100 區域有變化
        frame2[0:100, 0:100] = (255, 255, 255)

        # ROI 在變化區域 → 高動態
        roi_hit = {"x": 0, "y": 0, "w": 100, "h": 100}
        score_hit = compute_motion_score(frame1, frame2, roi_hit)

        # ROI 在沒變化的區域 → 低動態
        roi_miss = {"x": 200, "y": 200, "w": 100, "h": 100}
        score_miss = compute_motion_score(frame1, frame2, roi_miss)

        assert score_hit > score_miss
        assert score_miss == 0.0


class TestSmoothTimeline:
    def test_smoothing_reduces_spikes(self):
        # 建立一個有突然峰值的時序
        timeline = [
            {"time": i * 0.1, "score": 0.01}
            for i in range(20)
        ]
        timeline[10]["score"] = 1.0  # 插入一個峰值

        smoothed = smooth_timeline(timeline, window_size=5)

        # 峰值應該被平滑化
        assert smoothed[10]["score"] < 1.0
        # 峰值的能量應該分散到鄰近點
        assert smoothed[11]["score"] > 0.01

    def test_empty_timeline(self):
        result = smooth_timeline([], window_size=5)
        assert result == []

    def test_short_timeline_unchanged(self):
        timeline = [{"time": 0.0, "score": 0.5}]
        result = smooth_timeline(timeline, window_size=5)
        assert len(result) == 1
        assert result[0]["score"] == 0.5

    def test_preserves_time_values(self):
        timeline = [
            {"time": float(i), "score": float(i) / 10}
            for i in range(10)
        ]
        smoothed = smooth_timeline(timeline, window_size=3)
        for i, point in enumerate(smoothed):
            assert point["time"] == float(i)
