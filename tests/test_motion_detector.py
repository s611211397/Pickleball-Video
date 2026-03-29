"""視覺動態分析模組的單元測試"""

import numpy as np
import pytest

from src.motion_detector import (
    _auto_frame_skip,
    _extract_roi_gray,
    compute_motion_score,
    smooth_timeline,
)


class TestComputeMotionScore:
    def _make_gray(self, w, h, value=0):
        """建立指定灰度值的灰階幀。"""
        return np.full((h, w), value, dtype=np.uint8)

    def test_identical_frames_zero_motion(self):
        gray = self._make_gray(200, 200, 128)
        score = compute_motion_score(gray, gray.copy())
        assert score == 0.0

    def test_completely_different_frames_high_motion(self):
        gray1 = self._make_gray(200, 200, 0)
        gray2 = self._make_gray(200, 200, 255)
        score = compute_motion_score(gray1, gray2)
        assert score > 0.5

    def test_partial_motion(self):
        gray1 = self._make_gray(200, 200, 100)
        gray2 = gray1.copy()
        gray2[:, 100:] = 200
        score = compute_motion_score(gray1, gray2)
        assert 0.1 < score < 0.9


class TestExtractRoiGray:
    def test_basic_extract(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        frame[50:150, 100:300] = (255, 255, 255)  # 白色區域
        roi = {"x": 100, "y": 50, "w": 200, "h": 100}
        gray = _extract_roi_gray(frame, roi, scale=1.0)
        assert gray.shape == (100, 200)
        assert gray.mean() == 255

    def test_extract_with_downscale(self):
        frame = np.zeros((800, 800, 3), dtype=np.uint8)
        roi = {"x": 0, "y": 0, "w": 800, "h": 800}
        gray = _extract_roi_gray(frame, roi, scale=0.5)
        assert gray.shape == (400, 400)

    def test_roi_isolates_region(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        frame[0:100, 0:100] = (255, 255, 255)

        roi_hit = {"x": 0, "y": 0, "w": 100, "h": 100}
        roi_miss = {"x": 200, "y": 200, "w": 100, "h": 100}

        gray_hit = _extract_roi_gray(frame, roi_hit, 1.0)
        gray_miss = _extract_roi_gray(frame, roi_miss, 1.0)

        assert gray_hit.mean() == 255
        assert gray_miss.mean() == 0


class TestAutoFrameSkip:
    def test_short_video(self):
        # 2 分鐘 @ 30fps
        skip = _auto_frame_skip(30.0, 30 * 120)
        assert skip == 3

    def test_medium_video(self):
        # 15 分鐘 @ 30fps
        skip = _auto_frame_skip(30.0, 30 * 900)
        assert skip == 5

    def test_long_video(self):
        # 60 分鐘 @ 30fps
        skip = _auto_frame_skip(30.0, 30 * 3600)
        assert skip == 8


class TestSmoothTimeline:
    def test_smoothing_reduces_spikes(self):
        timeline = [
            {"time": i * 0.1, "score": 0.01}
            for i in range(20)
        ]
        timeline[10]["score"] = 1.0

        smoothed = smooth_timeline(timeline, window_size=5)
        assert smoothed[10]["score"] < 1.0
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
