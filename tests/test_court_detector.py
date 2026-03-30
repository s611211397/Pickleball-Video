"""Tests for court_detector module."""

import numpy as np
import pytest

from src.court_detector import (
    _detect_court_lines,
    _detect_court_surface,
    _iou,
    _merge_overlapping_courts,
    detect_courts,
    draw_courts_on_frame,
)


def _make_blank_frame(h=480, w=640):
    """建立空白 BGR 影像。"""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_blue_court_frame(h=480, w=640):
    """建立含藍色球場的影像。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # 畫一個藍色矩形模擬球場 (BGR: 200, 100, 0 → 藍色)
    frame[100:400, 80:560] = (200, 100, 0)
    # 畫白色線條模擬球場邊線
    frame[100:105, 80:560] = (255, 255, 255)
    frame[395:400, 80:560] = (255, 255, 255)
    frame[100:400, 80:85] = (255, 255, 255)
    frame[100:400, 555:560] = (255, 255, 255)
    return frame


class TestIoU:
    def test_identical_rects(self):
        r = {"x": 0, "y": 0, "w": 100, "h": 100}
        assert _iou(r, r) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = {"x": 0, "y": 0, "w": 50, "h": 50}
        b = {"x": 200, "y": 200, "w": 50, "h": 50}
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"x": 0, "y": 0, "w": 100, "h": 100}
        b = {"x": 50, "y": 50, "w": 100, "h": 100}
        # intersection: 50*50=2500, union: 10000+10000-2500=17500
        assert _iou(a, b) == pytest.approx(2500 / 17500)

    def test_zero_area(self):
        a = {"x": 0, "y": 0, "w": 0, "h": 0}
        b = {"x": 0, "y": 0, "w": 100, "h": 100}
        assert _iou(a, b) == pytest.approx(0.0)


class TestMergeOverlapping:
    def test_empty(self):
        assert _merge_overlapping_courts([]) == []

    def test_single(self):
        courts = [{"x": 0, "y": 0, "w": 100, "h": 100}]
        assert _merge_overlapping_courts(courts) == courts

    def test_no_overlap(self):
        courts = [
            {"x": 0, "y": 0, "w": 50, "h": 50},
            {"x": 200, "y": 200, "w": 50, "h": 50},
        ]
        result = _merge_overlapping_courts(courts)
        assert len(result) == 2

    def test_overlapping_merged(self):
        courts = [
            {"x": 0, "y": 0, "w": 100, "h": 100},
            {"x": 10, "y": 10, "w": 100, "h": 100},
        ]
        result = _merge_overlapping_courts(courts, iou_threshold=0.1)
        assert len(result) == 1
        assert result[0]["x"] == 0
        assert result[0]["y"] == 0


class TestDetectCourts:
    def test_blank_frame_no_courts(self):
        frame = _make_blank_frame()
        courts = detect_courts(frame)
        assert courts == []

    def test_blue_court_detected(self):
        frame = _make_blue_court_frame()
        courts = detect_courts(frame, min_court_ratio=0.01)
        assert len(courts) >= 1
        # 偵測到的球場應大致在藍色區域
        court = courts[0]
        assert court["x"] < 120
        assert court["y"] < 140
        assert court["w"] > 200
        assert court["h"] > 100

    def test_returns_sorted_by_area(self):
        # 建立含兩塊不同大小色塊的影像
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame[50:150, 50:150] = (200, 100, 0)   # 小藍
        frame[200:500, 200:700] = (200, 100, 0)  # 大藍
        courts = detect_courts(frame, min_court_ratio=0.005)
        if len(courts) >= 2:
            assert courts[0]["w"] * courts[0]["h"] >= courts[1]["w"] * courts[1]["h"]


class TestDetectCourtLines:
    def test_white_lines_detected(self):
        import cv2
        frame = _make_blank_frame()
        frame[100, 50:200] = (255, 255, 255)  # 白色水平線
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _detect_court_lines(hsv, gray)
        assert mask.shape == (480, 640)
        assert np.any(mask > 0)


class TestDetectCourtSurface:
    def test_blue_surface(self):
        import cv2
        frame = _make_blank_frame()
        frame[100:300, 100:400] = (200, 100, 0)  # 藍色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _detect_court_surface(hsv)
        assert np.count_nonzero(mask[100:300, 100:400]) > 0

    def test_green_surface(self):
        import cv2
        frame = _make_blank_frame()
        frame[100:300, 100:400] = (0, 150, 0)  # 綠色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _detect_court_surface(hsv)
        assert np.count_nonzero(mask[100:300, 100:400]) > 0


class TestDrawCourtsOnFrame:
    def test_draw_does_not_modify_original(self):
        frame = _make_blank_frame()
        original = frame.copy()
        courts = [{"x": 10, "y": 10, "w": 100, "h": 100}]
        result = draw_courts_on_frame(frame, courts)
        np.testing.assert_array_equal(frame, original)
        assert result.shape == frame.shape

    def test_draw_with_selection(self):
        frame = _make_blank_frame()
        courts = [
            {"x": 10, "y": 20, "w": 100, "h": 100},
            {"x": 200, "y": 20, "w": 100, "h": 100},
        ]
        result = draw_courts_on_frame(frame, courts, selected_idx=0)
        assert result is not frame

    def test_draw_empty_courts(self):
        frame = _make_blank_frame()
        result = draw_courts_on_frame(frame, [])
        np.testing.assert_array_equal(result, frame)
