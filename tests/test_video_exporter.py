"""影片切片輸出模組的單元測試"""

import pytest

from src.video_exporter import _merge_overlapping_ranges


class TestMergeOverlappingRanges:
    def test_no_overlap(self):
        ranges = [(0, 5), (10, 15), (20, 25)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 5), (10, 15), (20, 25)]

    def test_full_overlap(self):
        ranges = [(0, 10), (3, 7)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 10)]

    def test_partial_overlap(self):
        ranges = [(0, 7), (5, 12)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 12)]

    def test_adjacent_ranges(self):
        ranges = [(0, 5), (5, 10)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 10)]

    def test_chain_overlap(self):
        ranges = [(0, 5), (3, 8), (7, 12)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 12)]

    def test_empty(self):
        assert _merge_overlapping_ranges([]) == []

    def test_single_range(self):
        assert _merge_overlapping_ranges([(0, 5)]) == [(0, 5)]

    def test_unsorted_input(self):
        ranges = [(10, 15), (0, 5), (3, 8)]
        result = _merge_overlapping_ranges(ranges)
        assert result == [(0, 8), (10, 15)]
