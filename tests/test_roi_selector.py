"""ROI 選取模組的單元測試"""

import json
import os
import tempfile

import pytest

from src.roi_selector import load_roi, validate_roi


class TestLoadRoi:
    def test_load_valid_roi(self, tmp_path):
        roi_file = tmp_path / "roi.json"
        roi_data = {"x": 100, "y": 50, "w": 800, "h": 600}
        roi_file.write_text(json.dumps(roi_data))

        result = load_roi(str(roi_file))
        assert result == roi_data

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_roi("/nonexistent/path/roi.json")

    def test_load_invalid_format(self, tmp_path):
        roi_file = tmp_path / "roi.json"
        roi_file.write_text(json.dumps({"x": 100, "y": 50}))  # missing w, h

        with pytest.raises(ValueError, match="格式錯誤"):
            load_roi(str(roi_file))

    def test_load_extra_fields_ok(self, tmp_path):
        roi_file = tmp_path / "roi.json"
        roi_data = {"x": 100, "y": 50, "w": 800, "h": 600, "label": "court1"}
        roi_file.write_text(json.dumps(roi_data))

        result = load_roi(str(roi_file))
        assert result["x"] == 100
        assert result["w"] == 800


class TestValidateRoi:
    def test_valid_roi(self):
        roi = {"x": 100, "y": 50, "w": 800, "h": 600}
        result = validate_roi(roi, 1920, 1080)
        assert result == roi

    def test_roi_exceeds_width(self):
        roi = {"x": 1500, "y": 50, "w": 800, "h": 600}
        result = validate_roi(roi, 1920, 1080)
        assert result["x"] + result["w"] <= 1920

    def test_roi_exceeds_height(self):
        roi = {"x": 100, "y": 800, "w": 400, "h": 600}
        result = validate_roi(roi, 1920, 1080)
        assert result["y"] + result["h"] <= 1080

    def test_roi_negative_x(self):
        roi = {"x": -10, "y": 50, "w": 800, "h": 600}
        result = validate_roi(roi, 1920, 1080)
        assert result["x"] >= 0

    def test_roi_completely_outside(self):
        roi = {"x": 2000, "y": 2000, "w": 100, "h": 100}
        with pytest.raises(ValueError):
            validate_roi(roi, 1920, 1080)
