"""音訊分析模組的單元測試"""

import numpy as np
import pytest

from src.audio_analyzer import _compute_short_time_energy, bandpass_filter


class TestBandpassFilter:
    def test_passes_in_band_frequency(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 2000 Hz 訊號（在 1000-4000 Hz 範圍內）
        signal = np.sin(2 * np.pi * 2000 * t)
        filtered = bandpass_filter(signal, sr, 1000, 4000)

        # 過濾後應保留大部分能量
        original_energy = np.sum(signal ** 2)
        filtered_energy = np.sum(filtered ** 2)
        assert filtered_energy > 0.5 * original_energy

    def test_blocks_low_frequency(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 200 Hz 訊號（低於 1000 Hz，應被過濾）
        signal = np.sin(2 * np.pi * 200 * t)
        filtered = bandpass_filter(signal, sr, 1000, 4000)

        original_energy = np.sum(signal ** 2)
        filtered_energy = np.sum(filtered ** 2)
        assert filtered_energy < 0.1 * original_energy

    def test_blocks_high_frequency(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 8000 Hz 訊號（高於 4000 Hz，應被過濾）
        signal = np.sin(2 * np.pi * 8000 * t)
        filtered = bandpass_filter(signal, sr, 1000, 4000)

        original_energy = np.sum(signal ** 2)
        filtered_energy = np.sum(filtered ** 2)
        assert filtered_energy < 0.1 * original_energy


class TestComputeShortTimeEnergy:
    def test_basic_energy(self):
        signal = np.ones(4096, dtype=np.float64)
        energy = _compute_short_time_energy(signal, frame_length=1024, hop_length=512)
        assert len(energy) > 0
        # 每個 frame 的能量 = sum of 1^2 * 1024 = 1024
        np.testing.assert_allclose(energy, 1024.0)

    def test_silent_signal(self):
        signal = np.zeros(4096, dtype=np.float64)
        energy = _compute_short_time_energy(signal, frame_length=1024, hop_length=512)
        np.testing.assert_allclose(energy, 0.0)

    def test_empty_signal(self):
        signal = np.array([], dtype=np.float64)
        energy = _compute_short_time_energy(signal, frame_length=1024, hop_length=512)
        assert len(energy) == 0

    def test_short_signal(self):
        # 訊號比 frame_length 短
        signal = np.ones(100, dtype=np.float64)
        energy = _compute_short_time_energy(signal, frame_length=1024, hop_length=512)
        assert len(energy) == 0

    def test_energy_reflects_amplitude(self):
        signal_loud = np.ones(4096, dtype=np.float64) * 2.0
        signal_quiet = np.ones(4096, dtype=np.float64) * 0.5

        energy_loud = _compute_short_time_energy(signal_loud, 1024, 512)
        energy_quiet = _compute_short_time_energy(signal_quiet, 1024, 512)

        assert energy_loud[0] > energy_quiet[0]
        # 能量應為振幅平方的比例
        np.testing.assert_allclose(energy_loud[0] / energy_quiet[0], 16.0, rtol=0.01)
