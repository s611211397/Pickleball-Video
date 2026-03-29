"""音訊分析模組 — 偵測匹克球擊球聲"""

import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .ffmpeg_utils import run_ffmpeg


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """從影片中抽取音軌為 WAV 檔。

    Args:
        video_path: 輸入影片路徑
        output_path: 輸出音軌路徑（None 則建立暫存檔）

    Returns:
        音軌檔案路徑
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    result = run_ffmpeg([
        "-i", video_path,
        "-vn",                  # 不要影像
        "-acodec", "pcm_s16le", # 16-bit WAV
        "-ar", "22050",         # 取樣率 22050 Hz（夠用且省空間）
        "-ac", "1",             # 單聲道
        "-y",                   # 覆蓋已存在檔案
        output_path,
    ])

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 抽取音軌失敗: {result.stderr}")

    return output_path


def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    low_freq: int = 1000,
    high_freq: int = 4000,
    order: int = 5,
) -> np.ndarray:
    """帶通濾波：只保留擊球聲頻率範圍。"""
    nyquist = sr / 2
    # 確保頻率在有效範圍內
    low = max(low_freq, 1) / nyquist
    high = min(high_freq, nyquist - 1) / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_hits(
    audio_path: str,
    bandpass_low: int = 1000,
    bandpass_high: int = 4000,
    energy_threshold: float = 0.5,
    min_hit_interval: float = 0.3,
) -> list[float]:
    """偵測擊球時間點。

    Args:
        audio_path: WAV 音軌路徑
        bandpass_low: 帶通濾波下限 (Hz)
        bandpass_high: 帶通濾波上限 (Hz)
        energy_threshold: 能量門檻（用於自適應門檻計算）
        min_hit_interval: 最小擊球間距 (秒)

    Returns:
        擊球時間點列表 [秒]
    """
    # 載入音軌
    y, sr = librosa.load(audio_path, sr=None)

    if len(y) == 0:
        return []

    # 帶通濾波
    y_filtered = bandpass_filter(y, sr, bandpass_low, bandpass_high)

    # 計算短時能量 — 使用向量化操作加速
    hop_length = 512
    frame_length = 1024
    energy = _compute_short_time_energy(y_filtered, frame_length, hop_length)

    if len(energy) == 0:
        return []

    # 正規化能量
    max_energy = energy.max()
    if max_energy > 0:
        energy = energy / max_energy

    # 自適應門檻：中位數 + 倍率 * 標準差
    # 這樣可以適應不同錄音環境的噪音水準
    median_e = np.median(energy)
    std_e = np.std(energy)
    adaptive_threshold = max(
        energy_threshold,
        median_e + energy_threshold * std_e * 3,
    )
    # 但不能超過 0.95，否則什麼都偵測不到
    adaptive_threshold = min(adaptive_threshold, 0.95)

    # 峰值偵測
    min_distance = max(1, int(min_hit_interval * sr / hop_length))
    peaks, _ = find_peaks(
        energy,
        height=adaptive_threshold,
        distance=min_distance,
        prominence=0.1,  # 峰值需有一定的突出度，過濾平坦的高能量段
    )

    # 轉換為時間點
    hit_times = [float(p * hop_length / sr) for p in peaks]

    return hit_times


def _compute_short_time_energy(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """向量化計算短時能量。比 list comprehension 快 10-50 倍。"""
    # 計算可以產生多少個完整的 frame
    n_frames = max(0, (len(signal) - frame_length) // hop_length + 1)
    if n_frames == 0:
        return np.array([])

    # 建立 strided view 避免複製記憶體
    shape = (n_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # 每個 frame 的能量 = sum of squares
    energy = np.sum(frames ** 2, axis=1)

    return energy
