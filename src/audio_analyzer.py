"""音訊分析模組 — 偵測匹克球擊球聲"""

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


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

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                  # 不要影像
        "-acodec", "pcm_s16le", # 16-bit WAV
        "-ar", "22050",         # 取樣率 22050 Hz（夠用且省空間）
        "-ac", "1",             # 單聲道
        "-y",                   # 覆蓋已存在檔案
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
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
    low = low_freq / nyquist
    high = high_freq / nyquist
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
        energy_threshold: 能量門檻（相對於最大值的比例）
        min_hit_interval: 最小擊球間距 (秒)

    Returns:
        擊球時間點列表 [秒]
    """
    # 載入音軌
    y, sr = librosa.load(audio_path, sr=None)

    # 帶通濾波
    y_filtered = bandpass_filter(y, sr, bandpass_low, bandpass_high)

    # 計算短時能量（窗口 ~23ms at 22050Hz）
    hop_length = 512
    frame_length = 1024
    energy = np.array([
        np.sum(y_filtered[i : i + frame_length] ** 2)
        for i in range(0, len(y_filtered) - frame_length, hop_length)
    ])

    # 正規化能量
    if energy.max() > 0:
        energy = energy / energy.max()

    # 自適應門檻：取中位數 + 倍率
    adaptive_threshold = max(
        energy_threshold,
        np.median(energy) + energy_threshold * np.std(energy),
    )

    # 峰值偵測
    min_distance = int(min_hit_interval * sr / hop_length)
    peaks, properties = find_peaks(
        energy,
        height=adaptive_threshold,
        distance=min_distance,
    )

    # 轉換為時間點
    hit_times = [float(p * hop_length / sr) for p in peaks]

    return hit_times
