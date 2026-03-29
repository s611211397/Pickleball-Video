"""影片切片輸出模組 — 根據 rally 區段切割並匯出影片"""

import subprocess
from pathlib import Path

from tqdm import tqdm

from .rally_detector import Segment


def export_segments(
    video_path: str,
    segments: list[Segment],
    output_dir: str,
    buffer_before: float = 2.0,
    buffer_after: float = 2.0,
    reencode: bool = False,
    output_format: str = "mp4",
) -> list[str]:
    """將 rally 區段切片輸出為獨立影片檔。

    Args:
        video_path: 原始影片路徑
        segments: rally 區段列表
        output_dir: 輸出目錄
        buffer_before: rally 前 buffer (秒)
        buffer_after: rally 後 buffer (秒)
        reencode: 是否重新編碼（精確切點）
        output_format: 輸出格式

    Returns:
        輸出檔案路徑列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 取得影片總時長
    duration = _get_video_duration(video_path)

    output_files = []
    for i, seg in enumerate(tqdm(segments, desc="匯出片段", unit="segment")):
        start = max(0, seg.start - buffer_before)
        end = min(duration, seg.end + buffer_after)

        filename = f"rally_{i + 1:03d}.{output_format}"
        filepath = str(output_path / filename)

        _cut_segment(video_path, filepath, start, end, reencode)
        output_files.append(filepath)

    return output_files


def merge_segments(
    segment_files: list[str],
    output_path: str,
    reencode: bool = False,
) -> str:
    """將多個片段合併成單一影片。

    Args:
        segment_files: 片段檔案路徑列表
        output_path: 合併後的輸出路徑
        reencode: 是否重新編碼

    Returns:
        合併影片路徑
    """
    if not segment_files:
        raise ValueError("沒有片段可合併")

    # 建立 FFmpeg concat 列表
    list_path = Path(output_path).parent / "_concat_list.txt"
    with open(list_path, "w") as f:
        for filepath in segment_files:
            # FFmpeg concat 需要用單引號跳脫特殊字元
            escaped = str(Path(filepath).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    if reencode:
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-y", output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)  # 清理暫存檔

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 合併失敗: {result.stderr}")

    return output_path


def _cut_segment(
    video_path: str,
    output_path: str,
    start: float,
    end: float,
    reencode: bool = False,
) -> None:
    """切割影片片段。"""
    if reencode:
        cmd = [
            "ffmpeg",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", video_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", video_path,
            "-c", "copy",
            "-y", output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 切割失敗 [{start:.1f}-{end:.1f}]: {result.stderr}")


def _get_video_duration(video_path: str) -> float:
    """取得影片總時長（秒）。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe 取得時長失敗: {result.stderr}")
    return float(result.stdout.strip())
