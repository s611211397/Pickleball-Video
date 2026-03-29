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

    # 計算每段的實際切割範圍（加 buffer），並處理重疊
    cut_ranges = []
    for seg in segments:
        start = max(0, seg.start - buffer_before)
        end = min(duration, seg.end + buffer_after)
        cut_ranges.append((start, end))

    # 合併有重疊的切割範圍
    cut_ranges = _merge_overlapping_ranges(cut_ranges)

    output_files = []
    for i, (start, end) in enumerate(tqdm(cut_ranges, desc="匯出片段", unit="segment")):
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

    # 只有一個片段時直接複製
    if len(segment_files) == 1:
        import shutil
        shutil.copy2(segment_files[0], output_path)
        return output_path

    # 建立 FFmpeg concat 列表
    list_path = Path(output_path).parent / "_concat_list.txt"
    with open(list_path, "w") as f:
        for filepath in segment_files:
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
    list_path.unlink(missing_ok=True)

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
    """切割影片片段。

    無重新編碼模式：-ss 放在 -i 前面（快速 seek，切點可能差幾幀）
    重新編碼模式：-ss 放在 -i 後面（精確到幀，但較慢）
    """
    if reencode:
        # 精確模式：-ss 在 -i 之後，確保幀精確
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-y", output_path,
        ]
    else:
        # 快速模式：-ss 在 -i 之前，利用 keyframe seek
        cmd = [
            "ffmpeg",
            "-ss", f"{start:.3f}",
            "-to", f"{end - start:.3f}",  # -to 相對於 -ss 起點
            "-i", video_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y", output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 切割失敗 [{start:.1f}-{end:.1f}]: {result.stderr}")


def _merge_overlapping_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合併有重疊或相鄰的時間範圍。"""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # 有重疊，合併
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


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
