"""影片切片輸出模組 — 根據 rally 區段切割並匯出影片"""

import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from .ffmpeg_utils import run_ffmpeg, run_ffprobe
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
    """將 rally 區段切片輸出為獨立影片檔。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    duration = _get_video_duration(video_path)

    cut_ranges = []
    for seg in segments:
        start = max(0, seg.start - buffer_before)
        end = min(duration, seg.end + buffer_after)
        cut_ranges.append((start, end))

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
    """將多個片段合併成單一影片。"""
    if not segment_files:
        raise ValueError("沒有片段可合併")

    if len(segment_files) == 1:
        shutil.copy2(segment_files[0], output_path)
        return output_path

    list_path = Path(output_path).parent / "_concat_list.txt"
    with open(list_path, "w") as f:
        for filepath in segment_files:
            escaped = str(Path(filepath).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    if reencode:
        args = [
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path,
        ]
    else:
        args = [
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-y", output_path,
        ]

    result = run_ffmpeg(args)
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
    """切割影片片段。"""
    if reencode:
        args = [
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
        # 注意：-ss 在 -i 前時，用 -t（持續時間）而非 -to（絕對時間）
        args = [
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{end - start:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y", output_path,
        ]

    result = run_ffmpeg(args)
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
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _get_video_duration(video_path: str) -> float:
    """取得影片總時長（秒）。

    優先用 ffprobe，找不到時用 OpenCV 替代。
    """
    result = run_ffprobe([
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ])

    if result is not None and result.returncode == 0:
        return float(result.stdout.strip())

    # ffprobe 不可用，用 OpenCV 替代
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    raise RuntimeError(f"無法取得影片時長: {video_path}")
