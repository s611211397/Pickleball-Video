"""FFmpeg 路徑解析 — 自動尋找系統或 Python 內建的 ffmpeg"""

import shutil
import subprocess


def get_ffmpeg_path() -> str:
    """取得可用的 ffmpeg 路徑。

    優先順序：
    1. 系統 PATH 中的 ffmpeg
    2. imageio-ffmpeg 內建的 ffmpeg
    """
    # 嘗試系統 ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 嘗試 imageio-ffmpeg 內建的
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise FileNotFoundError(
        "找不到 FFmpeg！請擇一安裝：\n"
        "  1. pip install imageio-ffmpeg（推薦，最簡單）\n"
        "  2. 從 https://ffmpeg.org/download.html 下載並加入 PATH"
    )


def get_ffprobe_path() -> str:
    """取得可用的 ffprobe 路徑。

    優先順序：
    1. 系統 PATH 中的 ffprobe
    2. 從 ffmpeg 路徑推導（同目錄下的 ffprobe）
    """
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe

    # imageio-ffmpeg 只提供 ffmpeg，不提供 ffprobe
    # 但 ffprobe 通常跟 ffmpeg 在同一個目錄
    try:
        from pathlib import Path
        ffmpeg = get_ffmpeg_path()
        ffprobe_candidate = str(Path(ffmpeg).parent / "ffprobe")
        if shutil.which(ffprobe_candidate):
            return ffprobe_candidate
        # Windows 上可能有 .exe
        ffprobe_candidate = str(Path(ffmpeg).parent / "ffprobe.exe")
        if Path(ffprobe_candidate).exists():
            return ffprobe_candidate
    except FileNotFoundError:
        pass

    # ffprobe 找不到，回傳 None 讓呼叫端用替代方案
    return None


def run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """執行 ffmpeg 指令，自動解析路徑。"""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg] + args
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def run_ffprobe(args: list[str], **kwargs) -> subprocess.CompletedProcess | None:
    """執行 ffprobe 指令，自動解析路徑。找不到 ffprobe 時回傳 None。"""
    ffprobe = get_ffprobe_path()
    if ffprobe is None:
        return None
    cmd = [ffprobe] + args
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)
