"""Pickleball Auto-Editor — 主程式入口"""

from pathlib import Path

import click
import yaml

from src.audio_analyzer import detect_hits, extract_audio
from src.motion_detector import analyze_video_motion
from src.rally_detector import detect_rallies
from src.roi_selector import load_roi, select_roi, validate_roi
from src.video_exporter import export_segments, merge_segments
from src.visualizer import export_segments_report, plot_timeline_cv2


def load_config(config_path: str = "config.yaml") -> dict:
    """載入設定檔。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"設定檔不存在: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("-o", "--output", default="output", help="輸出目錄")
@click.option("--config", "config_path", default="config.yaml", help="設定檔路徑")
@click.option("--set-roi", is_flag=True, help="設定球場 ROI（互動式框選）")
@click.option("--roi-config", default=None, help="指定 ROI 設定檔路徑")
@click.option("--no-audio", is_flag=True, help="停用音訊分析（純視覺模式）")
@click.option("--preview", is_flag=True, help="只顯示偵測結果，不輸出影片")
@click.option("--visualize", is_flag=True, help="顯示動態量時序圖（方便調參）")
@click.option("--report", default=None, help="匯出偵測報告 JSON 路徑")
def main(video_path, output, config_path, set_roi, roi_config, no_audio, preview, visualize, report):
    """Pickleball Auto-Editor — 自動剪輯匹克球對戰影片。

    VIDEO_PATH: 輸入影片檔案路徑

    \b
    範例用法：
      # 第一次使用，框選球場範圍
      python main.py --set-roi input.mp4

      # 預覽偵測結果（不輸出影片）
      python main.py input.mp4 --preview --visualize

      # 正式剪輯
      python main.py input.mp4 -o output/

      # 純視覺模式（不用音訊）
      python main.py input.mp4 -o output/ --no-audio
    """
    # 載入設定
    config = load_config(config_path)

    # === Step 1: ROI 設定 ===
    roi_file = roi_config or config["roi"]["config_file"]

    if set_roi:
        click.echo("[1/5] 請在彈出的視窗中框選你的球場範圍...")
        roi = select_roi(video_path, roi_file)
    else:
        try:
            roi = load_roi(roi_file)
        except FileNotFoundError:
            click.echo("[!] 找不到 ROI 設定檔，請先執行 --set-roi")
            click.echo(f"    python main.py --set-roi {video_path}")
            return

    click.echo(f"[1/5] ROI: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")

    # === Step 2: 視覺動態分析 ===
    motion_cfg = config["motion"]
    click.echo("[2/5] 分析影片動態...")
    motion_timeline = analyze_video_motion(
        video_path,
        roi,
        frame_skip=motion_cfg["frame_skip"],
        gaussian_kernel=motion_cfg["gaussian_kernel"],
        smooth_window=motion_cfg.get("smooth_window", 5),
    )
    click.echo(f"      分析了 {len(motion_timeline)} 個時間點")

    # === Step 3: 音訊分析（可選）===
    hit_times = None
    audio_cfg = config["audio"]

    if audio_cfg["enabled"] and not no_audio:
        click.echo("[3/5] 分析音訊擊球聲...")
        audio_path = extract_audio(video_path)
        hit_times = detect_hits(
            audio_path,
            bandpass_low=audio_cfg["bandpass_low"],
            bandpass_high=audio_cfg["bandpass_high"],
            energy_threshold=audio_cfg["energy_threshold"],
            min_hit_interval=audio_cfg["min_hit_interval"],
        )
        click.echo(f"      偵測到 {len(hit_times)} 次擊球聲")
        Path(audio_path).unlink(missing_ok=True)
    else:
        click.echo("[3/5] 音訊分析已停用，使用純視覺模式")

    # === Step 4: Rally 偵測 ===
    rally_cfg = config["rally"]
    click.echo("[4/5] 偵測 rally 區段...")
    segments = detect_rallies(
        motion_timeline,
        hit_times=hit_times,
        gap_threshold=rally_cfg["gap_threshold"],
        min_duration=rally_cfg["min_duration"],
        activity_threshold=rally_cfg["activity_threshold"],
        motion_weight=rally_cfg["motion_weight"],
        audio_weight=rally_cfg["audio_weight"],
        motion_threshold=motion_cfg["threshold"],
    )
    click.echo(f"      偵測到 {len(segments)} 段 rally")

    if not segments:
        click.echo("[!] 未偵測到任何 rally，請嘗試調整 config.yaml 中的參數")
        click.echo("    提示：降低 motion.threshold 或 rally.activity_threshold")
        if visualize:
            plot_timeline_cv2(
                motion_timeline, segments, hit_times,
                motion_threshold=motion_cfg["threshold"],
            )
        return

    # 顯示偵測結果
    total_rally_time = sum(seg.duration for seg in segments)
    video_total = motion_timeline[-1]["time"] if motion_timeline else 0

    click.echo(f"\n{'='*50}")
    click.echo(f"  偵測結果：{len(segments)} 段 rally")
    click.echo(f"{'='*50}")
    for i, seg in enumerate(segments):
        start_m, start_s = divmod(int(seg.start), 60)
        end_m, end_s = divmod(int(seg.end), 60)
        click.echo(
            f"  Rally {i + 1:3d}: {start_m}:{start_s:02d} - {end_m}:{end_s:02d}  ({seg.duration:.1f}s)"
        )
    click.echo(f"{'='*50}")
    click.echo(f"  Rally 總時間: {total_rally_time:.1f}s / {video_total:.1f}s")
    if video_total > 0:
        click.echo(f"  覆蓋率: {total_rally_time / video_total * 100:.1f}%")
    click.echo(f"{'='*50}\n")

    # 可視化
    if visualize:
        chart_path = str(Path(output) / "timeline.png")
        plot_timeline_cv2(
            motion_timeline, segments, hit_times,
            motion_threshold=motion_cfg["threshold"],
            output_path=chart_path,
        )
        click.echo(f"  時序圖已儲存至: {chart_path}")

    # 匯出報告
    if report:
        export_segments_report(segments, report, video_duration=video_total)
        click.echo(f"  偵測報告已儲存至: {report}")

    if preview:
        click.echo("\n（預覽模式，不輸出影片）")
        return

    # === Step 5: 輸出影片 ===
    output_cfg = config["output"]
    click.echo(f"[5/5] 切割並匯出影片至 {output}/...")

    segment_files = export_segments(
        video_path,
        segments,
        output,
        buffer_before=output_cfg["buffer_before"],
        buffer_after=output_cfg["buffer_after"],
        reencode=output_cfg["reencode"],
        output_format=output_cfg["format"],
    )

    if output_cfg["mode"] == "merged":
        merged_path = str(Path(output) / f"merged_rallies.{output_cfg['format']}")
        click.echo("      合併所有片段...")
        merge_segments(segment_files, merged_path, reencode=output_cfg["reencode"])
        click.echo(f"\n  完成！合併影片: {merged_path}")
    else:
        click.echo(f"\n  完成！{len(segment_files)} 個片段已輸出至 {output}/")


if __name__ == "__main__":
    main()
