"""參數調校可視化工具 — 繪製動態量曲線和 rally 偵測結果，方便調參"""

import json
from pathlib import Path

import cv2
import numpy as np

from .rally_detector import Segment


def plot_timeline_cv2(
    motion_timeline: list[dict],
    segments: list[Segment],
    hit_times: list[float] | None = None,
    motion_threshold: float = 0.08,
    output_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """用 OpenCV 繪製動態量時序圖 + rally 區段標記。

    不依賴 matplotlib，減少一個重型依賴。

    Args:
        motion_timeline: 動態時序資料
        segments: 偵測到的 rally 區段
        hit_times: 擊球時間點（可為 None）
        motion_threshold: 動態門檻線
        output_path: 輸出圖片路徑（None 則不儲存）
        show: 是否顯示視窗

    Returns:
        繪製好的圖片 (numpy array)
    """
    if not motion_timeline:
        raise ValueError("動態時序資料為空")

    # 圖片尺寸
    img_w, img_h = 1600, 500
    margin_l, margin_r, margin_t, margin_b = 80, 40, 40, 60
    plot_w = img_w - margin_l - margin_r
    plot_h = img_h - margin_t - margin_b

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # 時間與分數範圍
    times = [p["time"] for p in motion_timeline]
    scores = [p["score"] for p in motion_timeline]
    t_min, t_max = times[0], times[-1]
    s_max = max(max(scores), motion_threshold * 2, 0.1)

    def to_px(t: float, s: float) -> tuple[int, int]:
        x = margin_l + int((t - t_min) / (t_max - t_min) * plot_w) if t_max > t_min else margin_l
        y = margin_t + plot_h - int(s / s_max * plot_h)
        y = max(margin_t, min(margin_t + plot_h, y))
        return x, y

    # 繪製 rally 區段背景（綠色半透明）
    overlay = img.copy()
    for seg in segments:
        x1 = to_px(seg.start, 0)[0]
        x2 = to_px(seg.end, 0)[0]
        cv2.rectangle(overlay, (x1, margin_t), (x2, margin_t + plot_h), (200, 255, 200), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # 繪製坐標軸
    cv2.line(img, (margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 1)
    cv2.line(img, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)

    # X 軸刻度（每 30 秒）
    tick_interval = max(30, int((t_max - t_min) / 20 / 30) * 30)
    t = 0.0
    while t <= t_max:
        if t >= t_min:
            px, _ = to_px(t, 0)
            cv2.line(img, (px, margin_t + plot_h), (px, margin_t + plot_h + 5), (0, 0, 0), 1)
            label = f"{int(t // 60)}:{int(t % 60):02d}"
            cv2.putText(img, label, (px - 15, margin_t + plot_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        t += tick_interval

    # 動態門檻線（紅色虛線）
    thresh_y = to_px(0, motion_threshold)[1]
    for x in range(margin_l, margin_l + plot_w, 10):
        cv2.line(img, (x, thresh_y), (min(x + 5, margin_l + plot_w), thresh_y), (0, 0, 255), 1)
    cv2.putText(img, f"threshold={motion_threshold}", (margin_l + plot_w - 150, thresh_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 繪製動態量曲線（藍色）
    for i in range(1, len(times)):
        pt1 = to_px(times[i - 1], scores[i - 1])
        pt2 = to_px(times[i], scores[i])
        cv2.line(img, pt1, pt2, (255, 100, 0), 1)

    # 繪製擊球聲時間點（橘色短豎線）
    if hit_times:
        for ht in hit_times:
            if t_min <= ht <= t_max:
                hx, _ = to_px(ht, 0)
                cv2.line(img, (hx, margin_t + plot_h - 15), (hx, margin_t + plot_h), (0, 140, 255), 2)

    # 標題和圖例
    cv2.putText(img, "Motion Score Timeline", (margin_l, margin_t - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "X: Time (min:sec)  Y: Motion Score", (margin_l, img_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # 圖例
    legend_x = img_w - 300
    legend_y = margin_t + 15
    cv2.rectangle(img, (legend_x - 5, legend_y - 12), (legend_x + 250, legend_y + 55), (240, 240, 240), -1)
    cv2.line(img, (legend_x, legend_y), (legend_x + 20, legend_y), (255, 100, 0), 2)
    cv2.putText(img, "Motion score", (legend_x + 25, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.rectangle(img, (legend_x, legend_y + 12), (legend_x + 20, legend_y + 22), (200, 255, 200), -1)
    cv2.putText(img, f"Rally segments ({len(segments)})", (legend_x + 25, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    if hit_times:
        cv2.line(img, (legend_x, legend_y + 30), (legend_x + 20, legend_y + 30), (0, 140, 255), 2)
        cv2.putText(img, f"Hit sounds ({len(hit_times)})", (legend_x + 25, legend_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.line(img, (legend_x, legend_y + 45), (legend_x + 20, legend_y + 45), (0, 0, 255), 1)
    cv2.putText(img, "Threshold", (legend_x + 25, legend_y + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)

    if show:
        cv2.imshow("Timeline - Press any key to close", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def export_segments_report(
    segments: list[Segment],
    output_path: str,
    video_duration: float | None = None,
) -> None:
    """匯出 rally 偵測報告為 JSON。

    Args:
        segments: rally 區段列表
        output_path: 輸出路徑
        video_duration: 影片總時長（用於計算覆蓋率）
    """
    report = {
        "total_rallies": len(segments),
        "total_rally_time": round(sum(seg.duration for seg in segments), 1),
        "segments": [
            {
                "index": i + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.duration, 2),
            }
            for i, seg in enumerate(segments)
        ],
    }

    if video_duration and video_duration > 0:
        report["video_duration"] = round(video_duration, 1)
        report["coverage_pct"] = round(
            report["total_rally_time"] / video_duration * 100, 1
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
