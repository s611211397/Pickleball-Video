"""球場自動偵測模組 — 從影片畫面中偵測匹克球場地"""

import cv2
import numpy as np


def detect_courts(frame: np.ndarray, min_court_ratio: float = 0.03) -> list[dict]:
    """偵測畫面中的球場區域。

    利用球場線條（白色/亮色線條在有色地面上）來偵測矩形場地。

    Args:
        frame: BGR 影片幀
        min_court_ratio: 球場最小面積佔畫面比例（過濾太小的雜訊）

    Returns:
        偵測到的球場 ROI 列表 [{"x", "y", "w", "h"}, ...]
        按面積由大到小排序
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    min_area = frame_area * min_court_ratio

    # 轉換色彩空間
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 策略一：偵測白色線條（球場邊線）
    line_mask = _detect_court_lines(hsv, gray)

    # 策略二：偵測有色球場表面（藍色/綠色）
    surface_mask = _detect_court_surface(hsv)

    # 合併兩個 mask
    combined = cv2.bitwise_or(line_mask, surface_mask)

    # 形態學操作：填補缺口、去除雜訊
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # 找輪廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    courts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # 過濾不合理的形狀
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.3 or aspect > 4.0:
            continue

        # 球場至少要有一定比例是實際內容（非空白）
        roi_mask = combined[y:y+ch, x:x+cw]
        fill_ratio = np.count_nonzero(roi_mask) / (cw * ch) if cw * ch > 0 else 0
        if fill_ratio < 0.15:
            continue

        courts.append({"x": x, "y": y, "w": cw, "h": ch})

    # 按面積由大到小排序
    courts.sort(key=lambda c: c["w"] * c["h"], reverse=True)

    # 合併重疊的偵測結果
    courts = _merge_overlapping_courts(courts)

    return courts


def _detect_court_lines(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """偵測白色球場線條。"""
    # 白色線條：高亮度、低飽和度
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 用 Canny 偵測邊緣，強化線條
    edges = cv2.Canny(gray, 50, 150)

    # 合併白色 mask 和邊緣
    combined = cv2.bitwise_or(white_mask, edges)

    # 膨脹讓線條更粗，方便後續填充
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=3)

    return combined


def _detect_court_surface(hsv: np.ndarray) -> np.ndarray:
    """偵測球場表面顏色（藍色/綠色）。"""
    # 藍色球場
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 綠色球場
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 紅色/橘色球場（某些場地）
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )

    return cv2.bitwise_or(cv2.bitwise_or(blue_mask, green_mask), red_mask)


def _merge_overlapping_courts(courts: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """合併 IoU 過高的重疊偵測結果。"""
    if len(courts) <= 1:
        return courts

    merged = []
    used = [False] * len(courts)

    for i, c1 in enumerate(courts):
        if used[i]:
            continue
        current = c1.copy()
        used[i] = True

        for j in range(i + 1, len(courts)):
            if used[j]:
                continue
            c2 = courts[j]
            if _iou(current, c2) > iou_threshold:
                # 合併：取包含兩者的最小矩形
                x1 = min(current["x"], c2["x"])
                y1 = min(current["y"], c2["y"])
                x2 = max(current["x"] + current["w"], c2["x"] + c2["w"])
                y2 = max(current["y"] + current["h"], c2["y"] + c2["h"])
                current = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                used[j] = True

        merged.append(current)

    return merged


def _iou(a: dict, b: dict) -> float:
    """計算兩個矩形的 IoU (Intersection over Union)。"""
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def draw_courts_on_frame(
    frame: np.ndarray,
    courts: list[dict],
    selected_idx: int | None = None,
) -> np.ndarray:
    """在畫面上標示偵測到的球場區域。"""
    vis = frame.copy()

    for i, court in enumerate(courts):
        x, y, w, h = court["x"], court["y"], court["w"], court["h"]
        is_selected = (i == selected_idx)

        color = (0, 255, 0) if is_selected else (0, 200, 255)
        thickness = 4 if is_selected else 2

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        # 標號
        label = f"Court {i + 1}"
        font_scale = 1.2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(vis, (x, y - th - 16), (x + tw + 10, y), color, -1)
        cv2.putText(vis, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    return vis
