"""Pickleball Auto-Editor — Streamlit Web UI"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from src.audio_analyzer import detect_hits, extract_audio
from src.court_detector import detect_courts, draw_courts_on_frame
from src.motion_detector import analyze_video_motion
from src.rally_detector import Segment, detect_rallies
from src.video_exporter import export_segments, merge_segments

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pickleball Auto-Editor",
    page_icon="🏓",
    layout="wide",
)

# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def get_first_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def draw_timeline_chart(motion_timeline, segments, hit_times, motion_threshold):
    from src.visualizer import plot_timeline_cv2
    img_bgr = plot_timeline_cv2(
        motion_timeline, segments, hit_times,
        motion_threshold=motion_threshold,
        output_path=None, show=False,
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def box_to_roi(box, img_w: int, img_h: int) -> dict | None:
    if isinstance(box, dict):
        x, y = int(box["left"]), int(box["top"])
        w, h = int(box["width"]), int(box["height"])
    else:
        left, upper, right, lower = box
        x, y = int(left), int(upper)
        w, h = int(right - left), int(lower - upper)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w < 10 or h < 10:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


# ─────────────────────────────────────────────
# 主介面
# ─────────────────────────────────────────────

st.title("🏓 匹克球自動剪輯工具")
st.caption("上傳影片 → 自動偵測球場 → 分析對戰片段 → 下載剪輯結果")

# ─── 側邊欄 ───
with st.sidebar:
    st.header("📹 影片上傳")
    uploaded = st.file_uploader(
        "拖拉影片到這裡",
        type=["mp4", "mov", "avi", "mkv"],
        help="支援 MP4、MOV、AVI、MKV 格式",
    )

    st.divider()
    st.header("⚙️ 參數設定")

    st.subheader("🔍 偵測靈敏度")
    motion_threshold = st.slider(
        "動態門檻", 0.01, 0.30, 0.08, 0.01,
        help="越低越靈敏，越高越嚴格",
    )
    gap_threshold = st.slider(
        "回合間隔（秒）", 1.0, 10.0, 4.0, 0.5,
        help="超過這個秒數沒動作 = 回合結束",
    )
    min_duration = st.slider(
        "最短回合長度（秒）", 1.0, 10.0, 3.0, 0.5,
        help="低於此秒數的活動會被忽略",
    )

    st.divider()
    st.subheader("🔊 音訊分析")
    use_audio = st.toggle("啟用擊球聲偵測", value=True)
    audio_weight = st.slider("音訊權重", 0.0, 1.0, 0.3, 0.05) if use_audio else 0.0

    st.divider()
    st.subheader("✂️ 輸出設定")
    buffer_sec = st.slider("前後保留秒數", 0.0, 5.0, 2.0, 0.5)
    output_mode = st.radio("輸出模式", ["合併成一個影片", "分開每段回合"])
    reencode = st.toggle("精確剪輯模式", value=False,
                         help="切點更精準，但較慢")

# ─── 主區域 ───

if uploaded is None:
    st.info("👈 請先在左側上傳匹克球影片")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ 上傳影片")
        st.markdown("從左側面板拖拉影片檔")
    with col2:
        st.markdown("### 2️⃣ 自動偵測球場")
        st.markdown("系統會自動辨識場地，多場地時讓你選")
    with col3:
        st.markdown("### 3️⃣ 一鍵剪輯")
        st.markdown("按下按鈕，自動產出精華影片")
    st.stop()

# ─── 影片已上傳 → 立即進入編輯 ───

# 儲存到暫存檔
if "video_path" not in st.session_state or st.session_state.get("_uploaded_name") != uploaded.name:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    tmp.write(uploaded.read())
    tmp.close()
    st.session_state["video_path"] = tmp.name
    st.session_state["_uploaded_name"] = uploaded.name
    # 清除所有舊狀態
    for k in ["motion_timeline", "hit_times", "segments", "output_files",
              "merged_path", "courts", "selected_court"]:
        st.session_state.pop(k, None)

video_path = st.session_state["video_path"]
video_info = get_video_info(video_path)
first_frame = get_first_frame(video_path)
fw, fh = video_info["width"], video_info["height"]

st.success(
    f"✅ **{uploaded.name}**　｜　"
    f"{fw}x{fh}　｜　"
    f"{video_info['fps']:.0f} FPS　｜　"
    f"{fmt_time(video_info['duration'])}"
)

if first_frame is None:
    st.error("無法讀取影片第一幀")
    st.stop()

# ─────────────────────────────────────────────
# Step 1: 球場偵測 + ROI 選取
# ─────────────────────────────────────────────

# 自動偵測球場（只跑一次，結果存 session_state）
if "courts" not in st.session_state:
    with st.spinner("🔍 自動偵測球場中..."):
        st.session_state["courts"] = detect_courts(first_frame)

courts = st.session_state["courts"]
roi = None

if len(courts) >= 2:
    # 偵測到多個場地 → 讓使用者選
    st.header("📐 偵測到多個球場，請選擇你的場地")
    st.caption(f"偵測到 {len(courts)} 個球場區域，請點選你要分析的場地。")

    # 顯示標示了球場的畫面
    annotated = draw_courts_on_frame(first_frame, courts, selected_idx=st.session_state.get("selected_court"))
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True)

    # 選擇按鈕
    cols = st.columns(len(courts))
    for i, court in enumerate(courts):
        with cols[i]:
            label = f"Court {i+1}（{court['w']}x{court['h']}）"
            if st.button(label, key=f"court_{i}", use_container_width=True,
                         type="primary" if st.session_state.get("selected_court") == i else "secondary"):
                st.session_state["selected_court"] = i
                # 清除舊分析結果
                for k in ["motion_timeline", "hit_times", "segments", "output_files", "merged_path"]:
                    st.session_state.pop(k, None)
                st.rerun()

    if "selected_court" in st.session_state:
        idx = st.session_state["selected_court"]
        roi = courts[idx]
        st.info(f"✅ 已選擇 **Court {idx+1}**：X={roi['x']}, Y={roi['y']}, 寬={roi['w']}, 高={roi['h']}")
    else:
        st.warning("👆 請選擇一個球場")
        st.stop()

    # 提供手動微調選項
    with st.expander("🔧 手動微調範圍"):
        st.caption("如果自動偵測不夠精確，可以在這裡拖拉調整。")
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        bg_image = Image.fromarray(frame_rgb)
        cropped = st_cropper(
            bg_image, realtime_update=True,
            box_color="#00FF00", aspect_ratio=None,
            return_type="box", key="roi_cropper_adjust",
        )
        manual_roi = box_to_roi(cropped, fw, fh)
        if manual_roi and st.button("使用手動選取的範圍"):
            roi = manual_roi
            st.session_state["selected_court"] = -1  # 標記為手動
            for k in ["motion_timeline", "hit_times", "segments", "output_files", "merged_path"]:
                st.session_state.pop(k, None)
            st.rerun()

elif len(courts) == 1:
    # 只偵測到一個場地 → 自動選取，顯示確認
    roi = courts[0]
    st.header("📐 已自動偵測到球場")

    annotated = draw_courts_on_frame(first_frame, courts, selected_idx=0)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True)
    st.info(f"✅ 自動選取球場範圍：X={roi['x']}, Y={roi['y']}, 寬={roi['w']}, 高={roi['h']}")

    # 提供手動調整選項
    with st.expander("🔧 手動調整範圍（如果自動偵測不準確）"):
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        bg_image = Image.fromarray(frame_rgb)
        cropped = st_cropper(
            bg_image, realtime_update=True,
            box_color="#00FF00", aspect_ratio=None,
            return_type="box", key="roi_cropper_single",
        )
        manual_roi = box_to_roi(cropped, fw, fh)
        if manual_roi and st.button("使用手動選取的範圍 "):
            roi = manual_roi
            st.rerun()

else:
    # 沒偵測到球場 → 手動框選
    st.header("📐 請手動框選球場範圍")
    st.caption("未自動偵測到球場，請拖動裁切框來選取你的球場。")

    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    bg_image = Image.fromarray(frame_rgb)
    cropped = st_cropper(
        bg_image, realtime_update=True,
        box_color="#00FF00", aspect_ratio=None,
        return_type="box", key="roi_cropper_manual",
    )
    roi = box_to_roi(cropped, fw, fh)

    if roi:
        st.info(f"✅ 已選取範圍：X={roi['x']}, Y={roi['y']}, 寬={roi['w']}, 高={roi['h']}")
    else:
        st.warning("請在上方畫面中調整裁切框來選取球場範圍")
        st.stop()

if roi is None:
    st.stop()

# ─────────────────────────────────────────────
# Step 2: 開始分析
# ─────────────────────────────────────────────
st.header("🔍 Step 2：分析影片")

if st.button("🚀 開始分析", type="primary", use_container_width=True):
    for k in ["motion_timeline", "hit_times", "segments", "output_files", "merged_path"]:
        st.session_state.pop(k, None)

    progress = st.progress(0, text="準備中...")

    # 2a: 視覺動態分析
    progress.progress(5, text="🎬 分析影片動態中...")

    def on_motion_progress(pct, msg):
        progress.progress(5 + int(pct * 45), text=f"🎬 {msg}")

    motion_timeline = analyze_video_motion(
        video_path, roi,
        frame_skip=0, gaussian_kernel=11, smooth_window=5,
        progress_callback=on_motion_progress,
    )
    st.session_state["motion_timeline"] = motion_timeline
    progress.progress(50, text=f"🎬 動態分析完成，{len(motion_timeline)} 個時間點")

    # 2b: 音訊分析
    hit_times = None
    if use_audio:
        progress.progress(55, text="🔊 分析音訊擊球聲中...")
        try:
            audio_path = extract_audio(video_path)
            hit_times = detect_hits(
                audio_path, bandpass_low=1000, bandpass_high=4000,
                energy_threshold=0.5, min_hit_interval=0.3,
            )
            Path(audio_path).unlink(missing_ok=True)
            progress.progress(80, text=f"🔊 偵測到 {len(hit_times)} 次擊球聲")
        except Exception as e:
            st.warning(f"音訊分析失敗（{e}），改用純視覺模式")
            hit_times = None
    st.session_state["hit_times"] = hit_times

    # 2c: Rally 偵測
    progress.progress(85, text="🏓 偵測回合中...")
    motion_w = 1.0 - audio_weight
    segments = detect_rallies(
        motion_timeline, hit_times=hit_times,
        gap_threshold=gap_threshold, min_duration=min_duration,
        activity_threshold=0.3, motion_weight=motion_w,
        audio_weight=audio_weight, motion_threshold=motion_threshold,
    )
    st.session_state["segments"] = segments
    progress.progress(100, text="✅ 分析完成！")

# ─────────────────────────────────────────────
# Step 3: 結果展示
# ─────────────────────────────────────────────
if "segments" in st.session_state:
    segments: list[Segment] = st.session_state["segments"]
    motion_timeline = st.session_state["motion_timeline"]
    hit_times = st.session_state.get("hit_times")

    st.header("📊 Step 3：偵測結果")

    if not segments:
        st.warning(
            "未偵測到任何回合。\n\n"
            "**建議：**\n"
            "- 降低左側「動態門檻」\n"
            "- 確認球場範圍有框對\n"
            "- 縮短「最短回合長度」"
        )
    else:
        total_rally = sum(seg.duration for seg in segments)
        video_total = motion_timeline[-1]["time"] if motion_timeline else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("偵測到的回合", f"{len(segments)} 段")
        c2.metric("回合總時間", fmt_time(total_rally))
        c3.metric("原始影片長度", fmt_time(video_total))
        c4.metric("保留比例", f"{total_rally / video_total * 100:.0f}%" if video_total > 0 else "N/A")

        st.subheader("動態時序圖")
        chart_img = draw_timeline_chart(motion_timeline, segments, hit_times, motion_threshold)
        st.image(chart_img, use_container_width=True)
        st.caption("🟢 綠色 = 回合　｜　🔵 藍線 = 動態分數　｜　🔴 紅虛線 = 門檻　｜　🟠 橘線 = 擊球聲")

        st.subheader("回合列表")
        table_data = [
            {"回合": f"# {i+1}", "開始": fmt_time(seg.start),
             "結束": fmt_time(seg.end), "長度": f"{seg.duration:.1f}s"}
            for i, seg in enumerate(segments)
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # ─── Step 4: 匯出 ───
        st.header("💾 Step 4：匯出剪輯影片")

        if st.button("✂️ 開始剪輯並匯出", type="primary", use_container_width=True):
            output_dir = tempfile.mkdtemp(prefix="pickleball_output_")
            progress2 = st.progress(0, text="切割影片中...")

            segment_files = export_segments(
                video_path, segments, output_dir,
                buffer_before=buffer_sec, buffer_after=buffer_sec,
                reencode=reencode,
            )
            progress2.progress(70, text="合併中..." if output_mode == "合併成一個影片" else "完成切割！")

            final_path = None
            if output_mode == "合併成一個影片":
                merged_path = str(Path(output_dir) / "merged_rallies.mp4")
                merge_segments(segment_files, merged_path, reencode=reencode)
                final_path = merged_path
                progress2.progress(100, text="✅ 匯出完成！")
            else:
                progress2.progress(100, text=f"✅ 匯出完成！共 {len(segment_files)} 個檔案")

            st.session_state["output_files"] = segment_files
            st.session_state["merged_path"] = final_path

        if "output_files" in st.session_state:
            st.success("剪輯完成！點擊下方按鈕下載。")
            if st.session_state.get("merged_path"):
                with open(st.session_state["merged_path"], "rb") as f:
                    st.download_button(
                        "⬇️ 下載合併影片", data=f,
                        file_name="pickleball_highlights.mp4",
                        mime="video/mp4", use_container_width=True,
                    )
            else:
                for i, fpath in enumerate(st.session_state["output_files"]):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            f"⬇️ 下載回合 {i+1}", data=f,
                            file_name=f"rally_{i+1:03d}.mp4",
                            mime="video/mp4", key=f"dl_{i}",
                        )

# ─── Footer ───
st.divider()
st.caption("Pickleball Auto-Editor  ·  視覺動態偵測 + 音訊擊球聲偵測  ·  固定機位最佳")
