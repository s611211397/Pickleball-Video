"""Pickleball Auto-Editor — Streamlit Web UI"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from src.audio_analyzer import detect_hits, extract_audio
from src.court_detector import detect_courts, draw_courts_on_frame, compute_roi_from_keypoints
from src.motion_detector import analyze_video_motion
from src.rally_detector import Segment, detect_rallies
from src.video_exporter import export_segments, merge_segments
from src.yolo_tracker import analyze_video_with_yolo
from src.dataset_manager import DatasetManager
from src.review_session import save_review_session, load_review_session, has_saved_session, delete_session

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pickleball Auto-Editor",
    page_icon="🏓",
    layout="wide",
)

# ─────────────────────────────────────────────
# 側邊欄：模型持續學習中心
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 YOLO 模型優化")
    st.caption("透過您「手動指正」的錯誤影像來提升 AI 的準確度！")
    
    # 算一下目前標註了多少張圖（DatasetManager 存到 dataset/labels/）
    label_path = Path("dataset/labels")
    num_labels = 0
    if label_path.exists():
        # 搜尋所有子目錄和根目錄的標註檔
        num_labels = len(list(label_path.rglob("*.txt")))
        
    st.metric("累積可訓練的「問題幀」數量", f"{num_labels} 張")
    
    if num_labels > 0:
        if st.button("🚀 開始 Fine-tune 訓練", use_container_width=True, type="primary"):
            import subprocess
            st.info(f"📊 使用 **{num_labels}** 張標註影像進行訓練")
            log_area = st.empty()
            status_area = st.empty()
            status_area.warning("⏳ 訓練進行中，請勿關閉或重新整理此視窗...")

            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", "train_model.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                log_lines = []
                with log_area.container():
                    log_display = st.empty()
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            line = line.rstrip()
                            log_lines.append(line)
                            # 只保留最近 30 行，避免 UI 太長
                            visible = log_lines[-30:]
                            log_display.code("\n".join(visible), language="bash")

                returncode = process.wait()
                if returncode == 0:
                    status_area.empty()
                    st.success("✅ 訓練完成！最新模型已部署。下次分析影片會自動套用。")
                    st.info("💡 模型已備份至 `models/pickleball_best.pt`，記得 Push 到 GitHub 永久保存！")
                    st.balloons()
                else:
                    status_area.empty()
                    st.error("❌ 訓練失敗，請查看上方 Log 輸出。")

            except Exception as e:
                status_area.empty()
                st.error(f"無法啟動訓練腳本: {e}")
    else:
        st.info("💡 目前還沒有問題幀資料。\n請多上傳比賽影片，並在「Step 2.5：軌跡迷失審核」步驟中，點選畫面指示正確位置來累積訓練素材！")

    st.divider()

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
              "merged_path", "courts", "selected_court", "manual_roi",
              "court_points", "court_points_single", "court_points_adjust",
              "tracking_data", "review_frames", "current_review_idx", "pending_annotations"]:
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
# 檢查是否有未完成的審核進度可恢復
# ─────────────────────────────────────────────
if "review_frames" not in st.session_state:
    saved_info = has_saved_session(uploaded.name)
    if saved_info is not None:
        st.info(
            f"📂 偵測到上次未完成的審核進度！\n\n"
            f"- 已審核：**{saved_info['current']}** / {saved_info['total']} 張\n"
            f"- 剩餘未審核：**{saved_info['remaining']}** 張\n"
            f"- 已偵測回合：**{saved_info['segments_count']}** 段"
        )
        resume_col, new_col = st.columns(2)
        with resume_col:
            if st.button("▶️ 繼續上次的審核", type="primary", use_container_width=True):
                session = load_review_session(uploaded.name)
                if session:
                    st.session_state["tracking_data"] = session["tracking_data"]
                    st.session_state["review_frames"] = session["review_frames"]
                    st.session_state["current_review_idx"] = session["current_review_idx"]
                    st.session_state["pending_annotations"] = []
                    st.session_state["segments"] = session["segments"]
                    st.session_state["motion_timeline"] = session["motion_timeline"]
                    st.session_state["hit_times"] = session["hit_times"]
                    if session["roi"]:
                        st.session_state["manual_roi"] = session["roi"]
                        st.session_state["selected_court"] = -1
                    st.rerun()
        with new_col:
            if st.button("🔄 重新分析（捨棄舊進度）", use_container_width=True):
                delete_session(uploaded.name)
                st.rerun()

# ─────────────────────────────────────────────
# Step 1: 球場偵測 + ROI 選取
# ─────────────────────────────────────────────

# 自動偵測球場（只跑一次，結果存 session_state）
if "courts" not in st.session_state:
    with st.spinner("🔍 自動偵測球場中..."):
        st.session_state["courts"] = detect_courts(first_frame)

courts = st.session_state["courts"]

# 優先從 session_state 恢復已選取的 ROI
roi = st.session_state.get("manual_roi", None)

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
        if idx == -1 and "manual_roi" in st.session_state:
            roi = st.session_state["manual_roi"]
            st.info(f"✅ 使用手動標記範圍：X={roi['x']}, Y={roi['y']}, 寬={roi['w']}, 高={roi['h']}")
        elif 0 <= idx < len(courts):
            roi = courts[idx]
            st.info(f"✅ 已選擇 **Court {idx+1}**：X={roi['x']}, Y={roi['y']}, 寬={roi['w']}, 高={roi['h']}")
    else:
        st.warning("👆 請選擇一個球場")
        st.stop()

    # 提供手動微調選項
    with st.expander("🔧 手動微調範圍"):
        st.caption("請依序點擊：**左上角 ➔ 右上角 ➔ 左下角 ➔ 右下角** 來自訂斜線邊界")
        if "court_points_adjust" not in st.session_state:
            st.session_state["court_points_adjust"] = []
            
        frame_rgb2 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        for i, pt in enumerate(st.session_state["court_points_adjust"]):
            cv2.circle(frame_rgb2, (pt["x"], pt["y"]), 8, (0, 0, 255), -1)
            cv2.putText(frame_rgb2, str(i+1), (pt["x"]+12, pt["y"]-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
        if len(st.session_state["court_points_adjust"]) == 4:
            pts = np.array([[[p["x"], p["y"]] for p in st.session_state["court_points_adjust"]]], dtype=np.int32)
            cv2.polylines(frame_rgb2, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        value = streamlit_image_coordinates(Image.fromarray(frame_rgb2), key="roi_points_adjust")
        
        if value is not None:
            new_pt = {"x": value["x"], "y": value["y"]}
            if new_pt not in st.session_state["court_points_adjust"]:
                if len(st.session_state["court_points_adjust"]) < 4:
                    st.session_state["court_points_adjust"].append(new_pt)
                    st.rerun()
                    
        if len(st.session_state["court_points_adjust"]) == 4:
            st.success("已集滿 4 個點！")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("清除重選", key="clear_pts_adjust"):
                    st.session_state["court_points_adjust"] = []
                    st.rerun()
            with c2:
                if st.button("使用這 4 個角落建立範圍", key="use_adjust_pts", type="primary"):
                    new_roi = compute_roi_from_keypoints(st.session_state["court_points_adjust"])
                    st.session_state["manual_roi"] = new_roi
                    st.session_state["selected_court"] = -1
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
        st.caption("請依序點擊：**左上角 ➔ 右上角 ➔ 左下角 ➔ 右下角** 來自訂斜線邊界")
        if "court_points_single" not in st.session_state:
            st.session_state["court_points_single"] = []
            
        frame_rgb2 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        for i, pt in enumerate(st.session_state["court_points_single"]):
            cv2.circle(frame_rgb2, (pt["x"], pt["y"]), 8, (0, 0, 255), -1)
            cv2.putText(frame_rgb2, str(i+1), (pt["x"]+12, pt["y"]-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
        if len(st.session_state["court_points_single"]) == 4:
            pts = np.array([[[p["x"], p["y"]] for p in st.session_state["court_points_single"]]], dtype=np.int32)
            cv2.polylines(frame_rgb2, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        value = streamlit_image_coordinates(Image.fromarray(frame_rgb2), key="roi_points_single")
        
        if value is not None:
            new_pt = {"x": value["x"], "y": value["y"]}
            if new_pt not in st.session_state["court_points_single"]:
                if len(st.session_state["court_points_single"]) < 4:
                    st.session_state["court_points_single"].append(new_pt)
                    st.rerun()
                    
        if len(st.session_state["court_points_single"]) == 4:
            st.success("已集滿 4 個點！")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("清除重選", key="clear_pts_single"):
                    st.session_state["court_points_single"] = []
                    st.rerun()
            with c2:
                if st.button("使用這 4 個角落建立範圍", type="primary"):
                    new_roi = compute_roi_from_keypoints(st.session_state["court_points_single"])
                    st.session_state["manual_roi"] = new_roi
                    st.session_state["selected_court"] = -1
                    for k in ["motion_timeline", "hit_times", "segments", "output_files", "merged_path"]:
                        st.session_state.pop(k, None)
                    st.rerun()

else:
    # 沒偵測到球場 → 手動框選
    st.header("📐 請手動標註球場範圍")
    st.caption("支援點擊斜線角點來畫出不規則（透視變形）的精確球場邊界。")

    st.write("請依序點擊：**左上角 ➔ 右上角 ➔ 左下角 ➔ 右下角**")
    if "court_points" not in st.session_state:
        st.session_state["court_points"] = []
        
    frame_rgb2 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    for i, pt in enumerate(st.session_state["court_points"]):
        cv2.circle(frame_rgb2, (pt["x"], pt["y"]), 8, (0, 0, 255), -1)
        cv2.putText(frame_rgb2, str(i+1), (pt["x"]+12, pt["y"]-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
    if len(st.session_state["court_points"]) == 4:
        pts = np.array([[[p["x"], p["y"]] for p in st.session_state["court_points"]]], dtype=np.int32)
        cv2.polylines(frame_rgb2, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    value = streamlit_image_coordinates(Image.fromarray(frame_rgb2), key="court_coords")
    
    if value is not None:
        new_pt = {"x": value["x"], "y": value["y"]}
        if new_pt not in st.session_state["court_points"]:
            if len(st.session_state["court_points"]) < 4:
                st.session_state["court_points"].append(new_pt)
                st.rerun()
                
    if len(st.session_state["court_points"]) == 4:
        st.success("已集滿 4 個點！")
        if st.button("清除重選", key="clear_pts"):
            st.session_state["court_points"] = []
            st.rerun()
        if st.button("使用這 4 個角落建立範圍", type="primary"):
            new_roi = compute_roi_from_keypoints(st.session_state["court_points"])
            st.session_state["manual_roi"] = new_roi
            st.session_state["selected_court"] = -1
            for k in ["motion_timeline", "hit_times", "segments", "output_files", "merged_path"]:
                st.session_state.pop(k, None)
            st.rerun()

    if roi is None:
        st.warning("請在上方畫面中點出球場範圍的 4 個重點")
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

    # 執行 YOLO 追蹤
    tracking_data, review_frames = analyze_video_with_yolo(video_path, roi, progress_callback=on_motion_progress)
    st.session_state["tracking_data"] = tracking_data
    st.session_state["review_frames"] = review_frames
    st.session_state["current_review_idx"] = 0
    st.session_state["pending_annotations"] = []

    # 同時執行傳統動態分析 (供後續相容)
    motion_timeline = analyze_video_motion(
        video_path, roi,
        frame_skip=0, gaussian_kernel=11, smooth_window=5,
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
        tracking_data=st.session_state.get("tracking_data", []),
        gap_threshold=gap_threshold, min_duration=min_duration,
        activity_threshold=0.3, motion_weight=motion_w,
