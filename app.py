"""Pickleball Auto-Editor — Streamlit Web UI"""

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
    
    # 算一下目前標註了多少張圖
    label_path = Path("dataset/labels/train")
    if label_path.exists():
        num_labels = len(list(label_path.glob("*.txt")))
    else:
        num_labels = 0
        
    st.metric("累積可訓練的「問題幀」數量", f"{num_labels} 張")
    
    if num_labels > 0:
        if st.button("🚀 開始 Fine-tune 訓練", use_container_width=True, type="primary"):
            st.warning("⚠️ 訓練期間將會占用 GPU 資源，且可能耗時數分鐘到數十分鐘。請勿關閉或重新整理此視窗。詳細進度會顯示於運行本程式的終端機。")
            with st.spinner("⏳ 正在訓練 YOLOv8 模型中，請查看終端機黑窗..."):
                import subprocess
                try:
                    # 使用 subprocess 呼叫，避免 Streamlit 迴圈阻塞輸出
                    result = subprocess.run(
                        ["python", "train_model.py"],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0:
                        st.success("✅ 訓練大成功！系統已將最新的完美權重部署完成。下次重新分析影片即會自動套用。")
                        st.info("💡 **別忘了備份模型**：最新優化的模型已獨立備份為 `models/pickleball_best.pt`，請記得找時間提交並 Push 到 GitHub 永久保存喔！")
                        st.balloons()
                    else:
                        st.error("❌ 訓練失敗。")
                        with st.expander("查看詳細錯誤 Log"):
                            st.code(result.stderr[-2000:], language="bash")
                except Exception as e:
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
                    roi = compute_roi_from_keypoints(st.session_state["court_points_adjust"])
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
                    roi = compute_roi_from_keypoints(st.session_state["court_points_single"])
                    st.session_state["selected_court"] = -1
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
            roi = compute_roi_from_keypoints(st.session_state["court_points"])
            st.session_state["selected_court"] = -1
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
        audio_weight=audio_weight, motion_threshold=motion_threshold,
    )
    st.session_state["segments"] = segments
    progress.progress(100, text="✅ 分析完成！")
    st.rerun()

# ─────────────────────────────────────────────
# Step 2.5: YOLO 軌跡審核 (快速翻頁模式)
# ─────────────────────────────────────────────
GRID_PAGE_SIZE = 12  # 保留，供未來用

if "review_frames" in st.session_state and "current_review_idx" in st.session_state:
    review_frames = st.session_state["review_frames"]
    tracking_data = st.session_state["tracking_data"]
    idx = st.session_state["current_review_idx"]

    if len(review_frames) > 0 and idx < len(review_frames):
        st.header("🧐 Step 2.5: 軌跡迷失審核")

        frame_idx = review_frames[idx]
        data = tracking_data[frame_idx]
        frame_bgr = data["frame"]

        # 進度列 + 右側儲存按鈕
        prog_col, save_col = st.columns([4, 1])
        with prog_col:
            st.progress(idx / len(review_frames),
                        text=f"第 {idx + 1} / {len(review_frames)} 張 (Frame {frame_idx})")
        with save_col:
            if st.button("💾 儲存並結束", use_container_width=True):
                st.session_state["current_review_idx"] = len(review_frames)
                # 不需要 st.rerun()，自然刷新即可

        # 顯示大圖（可點擊標記球的位置）
        if frame_bgr is not None:
            orig_h, orig_w = frame_bgr.shape[:2]
            scale = 1.0
            if orig_w > 960:
                scale = 960 / orig_w
                disp_bgr = cv2.resize(frame_bgr, (960, int(orig_h * scale)))
            else:
                disp_bgr = frame_bgr
                
            frame_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
            st.caption("👆 **直接點擊球的位置** 來標記座標，系統會嘗試「自動追蹤」後續緊接的迷失畫面！")
            
            click_val = streamlit_image_coordinates(Image.fromarray(frame_rgb), key=f"review_{idx}")
            if click_val is not None:
                # 1. 換算回原始解析度的真實座標
                orig_x = click_val["x"] / scale
                orig_y = click_val["y"] / scale
                bw, bh = 30 / scale, 30 / scale  # 點擊範圍擴大，以利後續追蹤
                
                box = {"x": orig_x - bw/2, "y": orig_y - bh/2, "w": bw, "h": bh}
                
                # 紀錄當前點擊的這一幀
                st.session_state["pending_annotations"].append(
                    (frame_bgr, f"{Path(video_path).stem}_f{frame_idx}", box)
                )
                tracking_data[frame_idx]["box"] = box
                tracking_data[frame_idx]["conf"] = 1.0
                tracking_data[frame_idx]["status"] = "DETECTED"
                
                adv_count = 1
                
                # 2. 自動追蹤後續相連幀 (Template Matching)
                orig_bgr = frame_bgr
                tx1 = max(0, int(box["x"]))
                ty1 = max(0, int(box["y"]))
                tx2 = min(orig_w, int(box["x"] + box["w"]))
                ty2 = min(orig_h, int(box["y"] + box["h"]))
                template = orig_bgr[ty1:ty2, tx1:tx2]
                prev_x, prev_y = int(box["x"]), int(box["y"])
                
                if template.size > 0:
                    for k in range(idx + 1, len(review_frames)):
                        if review_frames[k] == review_frames[k-1] + 1:
                            next_fi = review_frames[k]
                            next_frm = tracking_data[next_fi]["frame"]
                            if next_frm is None:
                                break
                                
                            # 在上一幀球的附近 150x150 範圍內尋找
                            search_margin = int(150 / scale)
                            sx1 = max(0, prev_x - search_margin)
                            sy1 = max(0, prev_y - search_margin)
                            sx2 = min(orig_w, prev_x + int(box["w"]) + search_margin)
                            sy2 = min(orig_h, prev_y + int(box["h"]) + search_margin)
                            
                            search_area = next_frm[sy1:sy2, sx1:sx2]
                            if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
                                break
                                
                            # 進行特徵比對
                            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            
                            if max_val > 0.55:  # 相似度閾值 (0.55 對快節奏球類已算穩健)
                                hit_x = sx1 + max_loc[0]
                                hit_y = sy1 + max_loc[1]
                                new_box = {"x": hit_x, "y": hit_y, "w": box["w"], "h": box["h"]}
                                
                                st.session_state["pending_annotations"].append(
                                    (next_frm, f"{Path(video_path).stem}_f{next_fi}", new_box)
                                )
                                tracking_data[next_fi]["box"] = new_box
                                tracking_data[next_fi]["conf"] = float(max_val)
                                tracking_data[next_fi]["status"] = "DETECTED"
                                
                                adv_count += 1
                                prev_x, prev_y = hit_x, hit_y
                                # 更新模板以適應球的形變與光影改變
                                template = next_frm[hit_y:hit_y+int(box["h"]), hit_x:hit_x+int(box["w"])]
                            else:
                                break # 追蹤失敗，中斷讓使用者針對這張再次點擊
                        else:
                            break # 不連號（代表中間本來就有抓到），跳出
                            
                st.session_state["current_review_idx"] += adv_count
                st.rerun()
        else:
            st.warning("此幀無影像資料")


        # 操作按鈕 — 大塊並排
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("✅ 有球（略過）", use_container_width=True):
                st.session_state["current_review_idx"] += 1
                st.rerun()
        with b2:
            if st.button("❌ 無球", use_container_width=True, type="primary"):
                st.session_state["pending_annotations"].append(
                    (data["frame"], f"{Path(video_path).stem}_f{frame_idx}", None)
                )
                st.session_state["current_review_idx"] += 1
                st.rerun()
        with b3:
            # 計算後續連號幀數，一次標記
            if st.button("❌❌ 連續無球", use_container_width=True, type="primary"):
                count = 1
                for k in range(idx + 1, len(review_frames)):
                    if review_frames[k] == review_frames[k - 1] + 1:
                        count += 1
                    else:
                        break
                for k_off in range(count):
                    fi = review_frames[idx + k_off]
                    st.session_state["pending_annotations"].append(
                        (tracking_data[fi]["frame"], f"{Path(video_path).stem}_f{fi}", None)
                    )
                st.session_state["current_review_idx"] += count
                st.rerun()
        with b4:
            if st.button("⏭ 略過", use_container_width=True):
                st.session_state["current_review_idx"] += 1
                st.rerun()

        st.stop()

    elif len(review_frames) > 0 and idx >= len(review_frames):
        if "pending_annotations" in st.session_state and len(st.session_state["pending_annotations"]) > 0:
            with st.spinner("💾 正在將標記寫入硬碟中，請稍候..."):
                dm = DatasetManager()
                for frm, name, box in st.session_state["pending_annotations"]:
                    dm.save_annotation(frm, name, box)
            st.session_state["pending_annotations"] = []
        st.success("✅ 問題軌跡審核告一段落，已存入 dataset 供日後訓練使用！")



if "review_frames" in st.session_state and "current_review_idx" in st.session_state:
    review_frames = st.session_state["review_frames"]
    tracking_data = st.session_state["tracking_data"]
    idx = st.session_state["current_review_idx"]

    if len(review_frames) > 0 and idx < len(review_frames):
        st.header("🧐 Step 2.5: 軌跡迷失審核 (批量審核模式)")

        # 計算目前頁面
        page_start = idx
        page_end   = min(idx + GRID_PAGE_SIZE, len(review_frames))
        page_frames = review_frames[page_start:page_end]

        # 進度提示
        col_prog, col_save = st.columns([3, 1])
        with col_prog:
            st.caption(
                f"📄 第 **{idx + 1} ~ {page_end}** 張 / 共 **{len(review_frames)}** 張"
                f"　｜　⏳ 每頁 {GRID_PAGE_SIZE} 張，點好後按「下一頁」"
            )
        with col_save:
            if st.button("💾 儲存並結束標註", use_container_width=True, type="primary"):
                st.session_state["current_review_idx"] = len(review_frames)
                st.rerun()

        st.divider()

        # ── 網格顯示 ─────────────────────────────────────────────
        grid_cols = st.columns(4)
        for i, frame_idx in enumerate(page_frames):
            data = tracking_data[frame_idx]
            frame_bgr = data["frame"]
            marked_key = f"marked_{frame_idx}"

            with grid_cols[i % 4]:
                # 已標記的顯示狀態角標
                already = st.session_state.get(marked_key)
                if already == "無球":
                    st.markdown("🔴 **無球**")
                elif already == "略過":
                    st.markdown("⬜ **略過**")
                else:
                    st.markdown("❓ *未標記*")

                if frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # 顯示縮圖（不需要點擊座標，只要設定有球/無球）
                    st.image(Image.fromarray(frame_rgb), caption=f"Frame {frame_idx}", use_container_width=True)
                else:
                    st.warning("無影像資料")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("無球", key=f"noball_{frame_idx}", use_container_width=True):
                        st.session_state["pending_annotations"].append(
                            (frame_bgr, f"{Path(video_path).stem}_f{frame_idx}", None)
                        )
                        st.session_state[marked_key] = "無球"
                        # 不 rerun 讓使用者繼續點其他格
                with b2:
                    if st.button("略過", key=f"skip_{frame_idx}", use_container_width=True):
                        st.session_state[marked_key] = "略過"

        st.divider()

        # ── 頁面導航 ─────────────────────────────────────────────
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("⏮ 這頁全部標「無球」", use_container_width=True):
                for frame_idx in page_frames:
                    frm_bgr = tracking_data[frame_idx]["frame"]
                    st.session_state["pending_annotations"].append(
                        (frm_bgr, f"{Path(video_path).stem}_f{frame_idx}", None)
                    )
                    st.session_state[f"marked_{frame_idx}"] = "無球"
                st.session_state["current_review_idx"] += len(page_frames)
                st.rerun()
        with nav2:
            st.progress((idx) / len(review_frames), text="標註進度")
        with nav3:
            if st.button("下一頁 ▶", use_container_width=True, type="primary"):
                st.session_state["current_review_idx"] += len(page_frames)
                st.rerun()

        st.stop()

    elif len(review_frames) > 0 and idx >= len(review_frames):
        # 全部批次寫入資料夾
        if "pending_annotations" in st.session_state and len(st.session_state["pending_annotations"]) > 0:
            with st.spinner("💾 正在將標記寫入硬碟中，請稍候..."):
                dm = DatasetManager()
                for frm, name, box in st.session_state["pending_annotations"]:
                    dm.save_annotation(frm, name, box)
            st.session_state["pending_annotations"] = []

        st.success("✅ 問題軌跡審核告一段落，已存入 dataset 供日後訓練使用！")


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
