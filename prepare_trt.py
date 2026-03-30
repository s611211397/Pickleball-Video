"""
一次性 TensorRT 引擎轉換腳本

用法：
    .\.venv\Scripts\python prepare_trt.py

轉換完成後，Streamlit 啟動時會自動載入引擎，不需要再次轉換。
"""

from pathlib import Path
from ultralytics import YOLO

MODEL_PT   = "yolov8n.pt"
IMGSZ      = 320
ENGINE_OUT = Path(MODEL_PT).with_name(
    Path(MODEL_PT).stem + f"_imgsz{IMGSZ}_fp16.engine"
)

if ENGINE_OUT.exists():
    print(f"✅ 引擎已存在：{ENGINE_OUT}，無需重新轉換。")
else:
    print(f"🔧 開始轉換 TensorRT 引擎，請耐心等待 3~8 分鐘...")
    print(f"   來源模型：{MODEL_PT}")
    print(f"   推論解析度：{IMGSZ}x{IMGSZ}、FP16 半精度、動態批次大小")

    model = YOLO(MODEL_PT)
    model.export(
        format="engine",
        imgsz=IMGSZ,
        half=True,
        device=0,
        dynamic=True,
    )

    # ultralytics 預設輸出在同目錄
    default_engine = Path(MODEL_PT).with_suffix(".engine")
    if default_engine.exists():
        default_engine.rename(ENGINE_OUT)

    if ENGINE_OUT.exists():
        print(f"\n✅ 轉換成功！引擎已儲存至：{ENGINE_OUT}")
        print("   現在可以啟動 Streamlit：.venv\\Scripts\\python -m streamlit run app.py")
    else:
        print("❌ 轉換失敗，請查看上方的錯誤訊息。")
