import os
from ultralytics import YOLO

def main():
    print("========================================")
    print("  Pickleball YOLOv8 Active Learning 訓練腳本 ")
    print("========================================")
    
    # 確保資料集配置檔存在
    yaml_path = os.path.abspath("dataset/dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ 找不到資料集定義檔: {yaml_path}")
        print("請先在 Streamlit UI 進行「人工審核」來標註資料！\n")
        return

    # 檢查有沒有之前的訓練權重，如果沒有，就從官方 pretrain 權重開始
    # 也可以固定每次都從 yolov8n.pt 開始 Fine-tune，避免對過度少量的資料 overfitting
    start_weights = 'yolov8n.pt'
    
    print(f"➡️ 讀取初始模型: {start_weights}")
    model = YOLO(start_weights)

    print("\n🚀 開始訓練模型 (Fine-tuning)...")
    # 訓練模型參數設定
    results = model.train(
        data=yaml_path,
        epochs=100,            # 最大輪數
        patience=20,           # Early stopping 檢查輪數
        imgsz=640,             # 輸入推論尺寸
        batch=16,              # 批次數量 (視顯示卡 VRAM 大小可調節為 8, 4 等等)
        project="dataset/runs",# 儲存主要路徑
        name="train",          # 子名稱
        exist_ok=True          # 每次覆蓋在同一個路徑下，以利 Tracker 直接抓到 best.pt
    )
    
    print("\n✅ 訓練完成！")
    print("最新權重已儲存在: dataset/runs/train/weights/best.pt")
    
    # 自動將產出模型複製到獨立資料夾以便 Git 追蹤
    import shutil
    os.makedirs("models", exist_ok=True)
    best_weight_path = "dataset/runs/train/weights/best.pt"
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, "models/pickleball_best.pt")
        print("💾 模型已自動備份至獨立資料夾: models/pickleball_best.pt (此檔案可提交至 Git)")

    print("下次在 Streamlit 進行影片分析時，系統將自動優先載入這個最新模型！")
    print("\n========================================")
    print(" 💡 溫馨提醒：")
    print("    您的 AI 變聰明了！為了永久保存進步成果，")
    print("    千萬別忘了將新的模型推送到 GitHub 上喔！")
    print("    未來也可以隨時呼叫我幫您跑『自動推送工作流程』來完成備份。")
    print("========================================\n")

if __name__ == "__main__":
    main()
