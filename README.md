# Pickleball Auto-Editor

自動剪輯匹克球影片工具 — 從固定機位拍攝的完整影片中，自動偵測並保留對戰（rally）畫面，剔除休息、聊天、撿球等無關片段。

## 核心功能

- **自動偵測 rally 起止點**：結合視覺動態分析 + 音訊擊球聲偵測
- **ROI 球場框選**：只分析你的球場區域，完全排除隔壁場干擾
- **智慧切片輸出**：每段 rally 前後自動保留 buffer，確保畫面完整
- **可調參數**：門檻值、間隔秒數、buffer 長度皆可自訂

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 第一次使用：設定球場 ROI（會跳出畫面讓你框選）
python main.py --set-roi input_video.mp4

# 自動剪輯
python main.py input_video.mp4 -o output_dir/

# 使用已儲存的 ROI 設定
python main.py input_video.mp4 -o output_dir/ --roi-config roi.json
```

## 技術架構

詳見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
