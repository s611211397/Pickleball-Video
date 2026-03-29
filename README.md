# Pickleball Auto-Editor 🏓

自動剪輯匹克球影片工具 — 從固定機位拍攝的完整影片中，自動偵測並保留對戰（rally）畫面，剔除休息、聊天、撿球等無關片段。

## 核心功能

- **自動偵測 rally 起止點**：結合視覺動態分析 + 音訊擊球聲偵測
- **ROI 球場框選**：只分析你的球場區域，完全排除隔壁場干擾
- **智慧切片輸出**：每段 rally 前後自動保留 buffer，確保畫面完整
- **可調參數**：門檻值、間隔秒數、buffer 長度皆可自訂

## 快速開始

### 方式一：網頁介面（推薦）

最簡單的使用方式，不需要任何指令：

```bash
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器會自動開啟，接著：
1. **上傳影片** — 左側面板拖拉影片進去
2. **框選球場** — 用滑桿調整綠色框，框住你的場地
3. **調整參數** — 左側有靈敏度、回合間隔等設定
4. **按下分析** — 等待分析完成
5. **下載成果** — 一鍵下載剪輯好的影片

### 方式二：命令列

適合需要批次處理或進階操作的使用者：

```bash
pip install -r requirements.txt

# 第一次使用：設定球場 ROI（會跳出畫面讓你框選）
python main.py --set-roi input_video.mp4

# 自動剪輯
python main.py input_video.mp4 -o output_dir/

# 預覽偵測結果（不輸出影片）+ 看時序圖
python main.py input_video.mp4 --preview --visualize

# 純視覺模式（不用音訊）
python main.py input_video.mp4 -o output_dir/ --no-audio
```

## 技術架構

詳見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
