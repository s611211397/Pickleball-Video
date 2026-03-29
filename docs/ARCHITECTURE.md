# 架構設計文件

## 系統總覽

```
輸入影片 (.mp4/.mov)
       │
       ├──→ [1] ROI 設定模組（首次使用 / 手動設定）
       │         └─→ roi.json（球場座標）
       │
       ├──→ [2] 視覺動態分析模組
       │         ├─ 讀取 ROI 區域
       │         ├─ 逐幀計算動態量（幀差法 / 光流法）
       │         └─→ 動態時序資料 (motion_timeline)
       │
       ├──→ [3] 音訊分析模組
       │         ├─ 抽取音軌
       │         ├─ 偵測擊球聲（短脈衝高振幅）
       │         └─→ 擊球時間點列表 (hit_events)
       │
       ├──→ [4] 融合判斷引擎
       │         ├─ 結合 motion_timeline + hit_events
       │         ├─ 狀態機：idle → rally → point_end → idle
       │         └─→ rally 起止時間列表 (segments)
       │
       └──→ [5] 影片切片輸出模組
                 ├─ 根據 segments 切片
                 ├─ 前後加 buffer
                 ├─ 可選：合併成單一影片 or 分段輸出
                 └─→ 最終影片
```

## 模組詳細設計

---

### 模組 1：ROI 設定 (`src/roi_selector.py`)

**目的**：讓使用者框選自己的球場範圍，排除隔壁場地干擾。

**設計**：
- 從影片中擷取第一幀，用 OpenCV 的 `selectROI()` 讓使用者拖曳框選
- 將座標儲存為 `roi.json`，格式：`{"x": 100, "y": 50, "w": 800, "h": 600}`
- 同機位同場地只需設定一次，之後重複使用

**進階（v2）**：
- 自動偵測球場線來推算 ROI
- 支援多邊形 ROI（非矩形球場角度）

---

### 模組 2：視覺動態分析 (`src/motion_detector.py`)

**目的**：逐幀偵測球場內的動態量，區分「有活動」與「靜止」狀態。

**方法選擇**：

| 方法 | 優點 | 缺點 | 建議 |
|------|------|------|------|
| 幀差法 (Frame Differencing) | 快速、簡單 | 對緩慢移動不敏感 | MVP 首選 |
| 背景減除 (MOG2/KNN) | 更穩定 | 需要學習期 | v2 考慮 |
| 光流法 (Optical Flow) | 精準追蹤運動方向 | 運算量大 | v2 考慮 |

**MVP 實作（幀差法）**：
```python
def compute_motion_score(frame_prev, frame_curr, roi):
    """計算 ROI 區域內的動態分數"""
    # 1. 裁切 ROI 區域
    crop_prev = frame_prev[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
    crop_curr = frame_curr[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

    # 2. 轉灰階
    gray_prev = cv2.cvtColor(crop_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(crop_curr, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊降噪
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)

    # 4. 幀差取絕對值
    diff = cv2.absdiff(gray_prev, gray_curr)

    # 5. 二值化 + 計算動態像素佔比
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh > 0) / thresh.size

    return motion_score
```

**輸出格式**：
```python
motion_timeline = [
    {"time": 0.0, "score": 0.02},   # 靜止
    {"time": 0.033, "score": 0.15},  # 有動態
    {"time": 0.066, "score": 0.23},  # 高動態
    ...
]
```

**關鍵參數**：
- `motion_threshold`: 動態分數門檻（建議 0.05-0.15，需依實際畫面調整）
- `frame_skip`: 每 N 幀分析一次（建議 2-3，加速處理）
- `gaussian_kernel`: 模糊核大小（建議 21x21）

---

### 模組 3：音訊分析 (`src/audio_analyzer.py`)

**目的**：偵測匹克球擊球的「啪」聲，作為輔助判斷依據。

**匹克球擊球聲特徵**：
- 頻率範圍：約 1000-4000 Hz（短促高頻）
- 持續時間：< 50ms
- 振幅：明顯高於背景噪音

**實作流程**：
```
音軌 → 帶通濾波 (1000-4000Hz) → 計算短時能量 →
峰值偵測 → 最小間距過濾 → 擊球時間點列表
```

**關鍵參數**：
- `bandpass_low`: 帶通濾波下限（建議 1000 Hz）
- `bandpass_high`: 帶通濾波上限（建議 4000 Hz）
- `energy_threshold`: 能量門檻（需依實際錄音調整）
- `min_hit_interval`: 最小擊球間距（建議 0.3 秒，避免重複偵測）

**降噪策略**：
- 帶通濾波：只保留擊球聲頻率範圍，過濾人聲（100-300Hz）和風聲（<500Hz）
- 自適應門檻：根據整段音訊的平均能量動態計算門檻
- 短時分析：用短窗口（20-50ms）偵測脈衝，長窗口的持續聲音（說話、音樂）會被平均掉

---

### 模組 4：融合判斷引擎 (`src/rally_detector.py`)

**目的**：結合視覺和音訊資訊，精準判斷每段 rally 的起止點。

**狀態機設計**：

```
         動態量上升 + 偵測到擊球聲
    ┌──────────────────────────────────┐
    │                                  ▼
 [IDLE] ◄──── 超過 gap_threshold ──── [RALLY]
    ▲           秒無活動                │
    │                                  │
    └──── rally 時長 < min_duration ───┘
              （太短，視為雜訊）
```

**判斷邏輯**：
```python
class RallyDetector:
    def __init__(self, config):
        self.state = "idle"
        self.gap_threshold = config.gap_threshold      # 幾秒無活動算結束 (預設 4 秒)
        self.min_rally_duration = config.min_duration   # 最短 rally 時長 (預設 3 秒)
        self.motion_weight = config.motion_weight       # 視覺權重 (預設 0.7)
        self.audio_weight = config.audio_weight         # 音訊權重 (預設 0.3)

    def compute_activity_score(self, motion_score, has_hit):
        """融合視覺和音訊分數"""
        audio_score = 1.0 if has_hit else 0.0
        return self.motion_weight * motion_score + self.audio_weight * audio_score
```

**融合策略的優勢**：

| 場景 | 視覺信號 | 音訊信號 | 融合結果 |
|------|----------|----------|----------|
| 正常 rally | 高動態 ✓ | 有擊球聲 ✓ | 高信心 rally ✓ |
| 球員走動聊天 | 中動態 | 無擊球聲 ✗ | 非 rally ✗ |
| 快速撿球 | 短暫高動態 | 無擊球聲 ✗ | 時長不足 → 排除 ✗ |
| 風吹晃動 | 低動態 | 無擊球聲 ✗ | 非 rally ✗ |
| 安靜的吊球 rally | 低動態 | 有擊球聲 ✓ | 音訊補救 → rally ✓ |

**關鍵參數**：
- `gap_threshold`: 無活動間隔門檻，幾秒算一分結束（建議 3-5 秒）
- `min_rally_duration`: 最短 rally 時長（建議 2-3 秒，過濾撿球）
- `activity_threshold`: 綜合活動分數門檻（建議 0.3）
- `motion_weight` / `audio_weight`: 視覺 vs 音訊的權重（建議 0.7 / 0.3）

---

### 模組 5：影片切片輸出 (`src/video_exporter.py`)

**目的**：根據偵測到的 rally 時段，切割並輸出最終影片。

**功能**：
- 每段 rally 前後加 buffer（預設 2 秒）
- 支援兩種輸出模式：
  - 分段輸出：每段 rally 一個檔案
  - 合併輸出：所有 rally 合成一個影片（中間可加轉場）
- 使用 FFmpeg 無重新編碼切片（速度快、無畫質損失）

**FFmpeg 切片策略**：
```bash
# 無重新編碼（快速，但切點可能不精確到幀）
ffmpeg -ss {start} -to {end} -i input.mp4 -c copy output_segment.mp4

# 重新編碼（精確到幀，但較慢）
ffmpeg -ss {start} -to {end} -i input.mp4 -c:v libx264 -c:a aac output_segment.mp4
```

MVP 先用無重新編碼模式，速度優先。

---

## 設定檔設計 (`config.yaml`)

```yaml
# ROI 設定
roi:
  config_file: "roi.json"     # ROI 座標檔路徑

# 視覺偵測參數
motion:
  threshold: 0.08             # 動態分數門檻
  frame_skip: 2               # 每 N 幀分析一次
  gaussian_kernel: 21         # 高斯模糊核大小

# 音訊偵測參數
audio:
  enabled: true               # 是否啟用音訊輔助
  bandpass_low: 1000          # 帶通濾波下限 (Hz)
  bandpass_high: 4000         # 帶通濾波上限 (Hz)
  energy_threshold: 0.5       # 能量門檻（相對值）
  min_hit_interval: 0.3       # 最小擊球間距 (秒)

# Rally 偵測參數
rally:
  gap_threshold: 4.0          # 無活動間隔 (秒)
  min_duration: 3.0           # 最短 rally 時長 (秒)
  activity_threshold: 0.3     # 綜合活動分數門檻
  motion_weight: 0.7          # 視覺權重
  audio_weight: 0.3           # 音訊權重

# 輸出設定
output:
  buffer_before: 2.0          # rally 前 buffer (秒)
  buffer_after: 2.0           # rally 後 buffer (秒)
  mode: "merged"              # "merged" 合併 / "segments" 分段
  reencode: false             # 是否重新編碼（精確切點）
  format: "mp4"               # 輸出格式
```

---

## 開發計劃

### Phase 1: MVP（核心功能）
1. **ROI 設定模組** — 手動框選球場範圍
2. **視覺動態偵測** — 幀差法偵測動態
3. **基礎 Rally 偵測** — 純視覺 + 狀態機
4. **影片切片輸出** — FFmpeg 切片

### Phase 2: 音訊強化
5. **音訊分析模組** — 擊球聲偵測
6. **融合判斷引擎** — 視覺 + 音訊混合判斷
7. **參數調校工具** — 可視化動態量曲線，方便調參

### Phase 3: 體驗優化
8. **自動 ROI 偵測** — 利用球場線自動框選
9. **批次處理** — 一次處理多個影片
10. **簡易 GUI** — 拖拉影片就能用
11. **預覽模式** — 剪輯前先預覽偵測結果

---

## 依賴套件

```
opencv-python>=4.8.0    # 視覺處理
numpy>=1.24.0           # 數值計算
librosa>=0.10.0         # 音訊分析
scipy>=1.11.0           # 信號處理（峰值偵測、濾波）
pyyaml>=6.0             # 設定檔解析
click>=8.1.0            # CLI 介面
tqdm>=4.65.0            # 進度條
```

外部工具：
- **FFmpeg** — 影片/音軌抽取與切片（需系統安裝）

---

## 預估信心度

| 方案 | 信心度 | 說明 |
|------|--------|------|
| 純音訊 | 70-75% | 隔壁場干擾、風聲問題 |
| 純視覺（ROI + 幀差法） | 85-90% | 走動聊天可能誤判 |
| **混合方案（視覺 + 音訊 + 狀態機）** | **90-95%** | 多層驗證大幅降低誤判 |

剩餘 5-10% 的邊界情況（如球員在場中做暖身操）可能需要手動微調，但作為 MVP 自用工具已經非常夠用。
