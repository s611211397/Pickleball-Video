---
description: 自動推送訓練好的 YOLO 模型至 GitHub 保存
---

這個自動化工作流程 (Workflow) 會幫您把這台電腦上最新訓練出來的 `models/pickleball_best.pt` 打包並自動推送到 GitHub 上。

// turbo-all
1. 將最新模型加入 Git 追蹤 `git add models/pickleball_best.pt`
2. 建立一個包含時間戳記的更新點 `git commit -m "Auto-backup: Update pickleball YOLO model"`
3. 將模型同步到雲端 `git push`
