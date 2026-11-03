# 314551126 · Lab 2 提交檔案

本目錄是準備繳交的內容，依作業格式分為 `Part1/`、`Part2/`、`Part3/`。

| 部分 | 主要檔案 | 重現入口 |
|---|---|---|
| Part 1 | `Part1/314551126_Part1.ipynb` | 在 Kaggle／Colab 準備 GPU 與指定模型權重後執行 Notebook；依平台設定模型路徑 |
| Part 2 | `Part2/314551126_report.pdf` | 在 Part2 執行 `bash run.sh` 或 `bash 314551126_run.sh` |
| Part 3 | `Part3/314551126_report.pdf` | 在 Part3 執行 `bash run.sh` |

Part 1 使用 torch 2.6.0；Part 2/3 使用 Python 3.11、torch 2.5.1+cu121，環境各自隔離。Part 2/3 都只需要一張 T4 等級 GPU；首次需要網路下載 Qwen3-4B、WikiText-2 及依賴。Part 2／3 的詳細環境與命令見各自的 README。

Part 2 的 `run.sh` 自動安裝依賴、從原始模型壓縮四個 cases、使用提供的 evaluator 評估，stdout 僅印四組 Compression Ratio／PPL；進度與日誌另外記錄。包含兩個 shell 檔名是為了同時符合題目內文 `run.sh` 與總提交清單 `StudentID_run.sh` 的要求。

Part 1 僅保留 Notebook，執行時另行提供題目指定的 MobileNetV2 權重。Part 2／3 保留壓縮與評估程式及已完成的 PDF 報告；重新執行後會產生結果 JSON、配置與 checkpoint。

大型 Qwen checkpoint 與校正候選由執行入口重建，不依賴本 workspace 的其他實驗資料夾。

2026-11-03 更新：Part 2 壓縮率及剪枝率均排除 lm_head；四個案例達標。Case 3 重建並完整重測，壓縮率精確50%、SVD壓縮率22.2727%、PPL25.7291。四個 case 均已在目前提交程式上完成重新壓縮與正式評估，數值與報告一致。Part 2 報告與比較表亦已同步。
