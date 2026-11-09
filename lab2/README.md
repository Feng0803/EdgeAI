# Lab 2 — 314551126

本目錄整理 Lab 2 的完整實作、報告與兩種 Part 3 版本。

## 目錄

- `Part1/`：量化作業 Notebook。
- `Part2/`：四個 LLM compression cases；正式入口為
  `bash 314551126_run.sh`，預設使用 `frozen/` 中已驗證的配置。
- `Part3/`：目前正式的 AdaptQuant-based calibration-free PTQ 版本；
  正式入口為 `bash run.sh`。
- `Part3_nf4/`：先前完成的 calibration-free NF4 W4A16 版本，保留作為
  baseline、結果比較與實作紀錄；入口同樣為 `bash run.sh`。
- `26502_AdaptQuant_Hardware_Alig.pdf`：Part 3 參考論文。

各部分的環境、方法、固定設定來源與重現方式，詳見各自的 README。
大型模型 checkpoint、虛擬環境與執行期間產物不納入版本控制。

## 作業繳交版本

正式繳交內容使用 `Part1/`、`Part2/` 與 `Part3/`。`Part3_nf4/` 與論文
留在 GitHub 作為替代方法及開發紀錄，不放入最終繳交 ZIP。
