# Part 3：Calibration-Free NF4 PTQ

本實驗採用 `Lab 2.md` 允許的替代 PTQ 方法：固定 NF4 W4A16 fake quantization，從原始 `Qwen/Qwen3-4B` 開始。不是 AdaptQuant 實作，也不使用 Part 2 的模型或校正資料。

已完成的實測結果列於下表，完整中文報告為 [314551126_report.pdf](314551126_report.pdf)。`13.98` 是題目提供的 AdaptQuant 比較值。

| 設定 | WikiText-2 PPL |
|---|---:|
| 原始 FP16，本次獨立控制實驗 | 13.6378 |
| 固定 NF4 W4A16 | **14.6063** |

依作業分級，NF4 達到 PPL ≤15，對應性能項 **15／15 分**；報告項的 5 分由助教評閱。

## 環境與重現

使用 Python 3.11、PyTorch 2.5.1+cu121、單張 NVIDIA T4 16GB。需要可用的 CUDA driver；首次執行需要網路下載模型及 WikiText-2。建議至少預留 30GB 磁碟空間供原始模型快取與 fake-quantized checkpoint 使用。程式預設 `cuda:0`，只使用一張 GPU。

在 `Part3/` 目錄直接重現：

```bash
bash run.sh
```

入口會自動建立本地 `.venv` 並安裝依賴，不需要 workspace 的其他資料夾。若已有符合版本的環境，可指定：

```bash
PART3_PYTHON=/absolute/path/to/python bash run.sh
```

`run.sh` 會安裝依賴，依序重新執行 FP16 控制實驗、NF4 量化、保存／重載一致性檢查及官方 PPL 評估，並更新結果 JSON。每次重建 NF4 都從原始模型開始，不對已量化權重再次量化。環境固定於 [requirements.txt](requirements.txt)，模型來源固定為 commit `1cfa9a7208912126459214e8b04321603b3df60c`。

也可以分步執行；以下 `python` 指向上述已安裝依賴的 Python 3.11：

```bash
python -u run_part3.py --mode fp16
python -u run_part3.py --mode nf4
```

已有 `artifacts/nf4/model/` 時，只重跑 PPL：

```bash
python -u run_part3.py --mode nf4 --eval-only
```

只重建模型而不評估可加 `--skip-eval`。執行記憶體不足時可加 `--rows-per-chunk 64`，這只改暫存 chunk，不改 block 邊界、格式或輸出值。模型與資料已快取時，可在命令前加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。

## 固定方法與 calibration-free 性質

| 項目 | 固定設定 |
|---|---|
| 模型 | 原始 Qwen/Qwen3-4B，FP16 載入 |
| 量化範圍 | `model.layers.*` 下全部 252 個 `nn.Linear` |
| 權重格式 | NF4 的固定 16 個 codepoints，最近鄰映射 |
| 分組 | 每 64 個連續權重，固定 block size |
| Scale | 直接對原始權重計算 FP32 block absmax |
| 重建 | `codebook[code] * scale` 後轉 FP16 |
| 其餘元件 | 共享 embedding／lm_head 保留原始 FP16；其他非 Linear 元件不改 |
| Activation | 沿用原模型 FP16 計算流程 |
| 額外程序 | 無 double quant、clipping search、逐層精度選擇或訓練 |

Codebook 源自 QLoRA 的 NF4 格式，本實驗沒有使用 LoRA 或 fine-tuning。量化函式只有原始權重輸入；不讀任何資料集，不蒐集 activation，也不以目標模型的輸出選擇參數、閾值或精度。Absmax 是事先固定的權重映射規則，不是搜尋得到的設定。

`run_part3.py` 在讀取權重前就記錄固定 policy；量化期間對所有模組安裝禁止 forward 的 hook，直到量化完成才移除。每層權重 MSE 只記錄診斷值，不影響任何決策。量化後固定 8 個 token 的 forward 只檢查 checkpoint 重載一致性，不作校正。正式評估才匯入 `evaluate.py` 並讀取 WikiText-2 test，結果不回流量化器。

此處使用題目明確允許的 **fake quantization**：NF4 重建值存回 FP16 tensor，執行原始 Linear；儲存檔案也是 FP16。邏輯格式為 4-bit 權重，並未實作 packed 4-bit 儲存、低精度 kernel 或推論加速，因此不宣稱實際檔案壓縮率。

## 評估與輸出

`run_eval.py` 和 `evaluate.py` 與 Part 2 提供的版本逐位元相同。包裝程式只替換模型載入入口，以及將實際 FP16 儲存的 compression rate 回報為 0；PPL 計算函式、WikiText-2 test 處理、sequence length 2048 均沿用原程式。每次完成 PPL 會在終端印出結果，並更新：

| 路徑 | 內容 |
|---|---|
| `results/fp16_result.json` | 獨立 FP16 對照 PPL |
| `results/nf4_result.json` | NF4 PPL、性能分級、146 段評估與 evaluator SHA-256 |
| `results/quantization_manifest.json` | 固定 policy、252 層統計、權重 MSE、重載一致性 |
| `artifacts/source.json` | 模型來源 commit |
| `artifacts/nf4/model/` | 可直接載入的 FP16 fake-quantized checkpoint |
| `314551126_report.pdf` | 方法、結果、觀察、限制及重現步驟 |

`artifacts/` 是可重建的大型輸出，提交程式時不必帶入模型權重。執行入口會在 `results/` 產生量化與評估的 JSON 紀錄；完成的 PDF 報告位於本目錄。

作業性能分級為 PPL ≤15：15 分、≤16：12 分、≤17：9 分、≤18：6 分、≤19：3 分，其他為 0；報告項的 5 分由助教評閱。

本方法固定 NF4，不實作 AdaptQuant 題名所述的 per-tile MX format selection 或 Blackwell kernel。未取得其可讀全文／原始實作，報告不推測其未確認配置。可比的項目是作業指定模型、評估流程與 PPL 分級。

## 來源

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)：NF4 格式來源。
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)：NF4 固定常數的參考來源；本程式不依賴此套件執行。
- `../Lab 2.md`：Part 3 條件、評估要求與分級。
