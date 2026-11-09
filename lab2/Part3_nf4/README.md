# Part 3：Calibration-Free NF4 W4A6 PTQ

本實驗採用作業允許的替代方法：固定 **NF4 W4A6 fake Post-Training Quantization**，從原始 `Qwen/Qwen3-4B` 開始。所有格式選擇都在載入 target model 前固定，不使用 calibration dataset、activation statistics、PPL feedback、threshold search、mixed-precision search 或 fine-tuning。

最終設定同時符合新版要求：WikiText-2 PPL 為 **14.6759**，Realized Effective Bit Rate 為 **5.25000007 bits ≤ 5.3 bits**。

| 設定 | WikiText-2 PPL | Effective bit rate |
|---|---:|---:|
| 原始 FP16（獨立控制實驗） | 13.6378 | 不適用 |
| 固定 NF4 W4A16（先前 weight-only 控制實驗） | 14.6063 | 10.25000007，不符合新版門檻 |
| **最終 NF4 W4A6** | **14.6759** | **5.25000007** |

依作業分級，最終 PPL ≤15，對應 performance 項最高級距。

## 1. 環境與一鍵重現

需求為 Python 3.11、PyTorch 2.5.1+cu121、單張 NVIDIA T4 16GB。首次執行需要網路下載原始模型與 WikiText-2；建議至少預留 30GB 暫存空間供 Hugging Face cache 與 fake-quantized checkpoint 使用。

在 `Part3/` 目錄執行：

```bash
bash run.sh
```

`run.sh` 會自動建立 `.venv`、安裝 `requirements.txt`，依序執行 FP16 控制實驗、W4A6 量化、checkpoint 保存／重載驗證，以及官方 WikiText-2 PPL 評估。無須事先提供 threshold 或 calibration 檔案。

已有符合版本的 Python 環境時可指定：

```bash
PART3_PYTHON=/absolute/path/to/python bash run.sh
```

也可分步執行：

```bash
python -u run_part3.py --mode fp16
python -u run_part3.py --mode nf4
```

已有 `artifacts/nf4/model/` 時，可只重新評估：

```bash
python -u run_part3.py --mode nf4 --eval-only
```

`--rows-per-chunk` 只控制暫存記憶體，不改變 64-value quantization block 邊界或結果。模型與資料已存在 cache 時，可加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。

## 2. 固定 W4A6 方法

量化範圍是 Qwen3-4B 的 36 個 decoder blocks 中全部 252 個 Linear projections：每層的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。依助教說明，未量化的 embedding、共享 `lm_head`、normalization 與其他非 Linear 元件不納入 effective-bit 計算。

### 2.1 Weight：NF4 4-bit

- 固定 16 個 NF4 非均勻 codepoints。
- 每 64 個連續 weights 為一組。
- 先以 FP32 計算 block absmax，再將 scale 以 FP16 儲存。
- weight 除以儲存後的 scale，映射至最近的 NF4 codepoint。
- 不使用 double quantization、clipping search 或逐層 precision selection。

### 2.2 Activation：symmetric INT6

- 每一個 target Linear 的輸入 activation 都使用固定 INT6 fake quantization。
- 每個 token 的最後一維按連續 64 個 values 分組，block 不跨 token。
- 每組以 FP32 計算 absmax，FP16 儲存 `scale = absmax / 31`。
- 固定整數範圍為 `[-31, 31]`：

```text
q = clamp(round(x / scale), -31, 31)
x_fake = q × scale
```

- activation scale 是依當下輸入動態計算，但 bit-width、block size、scale rule、rounding 與整數範圍在執行前已固定；它不使用 calibration data，也不依 target-model PPL 選參數。

### 2.3 為什麼是 calibration-free

`nf4.py` 中的 `POLICY` 是常數，並在 target model weights、dataset 或 evaluation result 被讀取前寫入 manifest。Weight quantization 期間禁止 target-model forward；activation quantizer 沒有蒐集或保存資料集統計，只對每次 Linear 輸入套用固定映射。WikiText-2 test 只在量化完成後交給原始 evaluator 計算 PPL，結果不會回傳調整 quantizer。

## 3. Realized Effective Bit Rate

所有 scale overhead 都列入。Qwen3-4B 的 252 個 target Linear 共有 3,633,315,840 個 weights；所有 Linear 每 token 的 input-feature 使用次數合計為 958,464。

### Weight effective bits

```text
NF4 codes             = 3,633,315,840 × 4  = 14,533,263,360 bits
FP16 block scales     =   56,770,560 × 16  =    908,328,960 bits
固定 NF4 codebook     =          16 × 32  =            512 bits
Weight total          =                         15,441,592,832 bits

Weight effective bits = 15,441,592,832 / 3,633,315,840
                      = 4.2500001409 bits
```

### Activation effective bits

每個 Linear input vector 都以 block size 64 分組，因此每 token 共 14,976 個 FP16 scales：

```text
INT6 activation codes = 958,464 × 6   = 5,750,784 bits
FP16 block scales     =  14,976 × 16  =   239,616 bits
Activation total      =                 5,990,400 bits

Activation effective bits = 5,990,400 / 958,464
                          = 6.25 bits
```

最終結果：

```text
Realized Effective Bit Rate
= (Weight effective bits + Activation effective bits) / 2
= (4.2500001409 + 6.25) / 2
= 5.2500000705 bits
<= 5.3 bits
```

程式會依實際 model structure 重新計算上述數字，並把完整分子、分母、scale 數量與門檻檢查寫入 `results/quantization_manifest.json` 和 `results/nf4_result.json`，不是只在文件中手動宣稱。

## 4. Fake quantization 說明

NVIDIA T4 沒有本方法對應的 NF4×INT6 kernel，因此依作業允許使用 fake quantization：

- NF4 weights 先映射成邏輯 4-bit code，再重建為 FP16 tensor 儲存與執行。
- INT6 activations 在每個 decoder Linear forward 前量化並重建成 FP16 tensor。
- checkpoint 重載後程式會重新安裝 252 個 activation fake-quant hooks。
- 固定 probe 的重載前後 logits 已驗證為 bitwise identical。

因此本實驗驗證的是 W4A6 數值格式對模型品質的影響，不宣稱實際 checkpoint 已 packed 成 4-bit，也不宣稱 T4 推論加速或真實檔案縮小。

## 5. 官方評估與輸出

`run_eval.py` 與 `evaluate.py` 保留作業提供的 PPL 流程：WikiText-2 test、sequence length 2048。量化 wrapper 只提供已完成 W4A6 fake quantization 的模型。

| 路徑 | 內容 |
|---|---|
| `results/fp16_result.json` | 原始 FP16 PPL 控制實驗 |
| `results/nf4_result.json` | W4A6 PPL、effective-bit accounting 與 evaluator hash |
| `results/quantization_manifest.json` | 固定 policy、252 層統計、bit-rate 分子分母、重載驗證 |
| `artifacts/nf4/model/` | 執行時產生的 FP16 fake-quantized checkpoint；不需放入提交 zip |
| `314551126_report.pdf` | 方法、結果與實驗觀察 |

最終已驗證結果：

```text
PPL: 14.6759
Realized Effective Bit Rate: 5.25000007
```

## 6. 來源

- QLoRA（Dettmers et al.）：NF4 codebook 的格式來源。
- bitsandbytes 0.48.1：NF4 固定常數的交叉核對來源；本程式不依賴它執行量化。
- 作業新版規格：fake quantization、calibration-free 要求、effective bit rate ≤5.3，以及 scale overhead 計算要求。
