# Part 2：Qwen3-4B 四個壓縮案例

## 單一執行入口

需求：Python 3.11（含 venv）、一張 CUDA GPU（實測 T4 16GB），以及首次安裝依賴、下載原始模型與 WikiText-2 的網路。

```bash
cd Part2
bash run.sh
```

不需要先手動安裝套件。入口自動建立本地 `.venv`，從 `requirements.txt` 安裝依賴，依序完成四個 case 的校正、壓縮、保存／重載驗證與官方 PPL 評估。`bash 314551126_run.sh` 執行相同流程。若已有符合版本的環境，可指定 `PART2_PYTHON=/absolute/path/to/python bash run.sh`。

基礎環境固定 Python 3.11、torch 2.5.1+cu121；其餘套件版本見 requirements。從原始模型建立校正候選需要較長時間；建議預留至少150GB暫存磁碟及30GB RAM。執行只使用指定的一張 GPU，不進行推論參數 offloading。

`stdout` 最後只印：

```text
Case 1
Compression Ratio: ...%
PPL: ...

Case 2
Compression Ratio: ...%
PPL: ...

Case 3
Compression Ratio: 50.00000000%
PPL: ...

Case 4
Compression Ratio: ...%
PPL: ...
```

依賴安裝與階段進度寫至 stderr；各 case 的完整過程存於 `logs/caseN.log`。任一 case 未符合題目條件、重載不一致或儲存計數不符，程式會以非零狀態結束，不用舊結果冒充成功。

## 方法與結果

已完成的實測值見 `314551126_report.pdf` 及本文件末段。重新執行後，最新數據會寫入 `results/all_cases.json` 和各 case 的結果 JSON。

| Case | 方法 | 限制 |
|---|---|---|
| 1 | packed NF4＋校正選擇的 GPTQ3 | CR >75%、PPL <15 |
| 2 | packed GPTQ2／3＋NF4 head | CR ≥81%、PPL <20 |
| 3 | activation-weighted SVD＋INT8／FP16 rows | SVD CR ≥20%、overall CR 50%（容許49.5%～50.5%）、PPL <50 |
| 4 | 物理 MLP 剪枝＋packed GPTQ2／3＋NF4 head | 剪枝率 ≥20%、overall CR ≥85%、PPL <50 |

四個案例都從相同原始 `Qwen/Qwen3-4B` commit `1cfa9a7208912126459214e8b04321603b3df60c` 開始。Case 4 的 GPTQ 使用重新收集的剪枝後統計；不把 Case 2 的模型或統計當作來源。校正與候選選擇只讀 WikiText-2 train，test 僅供最後正式評估。

Case 1：32×512 fit、4×512 holdout tokens，group128、1% damping，候選為 MLP projections；其餘 Linear NF4＋double quant。Case 2：64×1024 fit、8×1024 holdout，group256、act-order、1% damping，依 block 輸出誤差增量／實際節省 bytes 配置 INT2／3。

Case 3：64×1024 fit、8×1024 holdout；MLP 加權分解 rank 從1024起、主要步長64，先滿足 SVD CR ≥22.27%，再配置 INT8／FP16 rows 精確達到總大小一半。每個 INT8 row 的 FP16 scale 均計入，不靠無用途 padding 湊大小。

Case 4：同樣64×1024 fit、8×1024 holdout；按中間神經元輸出貢獻選擇寬度7424／7168／6912／6656／6400，進行閉式補償。同步刪除 gate/up rows 與 down columns，使實際權重數減少至少22.45%，再重新校正並量化。

## 真實儲存與壓縮率

依助教最後說明，原始目標為 **252個 decoder `nn.Linear`**，原始 FP16 weight／bias 共 **7,266,631,680 bytes**。分子與分母均排除 **lm_head、embedding、normalization**；完整模型仍保留輸出層。

共享權重應依實際 Linear 運算計數，不能只依儲存是否共用去重。本模型的 decoder projections 各計一次。

```text
CR = 1 - compressed_linear_bytes / original_linear_bytes
```

壓縮後分子計入同一組原始 decoder Linear 的全部 tensor payload：packed codes、scales、zero points、排列索引、codebooks、NF4 state、SVD factors 與 bias。head 的權重與 state 排除。儲存為 `model.safetensors`，格式資訊及張量形狀另存於 `manifest.json`。完整檔案的非 Linear tensors 與 header 大小另行記錄，不混用兩種範圍。

INT2 每4個 codes 存1 byte，INT3 每8個 codes 存3 bytes。推論僅暫時重建當前層的 FP16 權重，沒有持久保留全模型 FP16 副本。Case 3 直接執行低 rank 兩次 Linear，Case 4 使用實際縮小的矩陣；不以 fake quantization 或零值 mask 宣稱 Part 2 儲存壓縮。

## 官方評估與輸出

`evaluate.py` 與 `run_eval.py` 保留作業提供的 bytes。各 wrapper 只提供壓縮後模型與自行計算的 compression callback；PPL 使用未修改的 WikiText-2 test 流程，sequence length2048。統一入口會再次核對 evaluator SHA-256 與實際 safetensors tensor byte ranges。

- `results/caseN_result.json`：正式 PPL、CR、門檻與儲存檢查。
- `results/caseN_manifest.json`：模型來源、校正設定、精度／rank／寬度配置、重載結果。
- `artifacts/caseN/`：可重建的候選快取及最終 checkpoint；case1 直接存於此處，case2～4 最終模型在其 `mixed/` 子目錄。
- `caseN/`：各 case 的完整實作與原始 evaluator；單獨執行方式見各 case 的 README。

只有統一入口 `run_all.py` 會更新上述 `results/` 彙整檔。單獨執行 case 時，結果留在該 case 的 `artifacts/` 工作目錄，位置見各 case 的 README。

已有本次生成的四個 checkpoints 時，可只重新評估：

```bash
bash run.sh --eval-only
```

若模型、資料已快取，可設定 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。中斷後再次執行相同命令會重用已完成且 fingerprint 相同的校正候選，並重新建立最終 checkpoint／重跑 PPL。個別 case 程式支援 `--work-dir`；修改校正參數時請透過該參數指定新的工作目錄，以免混用統計。

提交包保留完成的 PDF 報告及重現程式。執行入口負責壓縮、官方評估，並在 `results/` 產生實測 JSON。

報告中的先前控制實驗也可重跑。在 `Part2/`，四個 case 的候選已建立後，依序執行：

```bash
.venv/bin/python case2/case2_gptq_mixed.py --work-dir artifacts/case2 --mode baseline3
.venv/bin/python case3/case3_svd.py --work-dir artifacts/case3 --mode svd
.venv/bin/python case4/case4_prune_gptq.py --work-dir artifacts/case4 --mode pruned
.venv/bin/python case4/case4_prune_gptq.py --work-dir artifacts/case4 --mode baseline3
```

控制實驗用於分析精度／剪枝取捨，部分設定本來就未達最終門檻；正式提交結果仍以 `run.sh` 的四個最終配置為準。

## 已確認的實測結果

目前程式已完成四個 case 的重新壓縮與完整官方評估：

| Case | 壓縮率 | PPL |
|---|---:|---:|
| 1 | 75.20874289% | 14.8505 |
| 2 | 81.87392902% | 16.1482 |
| 3 | 精確50% | 25.7291 |
| 4 | 86.27610508% | 18.8324 |

Case 3 的 SVD 壓縮率為22.27272727%；Case 4 的物理剪枝率為22.45670996%。全部條件通過。

`linear_accounting.py` 直接核對實際 safetensors header。`original_specs` 與 `formats` 保留完整重建資訊，所以仍有 lm_head；計數範圍由 helper 明確排除。
