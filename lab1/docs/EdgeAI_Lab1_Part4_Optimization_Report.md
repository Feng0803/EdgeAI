# EdgeAI Lab 1 — Part 4：CUDA SGEMM 最佳化報告

> **測試 GPU：** NVIDIA Tesla T4（Compute Capability 7.5，40 SM）  
> **CUDA：** 12.2；**目標架構：** `sm_75`；**資料精度：** FP32  
> **專案程式碼：** [Feng0803/EdgeAI — `feat/lab1-part4-padding`](https://github.com/Feng0803/EdgeAI/tree/feat/lab1-part4-padding)  
> **最終最佳化 Commit：** [`b0e1725`](https://github.com/Feng0803/EdgeAI/commit/b0e1725a0acfcd0dcdf7be25bee69b2255cb9b56)  
> **評分依據：** `scripts/grade.sh`，在 `4096×4096×4096` 上比較 Optimization Chain 的相鄰版本。  
> **狀態：** `scripts/check_all.sh` 最後為 **ALL PASSED**；`scripts/grade.sh` 回報 **Part 4 points: 10 / 10**。此為本地腳本結果，最終成績仍以課程驗收為準。

## 1. 目標與實驗方法

本次 Part 4 從 Part 3 表現最好的 `reg2d_128x128x8_8x8` 開始，在不使用 Tensor Cores、TF32 或第三方矩陣乘法核心函式庫的前提下，逐步實作三項最佳化。每個新 Kernel 保留前一版已完成的修改，只額外加入一個主要最佳化因素，使效能差異可由受控比較解釋。

運算定義為：

\[
C \leftarrow \alpha AB + \beta C,\quad
A\in\mathbb{R}^{M\times K},\;B\in\mathbb{R}^{K\times N},\;C\in\mathbb{R}^{M\times N}.
\]

矩陣皆為 Row-major FP32，計算使用 CUDA FP32 Core。所有 Part 4 Kernel 的 Tile 設定相同：

| 參數 | 設定 | 意義 |
|---|---:|---|
| `BM` | 128 | 每個 CTA 處理的 C Tile 行數 |
| `BN` | 128 | 每個 CTA 處理的 C Tile 列數 |
| `BK` | 8 | 每次 K 軸載入的深度 |
| `TM` | 8 | 每個 Thread 計算的輸出行數 |
| `TN` | 8 | 每個 Thread 計算的輸出列數 |
| Threads per CTA | 256 | `(BM/TM) × (BN/TN)` |

各版本的 Register Blocking 與 FMA 計算方式不變。對於支援的矩陣形狀，Launcher 直接執行各 Kernel；若 M、N、K 不滿足 Tile 整除條件，Launcher 回傳 `false`，由測試程式記錄 `SKIPPED`。這種行為與 Part 3 原始 Reg2D 的設計一致，並非運算錯誤。

本次 `kPart4Chain` 為：

```cpp
static const std::vector<const char*> kPart4Chain = {
    "reg2d_128x128x8_8x8",
    "custom_padA",
    "custom_padA_align16",
    "custom_padA_align16_swizzle8",
};
```

三項最佳化依序為：

1. **Opt1 — Shared Memory Padding +1：** 將轉置存放的 A Tile 改為 `As[BK][BM + 1]`，嘗試減少 Shared Memory Bank Conflicts。
2. **Opt2 — 16-byte Aligned Padding +4：** 以 `As[BK][BM + 4]` 取代 `+1` 的列距，使每一列保持 16-byte 對齊，嘗試兼顧 Bank Mapping 與 Memory Instruction 寬度。
3. **Opt3 — Thread Block Swizzling：** 保留 Opt2 的 Shared Memory Layout，僅改變 CTA 到輸出 Tile 的對應順序，對 M 方向的相鄰 8 個 Tile 進行分組。

## 2. 正確性與計分方法

### 2.1 正確性驗證

`bash scripts/check_all.sh` 在 15 種矩陣形狀與兩個 `beta` 設定（0、0.5）下測試全部註冊 Kernel。對 Part 4 的四個 Chain Entry，在所有支援形狀上均顯示 **PASS**，不支援形狀依 Launcher 限制顯示 **SKIPPED**；未觀察到 `FAIL`。

完整測試最後輸出如下：

```text
SANITIZER memcheck 1023 257 511: clean
SANITIZER memcheck 7 13 3: clean
SANITIZER memcheck 1000 1 1001: clean
SANITIZER memcheck 128 128 128: clean
SANITIZER memcheck 256 256 256: clean
SANITIZER racecheck 128 128 128: clean
SANITIZER racecheck 1023 257 511: clean
SANITIZER racecheck 256 256 256: clean
ALL PASSED
```

`grade.sh` 另對 15 種形狀 × 2 個 `beta` 做 Correctness 驗證，並檢查指定 Kernel 的 Compute Sanitizer。輸出顯示 Part 3 六個 Reg2D 配置均 PASS，且 Part 4 評分 Chain 可完成測量。

### 2.2 效能評分

`scripts/grade.sh` 的 Part 4 比較條件為：

- 矩陣形狀：`M=N=K=4096`。
- 每個版本進行 **3 rounds**，每輪 **20 次 timed launches**，使用當輪中位數；再取三輪中最佳的中位數。
- 每一步與前一個 **正確** 的版本比較；`speedup ≥ 1.05×` 得 5 分。未達門檻但報告有說明原因可得 2.5 分。Part 4 總分上限 10 分。

本文以 `grade.sh` 的數值作為**正式成績表**，不混用不同批次手動 Benchmark 的時間計算加速比。

## 3. Optimization 1：Shared Memory Padding +1

### 3.1 動機與實作

Part 3 的 Reg2D 將 A Tile 轉置後放在 Shared Memory：`As[k][m]` 保存 `A[m][k]`。其宣告為：

```cpp
// Baseline
__shared__ float As[BK][BM];

// Opt1：僅改變 A Tile 的 Physical Row Pitch
__shared__ float As[BK][BM + 1];
```

透過每列多保留一個 `float`，嘗試讓轉置寫入時的 Bank Mapping 不再集中在相同 Bank。有效資料依然只有 `BK×BM`，額外格子不參與計算；Global Memory Load、FMA 累積及輸出公式皆保持不變。

`BM=128` 時，每列由 128 個 float（512 B）變成 129 個 float（516 B）。

### 3.2 實測與失敗原因

`grade.sh` 中：

- Baseline：**46.941 ms**。
- Opt1：**49.248 ms**。
- Speedup：`46.941 / 49.248 = 0.953×`，相對原始版本慢約 **4.9%**（時間增加）。

Nsight Compute（**1024³**，與正式 4096³ 計時不同）顯示：

- Shared **Store** Bank Conflicts：1,835,008 → 786,432，確實下降約 57.1%。
- Shared **Load** Bank Conflicts：維持 4,194,304。
- Shared Load Instructions：2,097,152 → 5,242,880，增加 **150%**。
- 128-bit Memory Instructions：2,097,152 → 1,048,576，減少 50%。
- `Stall MIO Throttle`：0.78 → 1.65 cycles/issued instruction。
- Registers/thread：119 → 119；Theoretical Occupancy 均為 50%。

**分析：** `+1` 的列距無法維持每列 16-byte 對齊，且觀察到 Compiler 發出的 Shared Memory Load 指令變多。雖然 Store Bank Conflicts 減少，但額外的 Shared Memory 指令與 MIO Pipeline 壓力抵銷其效益，符合最終變慢的結果。這是由 Source-Level 受控修改與 NCU Counter 共同支持的解釋；沒有進行單一指令的因果隔離，因此不宣稱它是唯一因素。

**結論：** Opt1 為**正確但未加速**的嘗試。依 `grade.sh`，需要在報告解釋才有 **2.5 分**；本節即對其失敗原因提供量測證據。

## 4. Optimization 2：16-byte Aligned Shared Memory Padding +4

### 4.1 動機與實作

Opt2 仍使用 Padding，但將 A Tile 的 Physical Row Pitch 改為：

```cpp
// Opt1
__shared__ float As[BK][BM + 1];

// Opt2
__shared__ float As[BK][BM + 4];
```

因 `sizeof(float)=4 B`，當 `BM=128` 時，`(128+4)×4=528 B`，為 16 B 的整數倍。此設計讓每列起點仍可在 16-byte 邊界，避免 Opt1 的非對齊 Row Pitch；有效矩陣元素與 FMA 順序不變。

### 4.2 實測結果

`grade.sh` 中：

- Opt1：**49.248 ms**。
- Opt2：**43.685 ms**。
- Speedup：`49.248 / 43.685 = 1.127×`，通過 1.05× 門檻，**5 分**。

### 4.3 Nsight Compute 證據

在同樣的 **1024³** Profile 中，Opt2 相較 Opt1：

- Shared Store Bank Conflicts：786,432 → **0**。
- Shared Load Instructions：5,242,880 → **2,097,152**，恢復至原始 Reg2D 的水準。
- 128-bit Memory Instructions：1,048,576 → **2,097,152**。
- `Stall MIO Throttle`：1.65 → **0.38** cycles/issued instruction。
- Registers/thread：兩者皆 119；Theoretical Occupancy 皆 50%；Opt2 Achieved Occupancy 約 37.39%。

這些 Counter 與「恢復較寬的 Shared Memory 存取、減輕 MIO 壓力」的假設一致，能合理解釋 Opt2 在 4096³ 正式計時的加速。注意 Shared **Load** Bank Conflicts 仍為 4,194,304 次，因此 `+4` **沒有**消除所有 Shared Memory 衝突。

## 5. Optimization 3：Thread Block Swizzling（GROUP_M=8）

### 5.1 動機與實作

Opt2 的輸出 Tile 由 `blockIdx.y`、`blockIdx.x` 直接決定。Opt3 保留同一個 `BM=BN=128` 的 Tile 形狀、Register Blocking、Shared Memory Layout、FMA 次序與 CTA 數量；僅更改 `linearBlock → (tileM,tileN)` 的映射，以便在同一組內先處理 8 個相鄰 M Tiles，再切換 N Tile。

核心程式如下（詳見 `src/kernels/custom.cuh`）：

```cpp
constexpr int GROUP_M = 8;
const int tilesM = M / BM;
const int tilesN = N / BN;
const int linearBlock = blockIdx.y * gridDim.x + blockIdx.x;
const int groupId = linearBlock / (GROUP_M * tilesN);
const int firstM = groupId * GROUP_M;
const int remainingM = tilesM - firstM;
const int groupHeight = remainingM < GROUP_M ? remainingM : GROUP_M;
const int inGroup = linearBlock % (GROUP_M * tilesN);
const int tileM = firstM + inGroup % groupHeight;
const int tileN = inGroup / groupHeight;
```

在 M Tiles 數量無法被 8 整除時，`groupHeight` 處理最後一組不足 8 Tiles 的情形，保持完整而不重複的 Tile 對應。後續 `rowBase`、`colBase`、`outRow`、`outCol` 改由 `tileM`/`tileN` 計算。這個 Best-Effort CTA Ordering 可能改善多個 M Tiles 對同一 B Tile 的 L2 重用。

### 5.2 實測結果

`grade.sh` 中：

- Opt2：**43.685 ms**。
- Opt3：**39.780 ms**。
- Speedup：`43.685 / 39.780 = 1.098×`，通過 1.05× 門檻，**5 分**。

額外的三輪手動 Benchmark 亦觀察到相同方向：Opt2 平均約 **42.433 ms**，Opt3 平均約 **39.318 ms**，約 `1.079×`。手動量測與 `grade.sh` 使用的測量配置不同，因此本文不將兩者混合計算。

### 5.3 Nsight Compute：Opt2 與 Opt3 的 4096³ 硬體計數器對照

為了檢驗 Swizzling 是否改善資料重用率，針對 **Opt2** (`custom_padA_align16`) 與 **Opt3** (`custom_padA_align16_swizzle8`) 使用相同的 `M=N=K=4096` 進行 Nsight Compute Profiling：

```bash
sudo /usr/local/cuda-12.2/bin/ncu --set full --clock-control none -f -o reports/part4/ncu_custom_padA_align16_4096 ./sgemm custom_padA_align16 4096 4096 4096 --profile
sudo /usr/local/cuda-12.2/bin/ncu --set full --clock-control none -f -o reports/part4/ncu_custom_padA_align16_swizzle8_4096 ./sgemm custom_padA_align16_swizzle8 4096 4096 4096 --profile
```

| Nsight Compute 指標（4096³） | Opt2：Padding +4 | Opt3：+ Swizzle8 | 差異 |
|---|---:|---:|---|
| **L2 Hit Rate** | **59.22%** | **87.81%** | **+28.59 百分點** |
| **DRAM Bytes Read** | **2.75 GB** | **1.40 GB** | **約 −49.1%** |
| DRAM Bandwidth | 93.73 GB/s | 50.26 GB/s | 約 −46.4% |
| L2 Read Sectors (`lts__t_sectors_srcunit_tex_op_read.sum`) | 136,236,096 | 136,298,128 | 約 +0.046% |
| L2 Read Lookup Miss Sectors (`...lookup_miss.sum`) | 59,046,948 | 25,731,416 | 約 −56.4% |
| Shared Load Bank Conflicts | 268,435,456 | 268,435,456 | 不變 |
| Shared Store Bank Conflicts | 0 | 0 | 不變 |
| Shared Load Instructions | 134,217,728 | 134,217,728 | 不變 |
| 128-bit Memory Instructions | 134,217,728 | 134,217,728 | 不變 |
| Registers / Thread | 119 | 115 | −4 |
| Theoretical Occupancy | 50.00% | 50.00% | 不變 |
| Achieved Occupancy | 49.18% | 49.13% | −0.05 百分點 |
| Warp Stall Long Scoreboard（NCU 顯示值） | 1.13 | 1.03 | 下降 |
| Warp Stall MIO Throttle（NCU 顯示值） | 0.73 | 0.98 | 上升 |

**Cache Reuse 的證據。** 兩個版本的 L2 Read Sectors 幾乎相同，表示從 L2 觀察到的讀取需求量接近；但 Opt3 的 L2 Hit Rate 提升 28.59 個百分點，Read Lookup Miss Sectors 減少約 56.4%，對應的 DRAM 讀取資料量從 2.75 GB 降至 1.40 GB。這組計數器**強烈支持** CTA Swizzling 讓資料更常留在 L2 Cache，降低重複訪問 DRAM 的需求，符合在同一 N Tile 下處理相鄰 8 個 M Tiles 以重用 B Tile 的設計目的。

**並非所有指標改善。** Shared Memory Bank Conflicts、Shared Load Instructions 與 128-bit Memory Instructions 完全相同，且 Occupancy 接近，說明 Opt3 的加速不太可能是因為 Shared Memory 存取次數或 Occupancy 的主要改善。`Stall Long Scoreboard` 稍降，但 `Stall MIO Throttle` 反而升高；這些現象提醒我們不應把效能歸因於所有 Memory Stall 都下降。Registers/Thread 由 119 降至 115，可能與編譯器對不同 Index Mapping 的程式碼產生不同暫存器配置有關，但在目前資料下不能單獨判定其貢獻。

**量測限制。** 兩份 NCU 資料均來自 `--set full` 的多次 Kernel Replay，並使用 `--clock-control none`，故未固定 GPU 時脈；計數器也可能受 Profile Replay 與 Cache 狀態影響。上述分析是一組相同 Shape、相同 Profiling 設定下的硬體證據，能支持 Cache Reuse 的解釋，但不是每個加速來源的嚴格因果分離。**效能加速比仍使用獨立的 `grade.sh`**：43.685 ms → 39.780 ms，**1.098×**；不得用 NCU Profiler 中的 Duration 重新計算正式分數。

## 6. 綜合 Benchmark 與評分

### 6.1 `grade.sh` 正式結果（4096³）

| Step | Kernel | 時間 (ms) | 相對前一步 Speedup | 依腳本得分 |
|---|---|---:|---:|---:|
| Baseline | `reg2d_128x128x8_8x8` | 46.941 | — | — |
| Opt1 | `custom_padA` | 49.248 | 0.953× | 2.5（需解釋未加速原因） |
| Opt2 | `custom_padA_align16` | 43.685 | 1.127× | 5 |
| Opt3 | `custom_padA_align16_swizzle8` | **39.780** | **1.098×** | **5** |
| **總計** | | | | **10 / 10（上限）** |

該腳本原始輸出：

```text
== Part 4 (chain at 4096 4096 4096, best of 3 x median of 20)
   0 reg2d_128x128x8_8x8        46.941 ms  starting point
   1 custom_padA                49.248 ms  0.953x vs reg2d_128x128x8_8x8: failed (needs an explanation in the report), 2.5 pts
   2 custom_padA_align16        43.685 ms  1.127x vs custom_padA: successful, 5 pts
   3 custom_padA_align16_swizzle8   39.780 ms  1.098x vs custom_padA_align16: successful, 5 pts
   Part 4 points: 10 / 10
```

### 6.2 整體效能

由原始 Reg2D 到最終 Swizzle8：

\[
\mathrm{Overall\ Speedup}=\frac{46.941}{39.780}\approx\boxed{1.180\times}.
\]

相對原始版本的**執行時間降低**約 `15.3%`（`(46.941-39.780)/46.941`）；以速率定義的**運算速度提升**約 `18.0%`。兩種百分比的分母不同，不應混用。

對 `4096³` 矩陣乘法，理論運算工作量 `2MNK ≈ 137.439 GFLOP`；根據 `grade.sh` 的 Kernel 時間估算，Baseline 與 Opt3 的吞吐量約分別為 `2,928` 與 `3,455 GFLOP/s`。該數值從時間計算，非獨立的 Profiler 硬體計數器。

### 6.3 三項 Padding Layout 的 NCU 對照（1024³）

| NCU 指標 | Baseline | Opt1：Pad +1 | Opt2：Pad +4 |
|---|---:|---:|---:|
| Registers / Thread | 119 | 119 | 119 |
| Theoretical Occupancy | 50.00% | 50.00% | 50.00% |
| Achieved Occupancy | 38.34% | 37.23% | 37.39% |
| Static Shared Memory / CTA | 8,192 B | 8,224 B | 8,320 B |
| Shared Load Bank Conflicts | 4,194,304 | 4,194,304 | 4,194,304 |
| Shared Store Bank Conflicts | 1,835,008 | 786,432 | **0** |
| Shared Load Instructions | 2,097,152 | 5,242,880 | 2,097,152 |
| 128-bit Memory Instructions | 2,097,152 | 1,048,576 | 2,097,152 |
| MIO Throttle（cycles / issued inst.） | 0.78 | 1.65 | **0.38** |

**重要區分：** 第 6.1 節的 Speedup 由 4096³ 非 Profile Benchmark 計算；本節的硬體 Counter 來自 1024³ NCU Profile，兩種結果不應視為相同 workload 下的直接測量。三次 NCU Profile 均以 `--clock-control none` 執行，Profiler 提示時脈未固定，因此不使用 NCU Duration 取代正式 Benchmark。

## 7. 結論與討論

本次 Part 4 的主要發現如下：

1. **減少 Bank Conflicts 不保證加速。** Opt1 雖降低 Shared Store Bank Conflicts，但伴隨 Shared Load Instructions 大幅增加及 MIO Throttle 上升，實際執行時間反而變長。
2. **Padding 必須考慮資料對齊與編譯後指令。** Opt2 的 `+4` Padding 同時保留 16-byte Row Pitch Alignment、恢復原始數量的 Shared Load Instructions，並將 Shared Store Bank Conflicts 降至 0，成功取得 1.127× 相對加速。
3. **Thread Block Scheduling 可改善 L2 Cache Reuse。** Opt3 只改 CTA-to-Tile Mapping，即在 4096³ 達成額外 1.098× 加速。4096³ NCU 比較顯示 L2 Hit Rate 由 59.22% 升至 87.81%，DRAM Bytes Read 由 2.75 GB 降到 1.40 GB；Shared Memory 讀取指令數與 Bank Conflicts 不變。這組結果強烈支持改善 L2 資料重用是主要加速機制。
4. **程式正確性不應為效能犧牲。** 四個 Chain Entry 都通過涵蓋不同形狀與兩個 Beta 值的測試，Compute Sanitizer 的 Memcheck 與 Racecheck 亦均 Clean。

總結：在 Tesla T4 上，以原始 Reg2D 為起點、逐步加入三項最佳化後，最終 4096³ 執行時間由 46.941 ms 降到 39.780 ms，**整體 Speedup 1.180×**。本地 `grade.sh` 回報 **Part 4 points: 10 / 10**，其中包含對 Opt1 未加速結果的量測與原因解釋。

## 8. 實驗重現與證據

在遠端 GPU 主機的 `~/EdgeAI/lab1`：

```bash
make -B ARCH="-arch=sm_75"
./sgemm --chain
bash scripts/check_all.sh
bash scripts/grade.sh
```

由於 `logs/`、`reports/` 被 Git Ignore，實驗數據保留在遠端主機，程式碼則在 GitHub 的 `feat/lab1-part4-padding` 分支：

| 檔案 | 作用 |
|---|---|
| `src/kernels/custom.cuh` | Opt1、Opt2、Opt3 的 Kernel 與 Launcher |
| `src/kernels.cuh` | Kernel Registry 與 `kPart4Chain` |
| `scripts/grade.sh` | 1.05× 門檻及本地評分流程 |
| `logs/part4/check_all_opt3.txt` | 完整 Correctness + Sanitizer 原始輸出 |
| `logs/part4/grade_opt3.txt` | 4096³ 正式評分原始輸出 |
| `logs/part4/ncu_reg2d_128x128x8_8x8_1024.txt` | Baseline NCU 完整指標 |
| `logs/part4/ncu_custom_padA_1024.txt` | Opt1 NCU 完整指標 |
| `logs/part4/ncu_custom_padA_align16_1024.txt` | Opt2 NCU 完整指標（1024³） |
| `logs/part4/ncu_custom_padA_align16_4096.txt` | Opt2 NCU 完整指標（4096³） |
| `logs/part4/ncu_custom_padA_align16_swizzle8_4096.txt` | Opt3 NCU 完整指標（4096³） |
| `logs/part4/opt3_ncu_comparison.tar.gz` | Opt2 / Opt3 NCU 文字報告原始封存 |

**建議若需輸出課堂提交 PDF：** 附上最終 `grade.sh`（10/10）、`check_all.sh`（ALL PASSED）與上述 NCU 關鍵欄位的**真實終端／Profiler 截圖**。目前 Markdown 已提供全部數值與分析，截圖需從實際執行環境擷取，不能以自行繪製的結果圖片假冒硬體截圖。

**已完成的額外實驗（非取得 10/10 的必要條件）：** Opt2 與 Opt3 已在相同 4096³ Workload 下完成 NCU 比較，L2 Hit Rate、L2 Read Sectors、DRAM Bytes Read 及 Warp Stall 均列於第 5.3 節。這補足了 Swizzling 的硬體計數器證據。