# Case 2

方法與環境見 [Part 2 README](../README.md)。以下命令均在 **Part2/** 目錄執行。

## 完整四個 cases

```sh
bash 314551126_run.sh
```

入口會建立 `.venv`、安裝套件並執行四個 cases。執行完成後，彙整結果位於 `results/case2_result.json`，配置位於 `results/case2_manifest.json`。

## 單獨執行 Case 2

若尚未建立環境，先執行：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

使用已有環境重建並評估：

```sh
.venv/bin/python case2/case2_gptq_mixed.py --work-dir artifacts/case2 --mode mixed
```

單獨執行時，結果寫入 `artifacts/case2/mixed/result.json`，配置寫入 `artifacts/case2/mixed/manifest.json`。只有完整入口 `run_all.py` 會把結果彙整到根目錄的 `results/`。

已有相同 work-dir 的 checkpoint 時可加 `--eval-only`。壓縮率計數排除 lm_head；完整模型重建資訊仍包含輸出層。
