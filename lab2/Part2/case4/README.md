# Case 4

方法與環境見 [Part 2 README](../README.md)。以下命令均在 **Part2/** 目錄執行。

## 完整四個 cases

```sh
bash 314551126_run.sh
```

入口會建立 `.venv`、安裝套件，並以 `frozen/case4_manifest.json` 固定的 pruning widths、補償方式與 INT2／INT3 allocation 執行，不重新搜尋配置。執行完成後，彙整結果位於 `results/case4_result.json`，配置位於 `results/case4_manifest.json`。

## 單獨執行 Case 4

若尚未建立環境，先執行：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

使用已有環境重建並評估：

```sh
.venv/bin/python case4/case4_prune_gptq.py --work-dir artifacts/case4 --mode mixed
```

單獨執行時，結果寫入 `artifacts/case4/mixed/result.json`，配置寫入 `artifacts/case4/mixed/manifest.json`。只有完整入口 `run_all.py` 會把結果彙整到根目錄的 `results/`。

已有相同 work-dir 的 checkpoint 時可加 `--eval-only`。壓縮率計數排除 lm_head；完整模型重建資訊仍包含輸出層。

若要重新測試多種 pruning widths、補償方式與 INT2／INT3 allocation，可在命令後加 `--search`；預設模式只建立固定配置需要的候選。
