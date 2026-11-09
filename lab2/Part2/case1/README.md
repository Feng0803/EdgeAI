# Case 1

方法與環境見 [Part 2 README](../README.md)。以下命令均在 **Part2/** 目錄執行。

## 完整四個 cases

```sh
bash 314551126_run.sh
```

入口會建立 `.venv`、安裝套件，並以 `frozen/case1_manifest.json` 的固定 25 個 GPTQ3 layers 執行四個 cases，不重新搜尋 layer allocation。執行完成後，彙整結果位於 `results/case1_result.json`，配置位於 `results/case1_manifest.json`。

## 單獨執行 Case 1

若尚未建立環境，先執行：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

使用已有環境重建並評估：

```sh
.venv/bin/python case1/case1_gptq_mixed.py --work-dir artifacts/case1
```

單獨執行時，結果寫入 `artifacts/case1/result.json`，配置寫入 `artifacts/case1/manifest.json`。只有完整入口 `run_all.py` 會把結果彙整到根目錄的 `results/`。

已有相同 work-dir 的 checkpoint 時可加 `--eval-only`。壓縮率計數排除 lm_head；完整模型重建資訊仍包含輸出層。

若要重新建立全部候選並重做誤差排序與 layer selection，可在命令後加 `--search`；預設模式只重建已固定配置需要的候選。
