# Weather Bot Qwen — ChatGPT 審計說明書

## 專案簡介

**Weather Bot Qwen** 是一個針對香港天文台（HKO）每日最高/最低氣溫的**概率預測研究系統**，並整合了 Polymarket 預測市場的**模擬交易（paper trading）層**。

> ⚠️ 本 zip **只包含原始碼與設定檔**，不包含任何資料（parquet、csv、grib、db 等）。下面會說明有哪些資料被排除，方便你推理。

---

## 目錄結構與用途

| 目錄 | 大小(來源) | 用途 |
|---|---|---|
| `features/` | 27 .py files | 特徵工程：從原始資料建構 ML 訓練集 |
| `models/` | ~30 .py + 模型權重 | 模型訓練（XGBoost/LightGBM）、推理、校準、評估 |
| `execution/` | 20+ .py | 交易引擎：Kelly 投注、CLOB 執行、策略閘道、投資組合管理 |
| `execution/gates/` | 5 .py | 策略閘道管線（entry/exit/sizing/rebalance） |
| `execution/ensemble/` | 策略組合 | 多模型融合引擎 |
| `layer_a/` | 18 .py | Layer A：每分鐘天氣/市場資料捕捉管線（capture → schema → storage → export → upload） |
| `inference/` | 3 .py | 即時推理（model 2a adapters + realtime base） |
| `app/` | 14 API + 19 frontend | FastAPI Web 儀表板 + React TypeScript 前端 |
| `scripts/` | 15 .py + 3 .ps1 | 輔助工具（HF 同步、校準、clob replay、diagnostic） |
| `tests/` | 12 .py | 單元測試與整合測試 |
| `tools/` | 2 .py | Smoke test + leakage audit |
| `config/` | 7 files | YAML/JSON 策略與特徵設定 |
| `docs/` | 12 .md | 設計文件與實作報告 |
| `data/` | **僅 .py** | 資料管線腳本（所有資料排除） |
| `.github/workflows/` | 3 .yml | CI/CD: hourly update, daily update, run strategies |

---

## 排除的資料檔案清單

以下原始資料已被排除，僅留說明供你參考架構。

### 1. HKO 觀測與預報 Parquet (~250 MB)

| 檔案 | 說明 |
|---|---|
| `hko_tmax_historical.parquet` | 歷史每日最高氣溫觀測 |
| `hko_history.parquet` | 完整歷史天氣觀測數據 |
| `hko_rainfall_15min.parquet` | 15 分鐘累積雨量 |
| `intraday_hko_10min.parquet` | 每 10 分鐘即時溫度觀測 |
| `daily_forecast_all.parquet` / `daily_forecast_clean.parquet` | HKO 每日九天天氣預報 |
| `forecastlog_hko.parquet` | 預報日誌 |
| `hk_weather_raw/` | 原始逐日天氣資料（parquet） |

### 2. ECMWF 氣象模式資料 (~95 MB，`data/ecmwf_ens/`)

| 檔案 | 說明 |
|---|---|
| `latest_ens_raw.grib` | ECMWF 集合預報原始 GRIB 檔 (~94 MB) |
| `latest_ens_hko.nc` | 轉換後的 NetCDF（HKO 格點抽取） |
| `daily_tmax.nc` | 每日最高溫 NetCDF |

### 3. 特徵儲存 Parquet (~1.5 GB)

這些是已產生的 ML 訓練集與特徵表：

| 檔案 | 大小 |
|---|---|
| `intraday_minute_ml_features.parquet` | 228 MB |
| `intraday_minute_ml_features_tmin_d.parquet` | 328 MB |
| `intraday_minute_ml_features_tmin_e.parquet` | 95 MB |
| `weather_minute_wide.parquet` | 95 MB |
| `wind_features_10min.parquet` | 127 MB |
| `feature_store_enhanced.parquet` | 13 MB |
| `model_2a/2b/3a/3b/4_feature_store.parquet` | 各 ~26-30 MB |
| 及其他特徵 parquet... | |

### 4. Layer A 每分鐘快照資料 (~Hundreds MB)

- `data/layer_a/` — 每分鐘整層捕捉快照（Hive 分割 `date=YYYY-MM-DD/`）
- `data/layer_a_weather/` — Layer A 天氣子集
- `data/layer_a_market/` — Layer A 市場子集
- `data/layer_a_quality/` — Layer A 品質檢查

### 5. 策略快照匯出 (`data/export/`, ~446 MB)

每日 `YYYY-MM-DD.csv` 格式的策略執行記錄，從 2026-06-29 到今日，每天約 20-50 MB。

### 6. 模型權重 (`models/*/` 下的 .txt, ~100 MB)

每個模型子目錄含 LightGBM 的文字權重檔（如 `upside_q10.txt`、`downside_q25.txt`），大小約 500KB ~ 3MB。本 zip 已包含這些權重。

### 7. SQLite 策略資料庫（~430 MB）

`data/strategy_snapshots.db` + WAL + SHM — 策略執行快照記錄。

### 8. 舊歸檔資料

- `archive_training_data/` (147 MB) — 舊版訓練資料備份
- `hk-temperature-layer-a/` (263 MB) — Layer A HF 匯出備份

### 9. 其他排除

- `node_modules/` (~218 MB, npm packages)
- 所有 `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`
- `.git/` (Git 歷史)
- `reports/` (審計報表，約 1 MB)
- 根目錄的 `hk-temperature-layer-a.zip` (180 MB)

---

## 資料流摘要（給 ChatGPT 的架構參考）

```
HKO 原始 API / 爬蟲
    │
    ├──→ data/*.py (download / scrape)
    │         ↓
    ├──→ data/raw/ (*.csv)          ← 原始資料（zip 外）
    ├──→ data/ecmwf_ens/ (grib/nc)  ← 氣象模式（zip 外）
    │         ↓ (features/build_*.py)
    ├──→ data/*_feature_store.parquet  ← 特徵表（zip 外）
    │         ↓ (models/train_*.py)
    ├──→ models/*/ (model weights)  ← 模型權重（zip 內，.txt）
    │         ↓
    ├──→ inference/*.py             ← 即時推理
    │         ↓ 預測結果
    ├──→ execution/*.py             ← 策略引擎 + Kelly 投注
    │         ↓
    ├──→ app/api/*.py               ← REST API
    │         ↓
    └──→ app/frontend/src/*.tsx     ← React 儀表板
```

另外有一個 **Layer A** 平行管線：

```
Layer A (layer_a/*.py)
  capture → schema → storage → export → upload
  │
  ├── data/layer_a/ (每分鐘天氣快照)
  ├── data/layer_a_market/ (每分鐘市場快照)
  └── data/layer_a_weather/ (每分鐘天氣子集)
```

---

## 檔案數量統計（zip 內）

類別僅供參考：

| 類別 | 約略數量 |
|---|---|
| Python (.py) | ~226 個 |
| TypeScript/React (.tsx/.ts/.css) | ~19 個 |
| 設定檔 (.yaml/.json/.toml) | ~15 個 |
| 文件 (.md/.txt) | ~25 個 |
| 模型權重 (.txt body) | ~100 個（~100 MB） |
| **zip 總大小** | **~100-200 MB**（主要是模型權重） |
| **不含權重 zip** | **~5 MB** |

---

## 建議分析順序

1. **`TECHNICAL_DOCUMENT_EN.md`** — 先讀技術概覽，了解整體系架構
2. **`CLAUDE.md`** — 專案規範與 HF 推送流程
3. **`features/` + `models/`** — 核心 ML 管線
4. **`execution/`** — 交易 / 策略層
5. **`layer_a/`** — 資料品質保證層
6. **`app/`** — 前端儀表板
7. **`scripts/` + `tests/` + `tools/`** — 工具與測試

---

*Generated on 2026-07-23 by Claude Code*
