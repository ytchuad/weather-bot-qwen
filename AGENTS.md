# AGENTS.md — Weather Bot Qwen

Probabilistic HKO temperature forecasting + intraday nowcasting research system,
with an optional paper-trading simulation layer. Default mode is **paper trading**;
no real-money/exchange logic is active by default.

## Repository layout (read these)

| Path | Role | Agent note |
|------|------|-----------|
| `app/` | Streamlit dashboard: `pages/` (UI), `services/` (data/market/model/strategy services), `config.py` (path constants), `frontend/` (React, build-only) | Primary runtime app. Start here for UI/serving work. |
| `inference/` | **Realtime inference entry point.** `realtime_inference_base.py` (shared loader/guardrails) + `model_2a_realtime_inference.py` (concrete model). Loads model specs + weights from `models/<model_dir>/`. | This is the HF-inference surface. |
| `models/` | Per-model spec dirs (`feature_list.json`, `best_threshold.json`, `*.json` calibrations) + training scripts (`train_*.py`, `promote_model.py`, `validate_*.py`). **Only spec/weight files are used at inference**; `oot_predictions.parquet` etc. were moved out (see below). | Edit specs here; don't recreate deleted training outputs. |
| `data/` | Runtime data (see `app/config.py` for exact paths) + build/scraper scripts (`build_*.py`, `scrape_*.py`, `download_*.py`). | Only runtime parquets/json here are live. |
| `config/` | `paper_strategies.json`, `portfolios.json` (runtime portfolio config, gitignored at deploy). | |
| `archive_training_data/` | **GITIGNORED. Training data + model outputs, kept for retraining only.** | **DO NOT read or edit files here.** They are not part of the agent's context. See its README. |
| `tests/`, `scripts/`, `tools/`, `execution/`, `monitoring/`, `reports/`, `backtest/`, `features/`, `dash/` | Supporting modules. | |

## Inference entry points

- Realtime: `inference/model_2a_realtime_inference.py` (uses `models/intraday_minute_ml_model_2a_v2/`).
- Dashboard models served via `app/services/model_service.py` and `app/config.py` `MODEL_DIR_MAP`.
- Runtime data paths are centralized in `app/config.py` (`INTRADAY_10MIN_PATH`, `RAIN_15MIN_PATH`, `FORWARD_TEST_LOG`, `HISTORICAL_TEMP_PATH`, `PERF_LOG_PATH`, `TRADE_AUDIT_PATH`, …). **Do not move/delete these data files.**

## Common commands

- Run dashboard: `streamlit run streamlit_app.py` (or `dashboard.py`).
- Run inference: invoke `inference/model_2a_realtime_inference.py`.
- Tests / lint: `pytest` (if configured), `ruff` per `pyproject.toml`.
- Retraining: use `models/train_*.py` then `models/promote_model.py` (training data lives in `archive_training_data/`, gitignored).

## Hard rules for this repo

1. **Never read `archive_training_data/`** — it is gitignored training artifacts, not code.
2. **Never commit diagnostic dumps** (`project_text_dump.txt`, `project_source_dump.txt`) — gitignored on purpose.
3. **Do not push to the Hugging Face remote** (`hf`). GitHub remote only if explicitly needed.
4. **Don't delete runtime data files** listed in `app/config.py`.
5. Stale branches were archived as `archive/branch-*` tags; keep `main` as the active branch.
