# features/log_model_performance.py
"""
每日模型效能監控：計算前一天的最高溫/最低溫預測誤差，追加至效能日誌。
可由 GitHub Actions 排程執行，或手動執行。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import lightgbm as lgb
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
OBS_PATH = Path('data/hko_tmax_historical.parquet')
MODEL_DIR = Path('models/intraday_ml')
FEATURE_LIST = MODEL_DIR / 'feature_list.json'
LOG_PATH = Path('data/model_performance_log.parquet')

def load_models():
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f'upside_q{q}'] = lgb.Booster(model_file=str(MODEL_DIR / f'upside_q{q}.txt'))
        models[f'downside_q{q}'] = lgb.Booster(model_file=str(MODEL_DIR / f'downside_q{q}.txt'))
    return models

def main(target_date=None):
    if target_date is None:
        # 預設為昨天
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        target_date = pd.to_datetime(target_date).strftime('%Y-%m-%d')

    # 讀取日內資料並過濾該日期
    df = pd.read_parquet(DATA_PATH)
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df_day = df[df['date_str'] == target_date].copy()
    if df_day.empty:
        logger.warning(f"日期 {target_date} 無日內資料，跳過")
        return

    # 讀取官方實際值
    obs = pd.read_parquet(OBS_PATH)
    obs['date'] = pd.to_datetime(obs['date']).dt.date
    target_dt = pd.to_datetime(target_date).date()
    actual = obs[obs['date'] == target_dt]
    if actual.empty:
        logger.warning(f"無官方實際值 for {target_date}")
        return
    actual_tmax = actual['tmax'].values[0]
    actual_tmin = actual['tmin'].values[0]

    # 載入模型
    with open(FEATURE_LIST, 'r') as f:
        feature_cols = json.load(f)
    models = load_models()

    # 取最後一筆快照 (最接近結算的狀態)
    last_snapshot = df_day.iloc[-1:]
    X = last_snapshot[feature_cols].fillna(0)

    # 預測
    upside_q50 = models['upside_q50'].predict(X)[0]
    upside_q10 = models['upside_q10'].predict(X)[0]
    upside_q90 = models['upside_q90'].predict(X)[0]
    downside_q50 = models['downside_q50'].predict(X)[0]
    downside_q10 = models['downside_q10'].predict(X)[0]
    downside_q90 = models['downside_q90'].predict(X)[0]

    max_so_far = last_snapshot['max_so_far'].values[0]
    min_so_far = last_snapshot['min_so_far'].values[0]
    pred_tmax_p50 = max_so_far + upside_q50
    pred_tmin_p50 = min_so_far - downside_q50

    # 計算誤差指標
    error_tmax = pred_tmax_p50 - actual_tmax
    error_tmin = pred_tmin_p50 - actual_tmin

    # 覆蓋率（基於最後快照的區間）
    lower_tmax = max_so_far + upside_q10
    upper_tmax = max_so_far + upside_q90
    covered_tmax = 1 if lower_tmax <= actual_tmax <= upper_tmax else 0

    lower_tmin = min_so_far - downside_q90
    upper_tmin = min_so_far - downside_q10
    covered_tmin = 1 if lower_tmin <= actual_tmin <= upper_tmin else 0

    # 記錄
    record = {
        'date': target_date,
        'actual_tmax': actual_tmax,
        'actual_tmin': actual_tmin,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmin_p50': pred_tmin_p50,
        'error_tmax': error_tmax,
        'error_tmin': error_tmin,
        'abs_error_tmax': abs(error_tmax),
        'abs_error_tmin': abs(error_tmin),
        'covered_tmax_80': covered_tmax,
        'covered_tmin_80': covered_tmin,
        'snapshot_time': last_snapshot['datetime'].values[0],
        'max_so_far': max_so_far,
        'min_so_far': min_so_far,
        'upside_q50': upside_q50,
        'downside_q50': downside_q50
    }
    df_record = pd.DataFrame([record])

    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        # 避免重複日期
        existing = existing[existing['date'] != target_date]
        df_record = pd.concat([existing, df_record], ignore_index=True)
    df_record.to_parquet(LOG_PATH, index=False)
    logger.info(f"已記錄 {target_date} 的效能指標：Tmax error={error_tmax:.2f}, Tmin error={error_tmin:.2f}, 覆蓋 Tmax={covered_tmax}, Tmin={covered_tmin}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()