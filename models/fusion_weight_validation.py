# models/fusion_weight_validation.py (v3 – 修正缺失 tmax)
"""
融合權重驗證腳本（修正版）：
在最終溫度空間評估不同融合權重的表現，尋找逐小時最優權重。
從 hko_tmax_historical.parquet 合併每日官方最高溫以進行正確驗證。
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import logging
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
DAILY_PATH = Path('data/hko_tmax_historical.parquet')
MODEL_DIR = Path('models/intraday_ml')
FEATURE_LIST_PATH = MODEL_DIR / 'feature_list.json'
OUTPUT_WEIGHTS = Path('models/fusion_weight_table.parquet')
REPORT_PATH = Path('reports/fusion_validation_report.xlsx')

def load_intraday_models():
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f'upside_q{q}'] = lgb.Booster(model_file=str(MODEL_DIR / f'upside_q{q}.txt'))
    return models

def main():
    logger.info("讀取日內 ML 訓練資料...")
    df = pd.read_parquet(DATA_PATH)
    logger.info("讀取官方每日極值...")
    daily = pd.read_parquet(DAILY_PATH)
    daily['date_daily'] = pd.to_datetime(daily['date']).dt.date
    df['date_daily'] = df['datetime'].dt.date
    df = df.merge(daily[['date_daily', 'tmax']], on='date_daily', how='left')
    df_valid = df[df['datetime'] >= '2025-01-01'].copy()
    if len(df_valid) < 5000:
        split_idx = int(len(df) * 0.85)
        df_valid = df.iloc[split_idx:].copy()
        logger.warning("使用最後 15% 資料作為驗證集")

    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_cols = json.load(f)

    intra_models = load_intraday_models()
    X = df_valid[feature_cols].fillna(0)

    # 日內預測 remaining_upside 的分位數
    intra_q10 = intra_models['upside_q10'].predict(X)
    intra_q50 = intra_models['upside_q50'].predict(X)
    intra_q90 = intra_models['upside_q90'].predict(X)

    max_so_far = df_valid['max_so_far'].values
    intra_tmax_q50 = max_so_far + intra_q50
    intra_tmax_q10 = max_so_far + intra_q10
    intra_tmax_q90 = max_so_far + intra_q90
    intra_std = np.maximum((intra_tmax_q90 - intra_tmax_q10) / 2.56, 0.2)

    prior_mean = df_valid['forecast_tmax'].values
    prior_std = np.full_like(prior_mean, 1.2)  # 長期模型標準差（暫用固定值，可更換）

    actual_tmax = df_valid['tmax'].values
    hour_valid = df_valid['hour'].values

    weights = np.arange(0.0, 1.1, 0.1)
    results_by_hour = []

    for h in range(24):
        mask = hour_valid == h
        if mask.sum() < 20:
            continue
        best_weight = 1.0
        best_loss = np.inf
        best_coverage = None
        best_mae = None
        row = {'Hour': h, 'Count': mask.sum()}

        for w in weights:
            post_mean = w * prior_mean[mask] + (1 - w) * intra_tmax_q50[mask]
            post_std = w * prior_std[mask] + (1 - w) * intra_std[mask]

            log_loss = -np.mean(norm.logpdf(actual_tmax[mask], loc=post_mean, scale=post_std))
            lower = post_mean - 1.28155 * post_std
            upper = post_mean + 1.28155 * post_std
            coverage = np.mean((actual_tmax[mask] >= lower) & (actual_tmax[mask] <= upper))
            mae = np.mean(np.abs(actual_tmax[mask] - post_mean))

            if log_loss < best_loss:
                best_loss = log_loss
                best_weight = w
                best_coverage = coverage
                best_mae = mae

        row['Best_Prior_Weight'] = best_weight
        row['Best_LogLoss'] = best_loss
        row['Coverage_80'] = best_coverage
        row['MAE'] = best_mae
        post_mean_best = best_weight * prior_mean[mask] + (1 - best_weight) * intra_tmax_q50[mask]
        post_std_best = best_weight * prior_std[mask] + (1 - best_weight) * intra_std[mask]
        row['Mean_MAE'] = np.mean(np.abs(actual_tmax[mask] - post_mean_best))
        row['Std_Mean'] = np.mean(post_std_best)
        row['Coverage_80_check'] = np.mean((actual_tmax[mask] >= post_mean_best - 1.28155 * post_std_best) &
                                          (actual_tmax[mask] <= post_mean_best + 1.28155 * post_std_best))
        results_by_hour.append(row)

    df_hourly = pd.DataFrame(results_by_hour).set_index('Hour')
    df_hourly.to_parquet(OUTPUT_WEIGHTS)
    with pd.ExcelWriter(REPORT_PATH) as writer:
        df_hourly.to_excel(writer, sheet_name='Fusion_Weights')

    logger.info(f"逐小時最優權重已存至 {OUTPUT_WEIGHTS} 與 {REPORT_PATH}")
    logger.info("凌晨 0-5 點建議權重 (長期模型佔比):")
    logger.info(df_hourly.loc[0:5, ['Best_Prior_Weight', 'Coverage_80', 'MAE']].to_string())

if __name__ == '__main__':
    main()