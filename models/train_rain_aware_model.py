# models/train_rain_aware_model.py
"""
比較基線模型 vs. 降雨感知模型，聚焦於暴雨/降溫情境。
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
MODEL_DIR_BASE = Path('models/intraday_ml_base')
MODEL_DIR_RAIN = Path('models/intraday_ml_rain')
REPORT_PATH = Path('reports/rain_model_comparison.json')
MODEL_DIR_BASE.mkdir(parents=True, exist_ok=True)
MODEL_DIR_RAIN.mkdir(parents=True, exist_ok=True)

# 特徵清單（從中央 schema 載入）
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.feature_schema import get_feature_list

BASELINE_FEATURES = get_feature_list("baseline")
_RAIN_AWARE_FEATURES = get_feature_list("rain_aware")
RAIN_FEATURES = _RAIN_AWARE_FEATURES[len(BASELINE_FEATURES):]  # 只取降雨相關增量

def get_split(df):
    train = df[df['datetime'] < '2025-01-01']
    valid = df[(df['datetime'] >= '2025-01-01') & (df['datetime'] < '2026-01-01')]
    test = df[df['datetime'] >= '2026-01-01']
    if len(valid) < 1000 or len(test) < 100:
        raise ValueError("驗證/測試集太小，檢查資料分割日期。")
    return train, valid, test

def train_quantile_models(X_train, y_train, X_valid, y_valid, output_dir, prefix):
    models = {}
    for alpha in [0.10, 0.25, 0.50, 0.75, 0.90]:
        q = int(alpha * 100)
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=alpha,
            max_depth=6, num_leaves=31, learning_rate=0.05,
            n_estimators=500, early_stopping_rounds=30,
            random_state=42, verbose=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        path = output_dir / f'{prefix}_q{q}.txt'
        model.booster_.save_model(str(path))
        models[f'{prefix}_q{q}'] = model
    return models

def evaluate_on_slice(models, X, y, slice_mask, slice_name, prefix):
    if slice_mask.sum() < 10:
        return None
    X_slice = X[slice_mask]
    y_slice = y[slice_mask]
    preds = {}
    for q in [10, 25, 50, 75, 90]:
        preds[f'q{q}'] = models[f'{prefix}_q{q}'].predict(X_slice)
    # 排序確保單調性
    stacked = np.column_stack([preds[f'q{q}'] for q in [10,25,50,75,90]])
    stacked.sort(axis=1)
    for i, q in enumerate([10,25,50,75,90]):
        preds[f'q{q}'] = stacked[:, i]
    mae = np.mean(np.abs(y_slice - preds['q50']))
    coverage = np.mean((y_slice >= preds['q10']) & (y_slice <= preds['q90']))
    # 關鍵：預測 remaining_upside > 0 但實際為 0 的誤判率
    false_pos_mask = (preds['q50'] > 0.1) & (y_slice == 0)
    false_pos_rate = false_pos_mask.sum() / len(y_slice) if len(y_slice) > 0 else 0
    return {
        'slice': slice_name,
        'count': int(slice_mask.sum()),
        'mae': round(mae, 4),
        'coverage': round(coverage, 4),
        'false_pos_rate': round(false_pos_rate, 4)
    }

def main():
    df = pd.read_parquet(DATA_PATH)
    train, valid, test = get_split(df)

    # 共用 y
    y_train_u = train['remaining_upside']
    y_valid_u = valid['remaining_upside']

    # ---------- 基線模型 ----------
    logger.info("訓練基線模型...")
    X_train_base = train[BASELINE_FEATURES].fillna(0)
    X_valid_base = valid[BASELINE_FEATURES].fillna(0)
    base_models = train_quantile_models(X_train_base, y_train_u, X_valid_base, y_valid_u, MODEL_DIR_BASE, 'upside')

    # ---------- 降雨感知模型 ----------
    logger.info("訓練降雨感知模型...")
    rain_feature_list = BASELINE_FEATURES + RAIN_FEATURES
    X_train_rain = train[rain_feature_list].fillna(0)
    X_valid_rain = valid[rain_feature_list].fillna(0)
    rain_models = train_quantile_models(X_train_rain, y_train_u, X_valid_rain, y_valid_u, MODEL_DIR_RAIN, 'upside')

    # ---------- 情境評估 ----------
    X_test_base = test[BASELINE_FEATURES].fillna(0)
    X_test_rain = test[rain_feature_list].fillna(0)
    y_test = test['remaining_upside'].values

    slices = {
        'all_data': np.ones(len(test), dtype=bool),
        'rain_present': test['rainfall_60m_filled'] > 0,
        'heavy_rain': test['rainfall_60m_filled'] >= 10,
        'post_peak_rain': test['post_peak_rain_flag'] == 1,
        'morning_peak_rain': test['morning_peak_then_rain_flag'] == 1,
        'no_rain': test['rainfall_60m_filled'] == 0,
    }

    report = {'base': {}, 'rain': {}}
    for version, models, X_test in [('base', base_models, X_test_base), ('rain', rain_models, X_test_rain)]:
        for name, mask in slices.items():
            metrics = evaluate_on_slice(models, X_test, y_test, mask, name, 'upside')
            if metrics:
                report[version][name] = metrics

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 摘要
    base_all = report['base'].get('all_data', {})
    rain_all = report['rain'].get('all_data', {})
    logger.info(f"基線整體 MAE: {base_all.get('mae', 'N/A')}, 覆蓋率: {base_all.get('coverage', 'N/A')}")
    logger.info(f"降雨整體 MAE: {rain_all.get('mae', 'N/A')}, 覆蓋率: {rain_all.get('coverage', 'N/A')}")
    post_peak_base = report['base'].get('post_peak_rain', {})
    post_peak_rain = report['rain'].get('post_peak_rain', {})
    logger.info(f"午後高峰降雨情境 - 基線 False Pos: {post_peak_base.get('false_pos_rate','N/A')}, 降雨模型 False Pos: {post_peak_rain.get('false_pos_rate','N/A')}")

if __name__ == '__main__':
    main()