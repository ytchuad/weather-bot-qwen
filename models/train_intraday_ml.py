# models/train_intraday_ml.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
MODEL_DIR = Path('models/intraday_ml')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def time_split(df):
    """按時間劃分訓練/驗證/測試集（fail-fast，絕不複用測試集）"""
    df = df.sort_values('datetime').copy()
    train = df[df['datetime'] < '2025-01-01']
    valid = df[(df['datetime'] >= '2025-01-01') & (df['datetime'] < '2026-01-01')]
    test = df[df['datetime'] >= '2026-01-01']

    if len(train) < 1000:
        raise ValueError(f"訓練集樣本數不足: {len(train)}，請檢查資料範圍。")
    if len(valid) < 1000:
        raise ValueError(f"驗證集樣本數不足: {len(valid)}，不可將測試集用作驗證。請調整日期或擴充資料。")
    if len(test) < 100:
        logger.warning(f"測試集樣本數較少: {len(test)}，最終評估可能不穩定。")

    logger.info(f"資料分割: train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test

def train_quantile_models(X_train, y_train, X_valid, y_valid, alphas, prefix, output_dir):
    """訓練多個分位數迴歸模型"""
    models = {}
    for alpha in alphas:
        logger.info(f"訓練 {prefix} quantile={alpha}")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            early_stopping_rounds=30,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        models[f'{prefix}_q{int(alpha*100)}'] = model
        model.booster_.save_model(str(output_dir / f'{prefix}_q{int(alpha*100)}.txt'))
    return models

def train_classifier(X_train, y_train, X_valid, y_valid, name, output_dir):
    """訓練二元分類器預測 is_upside_zero / is_downside_zero"""
    logger.info(f"訓練 {name} 分類器")
    model = lgb.LGBMClassifier(
        objective='binary',
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        early_stopping_rounds=30,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    model.booster_.save_model(str(output_dir / f'{name}.txt'))
    return model

def enforce_monotonicity(preds_dict, alpha_order):
    """強制分位數預測單調性 (p10 <= p25 <= p50 <= p75 <= p90)"""
    # 將預測結果組合成矩陣，然後逐行排序
    preds_matrix = np.column_stack([preds_dict[f'q{int(a*100)}'] for a in alpha_order])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(alpha_order):
        preds_dict[f'q{int(a*100)}'] = preds_matrix[:, i]
    return preds_dict

def main():
    logger.info(f"讀取訓練資料 {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    
    feature_cols = [
        'temp', 'max_so_far', 'temp_change_30min', 'temp_change_60min',
        'hour', 'minute', 'minutes_since_midnight', 'month', 'day_of_year',
        'day_sin', 'day_cos', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'max_bucket', 'time_since_max_so_far', 'forecast_tmax', 'forecast_tmin',
        'range_so_far', 'temp_change_120min'
    ]
    
    X = df[feature_cols]
    y_upside = df['remaining_upside']
    y_downside = df['remaining_downside']
    y_upside_zero = df['is_upside_zero']
    y_downside_zero = df['is_downside_zero']
    
    train, valid, test = time_split(df)
    X_train = train[feature_cols]
    y_train_u = train['remaining_upside']
    y_train_d = train['remaining_downside']
    X_valid = valid[feature_cols]
    y_valid_u = valid['remaining_upside']
    y_valid_d = valid['remaining_downside']
    X_test = test[feature_cols]
    
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # ---- 最高溫 upside ----
    logger.info("=== 訓練最高溫分位數模型 ===")
    upside_models = train_quantile_models(X_train, y_train_u, X_valid, y_valid_u, alphas, 'upside', MODEL_DIR)
    upside_clf = train_classifier(X_train, train['is_upside_zero'], X_valid, valid['is_upside_zero'], 'upside_zero', MODEL_DIR)
    
    # ---- 最低溫 downside ----
    logger.info("=== 訓練最低溫分位數模型 ===")
    downside_models = train_quantile_models(X_train, y_train_d, X_valid, y_valid_d, alphas, 'downside', MODEL_DIR)
    downside_clf = train_classifier(X_train, train['is_downside_zero'], X_valid, valid['is_downside_zero'], 'downside_zero', MODEL_DIR)
    
    # ---- 測試集評估 ----
    logger.info("=== 測試集評估 ===")
    # 預測並套用單調性
    upside_test_preds = {}
    for a in alphas:
        model = upside_models[f'upside_q{int(a*100)}']
        upside_test_preds[f'q{int(a*100)}'] = model.predict(X_test)
    upside_test_preds = enforce_monotonicity(upside_test_preds, alphas)
    
    # 計算覆蓋率等指標
    test_actual = test['remaining_upside'].values
    in_80_interval = (test_actual >= upside_test_preds['q10']) & (test_actual <= upside_test_preds['q90'])
    coverage_80 = in_80_interval.mean()
    mae_q50 = np.mean(np.abs(test_actual - upside_test_preds['q50']))
    
    logger.info(f"最高溫 80% 區間覆蓋率: {coverage_80:.3f} (目標 0.80)")
    logger.info(f"最高溫 Q50 MAE: {mae_q50:.3f} °C")
    
    # 檢查凌晨預測
    mask_midnight = test['hour'] <= 2
    if mask_midnight.any():
        sample = test[mask_midnight].iloc[0]
        preds = {}
        for a in alphas:
            preds[f'q{int(a*100)}'] = upside_models[f'upside_q{int(a*100)}'].predict(sample[feature_cols].values.reshape(1, -1))[0]
        logger.info(f"凌晨預測範例: max_so_far={sample['max_so_far']:.1f}, forecast_tmax={sample['forecast_tmax']:.1f}, pred_tmax_p50={sample['max_so_far'] + preds['q50']:.1f}")
    
    # 儲存特徵清單
    with open(MODEL_DIR / 'feature_list.json', 'w') as f:
        json.dump(feature_cols, f)
    
    logger.info("✅ 所有模型已儲存至 models/intraday_ml/")

if __name__ == '__main__':
    main()