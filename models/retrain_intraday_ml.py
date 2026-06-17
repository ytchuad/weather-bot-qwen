# models/retrain_intraday_ml.py
"""自動重訓練日內 LightGBM 模型，比較效能後決定是否更新"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import logging
from datetime import datetime
from model_registry import register_new_version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
MODEL_DIR = Path('models/intraday_ml')
FEATURE_LIST = MODEL_DIR / 'feature_list.json'
OUTPUT_DIR = Path('models/temp_new_models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_single_quantile(X_train, y_train, X_val, y_val, alpha, name):
    model = lgb.LGBMRegressor(
        objective='quantile', alpha=alpha,
        max_depth=6, num_leaves=31, learning_rate=0.05,
        n_estimators=500, early_stopping_rounds=30,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    path = OUTPUT_DIR / f'{name}.txt'
    model.booster_.save_model(str(path))
    # 計算 pinball loss
    preds = model.predict(X_val)
    diff = y_val - preds
    pinball = np.where(diff >= 0, alpha * diff, (alpha - 1) * diff).mean()
    return pinball

def main():
    # 載入資料
    df = pd.read_parquet(DATA_PATH)
    # 排除劣質日
    quality_path = Path('reports/intraday_data_quality_report.csv')
    if quality_path.exists():
        bad_dates = pd.read_csv(quality_path)
        bad_dates = bad_dates[bad_dates['status'] == 'bad']['date'].tolist()
        bad_dates = pd.to_datetime(bad_dates).date
        df = df[~df['date'].isin(bad_dates)]

    # 時間分割 (最後一年作為驗證，之前作為訓練)
    df = df.sort_values('datetime')
    split_date = df['datetime'].max() - pd.DateOffset(years=1)
    train = df[df['datetime'] <= split_date]
    val = df[df['datetime'] > split_date]

    with open(FEATURE_LIST, 'r') as f:
        features = json.load(f)

    X_train = train[features].fillna(0)
    y_train_upside = train['remaining_upside']
    y_train_downside = train['remaining_downside']
    X_val = val[features].fillna(0)
    y_val_upside = val['remaining_upside']
    y_val_downside = val['remaining_downside']

    metrics = {'pinball_loss': {}}
    artifacts = []

    # 訓練 upside 分位數模型
    for q in [10, 25, 50, 75, 90]:
        alpha = q / 100.0
        loss = train_single_quantile(X_train, y_train_upside, X_val, y_val_upside, alpha, f'upside_q{q}')
        metrics['pinball_loss'][f'upside_q{q}'] = loss
        artifacts.append(str(OUTPUT_DIR / f'upside_q{q}.txt'))

    # 訓練 downside 分位數模型
    for q in [10, 25, 50, 75, 90]:
        alpha = q / 100.0
        loss = train_single_quantile(X_train, y_train_downside, X_val, y_val_downside, alpha, f'downside_q{q}')
        metrics['pinball_loss'][f'downside_q{q}'] = loss
        artifacts.append(str(OUTPUT_DIR / f'downside_q{q}.txt'))

    # 計算整體覆蓋率（簡化，僅用 q50 示範）
    model_q50 = lgb.Booster(model_file=str(OUTPUT_DIR / 'upside_q50.txt'))
    preds = model_q50.predict(X_val)
    val['pred_tmax'] = val['max_so_far'] + preds
    # 需要 tmax 欄位，從合併取得（若無則跳過）
    if 'tmax' in val.columns:
        error = val['tmax'] - val['pred_tmax']
        mae = np.mean(np.abs(error))
        metrics['tmax_mae'] = mae
        logger.info(f"新模型 Tmax MAE: {mae:.3f}")

    # 註冊版本
    config = {
        'training_start': str(train['datetime'].min().date()),
        'training_end': str(train['datetime'].max().date()),
        'validation_period': f"{val['datetime'].min().date()} to {val['datetime'].max().date()}"
    }
    promoted, entry = register_new_version('intraday_tmax_upside_lgbm', metrics, artifacts, config)
    if promoted:
        logger.info("New model is better than active. Use promote_model.py to promote:")
        logger.info("  python models/promote_model.py --candidate-dir %s", entry.get('artifact_path', str(OUTPUT_DIR)))
        # DO NOT copy directly to MODEL_DIR — promotion must go through promote_model.py
        # which validates: leakage audit, smoke test, feature consistency, backup, registry
    else:
        logger.info("New model did not beat active model. Keeping current active model.")

if __name__ == '__main__':
    import shutil
    main()