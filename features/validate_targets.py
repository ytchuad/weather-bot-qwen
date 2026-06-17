# features/validate_targets.py
"""驗證訓練集中 remaining_upside / remaining_downside 的計算正確性"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ML_TRAIN = Path('data/intraday_ml_train.parquet')
LONG_TRAIN_MAX = Path('data/training_set_max.parquet')
LONG_TRAIN_MIN = Path('data/training_set_min.parquet')
OBS_PATH = Path('data/hko_tmax_historical.parquet')

def check_remaining_targets(df, max_so_far_col, official_tmax_col, target_col, zero_col, label):
    """檢查 remaining target 的計算"""
    if official_tmax_col not in df.columns:
        logger.warning(f"{label}: 缺少 {official_tmax_col}，無法驗證。")
        return
    # 重新計算
    expected = (df[official_tmax_col] - df[max_so_far_col]).clip(lower=0)
    diff = (df[target_col] - expected).abs()
    max_diff = diff.max()
    neg_count = (df[target_col] < 0).sum()
    missing_count = df[target_col].isna().sum()
    zero_match = ((df[target_col] <= 0) == (df[zero_col] == 1)).mean()
    logger.info(f"{label}: max_diff={max_diff:.6f}, 負值數量={neg_count}, 缺失={missing_count}, zero匹配率={zero_match:.4f}")
    if max_diff > 0.01 or neg_count > 0:
        logger.error(f"{label}: 目標計算可能有誤！")
    else:
        logger.info(f"{label}: ✅ 目標計算正確")

def main():
    # 日內 ML 訓練集
    if ML_TRAIN.exists():
        df = pd.read_parquet(ML_TRAIN)
        if 'tmax' in df.columns:
            check_remaining_targets(df, 'max_so_far', 'tmax', 'remaining_upside', 'is_upside_zero', 'intraday_upside')
        if 'tmin' in df.columns:
            check_remaining_targets(df, 'min_so_far', 'tmin', 'remaining_downside', 'is_downside_zero', 'intraday_downside')
    # 長期模型訓練集
    if LONG_TRAIN_MAX.exists():
        df_long = pd.read_parquet(LONG_TRAIN_MAX)
        # 長期模型沒有 remaining_*，但我們可檢查 tmax 本身
        logger.info(f"長期最高溫訓練集 tmax 缺失: {df_long['tmax'].isna().sum()}, 負值: {(df_long['tmax']<0).sum()}")
    if LONG_TRAIN_MIN.exists():
        df_long_min = pd.read_parquet(LONG_TRAIN_MIN)
        logger.info(f"長期最低溫訓練集 tmin 缺失: {df_long_min['tmin'].isna().sum()}, 負值: {(df_long_min['tmin']<0).sum()}")

if __name__ == '__main__':
    main()