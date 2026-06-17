# features/build_rainfall_features.py
"""將累積雨量轉為間隔雨量，並計算滾動降雨特徵"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAINFALL_PATH = Path('data/hko_rainfall_15min.parquet')
OUTPUT_PATH = Path('data/hko_rainfall_15min_features.parquet')

def main():
    df = pd.read_parquet(RAINFALL_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df['date'] = df['datetime'].dt.date

    # 將模糊的 'rainfall' 重新命名為精確語意名稱
    # 此欄位代表自午夜起的累積雨量（mm），每 15 分鐘更新一次
    df = df.rename(columns={"rainfall": "rainfall_accumulated_since_midnight"})

    # 間隔雨量（每 15 分鐘）
    df['rainfall_interval_15m'] = df.groupby('date')['rainfall_accumulated_since_midnight'].diff().fillna(df['rainfall_accumulated_since_midnight'])
    # 異常負值（可能是資料更正）→ 裁剪為 0 並記錄
    neg_mask = df['rainfall_interval_15m'] < 0
    if neg_mask.any():
        logger.warning(f"發現 {neg_mask.sum()} 筆負的間隔雨量，已裁剪為 0")
        df['rainfall_interval_15m'] = df['rainfall_interval_15m'].clip(lower=0)

    # 滾動累積雨量（以 15 分鐘為單位，分別取 30/60/120 分鐘）
    # 注意：rolling 只會使用當前時刻及之前的觀測
    for window_min in [30, 60, 120]:
        periods = window_min // 15
        col_name = f'rainfall_{window_min}m'
        df[col_name] = df.groupby('date')['rainfall_interval_15m'].transform(
            lambda x: x.rolling(periods, min_periods=1).sum()
        )

    # 滾動最大間隔雨量（捕捉降雨強度）
    for window_min in [30, 60]:
        periods = window_min // 15
        col_name = f'rainfall_max_{window_min}m'
        df[col_name] = df.groupby('date')['rainfall_interval_15m'].transform(
            lambda x: x.rolling(periods, min_periods=1).max()
        )

    # 當日累積雨量（即原始 rainfall，已存在，可留作對照）

    # 儲存
    df = df.drop(columns=['date'])
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"降雨特徵已儲存至 {OUTPUT_PATH}，欄位: {list(df.columns)}")
    logger.info(f"範例:\n{df.tail(10).to_string(index=False)}")

if __name__ == '__main__':
    main()