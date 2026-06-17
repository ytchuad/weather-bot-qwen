# features/build_intraday_data_quality_report.py
"""產出逐日資料品質報告，用於訓練前排除劣質日"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTRADAY_PATH = Path('data/intraday_hko_10min.parquet')
DAILY_PATH = Path('data/hko_tmax_historical.parquet')
OUTPUT = Path('reports/intraday_data_quality_report.csv')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(INTRADAY_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    daily['date'] = pd.to_datetime(daily['date']).dt.date
    df['date'] = df['datetime'].dt.date

    report = []
    for date, group in df.groupby('date'):
        row_count = len(group)
        expected = 144
        missing_rate = 1 - row_count / expected
        first = group['datetime'].min()
        last = group['datetime'].max()
        derived_max = group['temp'].max()
        derived_min = group['temp'].min()

        official = daily[daily['date'] == date]
        if not official.empty:
            off_tmax = official.iloc[0]['tmax']
            off_tmin = official.iloc[0]['tmin']
            diff_max = abs(off_tmax - derived_max)
            diff_min = abs(off_tmin - derived_min)
        else:
            off_tmax = off_tmin = diff_max = diff_min = np.nan

        status = 'good'
        if missing_rate > 0.15:
            status = 'bad'
        elif missing_rate > 0.05:
            status = 'watch'

        report.append({
            'date': date,
            'row_count': row_count,
            'expected_rows': expected,
            'missing_rate': round(missing_rate, 4),
            'first_obs': first,
            'last_obs': last,
            'official_tmax': off_tmax,
            'derived_tmax': derived_max,
            'diff_tmax': round(diff_max, 2) if not np.isnan(diff_max) else None,
            'official_tmin': off_tmin,
            'derived_tmin': derived_min,
            'diff_tmin': round(diff_min, 2) if not np.isnan(diff_min) else None,
            'status': status
        })

    df_report = pd.DataFrame(report).sort_values('date')
    df_report.to_csv(OUTPUT, index=False)
    logger.info(f"報告已存至 {OUTPUT}，總天數: {len(df_report)}，劣質天: {(df_report['status']=='bad').sum()}")

if __name__ == '__main__':
    main()