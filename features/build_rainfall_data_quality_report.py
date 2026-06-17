# features/build_rainfall_data_quality_report.py
"""降雨資料品質報告：檢查 15 分鐘累積雨量的完整性與合理性"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAINFALL_PATH = Path('data/hko_rainfall_15min.parquet')
REPORT_CSV = Path('reports/rainfall_data_quality_report.csv')
SUMMARY_JSON = Path('reports/rainfall_data_quality_summary.json')

def main():
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 讀取資料
    df = pd.read_parquet(RAINFALL_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # 每日檢查
    records = []
    for dt, grp in df.groupby('date'):
        grp = grp.sort_values('datetime')
        row_count = len(grp)
        expected = 96  # 每 15 分鐘一筆，全日共 96 筆
        missing_rate = round(1 - row_count / expected, 4)

        first_time = grp['datetime'].min()
        last_time = grp['datetime'].max()
        rain_min = grp['rainfall'].min()
        rain_max = grp['rainfall'].max()
        final_rain = grp['rainfall'].iloc[-1] if row_count > 0 else np.nan

        # 負增量檢查
        diff = grp['rainfall'].diff()
        neg_inc_count = (diff < 0).sum()
        zero_inc_count = (diff == 0).sum()
        dup_count = grp['datetime'].duplicated().sum()

        # 判斷品質
        status = 'good'
        if missing_rate > 0.15:
            status = 'bad'
        elif missing_rate > 0.05 or neg_inc_count > 5:
            status = 'watch'

        records.append({
            'date': dt,
            'expected_rows': expected,
            'actual_rows': row_count,
            'first_obs_time': first_time,
            'last_obs_time': last_time,
            'missing_rate': missing_rate,
            'min_rainfall': rain_min,
            'max_rainfall': rain_max,
            'final_rainfall_since_midnight': final_rain,
            'negative_increment_count': neg_inc_count,
            'zero_increment_count': zero_inc_count,
            'duplicate_timestamp_count': dup_count,
            'data_quality_status': status
        })

    report = pd.DataFrame(records).sort_values('date')
    report.to_csv(REPORT_CSV, index=False)
    logger.info(f"每日品質報告已儲存至 {REPORT_CSV}")

    # 摘要 JSON
    total_days = len(report)
    bad_days = report[report['data_quality_status'] == 'bad']
    watch_days = report[report['data_quality_status'] == 'watch']
    summary = {
        'total_days': total_days,
        'bad_days': len(bad_days),
        'watch_days': len(watch_days),
        'max_missing_rate': report['missing_rate'].max(),
        'days_with_negative_increments': int((report['negative_increment_count'] > 0).sum()),
        'worst_dates_by_missing_rate': report.nlargest(5, 'missing_rate')['date'].astype(str).tolist(),
        'worst_dates_by_neg_inc': report.nlargest(5, 'negative_increment_count')['date'].astype(str).tolist()
    }
    import json
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"摘要已存至 {SUMMARY_JSON}")

    # 印出重要統計
    logger.info(f"總天數: {total_days}, 劣質天: {len(bad_days)}, 觀察天: {len(watch_days)}")
    if not bad_days.empty:
        logger.info(f"劣質日期:\n{bad_days[['date', 'missing_rate', 'negative_increment_count']].to_string(index=False)}")

if __name__ == '__main__':
    main()