# data/update_intraday_data.py
"""增量更新 intraday_hko_10min.parquet，從 HKO 即時 CSV 抓取最新觀測"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INTRADAY_PATH = Path('data/intraday_hko_10min.parquet')
CSV_URL = "https://www.hko.gov.hk/wxinfo/awsgis/hko.csv"

def fetch_live_hko_csv():
    try:
        resp = requests.get(CSV_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().split('\n')
        records = []
        for line in lines[1:]:  # 跳過標頭
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                dt = pd.to_datetime(parts[0].strip(), format='%Y/%m/%d %H:%M')
                temp = float(parts[1].strip())
                records.append({'datetime': dt, 'temp': temp})
            except:
                continue
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"無法抓取即時 CSV: {e}")
        return None

def main():
    logger.info("下載 HKO 即時氣溫...")
    df_live = fetch_live_hko_csv()
    if df_live is None or df_live.empty:
        logger.warning("無即時資料，終止。")
        return

    # 讀取現有歷史資料（若存在）
    if INTRADAY_PATH.exists():
        df_hist = pd.read_parquet(INTRADAY_PATH)
        # 合併：移除重複 datetime，保留最新的數值
        df_combined = pd.concat([df_hist, df_live], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset='datetime', keep='last')
    else:
        df_combined = df_live

    # 排序並儲存
    df_combined = df_combined.sort_values('datetime')
    df_combined.to_parquet(INTRADAY_PATH, index=False)
    logger.info(f"更新完成，資料範圍: {df_combined['datetime'].min()} 至 {df_combined['datetime'].max()}，總筆數: {len(df_combined)}")

if __name__ == '__main__':
    main()