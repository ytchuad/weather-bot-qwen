# data/download_rainfall.py
"""
下載 HKO 累積雨量 (15 分鐘) — 香港天文台測站
資料來源: i-lens.hk
歷史: history_chart.php?chart_type=STATION_ACCUM_RAIN&date=YYYY-MM-DD
即時: instant_chart.php?chart_type=STATION_ACCUM_RAIN
"""
import requests
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HISTORY_URL = "https://i-lens.hk/hkweather/history_chart.php?chart_type=STATION_ACCUM_RAIN&date={date_str}"
INSTANT_URL = "https://i-lens.hk/hkweather/instant_chart.php?chart_type=STATION_ACCUM_RAIN"
OUTPUT_PATH = Path('data/hko_rainfall_15min.parquet')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def parse_rainfall_from_html(html: str, target_date: date = None):
    """
    從 HTML 中提取 '香港天文台' 的累積雨量陣列。
    回傳 list of dict: [{'datetime': datetime, 'rainfall': float}, ...]
    """
    # 尋找 {name: '香港天文台', data: [...]}
    pattern = r"\{name:\s*'香港天文台'\s*,\s*data:\s*(\[.*?\])\s*\}"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        logger.warning("找不到香港天文台雨量資料")
        return []

    data_str = match.group(1)
    # 將 JavaScript 的 Date.UTC(...) 和 null 轉換成可解析的格式
    # 先將 Date.UTC(y,m,d,h,min) 替換成 "y-m-d h:min"
    # 注意 JS 月份是 0-based
    def replace_utc(m):
        y = int(m.group(1))
        mo = int(m.group(2)) + 1   # JS 月份 0‑11 → 1‑12
        d = int(m.group(3))
        h = int(m.group(4))
        mi = int(m.group(5))
        return f'"{y}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}"'
    
    data_str = re.sub(r"Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\)", replace_utc, data_str)
    # 將 null 替換成 "null" (已為 null)
    # 現在 data_str 應該是一個類似 [["2023-6-1 0:00",0], ...] 的 JSON 字串
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失敗: {e}")
        return []

    records = []
    for item in data:
        if len(item) != 2:
            continue
        dt_str, val = item
        if val is None:
            continue
        try:
            dt = pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M')
            records.append({'datetime': dt, 'rainfall': float(val)})
        except:
            continue
    return records

def download_historical(start_date: date, end_date: date):
    """下載歷史範圍 (含起訖) 的雨量資料"""
    all_records = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        logger.info(f"下載 {date_str} ...")
        try:
            resp = requests.get(HISTORY_URL.format(date_str=date_str), headers=HEADERS, timeout=15)
            resp.raise_for_status()
            records = parse_rainfall_from_html(resp.text, current)
            if not records:
                logger.warning(f"{date_str} 無資料，可能尚未生成或測站無數據")
            all_records.extend(records)
        except Exception as e:
            logger.error(f"下載 {date_str} 失敗: {e}")
        current += timedelta(days=1)
        time.sleep(0.3)  # 禮貌性延遲
    return all_records

def download_live():
    """下載今日即時雨量 (可能包含部分昨天的尾巴)"""
    logger.info("下載即時雨量...")
    try:
        resp = requests.get(INSTANT_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        records = parse_rainfall_from_html(resp.text)
        return records
    except Exception as e:
        logger.error(f"即時下載失敗: {e}")
        return []

def main():
    # 1. 載入現有資料 (若存在)
    if OUTPUT_PATH.exists():
        df_old = pd.read_parquet(OUTPUT_PATH)
        logger.info(f"現有資料 {len(df_old)} 筆，日期範圍 {df_old['datetime'].min()} ~ {df_old['datetime'].max()}")
    else:
        df_old = pd.DataFrame(columns=['datetime', 'rainfall'])

    # 2. 決定歷史下載範圍
    today = date.today()
    hist_start = date(2023, 6, 1)  # 最早可用日期
    # 歷史結束為昨天
    hist_end = today - timedelta(days=1)

    if hist_start <= hist_end:
        # 檢查是否有舊資料的最後日期，避免重複下載
        if not df_old.empty:
            last_date = df_old['datetime'].max().date()
            if last_date >= hist_start:
                # 若已有部分歷史，從最後日期的下一天開始
                hist_start = last_date + timedelta(days=1)
                if hist_start > hist_end:
                    logger.info("歷史資料已是最新，無需下載歷史。")
                    hist_start = None  # 跳過歷史
        if hist_start and hist_start <= hist_end:
            logger.info(f"下載歷史範圍: {hist_start} ~ {hist_end}")
            hist_records = download_historical(hist_start, hist_end)
            if hist_records:
                df_hist = pd.DataFrame(hist_records)
                df_old = pd.concat([df_old, df_hist], ignore_index=True)
                # 去重 (保留最後一筆)
                df_old = df_old.drop_duplicates(subset='datetime', keep='last')
                df_old = df_old.sort_values('datetime')
                logger.info(f"歷史更新後總筆數: {len(df_old)}")

    # 3. 下載即時資料 (今日)，可能包含昨天 23:xx 的資料，但無妨
    live_records = download_live()
    if live_records:
        df_live = pd.DataFrame(live_records)
        df_old = pd.concat([df_old, df_live], ignore_index=True)
        df_old = df_old.drop_duplicates(subset='datetime', keep='last')
        df_old = df_old.sort_values('datetime')
        logger.info(f"即時更新後總筆數: {len(df_old)}")

    # 4. 過濾掉未來時間 (即時頁面可能包含 null 而沒抓進，但安全起見)
    df_old = df_old[df_old['datetime'] <= pd.Timestamp.now()]

    # 5. 儲存
    df_old.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"雨量資料已儲存至 {OUTPUT_PATH}，最新時間: {df_old['datetime'].max()}")

if __name__ == '__main__':
    main()