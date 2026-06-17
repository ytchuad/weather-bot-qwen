# data/build_hko_forecast_dataset.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUT_PATH = Path('data/hko_historical_forecasts.parquet')
START_DATE = datetime(2016, 6, 6)
END_DATE = datetime(2026, 4, 30)

def parse_numeric(val):
    """
    穩健解析數值欄位。
    處理單個數字 (如 "33") 或區間 (如 "28-33")，區間將取平均值。
    忽略 "不詳" 或空值。
    """
    if pd.isna(val) or not val or str(val).strip() in ['不詳', 'N/A', '']:
        return np.nan
    nums = re.findall(r'\d+\.?\d*', str(val))
    if nums:
        return np.mean([float(n) for n in nums])
    return np.nan

def scrape_date(target_date_str):
    url = f"https://i-lens.hk/hkweather/daily_extract.php?date={target_date_str}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        # i-lens 網頁可能使用 Big5 或 UTF-8，讓 requests 自動推斷編碼
        if resp.encoding == 'ISO-8859-1':
            resp.encoding = resp.apparent_encoding
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找包含目標表頭的表格
        tables = soup.find_all('table')
        target_table = None
        for table in tables:
            if '發佈日期' in table.text and '預測最高氣溫' in table.text:
                target_table = table
                break
                
        if not target_table:
            return pd.DataFrame()
            
        rows = target_table.find_all('tr')
        if not rows:
            return pd.DataFrame()
            
        # 提取表頭
        header_row = rows[0].find_all(['th', 'td'])
        headers = [h.get_text(strip=True) for h in header_row]
        
        col_map = {
            '發佈日期': 'publish_date',
            '發佈時間': 'publish_time',
            '預測最低氣溫': 'predicted_min_temp',
            '預測最高氣溫': 'predicted_max_temp',
            '預測最低相對濕度': 'predicted_min_rh',
            '預測最高相對濕度': 'predicted_max_rh',
            '預測風向及風力': 'predicted_wind',
            '預測天氣狀況': 'predicted_weather'
        }
        
        # 建立欄位索引映射
        idx_map = {}
        for i, h in enumerate(headers):
            if h in col_map:
                idx_map[col_map[h]] = i
                
        data = []
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) < len(headers):
                continue
            
            record = {'target_date': target_date_str}
            for col_name, idx in idx_map.items():
                val = cols[idx].get_text(strip=True)
                record[col_name] = val
            data.append(record)
            
        return pd.DataFrame(data)
    except Exception as e:
        logging.warning(f"抓取 {target_date_str} 時發生錯誤: {e}")
        return pd.DataFrame()

def main():
    all_dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
    
    # 斷點續傳支援
    existing_df = pd.DataFrame()
    processed_dates = set()
    if OUT_PATH.exists():
        logging.info("發現現有 Parquet 檔案，準備斷點續傳...")
        existing_df = pd.read_parquet(OUT_PATH)
        processed_dates = set(existing_df['target_date'].unique())
        logging.info(f"已處理 {len(processed_dates)} 個日期。")
        
    new_dfs = [existing_df]
    count = 0
    
    logging.info(f"開始爬取 {len(all_dates)} 個日期的歷史預報...")
    for d in all_dates:
        d_str = d.strftime('%Y-%m-%d')
        if d_str in processed_dates:
            continue
            
        df = scrape_date(d_str)
        if not df.empty:
            new_dfs.append(df)
            
        count += 1
        if count % 100 == 0:
            logging.info(f"進度: 已檢查 {count}/{len(all_dates)} 個日期。最新: {d_str}")
            # 儲存檢查點
            temp_df = pd.concat(new_dfs, ignore_index=True)
            temp_df.to_parquet(OUT_PATH, index=False)
            
        time.sleep(0.4) # 禮貌性延遲
        
    # 最終合併與儲存
    final_df = pd.concat(new_dfs, ignore_index=True)
    
    # 轉換數值欄位
    num_cols = ['predicted_min_temp', 'predicted_max_temp', 'predicted_min_rh', 'predicted_max_rh']
    for col in num_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].apply(parse_numeric)
            
    final_df.to_parquet(OUT_PATH, index=False)
    logging.info(f"完成！總紀錄數: {len(final_df)}。已儲存至 {OUT_PATH}")
    logging.info(f"資料日期範圍: {final_df['target_date'].min()} 至 {final_df['target_date'].max()}")

if __name__ == "__main__":
    main()