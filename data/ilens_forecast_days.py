import os
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Optional, Set
import random

# ========== 設定 ==========
BASE_URL = "https://i-lens.hk/hkweather/daily_extract.php"
START_DATE = datetime(1998, 5, 1)
END_DATE = datetime(2026, 6, 23)

OUTPUT_FILE = "./hk_daily_forecast/daily_forecast_all.parquet"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 安全設定：保持低並發
SEMAPHORE_LIMIT = 15
MIN_DELAY = 0.3
MAX_DELAY = 0.8

# ========== 精準定位的 BeautifulSoup 解析函數 ==========
async def parse_html_table_async(html: str, target_date: str) -> Optional[pd.DataFrame]:
    def _sync_parse():
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 方法 1：尋找 <h2> 標題「香港天文台對 ... 所作出的天氣預測」
        h2_tag = soup.find('h2', string=re.compile('香港天文台對.*所作出的天氣預測'))
        target_table = None
        
        if h2_tag:
            # 找該 h2 之後的第一個 <table>
            target_table = h2_tag.find_next('table')
        
        # 方法 2（備用）：若找不到 h2，找包含「發佈日期」表頭的表格
        if not target_table:
            tables = soup.find_all('table')
            for table in tables:
                headers = table.find_all('th')
                header_texts = [h.get_text(strip=True) for h in headers]
                if any('發佈日期' in h for h in header_texts):
                    target_table = table
                    break
        
        if not target_table:
            return None
        
        # 開始解析該表格的資料列
        rows = target_table.find_all('tr')
        if len(rows) < 2:
            return None
        
        results = []
        for row in rows[1:]:  # 跳過表頭
            cols = row.find_all('td')
            # 正常預測表格有 8 或 9 個欄位
            if len(cols) < 8:
                continue
            
            # 提取純文字
            clean_cols = [col.get_text(strip=True) for col in cols]
            
            # 🔥 關鍵防呆：第一個欄位必須是「日期」格式 (YYYY-MM-DD)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', clean_cols[0]):
                continue  # 若是站名（如吉澳）則跳過
            
            # 依據欄位數映射（0:日期, 1:時間, 2:低溫, 3:高溫, 4:低濕, 5:高濕, 6:風力）
            record = {
                'query_date': target_date,
                'forecast_issue_date': clean_cols[0],
                'forecast_issue_time': clean_cols[1],
                'forecast_min_temp': clean_cols[2] if len(clean_cols) > 2 else None,
                'forecast_max_temp': clean_cols[3] if len(clean_cols) > 3 else None,
                'forecast_min_rh': clean_cols[4] if len(clean_cols) > 4 else None,
                'forecast_max_rh': clean_cols[5] if len(clean_cols) > 5 else None,
                'forecast_wind': clean_cols[6] if len(clean_cols) > 6 else None,
            }
            
            # 降雨機率與天氣狀況（根據長度自動判斷）
            if len(clean_cols) == 9:
                record['forecast_rain_prob'] = clean_cols[7]
                record['forecast_weather_desc'] = clean_cols[8]
            else:
                record['forecast_rain_prob'] = None
                record['forecast_weather_desc'] = clean_cols[7] if len(clean_cols) > 7 else None
            
            results.append(record)
        
        if not results:
            return None
        return pd.DataFrame(results)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_parse)

# ========== 非同步抓取（含內容驗證） ==========
async def fetch_date_forecast(
    session: aiohttp.ClientSession,
    date: datetime,
    semaphore: asyncio.Semaphore,
    processed_dates: Set[str]
) -> Optional[pd.DataFrame]:
    date_str = date.strftime("%Y-%m-%d")
    
    if date_str in processed_dates:
        return None
    
    params = {"date": date_str}
    
    async with semaphore:
        try:
            async with session.get(BASE_URL, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
        except Exception:
            return None
        
        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        
        if not html or len(html) < 500:
            return None
        
        df = await parse_html_table_async(html, date_str)
        return df

# ========== 主程式 ==========
async def main():
    processed_dates = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_parquet(OUTPUT_FILE)
            processed_dates = set(existing_df['query_date'].unique())
            print(f"📂 跳過已存在的 {len(processed_dates)} 天")
        except Exception:
            pass
    
    tasks = []
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    
    current_date = START_DATE
    total_days = (END_DATE - START_DATE).days + 1
    total_to_fetch = total_days - len(processed_dates)
    
    if total_to_fetch <= 0:
        print("✅ 已完成")
        return
    
    print(f"📊 需抓取 {total_to_fetch} 天 (並發: {SEMAPHORE_LIMIT})")
    
    async with aiohttp.ClientSession() as session:
        while current_date <= END_DATE:
            date_str = current_date.strftime("%Y-%m-%d")
            if date_str not in processed_dates:
                task = fetch_date_forecast(session, current_date, semaphore, processed_dates)
                tasks.append(task)
            current_date += timedelta(days=1)
        
        results = []
        completed = 0
        total_tasks = len(tasks)
        print(f"🚀 開始使用 BeautifulSoup 解析，共 {total_tasks} 個請求...")
        
        for coro in asyncio.as_completed(tasks):
            df = await coro
            completed += 1
            if df is not None and not df.empty:
                results.append(df)
                if len(results) % 50 == 0:
                    print(f"✅ 已累積 {len(results)} 個有效日期 ({completed}/{total_tasks})")
            else:
                if completed % 500 == 0:
                    print(f"⏳ 進度: {completed}/{total_tasks}")
        
        if not results:
            print("⚠️ 完全沒有抓到資料！請確認 BeautifulSoup 已安裝 (pip install beautifulsoup4)")
            return
        
        new_df = pd.concat(results, ignore_index=True)
        
        if os.path.exists(OUTPUT_FILE):
            old_df = pd.read_parquet(OUTPUT_FILE)
            final_df = pd.concat([old_df, new_df], ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset=['forecast_issue_date', 'forecast_issue_time', 'query_date'],
                keep='last'
            )
        else:
            final_df = new_df
        
        final_df['forecast_datetime'] = pd.to_datetime(
            final_df['forecast_issue_date'] + ' ' + final_df['forecast_issue_time'],
            errors='coerce'
        )
        
        numeric_cols = ['forecast_min_temp', 'forecast_max_temp', 'forecast_min_rh', 'forecast_max_rh']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n🎉 完成！總共 {len(final_df)} 筆預測記錄")
        print(f"📁 儲存於: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())