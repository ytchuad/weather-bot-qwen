"""
scrape_wind_kings_park.py (極簡穩健版)

使用 re.findall 直接從 HTML 抓取所有 Date.UTC 數據點。
適用於京士柏 (King's Park) 10 分鐘平均風速。
如果某天真的沒有數據，回傳空檔案。
"""

import os
import re
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import random

# ========== 設定 ==========
BASE_URL = "https://i-lens.hk/hkweather/history_chart.php"
# 既然 2016-12-08 有數據（雖然是 0），我們就從頭開始抓，確保連續性
START_DATE = datetime(2016, 12, 8)
END_DATE = datetime(2026, 6, 23)

OUTPUT_DIR = "./hk_weather_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEMAPHORE_LIMIT = 10
MIN_DELAY = 0.3
MAX_DELAY = 0.8
TARGET_STATION = "京士柏"

# ========== 萬無一失的解析函數 ==========
def parse_wind_from_html(html: str) -> List[Tuple[datetime, float]]:
    """
    策略：先在 HTML 中定位 '京士柏'，然後從該位置往後抓取所有 Date.UTC。
    這種做法完全無視 JSON 結構的換行與巢狀，極度穩健。
    """
    # 1. 檢查是否有京士柏的數據
    if TARGET_STATION not in html:
        return []
    
    # 2. 只取京士柏之後的部分 (避免抓到其他站的數據)
    # 注意：如果京士柏的 data 陣列很大，re.findall 會抓取所有符合的片段
    idx = html.find(TARGET_STATION)
    sub_html = html[idx:]
    
    # 3. 提取所有 [Date.UTC(...), value] 元組
    pattern = r"\[Date\.UTC\(([^)]+)\)\s*,\s*([\d.]+)\]"
    entries = re.findall(pattern, sub_html)
    
    if not entries:
        return []
    
    results = []
    # 4. 只取前 1440 筆 (一天的完整數據)，避免抓到京士柏之後其他站的數據
    # 通常京士柏的數據會完整排列在前 1440 筆
    for utc_args, value_str in entries[:1440]:
        parts = utc_args.split(',')
        if len(parts) == 5:
            y, m, d, h, min_ = map(int, parts)
            # JavaScript 月份從 0 開始，需 +1
            dt = datetime(y, m + 1, d, h, min_)
            results.append((dt, float(value_str)))
    
    return results

# ========== 非同步抓取任務 ==========
async def fetch_wind_date(session: aiohttp.ClientSession, date: datetime, semaphore: asyncio.Semaphore):
    date_str = date.strftime("%Y-%m-%d")
    filepath = os.path.join(OUTPUT_DIR, f"{date_str}_wind.parquet")
    
    # 斷點續傳
    if os.path.exists(filepath):
        return None
    
    params = {"chart_type": "ALL_WIND", "date": date_str}
    
    async with semaphore:
        try:
            async with session.get(BASE_URL, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
        except Exception as e:
            print(f"⚠️ 請求失敗 {date_str}: {e}")
            return None
        
        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        
        # 過濾空頁面
        if not html or len(html) < 500:
            return None
        
        data = parse_wind_from_html(html)
        
        # 如果完全沒有數據，靜默跳過
        if not data:
            return None
        
        # 如果數據少於 100 筆，視為異常，跳過不存（避免存到錯誤的 headers）
        if len(data) < 100:
            print(f"⚠️ {date_str} 僅抓到 {len(data)} 筆 (少於預期 1440)，跳過不存")
            return None
        
        df = pd.DataFrame(data, columns=["timestamp", "value"])
        df["date"] = date_str
        df.to_parquet(filepath, index=False)
        return df

# ========== 主程式 ==========
async def main():
    print("=" * 60)
    print("🌀 開始抓取京士柏 10 分鐘平均風速 (極簡穩健版)")
    print(f"📅 範圍: {START_DATE.strftime('%Y-%m-%d')} ~ {END_DATE.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    tasks = []
    
    current_date = START_DATE
    total_days = (END_DATE - START_DATE).days + 1
    
    async with aiohttp.ClientSession() as session:
        while current_date <= END_DATE:
            tasks.append(fetch_wind_date(session, current_date, semaphore))
            current_date += timedelta(days=1)
        
        results = []
        completed = 0
        total_tasks = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            df = await coro
            completed += 1
            if df is not None:
                results.append(df)
                if len(results) % 50 == 0:
                    print(f"✅ 已成功抓取 {len(results)} 天 ({completed}/{total_tasks})")
            else:
                if completed % 200 == 0:
                    print(f"⏳ 進度: {completed}/{total_tasks}")
    
    print(f"\n🎉 抓取完成！新增 {len(results)} 天風速數據")
    print(f"📁 儲存於: {OUTPUT_DIR}/*_wind.parquet")

if __name__ == "__main__":
    asyncio.run(main())