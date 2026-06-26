import re
import os
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import json

# ========== 設定 ==========
BASE_URL = "https://i-lens.hk/hkweather/history_chart.php"
START_DATE = datetime(2016, 12, 8)
END_DATE = datetime(2026, 6, 23)

# 圖表類型對照
CHART_TYPES = {
    "": "temperature",
    "ALL_RH": "humidity",
    "ALL_DT": "dew_point",
    "DG_MSLP": "pressure",
}

OUTPUT_DIR = "./hk_weather_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 並發控制：同時最多 15 個請求（避免被視為 DDoS）
SEMAPHORE_LIMIT = 15
# 單次請求超時（秒）
REQUEST_TIMEOUT = 30

# ========== 解析函數（與之前相同） ==========
def parse_date_utc(utc_str: str) -> datetime:
    pattern = r"Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\)"
    match = re.search(pattern, utc_str)
    if not match:
        raise ValueError(f"無法解析: {utc_str}")
    year, month, day, hour, minute = map(int, match.groups())
    return datetime(year, month + 1, day, hour, minute)

def extract_data_from_html(html: str) -> List[Tuple[datetime, float]]:
    pattern = r"data:\s*\[\s*((?:\[Date\.UTC\([^)]+\)\s*,\s*[\d.]+\]\s*,?\s*)+)\]"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        return []
    data_block = match.group(1)
    entry_pattern = r"\[Date\.UTC\(([^)]+)\)\s*,\s*([\d.]+)\]"
    entries = re.findall(entry_pattern, data_block)
    results = []
    for utc_args, value_str in entries:
        utc_str = f"Date.UTC({utc_args})"
        dt = parse_date_utc(utc_str)
        results.append((dt, float(value_str)))
    return results

# ========== 非同步單日抓取任務 ==========
async def fetch_date_data(session: aiohttp.ClientSession, date: datetime, chart_type: str, semaphore: asyncio.Semaphore):
    date_str = date.strftime("%Y-%m-%d")
    data_name = CHART_TYPES[chart_type]
    filepath = os.path.join(OUTPUT_DIR, f"{date_str}_{data_name}.parquet")
    
    # 若檔案已存在，直接跳過（斷點續傳）
    if os.path.exists(filepath):
        # 可選擇略過或檢查檔案大小，此處直接跳過
        return f"⏭️ 跳過 {date_str} ({data_name}) - 已存在"

    params = {"date": date_str}
    if chart_type:
        params["chart_type"] = chart_type

    # 使用 Semaphore 控制並發數量
    async with semaphore:
        try:
            async with session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()
                html = await resp.text()
        except Exception as e:
            return f"❌ 失敗 {date_str} ({data_name}): {e}"

        data = extract_data_from_html(html)
        if not data:
            return f"⚠️ 無數據 {date_str} ({data_name})"

        df = pd.DataFrame(data, columns=["timestamp", "value"])
        df["date"] = date_str
        df.to_parquet(filepath, index=False)
        return f"✅ 儲存 {date_str} ({data_name}) - {len(df)} 筆"

# ========== 主程式（非同步執行） ==========
async def main():
    # 建立所有任務清單
    tasks = []
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    
    # 建立共用的 ClientSession（重用 TCP 連線，加速）
    async with aiohttp.ClientSession() as session:
        current_date = START_DATE
        total_days = (END_DATE - START_DATE).days + 1
        total_tasks = total_days * len(CHART_TYPES)
        
        print(f"📊 總共將處理 {total_days} 天，共 {total_tasks} 個請求 (並發上限: {SEMAPHORE_LIMIT})")
        
        # 生成所有任務
        while current_date <= END_DATE:
            for chart_type in CHART_TYPES.keys():
                task = fetch_date_data(session, current_date, chart_type, semaphore)
                tasks.append(task)
            current_date += timedelta(days=1)
        
        # 全部同時執行（非同步）
        # 為了觀看進度，使用 asyncio.as_completed
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            if completed % 50 == 0 or "❌" in result or "✅" in result:
                print(f"[{completed}/{total_tasks}] {result}")
            else:
                print(result)  # 顯示跳過訊息

        print(f"\n🎉 全部完成！檔案存放於 {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())