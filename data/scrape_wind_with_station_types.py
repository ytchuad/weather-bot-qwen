#!/usr/bin/env python3
"""
非同步爬取香港風速歷史圖表資料（2016-12-08 ~ 2026-06-23）
輸出格式：{date}_wind_all.parquet，包含 timestamp, station_type, station, wind_speed
"""

import asyncio
import re
import os
import logging
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- 設定 ----------
BASE_URL = "https://i-lens.hk/hkweather/history_chart.php"
START_DATE = datetime(2016, 12, 8)
END_DATE = datetime(2026, 6, 23)
OUTPUT_DIR = "./wind_data"          # 存放 parquet 的資料夾
CONCURRENCY = 5                     # 同時連線數
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
RETRY_LIMIT = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------- 工具函式 ----------
def extract_station_type(title: str) -> str | None:
    """從圖表標題中抽取分類名稱。
    
    對應規則：
    - "參考" / "香港八個參考測風站" → "參考"
    - "維多利亞港" → "維多利亞港"
    - "離岸及高地" / "離岸" / "高山" → "離岸及高地"（合併為單一群組）
    - 其他 → None（丟棄，不產生 station_type）
    """
    if "參考" in title:
        return "參考"
    if "維多利亞港" in title:
        return "維多利亞港"
    if any(k in title for k in ("離岸及高地", "離岸", "高山")):
        return "離岸及高地"
    return None

def parse_highcharts_config(script_text: str):
    """
    從一段 JavaScript 字串中解析所有 Highcharts.chart 呼叫，
    抽出 title, series, data
    回傳 list of dict: {timestamp, station_type, station, wind_speed}
    """
    records = []

    # 找到每個 Highcharts.chart('id', { ... }); 的起始位置
    for m in re.finditer(r"Highcharts\.chart\([^)]+?,\s*\{", script_text):
        start = m.end() - 1  # '{' 的位置

        # 尋找對應的結束點：下一個 addMaxMin(...) 或 dimSeries(...) 或 )});
        # 使用簡單的堆疊計數器找到最外層的 '});' 結束
        brace_count = 0
        i = start
        while i < len(script_text):
            ch = script_text[i]
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    # 找到最外層 '}'，往後應該是 ');' 結尾
                    end = script_text.find(');', i) + 1
                    break
            i += 1
        else:
            # 沒找到完整物件，跳過
            continue

        config_str = script_text[start:end+1]   # 包含最後的 );

        # 擷取 title
        title_m = re.search(r"text\s*:\s*'([^']+)'", config_str)
        if not title_m:
            continue
        title = title_m.group(1)
        station_type = extract_station_type(title)
        
        # 跳過無法識別的圖表類型
        if station_type is None:
            continue

        # 擷取 series 陣列內容
        # 方式：找到 "series:" 後面的 '['，再用堆疊抓出整個陣列
        series_match = re.search(r"series\s*:\s*\[", config_str)
        if not series_match:
            continue
        arr_start = series_match.end() - 1  # '[' 位置
        bracket_count = 0
        i = arr_start
        while i < len(config_str):
            if config_str[i] == '[':
                bracket_count += 1
            elif config_str[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    arr_end = i
                    break
            i += 1
        else:
            continue
        series_content = config_str[arr_start+1:arr_end]

        # 在 series_content 中逐一匹配每個 {name:'...', data:[...]}
        # 為了簡便，用 re.finditer 抓取 name 及其後緊接的 data
        for s in re.finditer(
            r"name\s*:\s*'([^']+)'\s*,\s*data\s*:\s*\[(.*?)\]\s*(?:\}\s*[,|\]])",
            series_content,
            re.DOTALL
        ):
            station = s.group(1)
            data_part = s.group(2)

            # 解析 data 中的 [Date.UTC(y,m,d,h,min), value]
            pts = re.findall(
                r"Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\)\s*,\s*(\d+)",
                data_part
            )
            for y_str, mon_str, d_str, h_str, min_str, val_str in pts:
                try:
                    ts = datetime(
                        int(y_str), int(mon_str)+1, int(d_str),
                        int(h_str), int(min_str)
                    )
                except ValueError:
                    continue
                records.append({
                    "timestamp": ts,
                    "station_type": station_type,
                    "station": station,
                    "wind_speed": int(val_str)
                })

    return records

async def fetch_and_parse(session: aiohttp.ClientSession, date_str: str):
    """下載某一天的頁面，解析並回傳 DataFrame"""
    params = {"date": date_str, "chart_type": "DG_WIND"}
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            async with session.get(
                BASE_URL, params=params, timeout=REQUEST_TIMEOUT
            ) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status
                    )
                html = await resp.text(encoding="utf-8", errors="ignore")
                break
        except Exception as e:
            logging.warning(f"{date_str} 第{attempt}次請求失敗: {e}")
            if attempt == RETRY_LIMIT:
                raise
            await asyncio.sleep(2 ** attempt)

    # 在執行緒池中解析（BeautifulSoup 為 CPU 密集）
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(None, _parse_html, html)
    return pd.DataFrame(records)

def _parse_html(html: str):
    """同步解析 HTML，回傳 records list"""
    soup = BeautifulSoup(html, "html.parser")
    all_records = []
    for script in soup.find_all("script"):
        if script.string:
            all_records.extend(parse_highcharts_config(script.string))
    return all_records

def output_path(date_str: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{date_str}_wind_all.parquet")

async def process_date(
    session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, date_str: str
):
    """處理單一日期：檢查是否已存在 -> 下載 -> 儲存"""
    path = output_path(date_str)
    if os.path.exists(path):
        logging.info(f"{date_str} 已存在，跳過")
        return

    async with semaphore:
        try:
            df = await fetch_and_parse(session, date_str)
        except Exception as e:
            logging.error(f"{date_str} 取得資料失敗: {e}")
            return

    if df.empty:
        logging.warning(f"{date_str} 沒有解析到任何資料")
        return

    # 寫入 Parquet
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_parquet(path, index=False)
        logging.info(f"{date_str} 成功寫入 {len(df)} 筆")
    except Exception as e:
        logging.error(f"{date_str} 寫入檔案失敗: {e}")

async def main():
    # 產生所有日期字串
    all_dates = []
    current = START_DATE
    while current <= END_DATE:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        connector=connector,
        timeout=REQUEST_TIMEOUT
    ) as session:
        tasks = [
            process_date(session, semaphore, d)
            for d in all_dates
        ]
        # 使用 gather 但允許部分失敗
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())