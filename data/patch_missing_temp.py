import os
import re
import time
import requests
import pandas as pd
from datetime import datetime

# ========== 設定 ==========
DATA_DIR = r"C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen\data\hk_weather_raw"
MISSING_LIST_FILE = "missing_temperature_dates.txt"
BASE_URL = "https://i-lens.hk/hkweather/history_chart.php"

os.makedirs(DATA_DIR, exist_ok=True)

# ========== 穩健的解析函數（使用括號計數） ==========
def extract_temperature_from_html(html: str):
    """
    利用括號計數，精確抓取 name: '氣溫' 底下的 data: [ ... ] 陣列
    這種方法比正則表達式更可靠，因為它不怕換行、不怕結尾逗號。
    """
    # 1. 嘗試多種可能的標籤寫法（避免因為空白字元差異而找不到）
    temp_idx = html.find("name: '氣溫'")
    if temp_idx == -1:
        temp_idx = html.find('name: "氣溫"')
    if temp_idx == -1:
        temp_idx = html.find("name:'氣溫'")
    if temp_idx == -1:
        return []

    # 2. 從該位置往後找 data: [
    data_idx = html.find("data: [", temp_idx)
    if data_idx == -1:
        return []

    # 3. 括號計數法：從 data: [ 的 "[" 開始，逐字元掃描，直到括號完全閉合
    bracket_count = 0
    end_idx = -1
    
    # 跳過 "data: [" 這 7 個字元，直接從 "[" 開始
    i = data_idx + len("data: [") - 1  # 此時 i 指向 "["
    
    while i < len(html):
        c = html[i]
        if c == '[':
            bracket_count += 1
        elif c == ']':
            bracket_count -= 1
            if bracket_count == 0:
                # 找到匹配的結尾
                end_idx = i
                break
        i += 1

    if end_idx == -1:
        return []

    # 4. 取出陣列內的完整內容（不含最外層的 [ 和 ]）
    block = html[data_idx + len("data: ["):end_idx]
    
    # 5. 用正則表達式抓出裡面的每一筆 [Date.UTC(...), value]
    #    注意：這裡只針對「區塊內部」做匹配，不會被外部結構干擾
    entry_pattern = r"\[Date\.UTC\(([^)]+)\)\s*,\s*([\d.]+)\]"
    entries = re.findall(entry_pattern, block)
    
    results = []
    for utc_args, value_str in entries:
        # 解析 Date.UTC 的參數 (year, month, day, hour, minute)
        # 注意：JavaScript 的 month 從 0 開始
        parts = utc_args.split(',')
        if len(parts) == 5:
            y, m, d, h, min_ = map(int, parts)
            # 轉換為 Python datetime (month 要 +1)
            dt = datetime(y, m + 1, d, h, min_)
            results.append((dt, float(value_str)))
    
    return results

def fetch_and_save_temperature(date_str: str):
    filepath = os.path.join(DATA_DIR, f"{date_str}_temperature.parquet")
    
    if os.path.exists(filepath):
        print(f"⏭️  跳過 {date_str} (已存在)")
        return True

    params = {"date": date_str}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ 請求失敗 {date_str}: {e}")
        return False

    html = resp.text
    data = extract_temperature_from_html(html)
    
    if not data:
        # 除錯用：若還是失敗，印出頁面中氣溫附近的片段供檢查
        temp_idx = html.find("氣溫")
        if temp_idx != -1:
            snippet = html[temp_idx:temp_idx+300].replace('\n', ' ')
            print(f"⚠️  找到 '氣溫' 但無法解析陣列，片段: {snippet}...")
        else:
            print(f"⚠️  {date_str} 頁面完全找不到 '氣溫' 字樣")
        return False

    if len(data) != 1440:
        print(f"⚠️  {date_str} 抓到 {len(data)} 筆 (異常，預期 1440)，仍強制儲存")

    df = pd.DataFrame(data, columns=["timestamp", "value"])
    df["date"] = date_str
    df.to_parquet(filepath, index=False)
    print(f"✅ 已補抓 {date_str} ({len(df)} 筆)")
    return True

# ========== 主程式 ==========
def main():
    if not os.path.exists(MISSING_LIST_FILE):
        print(f"❌ 找不到清單檔案: {MISSING_LIST_FILE}")
        return

    with open(MISSING_LIST_FILE, "r") as f:
        missing_dates = [line.strip() for line in f.readlines() if line.strip()]

    print(f"📋 讀取到 {len(missing_dates)} 個遺失日期需補抓")
    
    success_count = 0
    fail_count = 0
    total = len(missing_dates)

    for idx, date_str in enumerate(missing_dates, 1):
        print(f"[{idx}/{total}] 處理 {date_str}...")
        result = fetch_and_save_temperature(date_str)
        if result:
            success_count += 1
        else:
            fail_count += 1
        
        # 動態延遲 0.5 ~ 1.5 秒
        time.sleep(0.5 + (idx % 10) * 0.1)

    print(f"\n🎉 補抓完成！")
    print(f"  成功: {success_count}")
    print(f"  失敗: {fail_count}")
    if fail_count > 0:
        print("  ⚠️  若有失敗，請將錯誤片段貼給我，我會再調整解析邏輯。")

if __name__ == "__main__":
    main()