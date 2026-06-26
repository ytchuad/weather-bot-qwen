import pandas as pd
import glob
import os
from datetime import datetime, timedelta

DATA_DIR = r"C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen\data\hk_weather_raw"
START = "2016-12-08"
END = "2026-06-23"

# 產生預期所有日期
all_dates = pd.date_range(start=START, end=END).strftime("%Y-%m-%d").tolist()

# 讀取既有的溫度檔案日期
files = glob.glob(os.path.join(DATA_DIR, "*_temperature.parquet"))
existing_dates = [os.path.basename(f).split('_')[0] for f in files]

missing = sorted(set(all_dates) - set(existing_dates))
print(f"✅ 溫度資料遺失天數: {len(missing)}")
if len(missing) > 0:
    print("前 20 個遺失日期:")
    print(missing[:20])
    # 將清單存檔，供爬蟲使用
    with open("missing_temperature_dates.txt", "w") as f:
        f.write("\n".join(missing))
    print("📁 完整遺失清單已儲存至 missing_temperature_dates.txt")