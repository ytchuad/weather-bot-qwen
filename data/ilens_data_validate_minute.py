import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# ========== 設定 ==========
DATA_DIR = r"C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen\data\hk_weather_raw"
START_DATE = "2016-12-08"
END_DATE = "2026-06-23"

# 各類數據的合理範圍（根據物理常識）
VALID_RANGES = {
    "temperature": (-10, 45),      # 香港氣溫極值約 0~38°C
    "humidity": (0, 100),          # 相對濕度 0~100%
    "dew_point": (-20, 35),        # 露點溫度（香港通常 5~30°C）
    "pressure": (980, 1050),       # 氣壓（hPa）
}

# ========== 1. 檢查檔案數量與日期覆蓋率 ==========
print("📂 掃描檔案中...")
all_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
print(f"總檔案數: {len(all_files)}")

# 按資料類型分類
files_by_type = {}
for f in all_files:
    basename = os.path.basename(f)
    # 檔名格式: YYYY-MM-DD_type.parquet
    parts = basename.split('_')
    if len(parts) >= 2:
        date_str = parts[0]
        data_type = parts[1].replace('.parquet', '')
        if data_type not in files_by_type:
            files_by_type[data_type] = []
        files_by_type[data_type].append((date_str, f))

print("\n📊 各類型檔案數量:")
for typ, lst in files_by_type.items():
    print(f"  {typ}: {len(lst)} 個檔案")

# 預期總天數
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
expected_days = (end_dt - start_dt).days + 1
print(f"\n📅 預期總天數: {expected_days} (從 {START_DATE} 到 {END_DATE})")

# 檢查每種類型的日期覆蓋率
for typ, lst in files_by_type.items():
    dates = set([item[0] for item in lst])
    print(f"  {typ} 實際涵蓋天數: {len(dates)} / {expected_days} (覆蓋率: {len(dates)/expected_days*100:.1f}%)")

# ========== 2. 逐一驗證數值合理性（分開檢查） ==========
print("\n🔍 開始驗證數值合理性...")

for typ, valid_range in VALID_RANGES.items():
    print(f"\n--- 檢查 {typ} ---")
    typ_files = [f for d, f in files_by_type.get(typ, [])]
    if not typ_files:
        print("  ⚠️ 無此類型檔案")
        continue
    
    all_values = []
    sample_df = None
    for f in typ_files[:5]:  # 抽樣前5個檔案加速
        df = pd.read_parquet(f)
        all_values.extend(df['value'].tolist())
        if sample_df is None:
            sample_df = df
    
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        lower, upper = valid_range
        out_of_range = [v for v in all_values if v < lower or v > upper]
        print(f"  數值範圍: {min_val:.2f} ~ {max_val:.2f}")
        print(f"  合理範圍應為: {lower} ~ {upper}")
        print(f"  超出範圍筆數 (抽樣): {len(out_of_range)}")
        if len(out_of_range) > 0:
            print(f"  異常範例: {out_of_range[:5]}")

# ========== 3. 檢查特定一天（例如 2026-06-23）的連續性 ==========
print("\n📋 檢查單日完整性 (以 2026-06-23 temperature 為例)...")
test_file = os.path.join(DATA_DIR, "2026-06-23_temperature.parquet")
if os.path.exists(test_file):
    df_test = pd.read_parquet(test_file)
    print(f"  檔案筆數: {len(df_test)}")
    if len(df_test) == 1440:
        print("  ✅ 完美！包含完整 1440 分鐘")
    else:
        print(f"  ⚠️ 筆數不足，缺少 {1440 - len(df_test)} 分鐘")
else:
    print("  ❌ 檔案不存在 (可能尚未下載到當天)")

# ========== 4. 總結 ==========
print("\n🎯 總結建議:")
print("1. 若 '溫度' 與 '濕度' 覆蓋率 > 95%，資料品質足夠建模。")
print("2. 若 '壓力' 與 '露點' 覆蓋率較低，不影響主模型，可先忽略。")
print("3. 請確認 '2026-06-23' 之後的日期是否已全部下載完畢。")