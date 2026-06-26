import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from tqdm import tqdm  # 若無 tqdm 可先 pip install tqdm

# ========== 設定 ==========
DATA_DIR = r"C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen\data\hk_weather_raw"
FORECAST_PATH = "./hk_daily_forecast/daily_forecast_clean.parquet"  # 使用剛清理完的檔案
OUTPUT_PATH = "./feature_store_10min_full.parquet"

print("📂 1. 讀取溫度資料...")
temp_files = glob.glob(os.path.join(DATA_DIR, "*_temperature.parquet"))
df_temp = pd.concat([pd.read_parquet(f) for f in tqdm(temp_files)]).drop_duplicates(subset=['timestamp'], keep='last')
df_temp = df_temp.sort_values('timestamp').reset_index(drop=True)
df_temp['date'] = df_temp['timestamp'].dt.date

# 計算每日最高溫（Label 用）
daily_max = df_temp.groupby('date')['value'].max().reset_index()
daily_max.columns = ['date', 'daily_max_temp']

# 合併每日最高溫
df_temp = df_temp.merge(daily_max, on='date', how='left')
df_temp['remaining_upside'] = df_temp['daily_max_temp'] - df_temp['value']

print(f"   ✅ 溫度數據筆數: {len(df_temp):,}")

print("📂 2. 讀取每日預測資料...")
df_forecast = pd.read_parquet(FORECAST_PATH)
df_forecast['forecast_datetime'] = pd.to_datetime(df_forecast['forecast_issue_date'] + ' ' + df_forecast['forecast_issue_time'], format='%Y-%m-%d %H:%M', errors='coerce')
df_forecast['target_date'] = pd.to_datetime(df_forecast['query_date']).dt.date

# 只保留必要欄位
df_forecast = df_forecast[['target_date', 'forecast_datetime', 'forecast_max_temp']].drop_duplicates()

print(f"   ✅ 預測數據筆數: {len(df_forecast):,}")

print("📂 3. 建立決策點（每 10 分鐘取樣）...")
# 過濾日期範圍：2016-12-08 ~ 2026-06-23
start_dt = datetime(2016, 12, 8)
end_dt = datetime(2026, 6, 23)

# 產生所有 10 分鐘間隔的時間點（僅在 06:00 ~ 23:50 之間，避免凌晨無交易量）
decision_times = []
current = start_dt.replace(hour=6, minute=0, second=0, microsecond=0)
while current <= end_dt:
    decision_times.append(current)
    current += timedelta(minutes=10)
    # 超過 23:50 則跳到隔天 06:00
    if current.hour >= 0 and current.hour < 6:
        current = current.replace(hour=6, minute=0, second=0, microsecond=0)
        current += timedelta(days=1)
    # 若超出 end_dt 則停止
    if current > end_dt:
        break

df_decisions = pd.DataFrame({'decision_time': decision_times})
df_decisions['decision_date'] = df_decisions['decision_time'].dt.date

print(f"   ✅ 決策點總數: {len(df_decisions):,}")

print("📂 4. 對齊觀測值（模擬 8 分鐘延遲，使用 T-8 的數據）...")
# 合併溫度數據：取決策時間前 8 分鐘的觀測值（即 timestamp <= decision_time - 8min）
# 為加速，我們用 merge_asof
df_temp['timestamp_dt'] = df_temp['timestamp']
df_decisions = df_decisions.sort_values('decision_time')
df_temp = df_temp.sort_values('timestamp')

# 計算對齊時間點 (T - 8 分鐘)
df_decisions['lookback_time'] = df_decisions['decision_time'] - timedelta(minutes=8)

# 使用 merge_asof 找到每個決策點最近的觀測值（必須 <= lookback_time）
merged = pd.merge_asof(
    df_decisions,
    df_temp[['timestamp', 'value', 'daily_max_temp', 'remaining_upside']],
    left_on='lookback_time',
    right_on='timestamp',
    direction='backward'  # 向後找（即找 <= lookback_time 的最新一筆）
)

# 捨棄無法對齊到觀測值的極早期點（例如 2016-12-08 06:00 之前）
merged = merged.dropna(subset=['value'])

print(f"   ✅ 對齊後樣本數: {len(merged):,}")

print("📂 5. 對齊每日預測（找該日期最新發布的預測）...")
# 將預測資料轉為以 target_date 和 forecast_datetime 為索引
df_forecast = df_forecast.sort_values(['target_date', 'forecast_datetime'])

# 對每個決策點，找出該 target_date 最新發布（但發布時間 <= decision_time）的預測
# 這裡用 apply 效率較低，但數據量 50 萬筆還算可接受
def get_latest_forecast(row):
    target_d = row['decision_date']
    dec_t = row['decision_time']
    # 找出該日期的所有預測
    sub = df_forecast[df_forecast['target_date'] == target_d]
    if sub.empty:
        return np.nan
    # 找出發布時間 <= 決策時間的最新一筆
    sub = sub[sub['forecast_datetime'] <= dec_t]
    if sub.empty:
        return np.nan
    return sub.iloc[-1]['forecast_max_temp']

# 若覺得 apply 太慢，可用 merge_asof 再次處理，此處為求簡潔先用 apply + tqdm
tqdm.pandas(desc="匹配預測數據")
merged['forecast_max_temp_aligned'] = merged.progress_apply(get_latest_forecast, axis=1)

print(f"   ✅ 匹配完成")

print("📂 6. 特徵工程（衍生變數）...")
# 計算過去 10 分鐘的斜率與波動（需重新計算，因為 merged 已對齊）
# 由於我們只有對齊點的單一數值，沒有過去 10 分鐘的原始資料，故需回頭從 df_temp 補
# 這裡簡化：先存檔，後續訓練時再補複雜特徵
# 但我們至少有這幾個基礎特徵：
merged['hour'] = merged['decision_time'].dt.hour
merged['minute'] = merged['decision_time'].dt.minute
merged['day_of_week'] = merged['decision_time'].dt.dayofweek
merged['forecast_gap'] = merged['forecast_max_temp_aligned'] - merged['value']

# 預測年齡（發布至今幾分鐘）— 因難以精確算出，先略過，後續可再強化

# 移除極端值（剩餘上行空間不合理為負，或過大）
merged = merged[merged['remaining_upside'] >= -2]  # 允許些微觀測誤差
merged = merged[merged['remaining_upside'] <= 15]  # 香港不可能升超過 15°C

print(f"   ✅ 最終樣本數: {len(merged):,}")

# 7. 儲存
merged.to_parquet(OUTPUT_PATH, index=False)
print(f"\n🎉 Feature Store 建立完成！")
print(f"📁 儲存於: {OUTPUT_PATH}")
print(f"📊 最終維度: {merged.shape}")
print(merged.head(10))