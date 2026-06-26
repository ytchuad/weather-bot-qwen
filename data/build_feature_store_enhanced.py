"""
build_feature_store_enhanced.py

建立強化版特徵儲存庫 (Feature Store) — 供 Model G 訓練使用。

輸出特徵 (共 15 個)：
  - 12 個 Model F 基礎特徵：value, humidity, pressure,
    temp_slope_30min, temp_volatility_30min,
    humid_delta_30min, pressure_delta_30min,
    forecast_gap, hour, minute, day_of_week, month
  - 3 個 Model C 錨定特徵 (新加入)：
    max_so_far, drop_from_max, time_since_max

資料來源：
  - 分鐘溫度/濕度/氣壓：hk_weather_raw/*.parquet
  - 每日官方預測：hk_daily_forecast/daily_forecast_clean.parquet

決策頻率：每 10 分鐘 (06:00 ~ 23:50)
時間延遲：嚴格遵守 T-8 分鐘 (模擬 i-lens 數據延遲)
時間範圍：2016-12-08 ~ 2026-06-23
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== 設定路徑 (自動適應執行目錄) ==========
import os
from pathlib import Path

# 取得腳本所在的目錄 (假設腳本放在 data 目錄下)
SCRIPT_DIR = Path(__file__).parent.absolute()

# 資料目錄：直接指向腳本所在目錄下的 hk_weather_raw
DATA_DIR = SCRIPT_DIR / "hk_weather_raw"
FORECAST_PATH = SCRIPT_DIR / "hk_daily_forecast" / "daily_forecast_clean.parquet"
OUTPUT_PATH = SCRIPT_DIR / "feature_store_enhanced.parquet"

print(f"📂 腳本目錄: {SCRIPT_DIR}")
print(f"📂 數據目錄: {DATA_DIR}")
print(f"📂 預測檔案: {FORECAST_PATH}")

# ========== 時間範圍設定 ==========
START_DATE = datetime(2016, 12, 8)
END_DATE = datetime(2026, 6, 23)

# 決策頻率：每 10 分鐘
DECISION_INTERVAL_MINUTES = 10
# 數據延遲 (分鐘)
DATA_LAG_MINUTES = 8
# 活躍時段 (只取 06:00 ~ 23:50)
ACTIVE_START_HOUR = 6
ACTIVE_END_HOUR = 23

print("=" * 60)
print("🚀 開始建立強化版 Feature Store (含錨定特徵)")
print("=" * 60)

# ======================================================================
# 步驟 1：讀取並合併所有原始觀測數據 (溫度, 濕度, 氣壓)
# ======================================================================
print("\n📂 步驟 1：讀取原始觀測數據...")

# 1.1 溫度
temp_files = glob.glob(str(DATA_DIR / "*_temperature.parquet"))
print(f"  找到 {len(temp_files)} 個溫度檔案")
df_temp = pd.concat([pd.read_parquet(f) for f in tqdm(temp_files, desc="讀取溫度")])
df_temp = df_temp.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
df_temp['date'] = df_temp['timestamp'].dt.date

# 1.2 濕度
humid_files = glob.glob(str(DATA_DIR / "*_humidity.parquet"))
print(f"  找到 {len(humid_files)} 個濕度檔案")
df_humid = pd.concat([pd.read_parquet(f) for f in tqdm(humid_files, desc="讀取濕度")])
df_humid = df_humid.drop_duplicates(subset=['timestamp'], keep='last')
df_humid = df_humid.rename(columns={'value': 'humidity'})
df_humid = df_humid[['timestamp', 'humidity']]

# 1.3 氣壓
press_files = glob.glob(str(DATA_DIR / "*_pressure.parquet"))
print(f"  找到 {len(press_files)} 個氣壓檔案")
df_press = pd.concat([pd.read_parquet(f) for f in tqdm(press_files, desc="讀取氣壓")])
df_press = df_press.drop_duplicates(subset=['timestamp'], keep='last')
df_press = df_press.rename(columns={'value': 'pressure'})
df_press = df_press[['timestamp', 'pressure']]

# 1.4 合併所有觀測值
df_obs = df_temp.merge(df_humid, on='timestamp', how='left').merge(df_press, on='timestamp', how='left')
print(f"✅ 合併完成，總觀測筆數: {len(df_obs):,}")
print(f"   溫度範圍: {df_obs['value'].min():.1f} ~ {df_obs['value'].max():.1f}°C")
print(f"   濕度範圍: {df_obs['humidity'].min():.1f} ~ {df_obs['humidity'].max():.1f}%")
print(f"   氣壓範圍: {df_obs['pressure'].min():.1f} ~ {df_obs['pressure'].max():.1f} hPa")

# ======================================================================
# 步驟 2：修正版錨定特徵 (參考 AI 代理建議)
# ======================================================================
print("\n📊 步驟 2：計算修正版錨定特徵...")

# 2.1 確保數據按日期與時間嚴格排序
df_obs = df_obs.sort_values(['date', 'timestamp']).copy()

# 2.2 計算累積最高溫 (max_so_far)
df_obs['max_so_far'] = df_obs.groupby('date')['value'].cummax()

# 2.3 計算「首次達到最高溫的時間」(is_new_high 邏輯)
# 只有當目前數值突破或達到之前的累積最高值時，才標記為「新高時點」
# 這避免了「數值持平」時被誤標記為新高的問題
df_obs['prev_max'] = df_obs.groupby('date')['max_so_far'].shift(1).fillna(-999)
df_obs['is_new_high'] = (df_obs['value'] >= df_obs['prev_max'])

# 記錄達到該最高溫的時間 (僅在 is_new_high 為 True 時記錄)
df_obs['max_so_far_time'] = np.where(
    (df_obs['value'] == df_obs['max_so_far']) & (df_obs['is_new_high']),
    df_obs['timestamp'],
    pd.NaT
)

# 將最高溫時間向前填充（fill forward）至同一天的所有後續分鐘
df_obs['max_so_far_time'] = df_obs.groupby('date')['max_so_far_time'].ffill()

# 2.4 計算 time_since_max (分鐘)
df_obs['time_since_max'] = (
    (df_obs['timestamp'] - pd.to_datetime(df_obs['max_so_far_time'])).dt.total_seconds() / 60
).fillna(0).clip(lower=0)

# 2.5 計算 drop_from_max
df_obs['drop_from_max'] = df_obs['max_so_far'] - df_obs['value']

print(f"   ✅ 修正版錨定特徵計算完成")
print(f"      max_so_far 範圍: {df_obs['max_so_far'].min():.1f} ~ {df_obs['max_so_far'].max():.1f}°C")
print(f"      time_since_max 中位數: {df_obs['time_since_max'].median():.1f} 分鐘")

# ======================================================================
# 步驟 3：計算每日最終最高溫 & 修正版 Label
# ======================================================================
print("\n📊 步驟 3：計算修正版 Label...")

daily_max = df_obs.groupby('date')['value'].max().reset_index()
daily_max.columns = ['date', 'daily_max_temp']
df_obs = df_obs.merge(daily_max, on='date', how='left')

# 🔥 關鍵修正點：
# Label = 每日最終最高溫 - 目前為止累積最高溫（而非當前溫度）
# 如此一來，下雨導致當前溫度暴跌時，Label 不會變大，反而維持低點。
df_obs['remaining_upside'] = df_obs['daily_max_temp'] - df_obs['max_so_far']
df_obs['remaining_upside'] = df_obs['remaining_upside'].clip(lower=0)  # 物理上不小於0

print(f"   ✅ Label 計算完成 (使用 max_so_far 錨定)")

# ======================================================================
# 步驟 4：讀取每日預測資料 (for forecast_gap)
# ======================================================================
print("\n📂 步驟 4：讀取每日官方預測...")
df_forecast = pd.read_parquet(FORECAST_PATH)
df_forecast['forecast_datetime'] = pd.to_datetime(
    df_forecast['forecast_issue_date'] + ' ' + df_forecast['forecast_issue_time'],
    format='%Y-%m-%d %H:%M',
    errors='coerce'
)
df_forecast['target_date'] = pd.to_datetime(df_forecast['query_date']).dt.date
df_forecast = df_forecast[['target_date', 'forecast_datetime', 'forecast_max_temp']].drop_duplicates()
print(f"   ✅ 預測資料筆數: {len(df_forecast):,}")

# ======================================================================
# 步驟 5：建立決策點時間軸 (每 10 分鐘)
# ======================================================================
print("\n📊 步驟 5：建立決策點時間軸...")
decision_times = []
current = START_DATE.replace(hour=ACTIVE_START_HOUR, minute=0, second=0, microsecond=0)

while current <= END_DATE:
    decision_times.append(current)
    current += timedelta(minutes=DECISION_INTERVAL_MINUTES)
    # 若超過 ACTIVE_END_HOUR:50，跳到隔天 ACTIVE_START_HOUR:00
    if current.hour >= ACTIVE_END_HOUR + 1:
        current = current.replace(hour=ACTIVE_START_HOUR, minute=0, second=0, microsecond=0)
        current += timedelta(days=1)
    if current > END_DATE:
        break

df_decisions = pd.DataFrame({'decision_time': decision_times})
df_decisions['decision_date'] = df_decisions['decision_time'].dt.date
df_decisions['lookback_time'] = df_decisions['decision_time'] - timedelta(minutes=DATA_LAG_MINUTES)

print(f"   ✅ 決策點總數: {len(df_decisions):,}")

# ======================================================================
# 步驟 6：對齊觀測值 (使用 merge_asof, T-8 延遲)
# ======================================================================
print("\n📊 步驟 6：對齊觀測值 (T-8 延遲)...")
df_obs_sorted = df_obs.sort_values('timestamp')
df_decisions_sorted = df_decisions.sort_values('lookback_time')

# 合併基礎數值 (溫度, 濕度, 氣壓, 錨定特徵, label)
merged = pd.merge_asof(
    df_decisions_sorted,
    df_obs_sorted[['timestamp', 'value', 'humidity', 'pressure', 
                   'max_so_far', 'drop_from_max', 'time_since_max', 
                   'daily_max_temp', 'remaining_upside']],
    left_on='lookback_time',
    right_on='timestamp',
    direction='backward'  # 向後找 (即找 <= lookback_time 的最新一筆)
)

# 移除無法對齊到觀測值的極早期點
before_drop = len(merged)
merged = merged.dropna(subset=['value'])
print(f"   ✅ 對齊完成，有效樣本: {len(merged):,} (移除 {before_drop - len(merged):,} 筆)")

# ======================================================================
# 步驟 7：計算滑動窗口特徵 (過去 30 分鐘)
# ======================================================================
print("\n📊 步驟 7：計算滑動窗口特徵...")

# 按日期分組進行滾動計算
def calc_rolling_features(group):
    group = group.sort_values('timestamp').reset_index(drop=True)
    group['temp_slope_30min'] = np.nan
    group['temp_volatility_30min'] = np.nan
    group['humid_delta_30min'] = np.nan
    group['pressure_delta_30min'] = np.nan
    
    for i in range(3, len(group)):  # 至少需要 3 個點 (30 分鐘)
        window = group.iloc[i-3:i]
        if len(window) >= 2:
            # 升溫斜率 (線性迴歸)
            x = np.arange(len(window))
            y = window['value'].values
            if len(y) == len(x) and len(y) > 1:
                slope, _ = np.polyfit(x, y, 1)
                group.loc[i, 'temp_slope_30min'] = slope
            # 溫度波動 (標準差)
            group.loc[i, 'temp_volatility_30min'] = window['value'].std()
            # 濕度變化
            if not window['humidity'].isna().all():
                group.loc[i, 'humid_delta_30min'] = window['humidity'].iloc[-1] - window['humidity'].iloc[0]
            # 氣壓變化
            if not window['pressure'].isna().all():
                group.loc[i, 'pressure_delta_30min'] = window['pressure'].iloc[-1] - window['pressure'].iloc[0]
    return group

merged = merged.sort_values(['decision_date', 'timestamp'])
merged = merged.groupby('decision_date', group_keys=False).apply(calc_rolling_features)

print(f"   ✅ 滑動窗口特徵計算完成")

# ======================================================================
# 步驟 8：對齊每日官方預測 (forecast_gap)
# ======================================================================
print("\n📊 步驟 8：對齊每日官方預測 (修正 gap 計算)...")

def get_latest_forecast(row):
    target_d = row['decision_date']
    dec_t = row['decision_time']
    sub = df_forecast[df_forecast['target_date'] == target_d]
    if sub.empty:
        return np.nan
    sub = sub[sub['forecast_datetime'] <= dec_t]
    if sub.empty:
        return np.nan
    return sub.iloc[-1]['forecast_max_temp']

tqdm.pandas(desc="匹配預測")
merged['forecast_max_temp_aligned'] = merged.progress_apply(get_latest_forecast, axis=1)

# 🔥 關鍵修正點：
# 舊版: forecast_gap = forecast_max - value (被雨天膨脹)
# 新版: forecast_gap_from_max = forecast_max - max_so_far (雨天不受影響)
merged['forecast_gap_from_max'] = merged['forecast_max_temp_aligned'] - merged['max_so_far']
# 若無預測資料，設為 0
merged['forecast_gap_from_max'] = merged['forecast_gap_from_max'].fillna(0)

# 為了保險，保留舊的 forecast_gap 但改名為備用（可選）
# merged['forecast_gap'] = merged['forecast_max_temp_aligned'] - merged['value']

print(f"   ✅ 預測對齊完成，forecast_gap_from_max 計算完成")

# ======================================================================
# 步驟 9：加入時間特徵
# ======================================================================
print("\n📊 步驟 9：加入時間特徵...")
merged['hour'] = merged['decision_time'].dt.hour
merged['minute'] = merged['decision_time'].dt.minute
merged['day_of_week'] = merged['decision_time'].dt.dayofweek  # 0=Monday, 6=Sunday
merged['month'] = merged['decision_time'].dt.month

print(f"   ✅ 時間特徵完成")

# ======================================================================
# 步驟 10：過濾極端值與整理輸出
# ======================================================================
print("\n📊 步驟 10：過濾與整理最終資料集...")

# 過濾極端值
merged = merged[merged['remaining_upside'] >= -2]
merged = merged[merged['remaining_upside'] <= 15]
merged = merged[merged['value'].between(0, 40)]
merged = merged[merged['humidity'].between(0, 100)]

# 刪除無法計算斜率的早期樣本 (前 3 個點)
merged = merged.dropna(subset=['temp_slope_30min'])

# 填補可能的 NaN (極少數情況)
for col in ['humid_delta_30min', 'pressure_delta_30min', 'forecast_gap']:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)

print(f"   ✅ 最終樣本數: {len(merged):,}")

# ======================================================================
# 步驟 11：儲存最終 Feature Store
# ======================================================================
print("\n💾 步驟 11：儲存 Feature Store...")

# 選擇最終輸出欄位 (共 15 個特徵 + 輔助欄位)
final_cols = [
    'decision_time', 'timestamp',  # 輔助 (timestamp 是 lookback_time)
    'value', 'humidity', 'pressure',
    'temp_slope_30min', 'temp_volatility_30min',
    'humid_delta_30min', 'pressure_delta_30min',
    'forecast_gap',
    'max_so_far', 'drop_from_max', 'time_since_max',  # ✅ Model C 錨定特徵
    'hour', 'minute', 'day_of_week', 'month',
    'remaining_upside'  # Label
]

df_final = merged[final_cols].copy()
df_final = df_final.dropna(subset=['max_so_far', 'drop_from_max', 'time_since_max'])

# 儲存為 Parquet
df_final.to_parquet(OUTPUT_PATH, index=False)

print(f"\n🎉 Feature Store 強化版建立完成！")
print(f"📁 儲存於: {OUTPUT_PATH}")
print(f"📊 最終維度: {df_final.shape}")
print(f"📋 特徵清單: {list(df_final.columns)}")
print("\n✅ 特徵統計摘要:")
print(df_final.describe())

print("\n" + "=" * 60)
print("✅ 所有步驟完成！現在請執行 train_model_g.py 開始訓練。")
print("=" * 60)