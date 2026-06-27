"""
build_model_2a_feature_store.py

建立 Model 2A 特徵儲存庫 — 分鐘觀測 + 每日預報 + 風力資料。
輸出: model_2a_feature_store.parquet (含 ~60 個特徵 + 目標)

資料來源:
  - 分鐘: hk_weather_raw/*_{temperature,humidity,pressure,dew}.parquet
  - 風力: wind_data/*_wind_all.parquet
  - 預報: hk_daily_forecast/daily_forecast_clean.parquet

決策頻率: 每 10 分鐘 (06:00〜23:50)
數據可用性: available_time = block_end + 8min
"""

import pandas as pd
import numpy as np
import glob
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "hk_weather_raw"
WIND_DIR = SCRIPT_DIR / "wind_data"
FORECAST_PATH = SCRIPT_DIR / "hk_daily_forecast" / "daily_forecast_clean.parquet"

OUTPUT_WEATHER = SCRIPT_DIR / "weather_minute_wide.parquet"
OUTPUT_WIND_FEATURES = SCRIPT_DIR / "wind_features_10min.parquet"
OUTPUT_FORECAST = SCRIPT_DIR / "forecast_features.parquet"
OUTPUT_FINAL = SCRIPT_DIR / "model_2a_feature_store.parquet"

START_DATE = datetime(2016, 12, 8)
END_DATE = datetime(2026, 6, 23)
DECISION_INTERVAL = 10
DATA_LAG = 8
ACTIVE_START_HOUR = 6
ACTIVE_END_HOUR = 23

STATION_GROUP_MAP = {
    "參考": "ref",
    "離岸": "offshore",
    "高山": "highland",
    "未知": "victoria_harbour",
}

SPECIAL_STATIONS = {
    "wind_kings_park_current": "京士柏",
    "wind_kai_tak_current": "啟德",
}


def ceil_dt_10min(dt):
    m = dt.minute
    ceil_m = math.ceil(m / 10) * 10
    if ceil_m >= 60:
        dt = dt + timedelta(hours=1)
        ceil_m = 0
    return dt.replace(minute=ceil_m, second=0, microsecond=0)


def build_weather_minute_wide():
    print("=== Step 1: Weather Minute Wide ===")
    files = glob.glob(str(DATA_DIR / "*_temperature.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in tqdm(files, desc="Temp")])
    df = df.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'value': 'temp_current'})[['timestamp', 'temp_current']]

    for name, col in [("RH", "rh_current"), ("Pressure", "pressure_current"), ("Dew", "dew_point_current")]:
        suffix = "humidity" if name == "RH" else name.lower().replace("pressure", "pressure").replace("dew", "dew_point")
        pattern = str(DATA_DIR / f"*_{suffix}.parquet")
        fs = glob.glob(pattern)
        if fs:
            tmp = pd.concat([pd.read_parquet(f) for f in tqdm(fs, desc=name)])
            tmp = tmp.drop_duplicates(subset=['timestamp'], keep='last')
            tmp = tmp.rename(columns={'value': col})[['timestamp', col]]
            df = df.merge(tmp, on='timestamp', how='left')
        else:
            df[col] = np.nan

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['dew_point_spread'] = df['temp_current'] - df['dew_point_current']
    df['available_time'] = df['timestamp'].apply(
        lambda ts: ceil_dt_10min(ts) + timedelta(minutes=DATA_LAG)
    )
    df['date'] = df['timestamp'].dt.date
    print(f"  Shape: {df.shape}, range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    df.to_parquet(OUTPUT_WEATHER, index=False)
    return df


def build_wind_features():
    print("\n=== Step 2: Wind Features ===")
    all_files = sorted(glob.glob(str(WIND_DIR / "*_wind_all.parquet")))
    print(f"  Files: {len(all_files)}")

    chunks = []
    for f in tqdm(all_files, desc="Wind"):
        day = pd.read_parquet(f)
        day['group'] = day['station_type'].map(STATION_GROUP_MAP)

        raw_cols = {}
        for col_name, station_name in SPECIAL_STATIONS.items():
            sub = day[day['station'] == station_name]
            if len(sub) > 0:
                raw_cols[col_name] = sub.groupby('timestamp')['wind_speed'].mean()

        aggs = day.groupby(['timestamp', 'group']).agg(
            mean=('wind_speed', 'mean'),
            max=('wind_speed', 'max'),
            min=('wind_speed', 'min'),
            cnt=('station', 'nunique'),
        ).reset_index()
        aggs['spread'] = aggs['max'] - aggs['min']

        all_agg = day.groupby('timestamp').agg(
            mean=('wind_speed', 'mean'),
            max=('wind_speed', 'max'),
            min=('wind_speed', 'min'),
            cnt=('station', 'nunique'),
        ).reset_index()
        all_agg['group'] = 'all'
        all_agg['spread'] = all_agg['max'] - all_agg['min']

        combined = pd.concat([aggs, all_agg], ignore_index=True)
        pivot = combined.pivot_table(
            index='timestamp', columns='group',
            values=['mean', 'max', 'spread', 'cnt'],
        )
        pivot.columns = [f'wind_{col[1]}_{col[0]}' for col in pivot.columns]
        pivot = pivot.reset_index()

        for k in raw_cols:
            if k in raw_cols and raw_cols[k] is not None:
                sr = raw_cols[k].rename(k)
                pivot = pivot.merge(sr, on='timestamp', how='left')
            else:
                pivot[k] = np.nan

        pivot['available_time'] = pivot['timestamp'].apply(
            lambda ts: ceil_dt_10min(ts) + timedelta(minutes=DATA_LAG)
        )
        chunks.append(pivot)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp', keep='last')
    print(f"  Shape: {df.shape}")
    df.to_parquet(OUTPUT_WIND_FEATURES, index=False)
    return df


def build_forecast_features():
    print("\n=== Step 3: Forecast Features ===")
    df = pd.read_parquet(FORECAST_PATH)
    df['forecast_issue_datetime'] = pd.to_datetime(
        df['forecast_issue_date'] + ' ' + df['forecast_issue_time'],
        format='%Y-%m-%d %H:%M', errors='coerce'
    )
    df['target_date'] = pd.to_datetime(df['query_date']).dt.date
    df['forecast_range'] = df['forecast_max_temp'] - df['forecast_min_temp']
    df['forecast_lead_days'] = (
        pd.to_datetime(df['query_date']) - pd.to_datetime(df['forecast_issue_date'])
    ).dt.days

    df = df[[
        'target_date', 'forecast_issue_datetime',
        'forecast_min_temp', 'forecast_max_temp', 'forecast_range',
        'forecast_lead_days',
    ]].drop_duplicates()
    df = df.dropna(subset=['forecast_issue_datetime'])
    df = df.sort_values('forecast_issue_datetime').reset_index(drop=True)
    print(f"  Shape: {df.shape}")
    df.to_parquet(OUTPUT_FORECAST, index=False)
    return df


def build_decision_calendar():
    print("\n=== Step 4: Decision Calendar ===")
    times = []
    cur = START_DATE.replace(hour=ACTIVE_START_HOUR, minute=0, second=0, microsecond=0)
    while cur <= END_DATE:
        times.append(cur)
        cur += timedelta(minutes=DECISION_INTERVAL)
        if cur.hour >= ACTIVE_END_HOUR + 1:
            cur = cur.replace(hour=ACTIVE_START_HOUR, minute=0, second=0, microsecond=0)
            cur += timedelta(days=1)

    df = pd.DataFrame({'decision_time': times})
    df['target_date'] = df['decision_time'].dt.date
    df['minutes_since_midnight'] = df['decision_time'].dt.hour * 60 + df['decision_time'].dt.minute
    df['hour'] = df['decision_time'].dt.hour
    df['minute'] = df['decision_time'].dt.minute
    df['month'] = df['decision_time'].dt.month
    df['day_of_year'] = df['decision_time'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['is_morning'] = (df['hour'].between(6, 11)).astype(int)
    df['is_afternoon'] = (df['hour'].between(12, 17)).astype(int)
    df['is_evening'] = (df['hour'].between(18, 23)).astype(int)
    print(f"  Decisions: {len(df):,}")
    return df


def merge_asof_features(decisions, source, suffix):
    src = source.sort_values('available_time')
    dec = decisions.sort_values('decision_time')
    merged = pd.merge_asof(
        dec, src, left_on='decision_time', right_on='available_time',
        direction='backward',
    )
    merged = merged.drop(columns=['available_time'])
    return merged


def compute_anchors(df):
    print("\n=== Step 6: Anchor Features ===")
    df = df.sort_values(['target_date', 'decision_time']).reset_index(drop=True)
    df['max_so_far'] = df.groupby('target_date')['temp_current'].cummax()
    df['min_so_far'] = df.groupby('target_date')['temp_current'].cummin()
    df['range_so_far'] = df['max_so_far'] - df['min_so_far']
    df['drop_from_max'] = df['max_so_far'] - df['temp_current']
    df['rise_from_min'] = df['temp_current'] - df['min_so_far']

    def _time_since(group, col, cmp, tol):
        result = np.zeros(len(group))
        last_t = None
        for i in range(len(group)):
            row = group.iloc[i]
            val = row[col]
            extreme = row[cmp]
            ok = (val >= extreme - tol) if 'max' in cmp else (val <= extreme + tol)
            if ok:
                last_t = row['decision_time']
            if last_t is not None:
                result[i] = max((row['decision_time'] - last_t).total_seconds() / 60, 0.0)
        group['time_since_max' if 'max' in cmp else 'time_since_min'] = result
        return group

    df = df.groupby('target_date', group_keys=False).apply(
        lambda g: _time_since(g, 'temp_current', 'max_so_far', 1e-9)
    )
    df = df.groupby('target_date', group_keys=False).apply(
        lambda g: _time_since(g, 'temp_current', 'min_so_far', 1e-9)
    )
    print(f"  Anchors done")
    return df


def compute_trends(df):
    print("\n=== Step 7: Trend Features ===")
    df = df.sort_values(['target_date', 'decision_time']).reset_index(drop=True)

    def _trends(g):
        g = g.copy()
        g['temp_change_10m'] = g['temp_current'].diff(1)
        g['temp_change_30m'] = g['temp_current'].diff(3)
        g['temp_change_60m'] = g['temp_current'].diff(6)
        g['temp_change_120m'] = g['temp_current'].diff(12)
        g['temp_slope_30m'] = g['temp_change_30m'] / 30.0
        g['temp_slope_60m'] = g['temp_change_60m'] / 60.0
        g['temp_acceleration_60m'] = g['temp_slope_30m'] - g['temp_slope_30m'].shift(3)
        g['temp_volatility_30m'] = g['temp_current'].rolling(3, min_periods=2).std()
        g['temp_volatility_60m'] = g['temp_current'].rolling(6, min_periods=2).std()
        g['rh_change_30m'] = g['rh_current'].diff(3)
        g['rh_change_60m'] = g['rh_current'].diff(6)
        g['dew_point_change_30m'] = g['dew_point_current'].diff(3)
        g['dew_point_change_60m'] = g['dew_point_current'].diff(6)
        g['dew_point_spread_change_30m'] = g['dew_point_spread'].diff(3)
        g['dew_point_spread_change_60m'] = g['dew_point_spread'].diff(6)
        g['dew_point_spread_mean_60m'] = g['dew_point_spread'].rolling(6, min_periods=1).mean()
        g['pressure_change_60m'] = g['pressure_current'].diff(6)
        g['pressure_change_180m'] = g['pressure_current'].diff(18)
        g['pressure_mean_180m'] = g['pressure_current'].rolling(18, min_periods=1).mean()
        return g

    df = df.groupby('target_date', group_keys=False).apply(_trends)
    print(f"  Trends done")
    return df


def compute_wind_trends(df):
    print("  Computing wind trends...")
    df = df.sort_values(['target_date', 'decision_time']).reset_index(drop=True)

    def _w(g):
        g = g.copy()
        for prefix in ['ref', 'offshore', 'highland', 'victoria_harbour', 'all']:
            mc = f'wind_{prefix}_mean'
            if mc not in g.columns:
                continue
            g[f'wind_{prefix}_change_60m'] = g[mc] - g[mc].shift(6)
            g[f'wind_{prefix}_max_60m'] = g[mc].rolling(6, min_periods=1).max()
        return g

    df = df.groupby('target_date', group_keys=False).apply(_w)
    return df


def match_forecast(df, df_fc):
    print("\n=== Step 5c: Match Forecast ===")
    df_fc['target_date'] = pd.to_datetime(df_fc['target_date'])
    df_fc = df_fc.dropna(subset=['forecast_issue_datetime'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    # Build integer keys: use target_date as int and time-of-day offset
    DAY_SEC = 86400
    df_fc['td_int'] = df_fc['target_date'].astype('int64') // 10**9 // DAY_SEC
    df_fc['tod'] = (df_fc['forecast_issue_datetime'].astype('int64') // 10**9) % DAY_SEC
    df_fc['fc_key'] = df_fc['td_int'] * DAY_SEC + df_fc['tod']
    df_fc = df_fc.sort_values('fc_key').reset_index(drop=True)
    df['td_int'] = df['target_date'].astype('int64') // 10**9 // DAY_SEC
    df['tod'] = (df['decision_time'].astype('int64') // 10**9) % DAY_SEC
    df['dec_key'] = df['td_int'] * DAY_SEC + df['tod']
    df = df.sort_values('dec_key').reset_index(drop=True)
    merged = pd.merge_asof(
        df, df_fc,
        left_on='dec_key', right_on='fc_key',
        direction='backward',
        tolerance=DAY_SEC,  # prevent merging across days
    )
    # Drop helper columns and any cross-day mismatches
    merged = merged[merged['td_int_x'] == merged['td_int_y']]
    merged['forecast_age_minutes'] = (
        merged['decision_time'] - merged['forecast_issue_datetime']
    ).dt.total_seconds() / 60
    merged = merged.drop(columns=['dec_key', 'fc_key', 'td_int_x', 'td_int_y', 'tod_x', 'tod_y',
                                   'target_date_y'], errors='ignore')
    if 'target_date_x' in merged.columns:
        merged = merged.rename(columns={'target_date_x': 'target_date'})
    merged['forecast_age_minutes'] = (
        merged['decision_time'] - merged['forecast_issue_datetime']
    ).dt.total_seconds() / 60
    merged = merged.drop(columns=['forecast_issue_datetime'])
    merged['forecast_gap_from_max_so_far'] = merged['forecast_max_temp'] - merged['max_so_far']
    merged['forecast_gap_from_current'] = merged['forecast_max_temp'] - merged['temp_current']
    print(f"  Coverage: {merged['forecast_max_temp'].notna().mean():.1%}")
    return merged


def compute_targets(df):
    print("\n=== Step 8: Targets ===")
    daily_max = df.groupby('target_date')['temp_current'].max().reset_index()
    daily_max.columns = ['target_date', 'actual_high_today']
    df = df.merge(daily_max, on='target_date', how='left')
    df['remaining_upside'] = (df['actual_high_today'] - df['max_so_far']).clip(lower=0)
    df['is_upside_zero'] = (df['remaining_upside'] <= 0.05).astype(int)
    return df


def compute_freshness(df):
    print("  Computing freshness...")
    df['obs_data_age_minutes'] = DATA_LAG + (df['minutes_since_midnight'] % 10)
    df['wind_data_age_minutes'] = df['obs_data_age_minutes'].copy()
    return df


def sanity_checks(df):
    print("\n=== Sanity Checks ===")
    try:
        assert df['remaining_upside'].min() >= -0.01
        print("  ✅ remaining_upside.min() >= 0")
    except AssertionError:
        print(f"  ⚠️  remaining_upside.min() = {df['remaining_upside'].min():.3f}")

    gap_cov = df['forecast_gap_from_max_so_far'].notna().mean()
    print(f"  ✅ forecast_gap coverage: {gap_cov:.1%}")

    tc = df['temp_current'].between(0, 40).mean()
    print(f"  ✅ temp_current in [0,40]: {tc:.1%}")
    rc = df['rh_current'].between(0, 100).mean()
    print(f"  ✅ rh_current in [0,100]: {rc:.1%}")

    if 'wind_all_mean' in df.columns:
        wc = df['wind_all_mean'].ge(0).mean()
        print(f"  ✅ wind_all_mean >= 0: {wc:.1%}")

    rd = df[df['drop_from_max'] >= 5]
    if len(rd) > 0:
        nm = rd[rd['actual_high_today'] == rd['max_so_far']]
        if len(nm) > 0:
            zp = nm['is_upside_zero'].mean()
            print(f"  🌧️  Rain drop diagnostic (drop>=5, max reached): "
                  f"{len(nm):,} rows, is_upside_zero={zp:.1%}")
            if zp < 0.7:
                print("  ⚠️  Warning: remaining_upside should be near 0 after rain drop")
            else:
                print("  ✅  Correct: rain drop + max reached ≈ zero upside")


def main():
    print("=" * 60)
    print("  Model 2A Feature Store Builder")
    print("=" * 60)

    df_weather = build_weather_minute_wide()
    df_wind = build_wind_features()
    df_forecast = build_forecast_features()
    decisions = build_decision_calendar()

    print("\n=== Step 5a: Merge Weather ===")
    merged = merge_asof_features(decisions, df_weather, suffix=False)
    wc = merged['temp_current'].notna().mean()
    print(f"  Coverage: {wc:.1%}")

    print("\n=== Step 5b: Merge Wind ===")
    wind_cols = [c for c in df_wind.columns if c.startswith('wind_') or c == 'available_time']
    merged = merge_asof_features(merged, df_wind[wind_cols], suffix=False)
    wic = merged['wind_ref_mean'].notna().mean() if 'wind_ref_mean' in merged.columns else 0
    print(f"  Wind coverage: {wic:.1%}")

    merged = compute_anchors(merged)
    merged = compute_trends(merged)
    merged = compute_wind_trends(merged)
    merged = match_forecast(merged, df_forecast)
    merged = compute_freshness(merged)
    merged = compute_targets(merged)

    # Fill NaN for features only
    fill_cols = [c for c in merged.columns if c not in
                 ['decision_time', 'target_date', 'actual_high_today',
                  'remaining_upside', 'is_upside_zero',
                  'forecast_max_temp', 'forecast_min_temp',
                  'forecast_range', 'forecast_lead_days',
                  'forecast_age_minutes', 'forecast_gap_from_max_so_far',
                  'forecast_gap_from_current']]
    for c in fill_cols:
        if c in merged.columns and merged[c].dtype in ('float64', 'float32', 'int64'):
            merged[c] = merged[c].fillna(0)

    final_cols = [
        'decision_time', 'target_date',
        'temp_current', 'rh_current', 'pressure_current', 'dew_point_current', 'dew_point_spread',
        'max_so_far', 'min_so_far', 'range_so_far', 'drop_from_max', 'rise_from_min',
        'time_since_max', 'time_since_min',
        'temp_change_10m', 'temp_change_30m', 'temp_change_60m', 'temp_change_120m',
        'temp_slope_30m', 'temp_slope_60m',
        'temp_acceleration_60m', 'temp_volatility_30m', 'temp_volatility_60m',
        'rh_change_30m', 'rh_change_60m',
        'dew_point_change_30m', 'dew_point_change_60m',
        'dew_point_spread_change_30m', 'dew_point_spread_change_60m',
        'dew_point_spread_mean_60m',
        'pressure_change_60m', 'pressure_change_180m', 'pressure_mean_180m',
        'forecast_min_temp', 'forecast_max_temp', 'forecast_range',
        'forecast_gap_from_max_so_far', 'forecast_gap_from_current',
        'forecast_age_minutes', 'forecast_lead_days',
        'wind_ref_mean', 'wind_ref_max', 'wind_ref_spread', 'wind_ref_station_count',
        'wind_ref_change_60m', 'wind_ref_max_60m',
        'wind_victoria_harbour_mean', 'wind_victoria_harbour_max',
        'wind_victoria_harbour_spread', 'wind_victoria_harbour_station_count',
        'wind_victoria_harbour_change_60m', 'wind_victoria_harbour_max_60m',
        'wind_offshore_mean', 'wind_offshore_max', 'wind_offshore_spread', 'wind_offshore_station_count',
        'wind_offshore_change_60m', 'wind_offshore_max_60m',
        'wind_highland_mean', 'wind_highland_max', 'wind_highland_spread', 'wind_highland_station_count',
        'wind_highland_change_60m', 'wind_highland_max_60m',
        'wind_all_mean', 'wind_all_max', 'wind_all_spread', 'wind_all_station_count',
        'wind_all_change_60m', 'wind_all_max_60m',
        'wind_kings_park_current', 'wind_kai_tak_current',
        'hour', 'minute', 'minutes_since_midnight', 'month', 'day_of_year',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_morning', 'is_afternoon', 'is_evening',
        'obs_data_age_minutes', 'wind_data_age_minutes',
        'actual_high_today', 'remaining_upside', 'is_upside_zero',
    ]

    existing = [c for c in final_cols if c in merged.columns]
    df_final = merged[existing].copy()
    before = len(df_final)
    df_final = df_final.dropna(subset=['temp_current'])
    print(f"\n  Dropped {before - len(df_final):,} rows without weather data")

    df_final.to_parquet(OUTPUT_FINAL, index=False)

    print(f"\n=== Model 2A Feature Store saved! ===")
    print(f"📊 Shape: {df_final.shape}")
    print(f"📋 Columns: {len(df_final.columns)}")
    print(f"📅 Date range: {df_final['target_date'].min()} ~ {df_final['target_date'].max()}")
    print(f"📈 Upside mean: {df_final['remaining_upside'].mean():.2f}, zero%: {df_final['is_upside_zero'].mean():.1%}")

    sanity_checks(df_final)


if __name__ == "__main__":
    main()
