# features/build_intraday_ml_dataset.py (v3 - complete pipeline with rainfall features and target validation)
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INTRADAY_PATH = Path('data/intraday_hko_10min.parquet')
DAILY_PATH = Path('data/hko_tmax_historical.parquet')
FORECAST_PATH = Path('data/hko_historical_forecasts.parquet')
RAINFALL_FEAT_PATH = Path('data/hko_rainfall_15min_features.parquet')
OUTPUT_PATH = Path('data/intraday_ml_train.parquet')


def add_time_features(df):
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
    df['remaining_minutes_to_midnight'] = 1440 - df['minutes_since_midnight']
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    return df


def add_intraday_state_features(df):
    df = df.sort_values(['date', 'datetime']).reset_index(drop=True)
    g = df.groupby('date', group_keys=False)

    # Cumulative extremes
    df['max_so_far'] = g['temp'].cummax()
    df['min_so_far'] = g['temp'].cummin()
    df['range_so_far'] = df['max_so_far'] - df['min_so_far']
    df['drop_from_max'] = df['max_so_far'] - df['temp']
    df['rise_from_min'] = df['temp'] - df['min_so_far']

    # Lag features and rolling stats at multiple windows
    lag_steps = {
        '10min': 1,
        '30min': 3,
        '60min': 6,
        '120min': 12,
    }
    for label, steps in lag_steps.items():
        df[f'temp_lag_{label}'] = g['temp'].shift(steps).fillna(df['temp'])
        df[f'temp_change_{label}'] = df['temp'] - df[f'temp_lag_{label}']
        df[f'rolling_mean_{label}'] = g['temp'].transform(
            lambda x, s=steps: x.rolling(s, min_periods=1).mean())
        df[f'rolling_std_{label}'] = g['temp'].transform(
            lambda x, s=steps: x.rolling(s, min_periods=1).std())

    # Time since extreme was first reached (point-in-time correct)
    df['time_since_max_so_far'] = _compute_minutes_since_extreme(df, extreme_col='max_so_far')
    df['time_since_min_so_far'] = _compute_minutes_since_extreme(df, extreme_col='min_so_far')

    # Discretised buckets
    df['max_bucket'] = (df['max_so_far'] // 0.5) * 0.5
    df['min_bucket'] = (df['min_so_far'] // 0.5) * 0.5

    return df


def _compute_minutes_since_extreme(df, extreme_col):
    """For each row, compute minutes since the current extreme value was first reached.

    This is point-in-time correct: at each row, we look at the extreme value
    as of that row and find when it was first achieved within the same day.
    """
    result = np.zeros(len(df))
    for date_val, group in df.groupby('date', sort=False):
        idx = group.index.values
        temps = group['temp'].values
        datetimes = group['datetime'].values
        extremes = group[extreme_col].values

        last_extreme_time = None
        current_extreme_val = None
        for i in range(len(idx)):
            # The extreme as of this row
            ext_val = extremes[i]
            row_temp = temps[i]

            # If this row's temp reaches the current extreme, update the timestamp
            if row_temp >= ext_val - 1e-9:
                last_extreme_time = datetimes[i]
                current_extreme_val = row_temp

            if last_extreme_time is not None:
                delta = (datetimes[i] - last_extreme_time).astype('timedelta64[ns]').astype(float) / 6e10
                result[idx[i]] = max(delta, 0.0)
            else:
                result[idx[i]] = 0.0

    return pd.Series(result, index=df.index)


def merge_forecast(df, forecast_df):
    forecast_df = forecast_df.copy()
    forecast_df['publish_date'] = pd.to_datetime(forecast_df['publish_date']).dt.date
    forecast_df['target_date'] = pd.to_datetime(forecast_df['target_date']).dt.date
    fc_d1 = forecast_df[forecast_df['publish_date'] == forecast_df['target_date'] - pd.Timedelta(days=1)]
    fc_d0 = forecast_df[forecast_df['publish_date'] == forecast_df['target_date']]
    fc_d1 = fc_d1[['target_date', 'predicted_max_temp', 'predicted_min_temp']].rename(
        columns={'predicted_max_temp': 'forecast_tmax_d1', 'predicted_min_temp': 'forecast_tmin_d1'})
    fc_d0 = fc_d0[['target_date', 'predicted_max_temp', 'predicted_min_temp']].rename(
        columns={'predicted_max_temp': 'forecast_tmax_d0', 'predicted_min_temp': 'forecast_tmin_d0'})
    df['date_dt'] = df['datetime'].dt.date
    df = df.merge(fc_d1, left_on='date_dt', right_on='target_date', how='left')
    df = df.merge(fc_d0, left_on='date_dt', right_on='target_date', how='left', suffixes=('', '_d0'))
    mask_after_1130 = (df['hour'] >= 12) | ((df['hour'] == 11) & (df['minute'] >= 30))
    df['forecast_tmax'] = np.where(mask_after_1130,
                                   df['forecast_tmax_d0'].fillna(df['forecast_tmax_d1']),
                                   df['forecast_tmax_d1'].fillna(df['forecast_tmax_d0']))
    df['forecast_tmin'] = np.where(mask_after_1130,
                                   df['forecast_tmin_d0'].fillna(df['forecast_tmin_d1']),
                                   df['forecast_tmin_d1'].fillna(df['forecast_tmin_d0']))
    # Clean up intermediate columns
    df = df.drop(columns=[c for c in ['date_dt', 'target_date', 'target_date_d0',
                                       'forecast_tmax_d1', 'forecast_tmin_d1',
                                       'forecast_tmax_d0', 'forecast_tmin_d0']
                            if c in df.columns], errors='ignore')
    return df


RAIN_COLS = ['rainfall_60m', 'rainfall_120m', 'rainfall_30m']


def merge_rainfall(df, rain_df):
    """Point-in-time asof merge: temperature snapshot T matches latest rain <= T.

    Leakage rules enforced:
    - rainfall_datetime is always <= datetime (direction='backward')
    - tolerance capped at 120 min (stale data dropped)
    - no daily totals, no full-day max/min, no rain-end time, no event totals

    Missing-data strategy:
    - Original rainfall columns (rainfall_60m, etc.) are kept with NaN for
      unmatched rows — missing is NOT the same as confirmed zero rain.
    - _filled variants replace NaN with 0 for model compatibility.
    - _missing_flag columns let the model distinguish "no data" from "no rain".
    - rain_data_gap_flag is a single summary indicator (1 = any gap).
    """
    df = df.copy()
    rain_df = rain_df.copy()

    df['datetime'] = pd.to_datetime(df['datetime'])
    rain_df['datetime'] = pd.to_datetime(rain_df['datetime'])

    # Rename rain timestamp so it survives the merge as a separate column
    rain_df = rain_df.rename(columns={'datetime': 'rainfall_datetime'})

    # Point-in-time merge: each temp snapshot gets the latest rainfall
    # observation that is <= the snapshot time
    merged = pd.merge_asof(
        df.sort_values('datetime'),
        rain_df.sort_values('rainfall_datetime'),
        left_on='datetime',
        right_on='rainfall_datetime',
        direction='backward',
        tolerance=pd.Timedelta('120min'),
    )

    # Freshness indicators
    merged['has_rainfall_data'] = merged['rainfall_datetime'].notna().astype(int)
    merged['rainfall_data_age_minutes'] = (
        merged['datetime'] - merged['rainfall_datetime']
    ).dt.total_seconds() / 60.0
    merged['has_recent_rainfall_obs'] = (
        merged['has_rainfall_data'] & (merged['rainfall_data_age_minutes'] <= 20)
    ).astype(int)

    # Keep original rainfall columns as-is (NaN = missing observation).
    # Create filled variants and missing-flag columns for model consumption.
    any_missing = pd.Series(False, index=merged.index)
    for col in RAIN_COLS:
        if col in merged.columns:
            merged[f'{col}_missing_flag'] = merged[col].isna().astype(int)
            merged[f'{col}_filled'] = merged[col].fillna(0)
            any_missing = any_missing | merged[col].isna()
        else:
            merged[f'{col}_missing_flag'] = 1
            merged[f'{col}_filled'] = 0
    merged['rain_data_gap_flag'] = any_missing.astype(int)

    # Fill auxiliary rain feature columns (internal use only, not in RAIN_COLS)
    for col in ['rainfall_interval_15m',
                'rainfall_max_30m', 'rainfall_max_60m']:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Drop the raw accumulated rainfall column — it is not a model feature
    # and its name is ambiguous (could be 'rainfall' or 'rainfall_accumulated_since_midnight')
    merged = merged.drop(columns=['rainfall', 'rainfall_accumulated_since_midnight'], errors='ignore')

    return merged


def add_rain_temperature_interactions(df):
    """加入雨溫互動特徵（使用 _filled 欄位避免遺失值混淆）"""
    df = df.copy()

    # Cooling magnitude: positive when temperature dropped over the window.
    # temp_change_60min = temp - temp_lag_60min, so negative means cooling.
    # We flip the sign and clip to get a positive cooling magnitude.
    df['cooling_60m'] = df.groupby('date')['temp'].transform(
        lambda x: (x.shift(6) - x).clip(lower=0))
    df['cooling_120m'] = df.groupby('date')['temp'].transform(
        lambda x: (x.shift(12) - x).clip(lower=0))

    # Rain-gated cooling: only count cooling magnitude when rain was present.
    # has_rain_60m is a binary flag derived from filled rainfall.
    has_rain_60m = (df['rainfall_60m_filled'] > 0).astype(float)
    df['rain_cooling_60m'] = has_rain_60m * df['cooling_60m']
    df['rain_cooling_120m'] = has_rain_60m * df['cooling_120m']

    # Fallback: if cooling_NaN (first N rows of each day), fill with 0
    df['cooling_60m'] = df['cooling_60m'].fillna(0)
    df['cooling_120m'] = df['cooling_120m'].fillna(0)
    df['rain_cooling_60m'] = df['rain_cooling_60m'].fillna(0)
    df['rain_cooling_120m'] = df['rain_cooling_120m'].fillna(0)

    # Remove old rain_cooling_30m (replaced by rain_cooling_60m / _120m)
    # Keep for backward compatibility: re-derive as before but with correct sign
    df['rain_cooling_30m'] = (
        (df['rainfall_30m_filled'] > 0).astype(float) *
        df.groupby('date')['temp'].transform(lambda x: (x.shift(3) - x).clip(lower=0)).fillna(0)
    )

    # 午後降雨指標：_peak 後出現明顯降雨 + 溫度已從高點回落
    # 避免僅因 time_since_max > 0 就標記所有雨天
    RAIN_HEAVY_THRESHOLD = 5.0       # mm (60min 內)
    DROP_FROM_MAX_THRESHOLD = 0.5    # °C 從最高點回落
    POST_PEAK_MINUTES_MIN = 30       # 至少 30 分鐘後才算 peak 後
    POST_PEAK_MINUTES_MAX = 240      # 最多 4 小時後的冷卻窗口

    condition_post_peak = (
        (df['rainfall_60m_filled'] > RAIN_HEAVY_THRESHOLD) &
        (df['drop_from_max'] >= DROP_FROM_MAX_THRESHOLD) &
        (df['time_since_max_so_far'].between(POST_PEAK_MINUTES_MIN, POST_PEAK_MINUTES_MAX))
    )
    df['post_peak_rain_flag'] = condition_post_peak.astype(int)

    # 上午高峰後降雨：相同條件但限制在上午/午後早期
    df['morning_peak_then_rain_flag'] = (
        condition_post_peak &
        df['datetime'].dt.hour.between(9, 14)
    ).astype(int)

    # 相容性別名（保留舊名稱以支援已訓練模型）
    df['morning_peak_rain_flag'] = df['morning_peak_then_rain_flag']
    # 剩餘日照
    df['remaining_minutes_to_18'] = np.maximum(0, 18*60 - df['minutes_since_midnight'])
    df['expected_rebound_daylight'] = df['rain_cooling_60m'] * df['remaining_minutes_to_18']
    return df


def add_targets(df):
    """計算目標變數與輔助欄位"""
    df['upside'] = (df['tmax'] - df['temp']).clip(lower=0)
    df['remaining_upside'] = (df['tmax'] - df['max_so_far']).clip(lower=0)
    df['remaining_downside'] = (df['min_so_far'] - df['tmin']).clip(lower=0)
    df['is_upside_zero'] = (df['remaining_upside'] <= 0.05).astype(int)
    df['is_downside_zero'] = (df['remaining_downside'] <= 0.05).astype(int)
    return df


def validate_required_columns(df):
    required = [
        'datetime', 'date', 'temp',
        'tmax', 'tmin',
        'max_so_far', 'min_so_far', 'range_so_far',
        'time_since_max_so_far', 'time_since_min_so_far',
        'upside', 'remaining_upside', 'remaining_downside',
        'is_upside_zero', 'is_downside_zero',
        'drop_from_max', 'rise_from_min',
        'has_rainfall_data', 'has_recent_rainfall_obs',
        'rainfall_60m_filled', 'rainfall_60m_missing_flag',
        'rainfall_120m_filled', 'rainfall_120m_missing_flag',
        'rainfall_30m_filled', 'rainfall_30m_missing_flag',
        'rain_data_gap_flag', 'rainfall_data_age_minutes',
        'rain_cooling_60m', 'rain_cooling_120m',
        'post_peak_rain_flag', 'morning_peak_then_rain_flag',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_missing = df[['remaining_upside', 'remaining_downside']].isna().sum().sum()
    if target_missing > 0:
        raise ValueError(f"Targets contain missing values: {target_missing}")


def main():
    # 1. Load intraday 10-minute temperature observations
    logger.info("Loading intraday temperature observations...")
    df = pd.read_parquet(INTRADAY_PATH)

    # 2. Load official daily Tmax/Tmin target data
    logger.info("Loading official daily Tmax/Tmin targets...")
    daily = pd.read_parquet(DAILY_PATH)
    daily['date'] = pd.to_datetime(daily['date']).dt.date

    # 3. Load historical forecasts
    logger.info("Loading historical forecasts...")
    forecast = pd.read_parquet(FORECAST_PATH)

    # 4. Load rainfall features
    rainfall = None
    if RAINFALL_FEAT_PATH.exists():
        logger.info("Loading rainfall features...")
        rainfall = pd.read_parquet(RAINFALL_FEAT_PATH)
    else:
        logger.warning("Rainfall features file not found, will use zero-filled columns")

    # 5. Add date column (needed for intraday state features grouping)
    df['date'] = df['datetime'].dt.date

    # 6. Add time features
    logger.info("Adding time features...")
    df = add_time_features(df)

    # 7. Add intraday state features
    logger.info("Adding intraday state features...")
    df = add_intraday_state_features(df)

    # 8. Merge official daily target data
    logger.info("Merging official daily target data...")
    df = df.merge(daily[['date', 'tmax', 'tmin']], on='date', how='left')
    # Drop rows where daily targets are unavailable (e.g. most recent days not yet published)
    before = len(df)
    df = df.dropna(subset=['tmax', 'tmin'])
    logger.info(f"Dropped {before - len(df):,} rows with missing daily targets")

    # 8. Merge rainfall features
    if rainfall is not None:
        logger.info("Merging rainfall features...")
        df = merge_rainfall(df, rainfall)
    else:
        logger.warning("No rainfall data available — all rainfall columns set to NaN / 0")
        for col in RAIN_COLS:
            df[col] = np.nan
            df[f'{col}_filled'] = 0
            df[f'{col}_missing_flag'] = 1
        df['rain_data_gap_flag'] = 1
        for col in ['rainfall', 'rainfall_interval_15m', 'rainfall_max_30m', 'rainfall_max_60m']:
            df[col] = 0
        df['has_rainfall_data'] = 0
        df['has_recent_rainfall_obs'] = 0
        df['rainfall_data_age_minutes'] = np.nan
        df['rainfall_datetime'] = pd.NaT

    # 9. Add rain-temperature interaction features
    logger.info("Adding rain-temperature interaction features...")
    df = add_rain_temperature_interactions(df)

    # 10. Merge forecast features (point-in-time)
    logger.info("Merging forecast data (point-in-time)...")
    df = merge_forecast(df, forecast)

    # 11. Add/calculate target labels
    logger.info("Building target labels...")
    df = add_targets(df)

    # 12. Validate required columns
    logger.info("Validating required columns...")
    validate_required_columns(df)

    # Build output column list (from central schema)
    _schema_path = Path(__file__).resolve().parent
    import sys as _sys
    _sys.path.insert(0, str(_schema_path))
    from feature_schema import get_feature_list
    feature_cols = get_feature_list("rain_aware")
    target_cols = ['upside', 'remaining_upside', 'remaining_downside', 'is_upside_zero', 'is_downside_zero']
    # Keep all columns needed for reference/validation plus features and targets
    extra_cols = ['tmax', 'tmin', 'min_so_far', 'time_since_min_so_far',
                  'has_rainfall_data', 'has_recent_rainfall_obs',
                  'rainfall_datetime',
                  'rainfall_60m', 'rainfall_60m_filled', 'rainfall_60m_missing_flag',
                  'rainfall_120m', 'rainfall_120m_filled', 'rainfall_120m_missing_flag',
                  'rainfall_30m', 'rainfall_30m_filled', 'rainfall_30m_missing_flag',
                  'rain_data_gap_flag', 'rainfall_data_age_minutes',
                  'cooling_60m', 'cooling_120m', 'rain_cooling_120m',
                  'expected_rebound_daylight',
                  'morning_peak_rain_flag',  # backward-compat alias
                  'rise_from_min',
                  'min_bucket',
                  'rolling_std_30min', 'rolling_std_60min', 'rolling_std_120min',
                  'rolling_mean_10min', 'rolling_mean_30min', 'rolling_mean_60min', 'rolling_mean_120min',
                  'temp_change_10min', 'temp_change_30min', 'temp_change_60min', 'temp_change_120min']
    keep_cols = list(set(feature_cols + target_cols + extra_cols + ['datetime', 'date']))
    df = df[keep_cols].dropna(subset=['remaining_upside', 'remaining_downside'])
    # Fill feature columns: rainfall age gets 999 (stale/missing), rest get 0
    age_cols = ['rainfall_data_age_minutes']
    other_feature_cols = [c for c in feature_cols if c not in age_cols]
    df[other_feature_cols] = df[other_feature_cols].fillna(0)
    df['rainfall_data_age_minutes'] = df['rainfall_data_age_minutes'].fillna(999)

    # Exclude bad quality days
    quality_path = Path('reports/intraday_data_quality_report.csv')
    if quality_path.exists():
        bad_dates = pd.read_csv(quality_path)
        bad_dates = bad_dates[bad_dates['status'] == 'bad']['date'].tolist()
        bad_dates = pd.to_datetime(bad_dates).date
        df = df[~df['date'].isin(bad_dates)]
        logger.info(f"已排除 {len(bad_dates)} 個劣質日")

    # 13. Save data/intraday_ml_train.parquet
    logger.info(f"Final training set size: {len(df):,} rows")
    logger.info(f"Minutes range: {df['minutes_since_midnight'].min()} - {df['minutes_since_midnight'].max()}")
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Done. Output written to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
