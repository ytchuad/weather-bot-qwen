"""
build_model_2b_feature_store.py

Build Model 2B feature store = Model 2A v2 feature store + observed rainfall features
merged at point-in-time availability.

Output: data/model_2b_feature_store.parquet
"""

import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TWO_A_STORE = Path("data/model_2a_feature_store.parquet")
RAIN_RAW_PATH = Path("data/hko_rainfall_15min.parquet")
OUTPUT_PATH = Path("data/model_2b_feature_store.parquet")
DATA_LAG = 8
GAP_TOLERANCE_MINUTES = 45


def ceil_dt_10min(dt):
    m = dt.minute
    ceil_m = math.ceil(m / 10) * 10
    if ceil_m >= 60:
        dt = dt + timedelta(hours=1)
        ceil_m = 0
    return dt.replace(minute=ceil_m, second=0, microsecond=0)


def build_rainfall_features():
    print("=== Building rainfall features from raw 15-min data ===")
    df = pd.read_parquet(RAIN_RAW_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    df = df.rename(columns={"rainfall": "rainfall_all_since_midnight"})

    df["rainfall_interval_15m"] = (
        df.groupby("date")["rainfall_all_since_midnight"].diff().fillna(df["rainfall_all_since_midnight"])
    )
    neg = df["rainfall_interval_15m"] < 0
    if neg.any():
        print(f"  Clipped {neg.sum()} negative intervals to 0")
        df["rainfall_interval_15m"] = df["rainfall_interval_15m"].clip(lower=0)

    # Rolling accumulations
    df["rainfall_15m"] = df["rainfall_interval_15m"]
    df["rainfall_30m"] = df.groupby("date")["rainfall_interval_15m"].transform(
        lambda x: x.rolling(2, min_periods=1).sum()
    )
    df["rainfall_60m"] = df.groupby("date")["rainfall_interval_15m"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    df["rainfall_120m"] = df.groupby("date")["rainfall_interval_15m"].transform(
        lambda x: x.rolling(8, min_periods=1).sum()
    )

    # Max intensity over 120 min
    df["rain_intensity_max_120m"] = df.groupby("date")["rainfall_interval_15m"].transform(
        lambda x: x.rolling(8, min_periods=1).max()
    )

    # Minutes since last rain (per date)
    def _mins_since_last_rain(g):
        g = g.copy()
        last_min = np.nan
        result = np.full(len(g), np.nan)
        for i in range(len(g)):
            dt = g.iloc[i]["datetime"]
            if g.iloc[i]["rainfall_interval_15m"] > 0:
                last_min = dt
            if not np.isnan(last_min) if not isinstance(last_min, float) else False:
                result[i] = (dt - last_min).total_seconds() / 60.0
            elif last_min is not None and not (isinstance(last_min, float) and np.isnan(last_min)):
                result[i] = (dt - last_min).total_seconds() / 60.0
        g["minutes_since_last_rain"] = result
        return g

    # Simpler approach: carry-forward using transform
    df["_last_rain_ts"] = df.groupby("date")["rainfall_interval_15m"].transform(
        lambda x: x.where(x > 0).ffill()
    )
    df["minutes_since_last_rain"] = np.where(
        df["_last_rain_ts"].notna() & (df["_last_rain_ts"] > 0),
        np.nan,
        np.nan,
    )
    del df["_last_rain_ts"]

    # More direct approach for minutes_since_last_rain
    df = df.sort_values(["date", "datetime"])
    last_rain_ts = {}
    msr = []
    for _, row in df.iterrows():
        d = row["date"]
        if row["rainfall_interval_15m"] > 0:
            last_rain_ts[d] = row["datetime"]
        if d in last_rain_ts:
            msr.append((row["datetime"] - last_rain_ts[d]).total_seconds() / 60.0)
        else:
            msr.append(np.nan)
    df["minutes_since_last_rain"] = msr

    # rain available time: ceil datetime to 10min + 8min lag
    df["rain_available_time"] = df["datetime"].apply(
        lambda ts: ceil_dt_10min(ts) + timedelta(minutes=DATA_LAG)
    )
    df["rain_timestamp"] = df["datetime"]

    keep = [
        "datetime", "rain_timestamp", "rain_available_time",
        "rainfall_all_since_midnight", "rainfall_interval_15m",
        "rainfall_15m", "rainfall_30m", "rainfall_60m", "rainfall_120m",
        "rain_intensity_max_120m", "minutes_since_last_rain",
    ]
    df = df[keep].copy()
    print(f"  Rainfall features: {len(df):,} rows, {df['datetime'].min()} ~ {df['datetime'].max()}")
    return df


def main():
    print("=" * 60)
    print("  Model 2B Feature Store Builder")
    print("=" * 60)

    # Load Model 2A v2 feature store
    print("\n=== Loading Model 2A v2 feature store ===")
    df = pd.read_parquet(TWO_A_STORE)
    print(f"  Shape: {df.shape}")
    print(f"  target_date: {df['target_date'].min()} ~ {df['target_date'].max()}")
    print(f"  decision_time: {df['decision_time'].min()} ~ {df['decision_time'].max()}")

    # Build rainfall features
    rain = build_rainfall_features()

    # Merge rainfall at point-in-time: rain_available_time <= decision_time
    print("\n=== Merging rainfall features (point-in-time) ===")
    df = df.sort_values("decision_time")
    rain_sorted = rain.sort_values("rain_available_time")

    df = pd.merge_asof(
        df,
        rain_sorted,
        left_on="decision_time",
        right_on="rain_available_time",
        direction="backward",
    )
    print(f"  After merge: {len(df):,} rows")

    # Compute rainfall_data_age_minutes
    df["rainfall_data_age_minutes"] = (
        df["decision_time"] - df["rain_timestamp"]
    ).dt.total_seconds() / 60.0

    # rain_data_gap_flag: 1 if no valid rainfall within tolerance
    no_rain_data = df["rain_timestamp"].isna()
    stale = df["rainfall_data_age_minutes"] > GAP_TOLERANCE_MINUTES
    df["rain_data_gap_flag"] = (no_rain_data | stale).astype(int)

    # Fill NaN rainfall with 0
    rain_cols = [
        "rainfall_15m", "rainfall_30m", "rainfall_60m", "rainfall_120m",
        "rainfall_all_since_midnight", "rain_intensity_max_120m",
        "minutes_since_last_rain", "rainfall_interval_15m",
    ]
    for c in rain_cols:
        pre = df[c].isna().sum()
        df[c] = df[c].fillna(0.0)
        if pre > 0:
            print(f"  Filled {pre} NaNs in {c}")

    # Pre-2023 rows: set rain_data_gap_flag = 1
    pre_2023 = pd.to_datetime(df["target_date"]) < pd.Timestamp("2023-06-01")
    df.loc[pre_2023, "rain_data_gap_flag"] = 1
    print(f"  Pre-2023 rows: {pre_2023.sum():,} (rain_data_gap_flag set to 1)")

    # Compute derived features
    print("\n=== Computing derived rainfall features ===")
    df["has_recent_rainfall_obs"] = (
        (df["rainfall_60m"] > 0) | (df["rainfall_120m"] > 0)
    ).astype(int)

    # rain_after_max_flag: recent rain AND drop_from_max >= 0.5
    df["rain_after_max_flag"] = (
        (df["has_recent_rainfall_obs"] == 1) & (df["drop_from_max"] >= 0.5)
    ).astype(int)

    # post_peak_rain_flag: recent rain AND drop >= 0.5 AND 30 <= time_since_max <= 240
    df["post_peak_rain_flag"] = (
        (df["has_recent_rainfall_obs"] == 1)
        & (df["drop_from_max"] >= 0.5)
        & (df["time_since_max"] >= 30)
        & (df["time_since_max"] <= 240)
    ).astype(int)

    # morning_peak_rain_flag: post_peak AND 9 <= hour <= 14
    df["morning_peak_rain_flag"] = (
        (df["post_peak_rain_flag"] == 1)
        & (df["hour"] >= 9) & (df["hour"] <= 14)
    ).astype(int)

    # heavy_recent_rain_flag
    df["heavy_recent_rain_flag"] = (df["rainfall_60m"] >= 10).astype(int)

    # rain_cooling_60m = rainfall_60m * max(-temp_change_60m, 0)
    df["rain_cooling_60m"] = df["rainfall_60m"] * np.maximum(0, -df["temp_change_60m"])

    # rain_cooling_120m = rainfall_120m * max(-temp_change_60m, 0)
    df["rain_cooling_120m"] = df["rainfall_120m"] * np.maximum(0, -df["temp_change_60m"])

    # Fill derived feature NaNs
    for c in ["has_recent_rainfall_obs", "rain_after_max_flag", "post_peak_rain_flag",
              "morning_peak_rain_flag", "heavy_recent_rain_flag",
              "rain_cooling_60m", "rain_cooling_120m", "rainfall_data_age_minutes"]:
        df[c] = df[c].fillna(0)

    # Fill rainfall_data_age_minutes with large sentinel for pre-rain era
    df["rainfall_data_age_minutes"] = df["rainfall_data_age_minutes"].fillna(9999.0)

    # Drop raw merge columns not needed in final store
    drop_cols = ["rain_timestamp", "rain_available_time", "datetime"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Save
    print(f"\n=== Saving Model 2B feature store ===")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  target_date range: {df['target_date'].min()} ~ {df['target_date'].max()}")
    print(f"  rain_data_gap_flag mean: {df['rain_data_gap_flag'].mean():.3f}")
    print(f"  has_recent_rainfall_obs mean: {df['has_recent_rainfall_obs'].mean():.3f}")
    print(f"  post_peak_rain_flag mean: {df['post_peak_rain_flag'].mean():.3f}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
