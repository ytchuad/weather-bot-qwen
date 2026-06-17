"""Build minute-level ML features from hko_history.parquet + hko_tmax_historical.parquet.

Output: data/intraday_minute_ml_features.parquet (~35 feature cols + targets).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HISTORY_PATH = Path("data/hko_history.parquet")
DAILY_PATH = Path("data/hko_tmax_historical.parquet")
OUTPUT_PATH = Path("data/intraday_minute_ml_features.parquet")

HKT = "Asia/Hong_Kong"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["as_of_datetime_hkt"].dt.hour
    df["minute"] = df["as_of_datetime_hkt"].dt.minute
    df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]
    df["remaining_minutes_to_midnight"] = 1440 - df["minutes_since_midnight"]
    df["month"] = df["as_of_datetime_hkt"].dt.month
    df["day_of_year"] = df["as_of_datetime_hkt"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 24)).astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    return df


def _compute_minutes_since_extreme(df: pd.DataFrame, extreme_col: str) -> pd.Series:
    result = np.zeros(len(df))
    for _date, group in df.groupby("target_date", sort=False):
        idx = group.index.values
        temps = group["temp_current"].values
        extremes = group[extreme_col].values
        dts = group["as_of_datetime_hkt"].values

        last_extreme_time = None
        for i in range(len(idx)):
            ext_val = extremes[i]
            row_temp = temps[i]
            if row_temp >= ext_val - 1e-9:
                last_extreme_time = dts[i]
            if last_extreme_time is not None:
                delta = (dts[i] - last_extreme_time).astype("timedelta64[ns]").astype(float) / 6e10
                result[idx[i]] = max(delta, 0.0)
    return pd.Series(result, index=df.index)


def add_intraday_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["target_date", "as_of_datetime_hkt"]).reset_index(drop=True)
    g = df.groupby("target_date", group_keys=False)
    df["max_so_far_1m"] = g["temp_current"].cummax()
    df["min_so_far_1m"] = g["temp_current"].cummin()
    df["range_so_far_1m"] = df["max_so_far_1m"] - df["min_so_far_1m"]
    df["drop_from_max_1m"] = df["max_so_far_1m"] - df["temp_current"]
    df["rise_from_min_1m"] = df["temp_current"] - df["min_so_far_1m"]
    df["time_since_max_1m"] = _compute_minutes_since_extreme(df, "max_so_far_1m")
    df["time_since_min_1m"] = _compute_minutes_since_extreme(df, "min_so_far_1m")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("target_date", group_keys=False)

    # Temperature changes
    df["temp_change_5m"] = g["temp_current"].diff(5)
    df["temp_change_15m"] = g["temp_current"].diff(15)
    df["temp_change_30m"] = g["temp_current"].diff(30)
    df["temp_change_60m"] = g["temp_current"].diff(60)

    # Temperature acceleration: (temp - 2*temp.shift(15) + temp.shift(30)) / 15
    df["temp_acceleration_30m"] = (
        df["temp_current"]
        - 2 * g["temp_current"].shift(15)
        + g["temp_current"].shift(30)
    ) / 15.0

    # Temperature rolling std
    df["temp_std_30m"] = g["temp_current"].transform(
        lambda x: x.rolling(30, min_periods=2).std()
    )
    df["temp_std_60m"] = g["temp_current"].transform(
        lambda x: x.rolling(60, min_periods=2).std()
    )

    # RH changes
    df["rh_change_15m"] = g["rh_current"].diff(15)
    df["rh_change_30m"] = g["rh_current"].diff(30)
    df["rh_change_60m"] = g["rh_current"].diff(60)

    # RH rolling mean
    df["rh_mean_30m"] = g["rh_current"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df["rh_mean_60m"] = g["rh_current"].transform(
        lambda x: x.rolling(60, min_periods=1).mean()
    )

    # RH rolling std
    df["rh_std_60m"] = g["rh_current"].transform(
        lambda x: x.rolling(60, min_periods=2).std()
    )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_x_rh"] = df["temp_current"] * df["rh_current"]

    # Magnus formula: dew point
    a, b = 17.27, 237.7
    gamma = (a * df["temp_current"]) / (b + df["temp_current"]) + np.log(
        df["rh_current"].clip(lower=0.01) / 100.0
    )
    df["dew_point_c"] = (b * gamma) / (a - gamma)
    df["dew_point_spread"] = df["temp_current"] - df["dew_point_c"]

    return df


def add_targets(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(daily[["target_date", "tmax", "tmin"]], on="target_date", how="left")
    df = df.rename(columns={"tmax": "official_tmax", "tmin": "official_tmin"})
    df["remaining_upside"] = (df["official_tmax"] - df["max_so_far_1m"]).clip(lower=0)
    df["is_upside_zero"] = (df["remaining_upside"] <= 0.05).astype(int)
    df["remaining_downside"] = (df["min_so_far_1m"] - df["official_tmin"]).clip(lower=0)
    df["is_downside_zero"] = (df["remaining_downside"] <= 0.05).astype(int)
    return df


def main():
    logger.info("Loading minute history...")
    df = pd.read_parquet(HISTORY_PATH)

    # Rename for clarity
    df = df.rename(columns={"datetime": "as_of_datetime_hkt", "temp": "temp_current", "rh": "rh_current"})

    # Ensure HKT timezone
    if df["as_of_datetime_hkt"].dt.tz is None:
        df["as_of_datetime_hkt"] = df["as_of_datetime_hkt"].dt.tz_localize(HKT)
    else:
        df["as_of_datetime_hkt"] = df["as_of_datetime_hkt"].dt.tz_convert(HKT)

    df = df.drop(columns=["time"], errors="ignore")

    # Build target_date from date column or derive from datetime
    if "date" in df.columns:
        df["target_date"] = df["date"].astype(str)
    else:
        df["target_date"] = df["as_of_datetime_hkt"].dt.strftime("%Y-%m-%d")

    logger.info(f"Loaded {len(df):,} rows, {df.target_date.nunique()} days")

    logger.info("Adding time features...")
    df = add_time_features(df)

    logger.info("Adding intraday state features...")
    df = add_intraday_state_features(df)

    logger.info("Adding trend features...")
    df = add_trend_features(df)

    logger.info("Adding interaction features...")
    df = add_interaction_features(df)

    logger.info("Loading official daily Tmax...")
    daily = pd.read_parquet(DAILY_PATH)
    daily["target_date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    logger.info("Adding target columns...")
    df = add_targets(df, daily)

    # Column order
    index_cols = ["target_date", "as_of_datetime_hkt", "minute_of_day"]
    df["minute_of_day"] = df["minutes_since_midnight"]
    obs_cols = ["temp_current", "rh_current"]
    state_cols = [
        "max_so_far_1m", "min_so_far_1m", "range_so_far_1m",
        "time_since_max_1m", "time_since_min_1m",
        "drop_from_max_1m", "rise_from_min_1m",
    ]
    trend_cols = [
        "temp_change_5m", "temp_change_15m", "temp_change_30m", "temp_change_60m",
        "temp_acceleration_30m", "temp_std_30m", "temp_std_60m",
        "rh_change_15m", "rh_change_30m", "rh_change_60m",
        "rh_mean_30m", "rh_mean_60m", "rh_std_60m",
    ]
    interaction_cols = ["temp_x_rh", "dew_point_c", "dew_point_spread"]
    time_cols = [
        "hour", "minute", "minutes_since_midnight", "remaining_minutes_to_midnight",
        "month", "day_of_year",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "is_morning", "is_afternoon", "is_evening", "is_night",
    ]
    target_cols = ["official_tmax", "official_tmin", "remaining_upside", "is_upside_zero", "remaining_downside", "is_downside_zero"]

    all_cols = index_cols + obs_cols + state_cols + trend_cols + interaction_cols + time_cols + target_cols
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing]

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Target coverage: {df.official_tmax.notna().sum():,} / {len(df):,} rows have official_tmax")
    logger.info(f"Writing to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
