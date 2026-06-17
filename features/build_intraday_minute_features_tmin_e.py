"""Build Model E morning Tmin features — predicting morning minimum (00:00-07:59 HKT).

Reuses Model D's 34 cross-midnight / evening / night cooling features (same features)
with labels specific to the morning minimum prediction task.

Core labels:
- morning_min_00_08: minimum temperature observed between 00:00-07:59 HKT
- remaining_morning_downside: min_so_far - morning_min_00_08 (>= 0)
- morning_low_reached: whether the morning low has been observed by this row

Downstream label (computed in training script):
- morning_low_survives_day: whether morning_min == official_tmin

Output: data/intraday_minute_ml_features_tmin_e.parquet (pre-cutoff rows only)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PATH = Path("data/intraday_minute_ml_features.parquet")
HISTORY_PATH = Path("data/hko_history.parquet")
OUTPUT_PATH = Path("data/intraday_minute_ml_features_tmin_e.parquet")

HKT = "Asia/Hong_Kong"

MODEL_D_FEATURE_COLS = [
    "temp_change_180m_crossday", "temp_change_360m_crossday", "temp_change_720m_crossday",
    "temp_slope_360m_crossday", "temp_slope_720m_crossday",
    "temp_min_360m_crossday", "temp_min_720m_crossday",
    "temp_range_360m_crossday", "temp_std_360m_crossday",
    "prev_18_temp", "prev_21_temp", "prev_2359_temp",
    "prev_evening_temp_change", "prev_evening_temp_min",
    "prev_evening_temp_range", "prev_evening_temp_slope",
    "prev_evening_rh_mean", "prev_evening_rh_max",
    "prev_evening_dew_point_mean",
    "prev_evening_rainfall_18_24", "prev_evening_rain_flag",
    "cooling_since_prev_18", "cooling_since_prev_21",
    "distance_to_prev_evening_min", "dew_point_floor_gap",
    "dew_point_spread_min_360m", "rh_mean_360m", "rh_max_360m",
    "is_before_evening_cooling_window", "daytime_warming_so_far",
    "afternoon_temp_drop_60m", "afternoon_temp_drop_120m",
]

MORNING_LABEL_COLS = [
    "morning_min_00_08", "remaining_morning_downside",
    "morning_low_reached",
]


def _dew_point(temp_c, rh_pct):
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + np.log(np.maximum(rh_pct, 0.01) / 100.0)
    return (b * gamma) / (a - gamma)


def _safediff(arr, lag):
    result = np.full(len(arr), np.nan)
    if len(arr) > lag:
        result[lag:] = arr[lag:] - arr[:-lag]
    return result


def _extract_specific_hour(raw, hour, minute, col="temp_current"):
    mask = (raw["_hour"] == hour) & (raw["_minute"] == minute)
    specific = raw.loc[mask, ["_date", col]].drop_duplicates(subset="_date", keep="last")
    specific = specific.set_index("_date")
    return specific[col]


def compute_cross_midnight_features(raw: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing cross-midnight rolling features (Group A)...")
    tc = raw["temp_current"].values
    rh = raw["rh_current"].values
    out = pd.DataFrame({"as_of_datetime_hkt": raw["as_of_datetime_hkt"]})

    out["temp_change_180m_crossday"] = _safediff(tc, 180)
    out["temp_change_360m_crossday"] = _safediff(tc, 360)
    out["temp_change_720m_crossday"] = _safediff(tc, 720)
    out["temp_slope_360m_crossday"] = _safediff(tc, 360) / 6.0
    out["temp_slope_720m_crossday"] = _safediff(tc, 720) / 12.0

    s_tc = pd.Series(tc)
    out["temp_min_360m_crossday"] = s_tc.rolling(360, min_periods=30).min()
    out["temp_min_720m_crossday"] = s_tc.rolling(720, min_periods=60).min()
    out["temp_range_360m_crossday"] = (
        s_tc.rolling(360, min_periods=30).max() - out["temp_min_360m_crossday"]
    )
    out["temp_std_360m_crossday"] = s_tc.rolling(360, min_periods=30).std()

    s_rh = pd.Series(rh)
    out["dew_point_spread_min_360m"] = (
        _dew_point(s_tc, s_rh)
    ).rolling(360, min_periods=30).min()
    out["rh_mean_360m"] = s_rh.rolling(360, min_periods=30).mean()
    out["rh_max_360m"] = s_rh.rolling(360, min_periods=30).max()

    return out


def compute_evening_features(raw: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing previous evening context features (Group B)...")
    raw = raw.copy()
    raw["_date"] = raw["as_of_datetime_hkt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    raw["_hour"] = raw["as_of_datetime_hkt"].dt.hour
    raw["_minute"] = raw["as_of_datetime_hkt"].dt.minute

    evening = raw[(raw["_hour"] >= 18) & (raw["_hour"] < 24)]
    if len(evening) == 0:
        logger.warning("No evening data found!")
        return pd.DataFrame()

    gb = evening.groupby("_date")
    even_df = pd.DataFrame(index=gb.size().index)
    even_df.index.name = "_date"

    even_df["prev_18_temp"] = _extract_specific_hour(raw, 18, 0)
    even_df["prev_21_temp"] = _extract_specific_hour(raw, 21, 0)
    even_df["prev_2359_temp"] = _extract_specific_hour(raw, 23, 59)

    def first_val(x):
        return x.iloc[0] if len(x) > 0 else np.nan
    def last_val(x):
        return x.iloc[-1] if len(x) > 0 else np.nan

    even_df["prev_evening_temp_change"] = gb["temp_current"].apply(
        lambda x: last_val(x) - first_val(x)
    )
    even_df["prev_evening_temp_min"] = gb["temp_current"].min()
    even_df["prev_evening_temp_max"] = gb["temp_current"].max()
    even_df["prev_evening_temp_range"] = (
        even_df["prev_evening_temp_max"] - even_df["prev_evening_temp_min"]
    )
    even_df["prev_evening_temp_slope"] = gb["temp_current"].apply(
        lambda x: (last_val(x) - first_val(x)) / len(x) if len(x) >= 2 else 0.0
    )
    even_df["prev_evening_rh_mean"] = gb["rh_current"].mean()
    even_df["prev_evening_rh_max"] = gb["rh_current"].max()

    first_temp = gb["temp_current"].apply(lambda x: first_val(x))
    first_rh = gb["rh_current"].apply(lambda x: first_val(x))
    even_df["prev_evening_dew_point_mean"] = _dew_point(first_temp.values, first_rh.values)

    even_df["prev_evening_rainfall_18_24"] = 0.0
    even_df["prev_evening_rain_flag"] = 0

    prev_cols = [c for c in MODEL_D_FEATURE_COLS if c.startswith("prev_") and c in even_df.columns]
    even_df = even_df[prev_cols]

    prev_map = even_df.copy()
    prev_map.index = prev_map.index + pd.Timedelta(days=1)

    return prev_map.reset_index().rename(columns={"_date": "_target_date_next"})


def compute_group_c(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing night cooling features (Group C)...")
    out = pd.DataFrame(index=df.index)
    out["cooling_since_prev_18"] = df["prev_18_temp"] - df["temp_current"]
    out["cooling_since_prev_21"] = df["prev_21_temp"] - df["temp_current"]
    out["distance_to_prev_evening_min"] = df["temp_current"] - df["prev_evening_temp_min"]
    out["dew_point_floor_gap"] = df["temp_current"] - df["dew_point_c"]
    out["dew_point_spread_min_360m"] = df["dew_point_spread_min_360m"]
    out["rh_mean_360m"] = df["rh_mean_360m"]
    out["rh_max_360m"] = df["rh_max_360m"]
    return out


def compute_group_d(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing evening re-low features (Group D)...")
    out = pd.DataFrame(index=df.index)
    out["is_before_evening_cooling_window"] = ((df["hour"] >= 0) & (df["hour"] < 18)).astype(int)
    out["daytime_warming_so_far"] = df["range_so_far_1m"]
    out["afternoon_temp_drop_60m"] = np.where(
        df["hour"] >= 12, np.maximum(0, -df["temp_change_60m"]), 0.0,
    )
    out["afternoon_temp_drop_120m"] = np.where(
        df["hour"] >= 12, np.maximum(0, -df["temp_change_60m"]), 0.0,
    )
    return out


def compute_morning_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing morning min labels...")
    out = pd.DataFrame(index=df.index)
    morning_mask = df["hour"] < 8
    morning_mins = df.loc[morning_mask].groupby("target_date")["temp_current"].min()
    morning_mins.name = "morning_min_00_08"
    out["morning_min_00_08"] = df["target_date"].map(morning_mins)

    out["remaining_morning_downside"] = np.maximum(
        0.0, df["min_so_far_1m"] - out["morning_min_00_08"]
    )
    out["morning_low_reached"] = (
        df["min_so_far_1m"] <= out["morning_min_00_08"] + 1e-9
    ).astype(int)

    return out


def main():
    logger.info("=" * 60)
    logger.info("Model E Morning Tmin — Feature Builder")
    logger.info("=" * 60)

    logger.info("Loading base minute features...")
    df = pd.read_parquet(BASE_PATH)
    logger.info(f"  {len(df):,} rows, {df.shape[1]} columns")

    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("as_of_datetime_hkt").reset_index(drop=True)

    logger.info("Loading raw hko_history for cross-day computation...")
    raw = pd.read_parquet(HISTORY_PATH)
    raw = raw.rename(columns={"datetime": "as_of_datetime_hkt", "temp": "temp_current", "rh": "rh_current"})
    if raw["as_of_datetime_hkt"].dt.tz is None:
        raw["as_of_datetime_hkt"] = raw["as_of_datetime_hkt"].dt.tz_localize(HKT)
    else:
        raw["as_of_datetime_hkt"] = raw["as_of_datetime_hkt"].dt.tz_convert(HKT)
    raw = raw.sort_values("as_of_datetime_hkt").reset_index(drop=True)

    cross = compute_cross_midnight_features(raw)
    df = df.merge(cross, on="as_of_datetime_hkt", how="left")

    evening_map = compute_evening_features(raw)
    df["_date"] = df["as_of_datetime_hkt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df = df.merge(evening_map, left_on="_date", right_on="_target_date_next", how="left")
    df = df.drop(columns=["_date", "_target_date_next"])

    c_features = compute_group_c(df)
    for c in c_features.columns:
        df[c] = c_features[c].values

    d_features = compute_group_d(df)
    for c in d_features.columns:
        df[c] = d_features[c].values

    labels = compute_morning_labels(df)
    for c in labels.columns:
        df[c] = labels[c].values

    fill_cols = [c for c in MODEL_D_FEATURE_COLS if c in df.columns]
    for c in fill_cols:
        df[c] = df[c].fillna(0)

    before = len(df)
    df = df[df["hour"] < 8].reset_index(drop=True)
    logger.info(f"  Pre-cutoff rows (hour < 8): {len(df):,} (kept {len(df)/before*100:.1f}%)")

    index_cols = ["target_date", "as_of_datetime_hkt", "minute_of_day"]
    obs_cols = [c for c in df.columns if c not in index_cols]
    all_cols = index_cols + obs_cols
    existing = [c for c in all_cols if c in df.columns]
    df_out = df[existing]

    logger.info(f"Final shape: {df_out.shape}")
    logger.info(f"Morning labels: {MORNING_LABEL_COLS}")

    logger.info(f"Writing to {OUTPUT_PATH}...")
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
