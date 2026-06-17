"""Build Model D Tmin-specific features on top of the base minute-level parquet.

Adds 34 cross-midnight / previous evening / night cooling / evening re-low features,
plus 2 auxiliary classifier labels (will_make_new_low_after_now, tmin_timing_bucket).

Output: data/intraday_minute_ml_features_tmin_d.parquet
  (base features + 34 new features + 6 targets + 2 auxiliary labels)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PATH = Path("data/intraday_minute_ml_features.parquet")
HISTORY_PATH = Path("data/hko_history.parquet")
OUTPUT_PATH = Path("data/intraday_minute_ml_features_tmin_d.parquet")

HKT = "Asia/Hong_Kong"

MODEL_D_FEATURE_COLS = [
    # Group A: cross-midnight rolling
    "temp_change_180m_crossday",
    "temp_change_360m_crossday",
    "temp_change_720m_crossday",
    "temp_slope_360m_crossday",
    "temp_slope_720m_crossday",
    "temp_min_360m_crossday",
    "temp_min_720m_crossday",
    "temp_range_360m_crossday",
    "temp_std_360m_crossday",
    # Group B: previous evening context
    "prev_18_temp",
    "prev_21_temp",
    "prev_2359_temp",
    "prev_evening_temp_change",
    "prev_evening_temp_min",
    "prev_evening_temp_range",
    "prev_evening_temp_slope",
    "prev_evening_rh_mean",
    "prev_evening_rh_max",
    "prev_evening_dew_point_mean",
    "prev_evening_rainfall_18_24",
    "prev_evening_rain_flag",
    # Group C: night cooling potential
    "cooling_since_prev_18",
    "cooling_since_prev_21",
    "distance_to_prev_evening_min",
    "dew_point_floor_gap",
    "dew_point_spread_min_360m",
    "rh_mean_360m",
    "rh_max_360m",
    # Group D: evening re-low risk
    "is_before_evening_cooling_window",
    "daytime_warming_so_far",
    "afternoon_temp_drop_60m",
    "afternoon_temp_drop_120m",
]

AUX_LABEL_COLS = [
    "will_make_new_low_after_now",
    "tmin_timing_bucket",
]


def _dew_point(temp_c, rh_pct):
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + np.log(np.maximum(rh_pct, 0.01) / 100.0)
    return (b * gamma) / (a - gamma)


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
        _dew_point(s_tc, s_rh)  # computed on raw values across day boundary
    ).rolling(360, min_periods=30).min()
    out["rh_mean_360m"] = s_rh.rolling(360, min_periods=30).mean()
    out["rh_max_360m"] = s_rh.rolling(360, min_periods=30).max()

    return out


def _safediff(arr, lag):
    result = np.full(len(arr), np.nan)
    if len(arr) > lag:
        result[lag:] = arr[lag:] - arr[:-lag]
    return result


def _extract_specific_hour(raw, hour, minute, col="temp_current"):
    """Extract value at a specific hour:minute for each date."""
    mask = (raw["_hour"] == hour) & (raw["_minute"] == minute)
    specific = raw.loc[mask, ["_date", col]].drop_duplicates(subset="_date", keep="last")
    specific = specific.set_index("_date")
    return specific[col]


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
        df["hour"] >= 12,
        np.maximum(0, -df["temp_change_60m"]),
        0.0,
    )
    out["afternoon_temp_drop_120m"] = np.where(
        df["hour"] >= 12,
        np.maximum(0, -df["temp_change_60m"]),
        0.0,
    )
    return out


def compute_will_make_new_low(df: pd.DataFrame) -> pd.Series:
    logger.info("Computing will_make_new_low_after_now label (forward-looking)...")
    result = np.zeros(len(df))
    for _date, group in df.groupby("target_date", sort=False):
        idx = group.index.values
        mins = group["min_so_far_1m"].values
        temps = group["temp_current"].values
        forward_min = np.minimum.accumulate(temps[::-1])[::-1]
        for i in range(len(idx)):
            if i < len(idx) - 1:
                result[idx[i]] = 1.0 if forward_min[i + 1] < mins[i] - 1e-9 else 0.0
    return pd.Series(result, index=df.index, name="will_make_new_low_after_now")


def compute_tmin_timing_bucket(df: pd.DataFrame) -> pd.Series:
    logger.info("Computing tmin_timing_bucket label (forward-looking)...")

    def bucket(h):
        if pd.isna(h):
            return -1
        h = int(h)
        if 0 <= h < 8:
            return 0
        if 8 <= h < 18:
            return 1
        return 2

    first_zero = (
        df[df["is_downside_zero"] == 1]
        .groupby("target_date")["hour"]
        .first()
    )
    day_buckets = first_zero.apply(bucket)
    return df["target_date"].map(day_buckets).fillna(-1).astype(int)


def main():
    logger.info("=" * 60)
    logger.info("Model D Tmin — Feature Builder")
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
    logger.info(f"  {len(raw):,} rows, {raw['as_of_datetime_hkt'].min()} to {raw['as_of_datetime_hkt'].max()}")

    cross = compute_cross_midnight_features(raw)
    df = df.merge(cross, on="as_of_datetime_hkt", how="left")
    logger.info(f"  After merging cross-midnight features: {df.shape}")

    evening_map = compute_evening_features(raw)
    df["_date"] = df["as_of_datetime_hkt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df = df.merge(evening_map, left_on="_date", right_on="_target_date_next", how="left")
    df = df.drop(columns=["_date", "_target_date_next"])
    logger.info(f"  After merging evening features: {df.shape}")

    c_features = compute_group_c(df)
    for c in c_features.columns:
        df[c] = c_features[c].values
    logger.info(f"  After Group C (night cooling): {df.shape}")

    d_features = compute_group_d(df)
    for c in d_features.columns:
        df[c] = d_features[c].values
    logger.info(f"  After Group D (evening re-low): {df.shape}")

    df["will_make_new_low_after_now"] = compute_will_make_new_low(df)
    df["tmin_timing_bucket"] = compute_tmin_timing_bucket(df)
    logger.info(f"  Auxiliary label distribution:")
    logger.info(f"    will_make_new_low_after_now: mean={df['will_make_new_low_after_now'].mean():.4f}")
    bucket_dist = df["tmin_timing_bucket"].value_counts().to_dict()
    logger.info(f"    tmin_timing_bucket: {bucket_dist}")

    fill_cols = [c for c in MODEL_D_FEATURE_COLS + AUX_LABEL_COLS if c in df.columns]
    for c in fill_cols:
        df[c] = df[c].fillna(0)

    index_cols = ["target_date", "as_of_datetime_hkt", "minute_of_day"]
    obs_cols = [c for c in df.columns if c not in index_cols and c not in ["target_date", "as_of_datetime_hkt", "minute_of_day"]]
    all_cols = index_cols + obs_cols
    existing = [c for c in all_cols if c in df.columns]
    df_out = df[existing]

    logger.info(f"Final shape: {df_out.shape}")
    logger.info(f"  Base features: 38")
    logger.info(f"  Model D features: {len(MODEL_D_FEATURE_COLS)}")
    logger.info(f"  Auxiliary labels: {len(AUX_LABEL_COLS)}")
    logger.info(f"  Target columns: 6")
    logger.info(f"  Total columns: {df_out.shape[1]}")

    logger.info(f"Writing to {OUTPUT_PATH}...")
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
