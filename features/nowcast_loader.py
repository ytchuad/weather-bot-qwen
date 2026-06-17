from __future__ import annotations

import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rainfall_nowcast_download_and_station_features import (
    aggregate_station_features_long,
    normalize_nowcast_df,
    pivot_station_features,
    DEFAULT_HKO_LAT,
    DEFAULT_HKO_LON,
)

logger = logging.getLogger(__name__)

LIVE_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
WIDE_ALL_PATH = Path("data/features/rainfall_nowcast/rainfall_nowcast_station_features_wide_all.parquet")

RADIUS_KM = 5.0
HEAVY_RAIN_MM = 5.0

KNOWN_NOWCAST_FEATURES = {
    "rain_nc_sum_0_60m",
    "rain_nc_sum_0_120m",
    "rain_nc_any_0_120m",
    "rain_nc_front_loaded_ratio",
    "rain_nc_heavy_0_120m",
    "rain_nc_valid_horizon_count",
    "rain_nc_missing_flag",
    "rain_nc_nearest_mm_sum_30m",
    "rain_nc_nearest_mm_sum_60m",
    "rain_nc_nearest_mm_sum_90m",
    "rain_nc_nearest_mm_sum_120m",
    "rain_nc_mean_r5km_sum_30m",
    "rain_nc_mean_r5km_sum_60m",
    "rain_nc_mean_r5km_sum_90m",
    "rain_nc_mean_r5km_sum_120m",
    "rain_nc_max_r5km_sum_30m",
    "rain_nc_max_r5km_sum_60m",
    "rain_nc_max_r5km_sum_90m",
    "rain_nc_max_r5km_sum_120m",
    "rain_nc_min_r5km_sum_30m",
    "rain_nc_min_r5km_sum_60m",
    "rain_nc_min_r5km_sum_90m",
    "rain_nc_min_r5km_sum_120m",
    "rain_nc_p90_r5km_sum_30m",
    "rain_nc_p90_r5km_sum_60m",
    "rain_nc_p90_r5km_sum_90m",
    "rain_nc_p90_r5km_sum_120m",
    "rain_nc_area_gt0_r5km_sum_30m",
    "rain_nc_area_gt0_r5km_sum_60m",
    "rain_nc_area_gt0_r5km_sum_90m",
    "rain_nc_area_gt0_r5km_sum_120m",
    "rain_nc_area_gt5_r5km_sum_30m",
    "rain_nc_area_gt5_r5km_sum_60m",
    "rain_nc_area_gt5_r5km_sum_90m",
    "rain_nc_area_gt5_r5km_sum_120m",
    "rain_nowcast_age_minutes",
    "rain_nowcast_missing_flag",
}


def _filter_to_known(features: dict) -> dict:
    return {k: v for k, v in features.items() if k in KNOWN_NOWCAST_FEATURES}


def _fetch_live() -> dict:
    resp = requests.get(LIVE_URL, timeout=10)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), dtype=str)
    norm = normalize_nowcast_df(raw, source_member="live")
    long_feat = aggregate_station_features_long(
        norm,
        station_lat=DEFAULT_HKO_LAT,
        station_lon=DEFAULT_HKO_LON,
        radius_km=RADIUS_KM,
        heavy_rain_mm=HEAVY_RAIN_MM,
    )
    wide = pivot_station_features(long_feat)
    if wide.empty:
        return {}
    row = wide.sort_values("issue_time").iloc[-1]
    issued = row["issue_time"]
    feat = row.drop("issue_time").to_dict()
    feat = _filter_to_known(feat)
    feat["rain_nowcast_missing_flag"] = 0
    feat["rain_nowcast_age_minutes"] = 0.0
    feat["_issue_time"] = issued
    return feat


def _read_wide_all() -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(WIDE_ALL_PATH)
        if "issue_time" not in df.columns:
            return None
        df = df.sort_values("issue_time")
        return df
    except Exception as exc:
        logger.warning("Cannot read %s: %s", WIDE_ALL_PATH, exc)
        return None


def get_nowcast_features(snapshot_time: datetime | None = None) -> dict:
    live = _fetch_live()
    if live:
        if snapshot_time is not None:
            issued = live.pop("_issue_time")
            age = (snapshot_time - issued).total_seconds() / 60.0
            live["rain_nowcast_age_minutes"] = max(0.0, age)
        return live

    wide = _read_wide_all()
    if wide is not None and snapshot_time is not None:
        idx = wide["issue_time"].searchsorted(snapshot_time)
        if idx >= len(wide):
            idx = len(wide) - 1
        row = wide.iloc[max(0, idx)]
        issued = row["issue_time"]
        feat = row.drop("issue_time").to_dict()
        feat = _filter_to_known(feat)
        age = (snapshot_time - issued).total_seconds() / 60.0
        feat["rain_nowcast_age_minutes"] = max(0.0, age)
        feat["rain_nowcast_missing_flag"] = 0
        return feat

    return {}
