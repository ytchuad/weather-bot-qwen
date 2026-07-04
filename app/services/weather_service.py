# app/services/weather_service.py
"""HKO weather data fetching — APIs + local parquet.

All functions use cachetools TTLCache for caching.
"""

from __future__ import annotations

import io
import logging
import time as _time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from cachetools import TTLCache, cached

from ..config import (
    HKT_OFFSET,
    HKO_RHRREAD_URL,
    HKO_AWS_CSV_URL,
    HKO_MAXMIN_URL,
    HKO_FORECAST_URL_TEMPLATE,
    INTRADAY_10MIN_PATH,
    RAIN_15MIN_PATH,
    CACHE_TTL_SHORT,
    CACHE_TTL_MEDIUM,
)

logger = logging.getLogger(__name__)

_short_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_SHORT)
_medium_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_MEDIUM)
_intraday_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_SHORT)
_ilens_forecast_cache = TTLCache(maxsize=1, ttl=CACHE_TTL_MEDIUM)

ILENS_FORECAST_URL = "https://i-lens.hk/hkweather/daily_extract.php"

logger = logging.getLogger(__name__)


def hkt_now() -> datetime:
    """Current time in Hong Kong (UTC+8, naive)."""
    return datetime.utcnow() + HKT_OFFSET


# ── live HKO API fetchers ────────────────────────────────────────────

@cached(_short_cache)
def fetch_live_hko_temp_rh() -> tuple[datetime | None, float | None, float | None]:
    """Return (datetime, temp_c, rh_pct) from HKO rhrread API."""
    try:
        r = requests.get(HKO_RHRREAD_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        temp_data = data.get("temperature", {}).get("data", [])
        rh_data = data.get("humidity", {}).get("data", [])

        hko_temp: float | None = None
        for t in temp_data:
            if "observatory" in t.get("place", "").lower():
                hko_temp = float(t.get("value", t.get("temp", np.nan)))
                break
        if hko_temp is None and temp_data:
            hko_temp = float(temp_data[0].get("value", temp_data[0].get("temp", np.nan)))

        hko_rh: float | None = None
        for h in rh_data:
            if "observatory" in h.get("place", "").lower():
                hko_rh = float(h.get("value", h.get("humidity", np.nan)))
                break
        if hko_rh is None and rh_data:
            hko_rh = float(rh_data[0].get("value", rh_data[0].get("humidity", np.nan)))

        rt = data["temperature"].get("recordTime", "")
        if rt:
            dt = pd.to_datetime(rt)
            if dt.tzinfo is not None:
                dt = dt.tz_convert("Asia/Hong_Kong").tz_localize(None)
            return dt.to_pydatetime(), hko_temp, hko_rh
        return hkt_now(), hko_temp, hko_rh
    except Exception as e:
        logger.warning("fetch_live_hko_temp_rh failed: %s", e)
        return None, None, None


@cached(_medium_cache)
def fetch_hko_data(target_date_str: str) -> dict:
    """Return max/min since midnight + forecast max/min + AWS hourly rows.
    
    Args:
        target_date_str: Date in YYYYMMDD or YYYY-MM-DD format
    """
    # Normalize to YYYYMMDD format
    if "-" in target_date_str:
        target_date_str = target_date_str.replace("-", "")
    
    max_since_midnight: float | None = None
    min_since_midnight: float | None = None
    forecast_max: float | None = None
    forecast_min: float | None = None
    aws_hourly: list[dict] = []

    # 1) AWS CSV - last 24 hours, calculate max/min since midnight
    try:
        r = requests.get(HKO_AWS_CSV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["datetime"] = pd.to_datetime(df["Date"], format="%Y/%m/%d %H:%M")
        # Filter for today (since midnight)
        today_date = pd.to_datetime(target_date_str, format="%Y%m%d")
        today_mask = df["datetime"].dt.date == today_date.date()
        today_data = df[today_mask]
        if not today_data.empty:
            max_since_midnight = float(today_data["Temp"].max())
            min_since_midnight = float(today_data["Temp"].min())
    except Exception as e:
        logger.debug("AWS CSV fetch: %s", e)

    # 2) AWS forecast JSON
    try:
        url = HKO_FORECAST_URL_TEMPLATE.format(ts=int(_time.time() * 1000))
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        hourly_max, hourly_min = -99.0, 99.0
        for entry in data.get("HourlyWeatherForecast", []):
            f_hour = str(entry.get("ForecastHour", ""))
            try:
                val = float(entry.get("ForecastTemperature", 0))
                if val < 10.0 or val > 45.0:
                    continue
                if len(f_hour) == 10:
                    dt_obj = datetime.strptime(f_hour, "%Y%m%d%H")
                elif len(f_hour) == 12:
                    dt_obj = datetime.strptime(f_hour, "%Y%m%d%H%M")
                else:
                    continue
                aws_hourly.append({"time": dt_obj, "temp": val})
                if f_hour[:8] == target_date_str:
                    if val > hourly_max:
                        hourly_max = val
                    if val < hourly_min:
                        hourly_min = val
            except Exception:
                continue
        if hourly_max > -99.0:
            forecast_max = hourly_max
        if hourly_min < 99.0:
            forecast_min = hourly_min
    except Exception as e:
        logger.debug("AWS forecast fetch: %s", e)

    return {
        "max_since_midnight": max_since_midnight,
        "min_since_midnight": min_since_midnight,
        "forecast_max": forecast_max,
        "forecast_min": forecast_min,
        "aws_hourly": aws_hourly,
    }


@cached(_short_cache)
def fetch_hko_intraday_csv(_cache_buster: int = 0) -> pd.DataFrame:
    """Fetch HKO AWS CSV and return DataFrame with datetime, temp, rh columns.

    The ``_cache_buster`` param doubles as Streamlit cache key and URL
    cache-busting parameter.  Callers should pass ``int(time.time())``.
    """
    ts = _cache_buster if _cache_buster > 0 else int(_time.time() * 1000)
    url = f"{HKO_AWS_CSV_URL}?_={ts}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        records: list[dict] = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                dt = pd.to_datetime(parts[0].strip(), format="%Y/%m/%d %H:%M")
                temp = float(parts[1].strip())
                rh = float(parts[2].strip()) if len(parts) >= 3 else 50.0
                records.append({"datetime": dt, "temp": temp, "rh": rh})
            except (ValueError, IndexError):
                continue
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("fetch_hko_intraday_csv: %s", e)
        return pd.DataFrame()


# ── intraday state builder ───────────────────────────────────────────

@cached(_intraday_cache)
def get_intraday_state(target_date_str: str) -> dict | None:
    """Build intraday state dict for a given date.

    Merges parquet history + live HKO CSV, returns {temp_now, min_so_far,
    max_so_far, time_now, temp_60m_ago, temp_120m_ago, temp_change_30m,
    temp_change_60m, rh_now, df_today} or None.
    """
    target_dt = pd.to_datetime(target_date_str, format="%Y%m%d")
    prev_dt = target_dt - pd.Timedelta(days=1)

    # Load parquet
    df_hist = pd.DataFrame()
    if INTRADAY_10MIN_PATH.exists():
        try:
            df_all = pd.read_parquet(INTRADAY_10MIN_PATH)
            mask = (df_all["datetime"].dt.date >= prev_dt.date()) & (
                df_all["datetime"].dt.date <= target_dt.date()
            )
            df_hist = df_all[mask].copy()
        except Exception:
            pass

    # Fetch live CSV (with cache-buster to force invalidation when needed)
    df_live = fetch_hko_intraday_csv(_cache_buster=int(_time.time() // 60))

    # Combine
    df_combined = pd.concat([df_hist, df_live], ignore_index=True)
    if df_combined.empty:
        return None
    df_combined = df_combined.drop_duplicates(subset="datetime").sort_values("datetime")

    df_today = df_combined[df_combined["datetime"].dt.date == target_dt.date()].copy()
    if df_today.empty:
        return None

    temp_now = float(df_today["temp"].iloc[-1])
    time_now: pd.Timestamp = df_today["datetime"].iloc[-1]

    # RH
    if "rh" in df_today.columns and df_today["rh"].notna().any():
        rh_now = float(df_today["rh"].dropna().iloc[-1])
    else:
        rh_now = 50.0

    max_so_far = float(df_today["temp"].cummax().iloc[-1])
    min_so_far = float(df_today["temp"].cummin().iloc[-1])

    # 30-min ago temp
    time_threshold_30 = time_now - pd.Timedelta(minutes=25)
    df_30m = df_combined[df_combined["datetime"] <= time_threshold_30]
    temp_30m_ago = float(df_30m["temp"].iloc[-1]) if not df_30m.empty else temp_now

    # 60-min ago temp
    time_threshold_60 = time_now - pd.Timedelta(minutes=55)
    df_60m = df_combined[df_combined["datetime"] <= time_threshold_60]
    temp_60m_ago = float(df_60m["temp"].iloc[-1]) if not df_60m.empty else temp_now

    # 120-min ago temp
    time_threshold_120 = time_now - pd.Timedelta(minutes=115)
    df_120m = df_combined[df_combined["datetime"] <= time_threshold_120]
    temp_120m_ago = float(df_120m["temp"].iloc[-1]) if not df_120m.empty else temp_now

    # Time since max/min were last observed
    time_since_max = 0.0
    time_since_min = 0.0
    if not df_today.empty:
        max_idxs = df_today.index[df_today["temp"] == max_so_far]
        if len(max_idxs) > 0:
            time_of_max = df_today.loc[max_idxs[-1], "datetime"]
            time_since_max = max(0.0, (time_now - time_of_max).total_seconds() / 60.0)
        min_idxs = df_today.index[df_today["temp"] == min_so_far]
        if len(min_idxs) > 0:
            time_of_min = df_today.loc[min_idxs[-1], "datetime"]
            time_since_min = max(0.0, (time_now - time_of_min).total_seconds() / 60.0)

    # Pre-compute buffer-derived features for Model 2A stability
    temp_volatility_60m = 0.0
    if len(df_today) >= 10:
        temp_vals = df_today["temp"].values[-60:] if len(df_today) >= 60 else df_today["temp"].values
        temp_volatility_60m = float(np.std(temp_vals[-60:], ddof=1) if len(temp_vals) >= 2 else 0.0)

    temp_acceleration_60m = 0.0
    if len(df_today) >= 60:
        _t30 = float(df_today["temp"].iloc[-30])
        _t60 = float(df_today["temp"].iloc[-60])
        _s30 = (temp_now - _t30) / 30.0
        _s60 = (_t30 - _t60) / 30.0
        temp_acceleration_60m = _s30 - _s60

    rh_change_60m = 0.0
    if "rh" in df_today.columns and len(df_today) >= 60 and df_today["rh"].iloc[-60] is not None:
        rh_change_60m = rh_now - float(df_today["rh"].iloc[-60])

    # Pre-compute dew point features for Model 2A stability
    dew_point_change_60m = 0.0
    dew_point_spread_change_60m = 0.0
    if len(df_today) >= 60 and rh_change_60m != 0.0:
        try:
            import math as _m
            _a, _b = 17.625, 243.04
            _t60 = float(df_today["temp"].iloc[-60])
            _rh60 = float(df_today["rh"].iloc[-60])
            _gamma = _m.log(rh_now / 100.0) + (_a * temp_now) / (_b + temp_now)
            _dp0 = (_b * _gamma) / (_a - _gamma)
            _gamma60 = _m.log(_rh60 / 100.0) + (_a * _t60) / (_b + _t60)
            _dp60 = (_b * _gamma60) / (_a - _gamma60)
            dew_point_change_60m = _dp0 - _dp60
            dew_point_spread_change_60m = (temp_now - _dp0) - (_t60 - _dp60)
        except Exception:
            pass

    return {
        "temp_now": temp_now,
        "temp_30m_ago": temp_30m_ago,
        "temp_60m_ago": temp_60m_ago,
        "temp_120m_ago": temp_120m_ago,
        "max_so_far": max_so_far,
        "min_so_far": min_so_far,
        "time_now": time_now,
        "df_today": df_today,
        "rh_now": rh_now,
        "temp_change_30m": temp_now - temp_30m_ago,
        "temp_change_60m": temp_now - temp_60m_ago,
        "time_since_max": time_since_max,
        "time_since_min": time_since_min,
        "temp_volatility_60m": temp_volatility_60m,
        "temp_acceleration_60m": temp_acceleration_60m,
        "rh_change_60m": rh_change_60m,
        "dew_point_change_60m": dew_point_change_60m,
        "dew_point_spread_change_60m": dew_point_spread_change_60m,
    }


# ── rainfall ─────────────────────────────────────────────────────────

INSTANT_RAIN_URL = "https://i-lens.hk/hkweather/instant_chart.php?chart_type=STATION_ACCUM_RAIN"
RAIN_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
}

def _parse_rainfall_from_html(html: str) -> list[dict]:
    """Parse rainfall data from i-lens HTML for King's Park station."""
    import re
    import json
    pattern = r"\{name:\s*'香港天文台'\s*,\s*data:\s*(\[.*?\])\s*\}"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        return []
    data_str = match.group(1)
    def replace_utc(m):
        y, mo, d, h, mi = int(m.group(1)), int(m.group(2)) + 1, int(m.group(3)), int(m.group(4)), int(m.group(5))
        return f'"{y}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}"'
    data_str = re.sub(r"Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\)", replace_utc, data_str)
    try:
        data = json.loads(data_str)
        return [{'datetime': pd.to_datetime(item[0]), 'rainfall': float(item[1])} for item in data if len(item) == 2 and item[1] is not None]
    except Exception:
        return []

@cached(_short_cache)
def fetch_rainfall_live() -> pd.DataFrame:
    """Fetch live rainfall data from i-lens; returns DataFrame with datetime/rainfall."""
    try:
        r = requests.get(INSTANT_RAIN_URL, headers=RAIN_HEADERS, timeout=10)
        r.raise_for_status()
        records = _parse_rainfall_from_html(r.text)
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        logger.warning("fetch_rainfall_live failed: %s", e)
        return pd.DataFrame()

@cached(_medium_cache)
def load_rain_15min() -> pd.DataFrame:
    if RAIN_15MIN_PATH.exists():
        try:
            return pd.read_parquet(RAIN_15MIN_PATH)
        except Exception:
            pass
    return pd.DataFrame()


# ── pressure data (Model F) ─────────────────────────────────────────────
PRESSURE_CACHE = TTLCache(maxsize=1, ttl=100)

@cached(PRESSURE_CACHE)
def get_pressure_with_ttl_cache() -> dict:
    """Fetch pressure and pressure_30m_ago from i-lens.hk with 100s TTL cache."""
    import re
    PRESSURE_URL = "https://i-lens.hk/hkweather/instant_chart.php?chart_type=DG_MSLP"
    try:
        r = requests.get(PRESSURE_URL, headers=RAIN_HEADERS, timeout=10)
        r.raise_for_status()
        html = r.text
        # Parse JavaScript data array: [[timestamp, pressure], ...]
        match = re.search(r"data:\s*\[(\[.*?\])\s*\]", html, re.DOTALL)
        if not match:
            return {"pressure": None, "pressure_30m_ago": None}
        data_str = match.group(1)
        # Extract values
        values = re.findall(r"\[\s*Date\.UTC\(\d+,\d+,\d+,\d+,\d+\)\s*,\s*([0-9.]+)\s*\]", data_str)
        if not values:
            return {"pressure": None, "pressure_30m_ago": None}
        pressures = [float(v) for v in values]
        if len(pressures) < 1:
            return {"pressure": None, "pressure_30m_ago": None}
        pressure = pressures[-1]  # Latest
        # Estimate 30-min ago (6 steps if 5-min intervals)
        idx_30m = max(0, len(pressures) - 7)
        pressure_30m_ago = pressures[idx_30m]
        return {"pressure": pressure, "pressure_30m_ago": pressure_30m_ago}
    except Exception as e:
        logger.warning("get_pressure_with_ttl_cache failed: %s", e)
        return {"pressure": None, "pressure_30m_ago": None}


def compute_rain_kwargs(
    target_date_str: str,
    now_dt: datetime,
    intra_df: pd.DataFrame | None = None,
) -> dict:
    """Compute rainfall features needed by intraday models.

    Returns dict with keys like rain_60m, rain_120m, rain_data_ok, plus
    prev_evening_* features when previous-day evening data is available.
    """
    rain_60m = 0.0
    rain_120m = 0.0
    rain_data_ok = False

    try:
        rain_df = fetch_rainfall_live()
        if not rain_df.empty:
            rain_df = rain_df[rain_df["datetime"] <= now_dt]
            if not rain_df.empty:
                now_rain = rain_df["rainfall"].iloc[-1]
                target_time_60 = now_dt - pd.Timedelta(minutes=60)
                prev_60 = rain_df[rain_df["datetime"] <= target_time_60]
                rain_60m_ago = prev_60["rainfall"].iloc[-1] if not prev_60.empty else 0.0
                rain_60m = max(0.0, now_rain - rain_60m_ago)

                target_time_120 = now_dt - pd.Timedelta(minutes=120)
                prev_120 = rain_df[rain_df["datetime"] <= target_time_120]
                rain_120m_ago = prev_120["rainfall"].iloc[-1] if not prev_120.empty else 0.0
                rain_120m = max(0.0, now_rain - rain_120m_ago)
                rain_data_ok = True
    except Exception as e:
        logger.warning("compute_rain_kwargs: %s", e)

    kwargs: dict[str, Any] = {
        "rain_60m": rain_60m,
        "rain_120m": rain_120m,
        "rain_data_ok": rain_data_ok,
        "rainfall_60m_missing_flag": 0 if rain_data_ok else 1,
        "rainfall_120m_missing_flag": 0 if rain_data_ok else 1,
    }

    # Previous-day evening features (required by Model D/E)
    try:
        target_dt = pd.to_datetime(target_date_str, format="%Y%m%d")
        prev_dt = target_dt - pd.Timedelta(days=1)
        if INTRADAY_10MIN_PATH.exists():
            df_prev_all = pd.read_parquet(INTRADAY_10MIN_PATH)
            prev_mask = df_prev_all["datetime"].dt.date == prev_dt.date()
            df_prev = df_prev_all[prev_mask].copy()
            if not df_prev.empty:
                prev_evening = df_prev[
                    (df_prev["datetime"].dt.hour >= 18)
                    & (df_prev["datetime"].dt.hour < 24)
                ]
                if not prev_evening.empty and "rh" in prev_evening.columns:
                    a, b = 17.27, 237.7
                    kwargs.update({
                        "prev_18_temp": _extract_at_hour(df_prev, 18, 0),
                        "prev_21_temp": _extract_at_hour(df_prev, 21, 0),
                        "prev_2359_temp": _extract_at_hour(df_prev, 23, 59),
                        "prev_evening_temp_change": float(prev_evening["temp"].iloc[-1] - prev_evening["temp"].iloc[0]),
                        "prev_evening_temp_min": float(prev_evening["temp"].min()),
                        "prev_evening_temp_range": float(prev_evening["temp"].max() - prev_evening["temp"].min()),
                        "prev_evening_temp_slope": float(prev_evening["temp"].iloc[-1] - prev_evening["temp"].iloc[0]) / max(len(prev_evening), 1),
                        "prev_evening_rh_mean": float(prev_evening["rh"].mean()),
                        "prev_evening_rh_max": float(prev_evening["rh"].max()),
                        "prev_evening_dew_point_mean": float(
                            (b * (a * prev_evening["temp"].iloc[0]) / (b + prev_evening["temp"].iloc[0])
                             + np.log(max(prev_evening["rh"].iloc[0], 0.01) / 100.0))
                            / (a - (a * prev_evening["temp"].iloc[0]) / (b + prev_evening["temp"].iloc[0]))
                        ),
                        "prev_evening_rainfall_18_24": 0.0,
                        "prev_evening_rain_flag": 0,
                    })
    except Exception:
        pass

    return kwargs


def _extract_at_hour(df: pd.DataFrame, hour: int, minute: int = 0) -> float:
    """Extract temperature at a specific hour:minute from intraday DataFrame."""
    mask = (df["datetime"].dt.hour == hour) & (df["datetime"].dt.minute == minute)
    subset = df.loc[mask, "temp"]
    return float(subset.iloc[0]) if len(subset) > 0 else 0.0


@cached(_medium_cache)
def fetch_hko_aws_forecast() -> tuple[float | None, float | None, list]:
    """Return (forecast_max, forecast_min, daily_forecast_list) from HKO AWS."""
    url = HKO_FORECAST_URL_TEMPLATE.format(ts=int(_time.time() * 1000))
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        daily = data.get("DailyForecast", [])
        if not daily:
            return None, None, []
        today = daily[0]
        forecast_max = float(today.get("ForecastMaximumTemperature", 0))
        forecast_min = float(today.get("ForecastMinimumTemperature", 0))
        return forecast_max, forecast_min, daily
    except Exception:
        return None, None, []


# ── rainfall live compute ────────────────────────────────────────────

_rain_live_cache = TTLCache(maxsize=2, ttl=CACHE_TTL_MEDIUM)


@cached(_rain_live_cache)
def compute_rain_kwargs_live() -> dict:
    """Compute rainfall features from live i-lens data only."""
    # Force fresh fetch instead of using cached value
    rain_df = _fetch_rainfall_live_uncached()

    if rain_df is None or not isinstance(rain_df, pd.DataFrame) or rain_df.empty:
        return {"rain_60m": 0.0, "rain_120m": 0.0, "rain_data_ok": False}

    rain_df = rain_df.sort_values("datetime")

    now_rain = float(rain_df["rainfall"].iloc[-1])
    now_dt = rain_df["datetime"].iloc[-1]

    t60 = now_dt - pd.Timedelta(minutes=60)
    t120 = now_dt - pd.Timedelta(minutes=120)

    prev_60 = rain_df[rain_df["datetime"] <= t60]
    prev_120 = rain_df[rain_df["datetime"] <= t120]

    rain_60m = max(0.0, now_rain - (float(prev_60["rainfall"].iloc[-1]) if not prev_60.empty else 0.0))
    rain_120m = max(0.0, now_rain - (float(prev_120["rainfall"].iloc[-1]) if not prev_120.empty else 0.0))

    return {
        "rain_60m": round(rain_60m, 1),
        "rain_120m": round(rain_120m, 1),
        "rain_data_ok": True,
    }


# Internal uncached version for compute_rain_kwargs_live
def _fetch_rainfall_live_uncached() -> pd.DataFrame:
    """Uncached rainfall fetch for compute_rain_kwargs_live to avoid cache pollution."""
    try:
        r = requests.get(INSTANT_RAIN_URL, headers=RAIN_HEADERS, timeout=10)
        r.raise_for_status()
        records = _parse_rainfall_from_html(r.text)
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        logger.warning("_fetch_rainfall_live_uncached failed: %s", e)
        return pd.DataFrame()


def get_accumulated_rain_today() -> float | None:
    """Get accumulated rainfall for today from i-lens STATION_ACCUM_RAIN."""
    try:
        r = requests.get(INSTANT_RAIN_URL, headers=RAIN_HEADERS, timeout=10)
        r.raise_for_status()
        df = _parse_rainfall_from_html(r.text)
        if not df:
            return None
        df = pd.DataFrame(df)
        today = hkt_now().date()
        today_mask = df["datetime"].dt.date == today
        today_data = df[today_mask]
        if today_data.empty:
            return 0.0
        # Accumulated rainfall is the latest value (total from midnight)
        return float(today_data["rainfall"].iloc[-1])
    except Exception:
        return None


# ── pressure live (HKO CSV) ────────────────────────────────────────────

HKO_PRESSURE_CSV_URL = "https://www.hko.gov.hk/wxinfo/awsgis/hko_pre.csv"

_pressure_cache = TTLCache(maxsize=1, ttl=CACHE_TTL_MEDIUM)
_last_pressure_kwargs: dict | None = None


@cached(_pressure_cache)
def fetch_pressure_live() -> pd.DataFrame:
    """Fetch 1-min pressure CSV from HKO, return DataFrame with datetime, pressure."""
    try:
        r = requests.get(HKO_PRESSURE_CSV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["datetime"] = pd.to_datetime(df["Date"], format="%Y/%m/%d %H:%M")
        df = df.drop(columns=["Date"]).dropna()
        df = df.rename(columns={"Pressure": "pressure"})
        return df
    except Exception as e:
        logger.warning("fetch_pressure_live failed: %s", e)
        return pd.DataFrame()


def compute_pressure_kwargs() -> dict:
    """Compute pressure_current, pressure_30m_ago, pressure_change_60m/180m from live CSV."""
    global _last_pressure_kwargs
    df = fetch_pressure_live()
    if df.empty:
        if _last_pressure_kwargs is not None:
            return _last_pressure_kwargs
        return {"pressure_current": None, "pressure_30m_ago": None,
                "pressure_change_60m": 0.0, "pressure_change_180m": 0.0}
    now = hkt_now()
    latest = float(df["pressure"].iloc[-1])
    t_30 = now - timedelta(minutes=30)
    idx_30 = (df["datetime"] - t_30).abs().idxmin()
    p_30 = float(df.loc[idx_30, "pressure"])
    t_60 = now - timedelta(minutes=60)
    idx_60 = (df["datetime"] - t_60).abs().idxmin()
    p_60 = float(df.loc[idx_60, "pressure"])
    t_180 = now - timedelta(minutes=180)
    idx_180 = (df["datetime"] - t_180).abs().idxmin()
    p_180 = float(df.loc[idx_180, "pressure"])
    result = {
        "pressure_current": latest,
        "pressure_30m_ago": p_30,
        "pressure_change_60m": latest - p_60,
        "pressure_change_180m": latest - p_180,
    }
    _last_pressure_kwargs = result
    return result


# ── wind live (i-Lens DG_WIND) ─────────────────────────────────────────

WIND_INSTANT_URL = "https://i-lens.hk/hkweather/instant_chart.php?chart_type=DG_WIND"
_WIND_STATION_GROUP_MAP = {"參考": "ref", "離岸": "offshore", "高山": "highland"}
_WIND_VICTORIA_HARBOUR_STATIONS = {"京士柏", "啟德", "九龍天星碼頭"}
_WIND_GROUP_GROUPS = ["ref", "offshore", "highland", "victoria_harbour"]


def _parse_wind_from_html(html: str) -> pd.DataFrame:
    """Parse i-Lens wind HTML into DataFrame with timestamp, station_type, station, wind_speed."""
    import re
    records = []
    for m in re.finditer(
        r"name\s*:\s*'([^']+)'\s*,\s*data\s*:\s*\[(.*?)\](?=\s*\]\s*\}\s*[,\)])",
        html, re.DOTALL
    ):
        station = m.group(1)
        data_part = m.group(2)
        station_type = "未知"
        for k in _WIND_STATION_GROUP_MAP:
            if k in station:
                station_type = k
                break
        pts = re.findall(
            r"Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\)\s*,\s*([0-9.]+)",
            data_part
        )
        for y_str, mon_str, d_str, h_str, min_str, val_str in pts:
            try:
                ts = datetime(int(y_str), int(mon_str) + 1, int(d_str), int(h_str), int(min_str))
                records.append({
                    "timestamp": ts,
                    "station_type": station_type,
                    "station": station,
                    "wind_speed": float(val_str),
                })
            except ValueError:
                continue
    return pd.DataFrame(records)


_wind_cache = TTLCache(maxsize=1, ttl=CACHE_TTL_MEDIUM)
_last_wind_kwargs: dict | None = None


@cached(_wind_cache)
def fetch_wind_live() -> pd.DataFrame:
    """Fetch live wind data from i-Lens DG_WIND, return DataFrame with all stations."""
    try:
        r = requests.get(WIND_INSTANT_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = _parse_wind_from_html(r.text)
        if df.empty:
            return df
        df["group"] = df["station_type"].map(_WIND_STATION_GROUP_MAP).fillna("victoria_harbour")
        df.loc[df["station"].isin(_WIND_VICTORIA_HARBOUR_STATIONS), "group"] = "victoria_harbour"
        return df
    except Exception as e:
        logger.warning("fetch_wind_live failed: %s", e)
        return pd.DataFrame()


def compute_wind_kwargs() -> dict:
    """Compute wind features from live i-Lens data."""
    global _last_wind_kwargs
    df = fetch_wind_live()
    if df.empty:
        if _last_wind_kwargs is not None:
            return _last_wind_kwargs
        return {
            "wind_ref_mean": 0.0, "wind_ref_max": 0.0,
            "wind_victoria_harbour_mean": 0.0, "wind_victoria_harbour_max": 0.0,
            "wind_highland_mean": 0.0, "wind_highland_max": 0.0,
            "wind_all_change_60m": 0.0, "wind_kings_park_current": 0.0,
        }
    result = {}
    for grp in _WIND_GROUP_GROUPS:
        sub = df[df["group"] == grp]
        if not sub.empty:
            result[f"wind_{grp}_mean"] = float(sub.groupby("timestamp")["wind_speed"].mean().mean())
            result[f"wind_{grp}_max"] = float(sub.groupby("timestamp")["wind_speed"].max().max())
        else:
            result[f"wind_{grp}_mean"] = 0.0
            result[f"wind_{grp}_max"] = 0.0
    # Kings Park
    kp = df[df["station"] == "京士柏"]
    result["wind_kings_park_current"] = float(kp["wind_speed"].iloc[-1]) if not kp.empty else 0.0
    # All stations — compute 60m change
    all_ts = df.groupby("timestamp")["wind_speed"].mean().reset_index()
    all_ts = all_ts.sort_values("timestamp")
    now = hkt_now()
    t_60 = now - timedelta(minutes=60)
    now_mean = all_ts["wind_speed"].iloc[-1] if not all_ts.empty else 0.0
    past = all_ts[all_ts["timestamp"] <= t_60]
    past_mean = past["wind_speed"].iloc[-1] if not past.empty else now_mean
    result["wind_all_change_60m"] = float(now_mean - past_mean)
    _last_wind_kwargs = result
    return result


@cached(_ilens_forecast_cache, key=lambda target_date_str, **kw: (target_date_str,))
def fetch_hko_ilens_forecast(target_date_str: str | None = None) -> dict | None:
    """Fetch HKO 9-day forecast from i-lens (same source as training data).

    Parses the same HTML table as ilens_forecast_days.py to extract
    the latest forecast revision (max forecast_issue_date) for the
    given target_date. Returns forecast_max_temp, forecast_min_temp.

    Training source: https://i-lens.hk/hkweather/daily_extract.php?date=YYYY-MM-DD
    """
    from bs4 import BeautifulSoup
    import re
    from datetime import timezone, timedelta

    tz_hkt = timezone(timedelta(hours=8))
    if target_date_str is None:
        target_date_str = (datetime.now(tz_hkt)).strftime("%Y-%m-%d")

    url = f"{ILENS_FORECAST_URL}?date={target_date_str}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
    except Exception:
        logger.warning("Failed to fetch i-lens forecast for %s", target_date_str)
        return None

    try:
        soup = BeautifulSoup(r.text, "html.parser")
        # Find table by h2 header "香港天文台對 ... 所作出的天氣預測"
        h2 = soup.find("h2", string=re.compile("香港天文台對.*所作出的天氣預測"))
        table = h2.find_next("table") if h2 else None
        if not table:
            tables = soup.find_all("table")
            for t in tables:
                headers = t.find_all("th")
                if any("發佈日期" in h.get_text(strip=True) for h in headers):
                    table = t
                    break
        if not table:
            return None

        rows = table.find_all("tr")
        best = None
        latest_issue_date = None
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 8:
                continue
            clean = [c.get_text(strip=True) for c in cols]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", clean[0]):
                continue
            # Track the row with the latest forecast_issue_date
            if latest_issue_date is None or clean[0] > latest_issue_date:
                latest_issue_date = clean[0]
                best = {
                    "forecast_issue_date": clean[0],
                    "forecast_issue_time": clean[1],
                    "forecast_tmin": float(clean[2]) if len(clean) > 2 and clean[2] else None,
                    "forecast_tmax": float(clean[3]) if len(clean) > 3 and clean[3] else None,
                }
        return best
    except Exception as e:
        logger.warning("Failed to parse i-lens forecast HTML: %s", e)
        return None


def get_nowcast_rainfall() -> float | None:
    """Get rainfall nowcast from HKO Gridded_rainfall_nowcast.csv."""
    NOWCAST_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
    try:
        r = requests.get(NOWCAST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None
        # Parse last non-empty line for latest nowcast value
        for line in reversed(lines):
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[-1]:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
        return None
    except Exception:
        return None
