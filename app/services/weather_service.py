# app/services/weather_service.py
"""HKO weather data fetching — APIs + local parquet.

All functions use ``@st.cache_data`` with appropriate TTLs.
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
import streamlit as st

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


def hkt_now() -> datetime:
    """Current time in Hong Kong (UTC+8, naive)."""
    return datetime.utcnow() + HKT_OFFSET


# ── live HKO API fetchers ────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SHORT)
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


@st.cache_data(ttl=CACHE_TTL_SHORT)
def fetch_hko_data(target_date_str: str) -> dict:
    """Return max/min since midnight + forecast max/min + AWS hourly rows."""
    max_since_midnight: float | None = None
    min_since_midnight: float | None = None
    forecast_max: float | None = None
    forecast_min: float | None = None
    aws_hourly: list[dict] = []

    # 1) since-midnight CSV
    try:
        r = requests.get(HKO_MAXMIN_URL, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
        max_col = next((c for c in df.columns if "maximum" in c.lower()), None)
        min_col = next((c for c in df.columns if "minimum" in c.lower()), None)
        station_col = next((c for c in df.columns if "station" in c.lower() or "place" in c.lower()), None)
        if max_col and min_col and station_col:
            for _, row in df.iterrows():
                if "observatory" in str(row[station_col]).lower() or "天文台" in str(row[station_col]):
                    if pd.notna(row[max_col]):
                        max_since_midnight = float(row[max_col])
                    if pd.notna(row[min_col]):
                        min_since_midnight = float(row[min_col])
                    break
    except Exception as e:
        logger.debug("maxmin CSV fetch: %s", e)

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


@st.cache_data(ttl=CACHE_TTL_SHORT)
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

@st.cache_data(ttl=CACHE_TTL_MEDIUM)
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

    # 60-min ago temp
    time_threshold_60 = time_now - pd.Timedelta(minutes=55)
    df_60m = df_combined[df_combined["datetime"] <= time_threshold_60]
    temp_60m_ago = float(df_60m["temp"].iloc[-1]) if not df_60m.empty else temp_now

    # 120-min ago temp
    time_threshold_120 = time_now - pd.Timedelta(minutes=115)
    df_120m = df_combined[df_combined["datetime"] <= time_threshold_120]
    temp_120m_ago = float(df_120m["temp"].iloc[-1]) if not df_120m.empty else temp_now

    return {
        "temp_now": temp_now,
        "temp_60m_ago": temp_60m_ago,
        "temp_120m_ago": temp_120m_ago,
        "max_so_far": max_so_far,
        "min_so_far": min_so_far,
        "time_now": time_now,
        "df_today": df_today,
        "rh_now": rh_now,
        "temp_change_30m": temp_now - temp_60m_ago,   # approximate fallback
        "temp_change_60m": temp_now - temp_60m_ago,
        "time_since_max": 0.0,
        "time_since_min": 0.0,
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

@st.cache_data(ttl=CACHE_TTL_SHORT)
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

@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def load_rain_15min() -> pd.DataFrame:
    if RAIN_15MIN_PATH.exists():
        try:
            return pd.read_parquet(RAIN_15MIN_PATH)
        except Exception:
            pass
    return pd.DataFrame()


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
        # Try parquet first (for historical continuity)
        rain_df = load_rain_15min()
        if rain_df.empty:
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


@st.cache_data(ttl=CACHE_TTL_MEDIUM)
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

from cachetools import TTLCache, cached

_rain_live_cache = TTLCache(maxsize=2, ttl=300)


@cached(_rain_live_cache)
def compute_rain_kwargs_live() -> dict:
    """Compute rainfall features from live i-lens data only."""
    rain_df = fetch_rainfall_live()

    if rain_df is None or rain_df.empty:
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
