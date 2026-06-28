# features/model_2a_feature_builder.py
"""Model 2A shared feature builder.

Used for:
1. Historical feature-store build
2. Real-time inference
3. Replay parity check

All features respect: available_time <= decision_time
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_model_2a_features(
    decision_time: pd.Timestamp,
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    spec: dict,
    mode: str,
) -> pd.DataFrame:
    """Build Model 2A feature vector for a single decision_time.

    Args:
        decision_time: The decision time for feature computation.
        weather_canonical: Canonical weather obs DataFrame.
        wind_canonical: Canonical wind obs DataFrame.
        forecast_canonical: Canonical forecast DataFrame.
        spec: Model spec dict.
        mode: 'historical', 'live', or 'replay'.

    Returns:
        DataFrame with single row of features indexed by decision_time.
    """
    _validate_canonical_inputs(weather_canonical, wind_canonical, forecast_canonical)

    weather = _filter_available(weather_canonical, decision_time, "available_time")
    wind = _filter_available(wind_canonical, decision_time, "available_time")
    forecast = _filter_available(forecast_canonical, decision_time, "available_time")

    target_date = decision_time.normalize().date()
    hour = decision_time.hour
    minute = decision_time.minute
    mins_midnight = hour * 60 + minute

    temp_clean = _get_latest_value(weather, "temp_current_clean")
    temp_raw = _get_latest_value(weather, "temp_current_raw")
    rh = _get_latest_value(weather, "rh_current")
    dp = _get_latest_value(weather, "dew_point_current")
    pressure = _get_latest_value(weather, "pressure_current")

    temp_current = temp_raw if not np.isnan(temp_raw) else temp_clean
    if np.isnan(temp_current):
        temp_current = np.nan

    weather_date = weather[weather["timestamp"].dt.date == target_date] if "timestamp" in weather.columns else weather
    weather_date = _filter_available(weather_date, decision_time, "available_time")
    if len(weather_date) > 0:
        obs_ts = weather_date["timestamp"].dropna()
        obs_age = (decision_time - obs_ts.max()).total_seconds() / 60 if len(obs_ts) > 0 else np.nan
    else:
        obs_age = np.nan

    date_weather = weather if mode == "historical" else weather[
        weather.get("timestamp", pd.Series([pd.NaT] * len(weather))).dt.date == target_date
    ]
    if mode == "live" and len(date_weather) == 0:
        date_weather = weather

    if len(date_weather) > 0:
        temp_today = date_weather["temp_current_clean"].dropna()
        if len(temp_today) > 0:
            max_so_far = temp_today.max()
            min_so_far = temp_today.min()
            if np.isnan(min_so_far):
                min_so_far = temp_current if not np.isnan(temp_current) else np.nan
        else:
            max_so_far = temp_current if not np.isnan(temp_current) else np.nan
            min_so_far = temp_current if not np.isnan(temp_current) else np.nan
    else:
        max_so_far = temp_current if not np.isnan(temp_current) else np.nan
        min_so_far = temp_current if not np.isnan(temp_current) else np.nan

    range_so_far = max_so_far - min_so_far if not (np.isnan(max_so_far) or np.isnan(min_so_far)) else np.nan
    drop_from_max = max_so_far - temp_current if not (np.isnan(max_so_far) or np.isnan(temp_current)) else np.nan
    rise_from_min = temp_current - min_so_far if not (np.isnan(temp_current) or np.isnan(min_so_far)) else np.nan

    # Temperature trends
    temp_col = "temp_current_clean"
    temp_slope_10m = _compute_slope(date_weather, temp_col, decision_time, 1)
    temp_slope_30m = _compute_slope(date_weather, temp_col, decision_time, 3)
    temp_slope_60m = _compute_slope(date_weather, temp_col, decision_time, 6)

    temp_diff_30m = _compute_diff(date_weather, temp_col, decision_time, 3)
    temp_diff_60m = _compute_diff(date_weather, temp_col, decision_time, 6)
    temp_acceleration_60m = temp_slope_30m - _compute_diff(date_weather, temp_col, decision_time, 3, shift=3)

    # RH trends
    rh_change_30m = _compute_diff(date_weather, "rh_current", decision_time, 3)
    rh_change_60m = _compute_diff(date_weather, "rh_current", decision_time, 6)

    # Dew point
    dew_point_spread_current = temp_current - dp if not (np.isnan(temp_current) or np.isnan(dp)) else np.nan
    dew_point_spread_change_60m = _compute_diff(
        date_weather, "dew_point_spread", decision_time, 6
    ) if "dew_point_spread" in date_weather.columns else _compute_derived_dew_point_spread_diff(
        date_weather, temp_col, "dew_point_current", decision_time
    )

    # Pressure trends
    pressure_change_30m = _compute_diff(date_weather, "pressure_current", decision_time, 3)
    pressure_change_60m = _compute_diff(date_weather, "pressure_current", decision_time, 6)

    # Time features
    since_start_min = mins_midnight - 360  # minutes since 06:00
    if since_start_min < 0:
        since_start_min = 0

    # Forecast features
    forecast_row = _get_latest_forecast(forecast, decision_time, target_date)
    forecast_max_temp = _safe_val(forecast_row, "forecast_max_temp")
    forecast_min_temp = _safe_val(forecast_row, "forecast_min_temp")
    forecast_range = (forecast_max_temp - forecast_min_temp
                      if not (np.isnan(forecast_max_temp) or np.isnan(forecast_min_temp)) else np.nan)
    forecast_max_minus_max_so_far = (forecast_max_temp - max_so_far
                                     if not (np.isnan(forecast_max_temp) or np.isnan(max_so_far)) else np.nan)

    forecast_age_minutes = np.nan
    forecast_issue_hour = np.nan
    if forecast_row is not None:
        fi = _safe_val(forecast_row, "forecast_issue_datetime")
        if fi is not None and not (isinstance(fi, float) and np.isnan(fi)):
            fi_ts = pd.Timestamp(fi)
            forecast_age_minutes = (decision_time - fi_ts).total_seconds() / 60
            forecast_issue_hour = fi_ts.hour

        else:
            fi_time = forecast_row.get("available_time")
            if fi_time is not None and not (isinstance(fi_time, float) and np.isnan(fi_time)):
                fi_ts = pd.Timestamp(fi_time)
                forecast_age_minutes = (decision_time - fi_ts).total_seconds() / 60
                forecast_issue_hour = fi_ts.hour

    # Wind features
    wind_10min = _filter_available(wind, decision_time)

    wind_ref_mean = _wind_group_stat(wind_10min, "ref", "mean")
    wind_ref_max = _wind_group_stat(wind_10min, "ref", "max")
    wind_vh_mean = _wind_group_stat(wind_10min, "victoria_harbour", "mean")
    wind_vh_max = _wind_group_stat(wind_10min, "victoria_harbour", "max")
    wind_offshore_max = _wind_group_stat(wind_10min, "offshore", "max")
    wind_highland_max = _wind_group_stat(wind_10min, "highland", "max")
    wind_kings_park_current = _wind_station_value(wind_10min, "京士柏")

    wind_all_mean_now = _wind_group_stat(wind_10min, None, "mean")
    wind_all_60m_ago = _wind_group_stat(
        _filter_available(wind, decision_time - pd.Timedelta(minutes=60)),
        None, "mean"
    )
    wind_all_change_60m = (
        wind_all_mean_now - wind_all_60m_ago
        if not (np.isnan(wind_all_mean_now) or np.isnan(wind_all_60m_ago))
        else np.nan
    )

    # Wind data age
    wind_ts = wind_10min["timestamp"].dropna() if "timestamp" in wind_10min.columns else pd.Series(dtype="datetime64[ns]")
    wind_data_age = (decision_time - wind_ts.max()).total_seconds() / 60 if len(wind_ts) > 0 else np.nan

    # Cyclical time features
    month = decision_time.month
    doy = decision_time.timetuple().tm_yday
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * doy / 365.25)
    day_cos = np.cos(2 * np.pi * doy / 365.25)
    is_morning = 1 if 6 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 18 else 0
    is_evening = 1 if 18 <= hour < 24 else 0

    features = {
        "temp_current": temp_current,
        "rh_current": rh,
        "dew_point_current": dp,
        "temp_current_today": temp_current,
        "max_so_far": max_so_far,
        "min_so_far": min_so_far,
        "range_so_far": range_so_far,
        "drop_from_max": drop_from_max,
        "rise_from_min": rise_from_min,
        "temp_slope_10m": temp_slope_10m,
        "temp_slope_30m": temp_slope_30m,
        "temp_slope_60m": temp_slope_60m,
        "temp_acceleration_60m": temp_acceleration_60m,
        "rh_change_30m": rh_change_30m,
        "rh_change_60m": rh_change_60m,
        "dew_point_spread_current": dew_point_spread_current,
        "dew_point_spread_change_60m": dew_point_spread_change_60m,
        "pressure_change_30m": pressure_change_30m,
        "pressure_change_60m": pressure_change_60m,
        "since_start_min": since_start_min,
        "forecast_max_temp": forecast_max_temp,
        "forecast_min_temp": forecast_min_temp,
        "forecast_range": forecast_range,
        "forecast_max_minus_max_so_far": forecast_max_minus_max_so_far,
        "forecast_age_minutes": forecast_age_minutes,
        "forecast_issue_hour": forecast_issue_hour,
        "wind_ref_max": wind_ref_max,
        "wind_ref_mean": wind_ref_mean,
        "wind_victoria_harbour_mean": wind_vh_mean,
        "wind_victoria_harbour_max": wind_vh_max,
        "wind_all_change_60m": wind_all_change_60m,
        "wind_offshore_max": wind_offshore_max,
        "wind_highland_max": wind_highland_max,
        "wind_kings_park_current": wind_kings_park_current,
        "obs_data_age_minutes": obs_age,
        "wind_data_age_minutes": wind_data_age,
        "minutes_since_midnight": mins_midnight,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_morning": is_morning,
        "is_afternoon": is_afternoon,
        "is_evening": is_evening,
    }

    feature_df = pd.DataFrame([features], index=[decision_time])
    feature_df.index.name = "decision_time"

    return feature_df


def _validate_canonical_inputs(*dfs: pd.DataFrame) -> None:
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        if "available_time" not in df.columns:
            raise ValueError(
                "Canonical source missing available_time column. "
                "Apply standardize_source before calling feature builder."
            )


def _filter_available(
    df: pd.DataFrame,
    decision_time: pd.Timestamp,
    time_col: str = "available_time",
) -> pd.DataFrame:
    """Filter rows where time_col <= decision_time."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if time_col not in df.columns:
        return df
    return df[pd.to_datetime(df[time_col]) <= decision_time].copy()


def _get_latest_value(df: pd.DataFrame, col: str) -> float:
    """Get the latest non-null value for a column based on available_time."""
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    valid = df.dropna(subset=[col])
    if len(valid) == 0:
        return np.nan
    if "available_time" in valid.columns:
        valid = valid.sort_values("available_time")
    return valid[col].iloc[-1]


def _compute_slope(
    df: pd.DataFrame, col: str, decision_time: pd.Timestamp, lag_steps: int
) -> float:
    """Compute slope over lag_steps * 10 minutes."""
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    valid = df.dropna(subset=[col]).sort_values("available_time")
    current = _get_latest_value(df, col)
    if len(valid) <= lag_steps or np.isnan(current):
        return np.nan
    past_val = valid[col].iloc[-(lag_steps + 1)] if len(valid) > lag_steps else valid[col].iloc[0]
    return (current - past_val) / (lag_steps * 10) if not np.isnan(past_val) else np.nan


def _compute_diff(
    df: pd.DataFrame, col: str, decision_time: pd.Timestamp, lag_steps: int, shift: int = 0
) -> float:
    """Compute difference over lag_steps * 10 minutes."""
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    valid = df.dropna(subset=[col]).sort_values("available_time")
    current = _get_latest_value(df, col)
    idx = -(lag_steps + 1 + shift) if shift > 0 else -(lag_steps + 1)
    if len(valid) <= max(lag_steps, shift + lag_steps) or np.isnan(current):
        return np.nan
    past_val = valid[col].iloc[idx] if abs(idx) <= len(valid) else valid[col].iloc[0]
    return current - past_val if not np.isnan(past_val) else np.nan


def _compute_derived_dew_point_spread_diff(
    df: pd.DataFrame, temp_col: str, dp_col: str, decision_time: pd.Timestamp
) -> float:
    """Compute dew_point_spread diff from temp and dp columns."""
    if df is None or len(df) == 0 or temp_col not in df.columns or dp_col not in df.columns:
        return np.nan
    valid = df.dropna(subset=[temp_col, dp_col]).sort_values("available_time")
    if len(valid) < 7:
        return np.nan
    t_now = _get_latest_value(df, temp_col)
    t_6ago = valid[temp_col].iloc[-7] if len(valid) >= 7 else valid[temp_col].iloc[0]
    dp_now = _get_latest_value(df, dp_col)
    dp_6ago = valid[dp_col].iloc[-7] if len(valid) >= 7 else valid[dp_col].iloc[0]
    spread_now = t_now - dp_now
    spread_6ago = t_6ago - dp_6ago
    if np.isnan(spread_now) or np.isnan(spread_6ago):
        return np.nan
    return spread_now - spread_6ago


def _get_latest_forecast(
    forecast: pd.DataFrame,
    decision_time: pd.Timestamp,
    target_date,
) -> pd.Series:
    """Get latest forecast for target date where issue_time <= decision_time."""
    if forecast is None or len(forecast) == 0:
        return None
    valid = forecast.copy()
    if "available_time" in valid.columns:
        valid = valid[pd.to_datetime(valid["available_time"]) <= decision_time]
    if "forecast_issue_datetime" in valid.columns:
        valid = valid[pd.to_datetime(valid["forecast_issue_datetime"], errors="coerce") <= decision_time]
    if len(valid) == 0:
        return None
    if "available_time" in valid.columns:
        valid = valid.sort_values("available_time", ascending=False)
    elif "forecast_issue_datetime" in valid.columns:
        valid = valid.sort_values("forecast_issue_datetime", ascending=False)
    if len(valid) == 0:
        return None
    return valid.iloc[0]


def _wind_group_stat(df: pd.DataFrame, group: str, stat: str) -> float:
    """Compute wind statistic for a station group."""
    if df is None or len(df) == 0 or "wind_speed" not in df.columns:
        return np.nan
    if group is not None:
        if "station_group" not in df.columns:
            return np.nan
        subset = df[df["station_group"] == group].dropna(subset=["wind_speed"])
    else:
        subset = df.dropna(subset=["wind_speed"])
    if len(subset) == 0:
        return np.nan
    if stat == "mean":
        return subset["wind_speed"].mean()
    elif stat == "max":
        return subset["wind_speed"].max()
    return np.nan


def _wind_station_value(df: pd.DataFrame, station_name: str) -> float:
    """Get wind speed for a specific station."""
    if df is None or len(df) == 0 or "wind_speed" not in df.columns:
        return np.nan
    if "station_id" in df.columns:
        subset = df[df["station_id"] == station_name].dropna(subset=["wind_speed"])
    elif "station" in df.columns:
        subset = df[df["station"] == station_name].dropna(subset=["wind_speed"])
    else:
        return np.nan
    if len(subset) == 0:
        return np.nan
    return subset["wind_speed"].iloc[-1]


def _safe_val(row, col):
    if row is None or col not in row.index:
        return np.nan
    val = row[col]
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    return val
