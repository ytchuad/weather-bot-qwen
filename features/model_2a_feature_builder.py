# features/model_2a_feature_builder.py
"""Model 2A shared feature builder.

Used for:
1. Historical feature-store build
2. Real-time inference
3. Replay parity check

All features respect: available_time <= decision_time
Feature names must EXACTLY match models/intraday_minute_ml_model_2a/feature_list.json
"""

import pandas as pd
import numpy as np


def build_model_2a_features(
    decision_time: pd.Timestamp,
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    spec: dict,
    mode: str,
) -> pd.DataFrame:
    """Build Model 2A feature vector for a single decision_time.

    Output features exactly match feature_list.json (45 features).
    """
    _validate_canonical_inputs(weather_canonical, wind_canonical, forecast_canonical)

    weather = _filter_available(weather_canonical, decision_time, "available_time")
    wind = _filter_available(wind_canonical, decision_time, "available_time")
    forecast = _filter_available(forecast_canonical, decision_time, "available_time")

    target_date = decision_time.normalize().date()
    hour = decision_time.hour
    minute = decision_time.minute
    mins_midnight = hour * 60 + minute

    # --- Current observations ---
    temp_raw = _get_latest_value(weather, "temp_current_raw")
    temp_clean = _get_latest_value(weather, "temp_current_clean")
    rh = _get_latest_value(weather, "rh_current")
    dp = _get_latest_value(weather, "dew_point_current")
    pressure = _get_latest_value(weather, "pressure_current")

    temp_current = temp_raw if not np.isnan(temp_raw) else temp_clean

    dew_point_spread = temp_current - dp if not (np.isnan(temp_current) or np.isnan(dp)) else np.nan

    # --- Date-filtered weather for trend / anchor computations ---
    if "timestamp" in weather.columns:
        weather_td = weather[pd.to_datetime(weather["timestamp"]).dt.date == target_date]
    else:
        weather_td = weather
    weather_td = _filter_available(weather_td, decision_time, "available_time")

    if len(weather_td) == 0:
        weather_td = weather
        weather_td = _filter_available(weather_td, decision_time, "available_time")

    # --- Anchor features (cumulative per target_date) ---
    temp_series = weather_td["temp_current_clean"].dropna().sort_index(
        key=lambda x: pd.to_datetime(weather_td.loc[x, "available_time"])
        if "available_time" in weather_td.columns else x
    ) if len(weather_td) > 0 else pd.Series(dtype=float)

    if len(temp_series) > 0:
        max_so_far = float(temp_series.max())
        min_so_far = float(temp_series.min())
    else:
        max_so_far = temp_current if not np.isnan(temp_current) else np.nan
        min_so_far = temp_current if not np.isnan(temp_current) else np.nan

    range_so_far = max_so_far - min_so_far if not (np.isnan(max_so_far) or np.isnan(min_so_far)) else np.nan
    drop_from_max = max_so_far - temp_current if not (np.isnan(max_so_far) or np.isnan(temp_current)) else np.nan

    # time_since_max: minutes since temp was at max_so_far
    time_since_max = _compute_time_since_extreme(weather_td, decision_time, max_so_far, "max")

    # --- Trend features (time-window based lookback) ---
    temp_col = "temp_current_clean"
    temp_30m_ago = _lookback_value(weather_td, temp_col, decision_time, 30)
    temp_60m_ago = _lookback_value(weather_td, temp_col, decision_time, 60)

    temp_change_30m = (temp_current - temp_30m_ago) if not (np.isnan(temp_current) or np.isnan(temp_30m_ago)) else np.nan
    temp_change_60m = (temp_current - temp_60m_ago) if not (np.isnan(temp_current) or np.isnan(temp_60m_ago)) else np.nan
    temp_slope_30m = temp_change_30m / 30.0 if not np.isnan(temp_change_30m) else np.nan
    temp_slope_60m = temp_change_60m / 60.0 if not np.isnan(temp_change_60m) else np.nan

    # acceleration: change in slope_30m over last 30 min
    temp_slope_30m_30m_ago = _compute_slope_from_window(weather_td, temp_col, decision_time, 60, 30)
    temp_acceleration_60m = (temp_slope_30m - temp_slope_30m_30m_ago) if not (np.isnan(temp_slope_30m) or np.isnan(temp_slope_30m_30m_ago)) else np.nan

    # volatility: std of temp over last 60 min
    temp_volatility_60m = _compute_rolling_std(weather_td, temp_col, decision_time, 60)

    # RH trends
    rh_60m_ago = _lookback_value(weather_td, "rh_current", decision_time, 60)
    rh_change_60m = (rh - rh_60m_ago) if not (np.isnan(rh) or np.isnan(rh_60m_ago)) else np.nan

    # Dew point trends
    dp_60m_ago = _lookback_value(weather_td, "dew_point_current", decision_time, 60)
    dew_point_change_60m = (dp - dp_60m_ago) if not (np.isnan(dp) or np.isnan(dp_60m_ago)) else np.nan

    dew_point_spread_60m_ago = _lookback_value(weather_td, "dew_point_spread", decision_time, 60) if "dew_point_spread" in weather_td.columns else np.nan
    if np.isnan(dew_point_spread_60m_ago) and not np.isnan(temp_60m_ago) and not np.isnan(dp_60m_ago):
        dew_point_spread_60m_ago = temp_60m_ago - dp_60m_ago
    dew_point_spread_change_60m = (dew_point_spread - dew_point_spread_60m_ago) if not (np.isnan(dew_point_spread) or np.isnan(dew_point_spread_60m_ago)) else np.nan

    # Pressure trends
    pressure_60m_ago = _lookback_value(weather_td, "pressure_current", decision_time, 60)
    pressure_change_60m = (pressure - pressure_60m_ago) if not (np.isnan(pressure) or np.isnan(pressure_60m_ago)) else np.nan

    pressure_180m_ago = _lookback_value(weather_td, "pressure_current", decision_time, 180)
    pressure_change_180m = (pressure - pressure_180m_ago) if not (np.isnan(pressure) or np.isnan(pressure_180m_ago)) else np.nan

    # --- Data freshness ---
    if "timestamp" in weather_td.columns:
        obs_ts = weather_td["timestamp"].dropna()
        obs_age = (decision_time - obs_ts.max()).total_seconds() / 60 if len(obs_ts) > 0 else np.nan
    else:
        obs_age = np.nan

    # --- Forecast features ---
    forecast_row = _get_latest_forecast(forecast, decision_time, target_date)
    forecast_max_temp = _safe_val(forecast_row, "forecast_max_temp")
    forecast_min_temp = _safe_val(forecast_row, "forecast_min_temp")
    forecast_range = (forecast_max_temp - forecast_min_temp
                      if not (np.isnan(forecast_max_temp) or np.isnan(forecast_min_temp)) else np.nan)
    forecast_gap_from_max_so_far = (forecast_max_temp - max_so_far
                                    if not (np.isnan(forecast_max_temp) or np.isnan(max_so_far)) else np.nan)

    forecast_age_minutes = np.nan
    forecast_lead_days = np.nan
    if forecast_row is not None:
        fi = _safe_val(forecast_row, "forecast_issue_datetime")
        if fi is not None and not (isinstance(fi, float) and np.isnan(fi)):
            fi_ts = pd.Timestamp(fi)
            forecast_age_minutes = (decision_time - fi_ts).total_seconds() / 60
            forecast_lead_days = max(0, (target_date - fi_ts.normalize().date()).days)
        else:
            fi_time = forecast_row.get("available_time")
            if fi_time is not None and not (isinstance(fi_time, float) and np.isnan(fi_time)):
                fi_ts = pd.Timestamp(fi_time)
                forecast_age_minutes = (decision_time - fi_ts).total_seconds() / 60
                forecast_lead_days = max(0, (target_date - fi_ts.normalize().date()).days)

    # --- Wind features ---
    wind_10min = _filter_available(wind, decision_time)

    wind_ref_mean = _wind_group_stat(wind_10min, "ref", "mean")
    wind_ref_max = _wind_group_stat(wind_10min, "ref", "max")
    wind_vh_mean = _wind_group_stat(wind_10min, "victoria_harbour", "mean")
    wind_vh_max = _wind_group_stat(wind_10min, "victoria_harbour", "max")
    wind_offshore_highland_mean = _wind_group_stat(wind_10min, "offshore_highland", "mean")
    wind_offshore_highland_max = _wind_group_stat(wind_10min, "offshore_highland", "max")
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

    if "timestamp" in wind_10min.columns:
        wind_ts = wind_10min["timestamp"].dropna()
        wind_data_age = (decision_time - wind_ts.max()).total_seconds() / 60 if len(wind_ts) > 0 else np.nan
    else:
        wind_data_age = np.nan

    # --- Cyclical time features ---
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
        "pressure_current": pressure,
        "dew_point_current": dp,
        "dew_point_spread": dew_point_spread,
        "max_so_far": max_so_far,
        "min_so_far": min_so_far,
        "range_so_far": range_so_far,
        "drop_from_max": drop_from_max,
        "time_since_max": time_since_max,
        "temp_change_30m": temp_change_30m,
        "temp_change_60m": temp_change_60m,
        "temp_slope_30m": temp_slope_30m,
        "temp_slope_60m": temp_slope_60m,
        "temp_acceleration_60m": temp_acceleration_60m,
        "temp_volatility_60m": temp_volatility_60m,
        "rh_change_60m": rh_change_60m,
        "dew_point_change_60m": dew_point_change_60m,
        "dew_point_spread_change_60m": dew_point_spread_change_60m,
        "pressure_change_60m": pressure_change_60m,
        "pressure_change_180m": pressure_change_180m,
        "forecast_min_temp": forecast_min_temp,
        "forecast_max_temp": forecast_max_temp,
        "forecast_range": forecast_range,
        "forecast_gap_from_max_so_far": forecast_gap_from_max_so_far,
        "forecast_age_minutes": forecast_age_minutes,
        "forecast_lead_days": forecast_lead_days,
        "wind_ref_mean": wind_ref_mean,
        "wind_ref_max": wind_ref_max,
        "wind_victoria_harbour_mean": wind_vh_mean,
        "wind_victoria_harbour_max": wind_vh_max,
        "wind_offshore_highland_mean": wind_offshore_highland_mean,
        "wind_offshore_highland_max": wind_offshore_highland_max,
        "wind_all_change_60m": wind_all_change_60m,
        "wind_kings_park_current": wind_kings_park_current,
        "minutes_since_midnight": mins_midnight,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_morning": is_morning,
        "is_afternoon": is_afternoon,
        "is_evening": is_evening,
        "obs_data_age_minutes": obs_age,
        "wind_data_age_minutes": wind_data_age,
    }

    feature_df = pd.DataFrame([features], index=[decision_time])
    feature_df.index.name = "decision_time"
    return feature_df


# --- Helper functions ---

def _validate_canonical_inputs(*dfs: pd.DataFrame) -> None:
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        if "available_time" not in df.columns:
            raise ValueError(
                "Canonical source missing available_time column. "
                "Apply standardize_source before calling feature builder."
            )


def _filter_available(df, decision_time, time_col="available_time"):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if time_col not in df.columns:
        return df
    return df[pd.to_datetime(df[time_col]) <= decision_time].copy()


def _get_latest_value(df, col):
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    valid = df.dropna(subset=[col])
    if len(valid) == 0:
        return np.nan
    if "available_time" in valid.columns:
        valid = valid.sort_values("available_time")
    return valid[col].iloc[-1]


def _lookback_value(df, col, decision_time, lookback_minutes):
    """Find value closest to (decision_time - lookback_minutes)."""
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    if "available_time" not in df.columns:
        if "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            return np.nan
    else:
        time_col = "available_time"

    target_time = decision_time - pd.Timedelta(minutes=lookback_minutes)
    valid = df.dropna(subset=[col]).copy()
    if len(valid) == 0:
        return np.nan

    valid["_time_diff"] = (pd.to_datetime(valid[time_col]) - target_time).abs()
    best_idx = valid["_time_diff"].idxmin()
    best = valid.loc[best_idx, col]
    return float(best) if not (isinstance(best, float) and np.isnan(best)) else np.nan


def _compute_slope_from_window(df, col, decision_time, window_start, window_end):
    """Compute slope using values at window_start and window_end minutes ago."""
    v_later = _lookback_value(df, col, decision_time, window_end)
    v_earlier = _lookback_value(df, col, decision_time, window_start)
    if np.isnan(v_later) or np.isnan(v_earlier):
        return np.nan
    return (v_later - v_earlier) / (window_start - window_end) if window_start > window_end else np.nan


def _compute_time_since_extreme(df, decision_time, extreme_val, mode):
    """Compute minutes since temp was at the extreme value."""
    if df is None or len(df) == 0:
        return np.nan
    temp_col = "temp_current_clean"
    if temp_col not in df.columns:
        return np.nan
    if "available_time" not in df.columns:
        time_col = "timestamp" if "timestamp" in df.columns else None
    else:
        time_col = "available_time"
    if time_col is None:
        return np.nan

    valid = df.dropna(subset=[temp_col]).copy()
    if len(valid) == 0:
        return np.nan

    if mode == "max":
        at_extreme = valid[temp_col] >= extreme_val - 1e-9
    else:
        at_extreme = valid[temp_col] <= extreme_val + 1e-9

    extreme_times = valid.loc[at_extreme, time_col]
    if len(extreme_times) == 0:
        return np.nan

    last_extreme = pd.to_datetime(extreme_times.max())
    minutes = (decision_time - last_extreme).total_seconds() / 60
    return max(0.0, minutes)


def _compute_rolling_std(df, col, decision_time, window_minutes):
    """Compute std of col values within the last window_minutes."""
    if df is None or len(df) == 0 or col not in df.columns:
        return np.nan
    if "available_time" not in df.columns:
        if "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            return np.nan
    else:
        time_col = "available_time"

    cutoff = decision_time - pd.Timedelta(minutes=window_minutes)
    window = df[pd.to_datetime(df[time_col]) >= cutoff].dropna(subset=[col])
    if len(window) < 2:
        return np.nan
    return float(window[col].std(ddof=1))


def _get_latest_forecast(forecast, decision_time, target_date):
    """Get latest forecast matching target_date where issue_time <= decision_time."""
    if forecast is None or len(forecast) == 0:
        return None

    valid = forecast.copy()

    if "target_date" in valid.columns:
        td_dt = pd.to_datetime(target_date).date()
        forecast_td = pd.to_datetime(valid["target_date"])
        valid = valid[forecast_td.dt.date == td_dt]
    if len(valid) == 0:
        return None
    if "available_time" in valid.columns:
        valid = valid[pd.to_datetime(valid["available_time"]) <= decision_time]
    if "forecast_issue_datetime" in valid.columns:
        valid = valid[pd.to_datetime(valid["forecast_issue_datetime"], errors="coerce") <= decision_time]
    if len(valid) == 0:
        return None

    sort_col = "forecast_issue_datetime" if "forecast_issue_datetime" in valid.columns else "available_time"
    valid = valid.sort_values(sort_col, ascending=False)
    return valid.iloc[0]


def _wind_group_stat(df, group, stat):
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
        return float(subset["wind_speed"].mean())
    elif stat == "max":
        return float(subset["wind_speed"].max())
    return np.nan


def _wind_station_value(df, station_name):
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
    return float(subset["wind_speed"].iloc[-1])


def _safe_val(row, col):
    if row is None or col not in row.index:
        return np.nan
    val = row[col]
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    return val
