# features/model_2a_feature_builder.py
"""Model 2A shared feature builder.

Used for:
1. Historical feature-store build
2. Real-time inference
3. Replay parity check

All features respect: available_time <= decision_time
This canonical builder is the Model 2A v2 builder.  Its feature names must
exactly match models/intraday_minute_ml_model_2a_v2/feature_list.json.  Model
2A v1 must not call this builder because its trained wind semantics differ.
"""

import pandas as pd
import numpy as np

from features.input_status import (
    DEFAULT_STALE_AFTER_MINUTES,
    InputStatus,
    build_forecast_input_status,
    build_observation_buffer_status,
    jsonable,
)


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


def build_model_2a_input_status(
    decision_time: pd.Timestamp,
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    spec: dict,
    mode: str,
) -> dict:
    """Build source-status metadata alongside the v2 numeric feature frame.

    This function deliberately does not alter or return the LightGBM feature
    vector.  It is safe to call after ``build_model_2a_features`` and records
    the provenance of zero-valued observations separately from missing data.
    """
    features = build_model_2a_features(
        decision_time=decision_time,
        weather_canonical=weather_canonical,
        wind_canonical=wind_canonical,
        forecast_canonical=forecast_canonical,
        spec=spec,
        mode=mode,
    )
    row = features.iloc[0].to_dict() if not features.empty else {}
    stale_after = float(
        spec.get("data_quality_rules", {})
        .get("max_data_age_minutes", DEFAULT_STALE_AFTER_MINUTES)
    )

    weather = _filter_available(weather_canonical, decision_time, "available_time")
    wind = _filter_available(wind_canonical, decision_time, "available_time")

    def _source_timestamp(frame: pd.DataFrame, value_column: str | None) -> pd.Timestamp | None:
        if frame is None or frame.empty:
            return None
        source_column = "timestamp" if "timestamp" in frame.columns else "available_time"
        if source_column not in frame.columns:
            return None
        subset = frame
        if value_column and value_column in subset.columns:
            subset = subset.loc[~subset[value_column].isna()]
        if subset.empty:
            return None
        timestamps = pd.to_datetime(subset[source_column], errors="coerce").dropna()
        return timestamps.max() if not timestamps.empty else None

    def _canonical_status(
        name: str,
        value: object,
        frame: pd.DataFrame,
        value_column: str | None,
        source_name: str,
        *,
        selector=None,
        observation_method: str = "direct_observation",
    ) -> dict:
        subset = frame
        if selector is not None and not frame.empty:
            subset = frame.loc[selector]
        source = _source_timestamp(subset, value_column)
        if subset.empty or (value_column and value_column in subset.columns and subset[value_column].dropna().empty):
            return InputStatus.fallback(
                None,
                fallback_method="unavailable",
                decision_timestamp=decision_time,
                source_name=source_name,
                observation_method="insufficient_history",
            ).to_dict()
        return InputStatus.from_value(
            value,
            source_timestamp=source,
            decision_timestamp=decision_time,
            source_name=source_name,
            stale_after_minutes=stale_after,
            observation_method=observation_method,
        ).to_dict()

    weather_status = {}
    for name, column in (
        ("temp_current", "temp_current_clean"),
        ("rh_current", "rh_current"),
        ("pressure_current", "pressure_current"),
        ("dew_point_current", "dew_point_current"),
    ):
        weather_status[name] = _canonical_status(
            name,
            row.get(name),
            weather,
            column,
            "hko_weather_obs",
        )

    for name in ("max_so_far", "min_so_far"):
        value = row.get(name)
        selector = None
        if "temp_current_clean" in weather.columns and value is not None and not pd.isna(value):
            selector = weather["temp_current_clean"].eq(float(value))
        weather_status[name] = _canonical_status(
            name,
            value,
            weather,
            "temp_current_clean",
            "hko_weather_obs",
            selector=selector,
        )

    weather_status["obs_data_age_minutes"] = InputStatus.from_value(
        row.get("obs_data_age_minutes"),
        source_timestamp=_source_timestamp(weather, "temp_current_clean"),
        decision_timestamp=decision_time,
        source_name="hko_weather_obs",
        stale_after_minutes=stale_after,
        observation_method="source_age",
    ).to_dict()

    wind_field_groups = {
        "reference": ("ref", "wind_ref_mean", "wind_ref_max"),
        "victoria_harbour": (
            "victoria_harbour",
            "wind_victoria_harbour_mean",
            "wind_victoria_harbour_max",
        ),
        "offshore_highland": (
            "offshore_highland",
            "wind_offshore_highland_mean",
            "wind_offshore_highland_max",
        ),
    }
    wind_status: dict[str, object] = {}
    wind_groups: dict[str, object] = {}
    for group_name, (canonical_group, mean_name, max_name) in wind_field_groups.items():
        selector = wind.get("station_group", pd.Series(index=wind.index, dtype=object)).eq(canonical_group)
        subset = wind.loc[selector] if not wind.empty else wind
        mean_status = _canonical_status(
            mean_name,
            row.get(mean_name),
            subset,
            "wind_speed",
            "i-lens_wind_obs",
        )
        max_status = _canonical_status(
            max_name,
            row.get(max_name),
            subset,
            "wind_speed",
            "i-lens_wind_obs",
        )
        wind_status[mean_name] = mean_status
        wind_status[max_name] = max_status
        wind_groups[group_name] = {"mean": mean_status, "max": max_status}

    kings_selector = wind.get("station_id", pd.Series(index=wind.index, dtype=object)).eq("京士柏")
    kings_status = _canonical_status(
        "wind_kings_park_current",
        row.get("wind_kings_park_current"),
        wind.loc[kings_selector] if not wind.empty else wind,
        "wind_speed",
        "i-lens_wind_obs",
    )
    wind_status["wind_kings_park_current"] = kings_status
    wind_groups["kings_park"] = {"current": kings_status}

    all_wind_source = _source_timestamp(wind, "wind_speed")
    change_status = InputStatus.from_value(
        row.get("wind_all_change_60m"),
        source_timestamp=all_wind_source,
        decision_timestamp=decision_time,
        source_name="i-lens_wind_obs",
        stale_after_minutes=stale_after,
        observation_method="derived_change",
    ).to_dict()
    wind_status["wind_all_change_60m"] = change_status
    wind_groups["aggregate_change"] = {"change_60m": change_status}
    wind_status["groups"] = wind_groups

    forecast_status = build_forecast_input_status(
        forecast_canonical,
        decision_timestamp=decision_time,
        target_date=decision_time.normalize(),
        stale_after_minutes=stale_after,
    )
    # Keep both the canonical forecast names and feature names convenient for
    # snapshot consumers without putting these dicts into the model vector.
    forecast_status["forecast_max_temp"] = forecast_status.get("forecast_max")
    forecast_status["forecast_min_temp"] = forecast_status.get("forecast_min")

    weather_buffer = weather.copy()
    if not weather_buffer.empty:
        if "temp" not in weather_buffer.columns:
            weather_buffer["temp"] = weather_buffer.get("temp_current_clean", np.nan)
        if "rh" not in weather_buffer.columns:
            weather_buffer["rh"] = weather_buffer.get("rh_current", np.nan)
    buffer_values = {
        "temp_now": row.get("temp_current"),
        "rh_now": row.get("rh_current"),
        "max_so_far": row.get("max_so_far"),
        "min_so_far": row.get("min_so_far"),
        "temp_30m_ago": _lookback_value(weather, "temp_current_clean", decision_time, 30),
        "temp_60m_ago": _lookback_value(weather, "temp_current_clean", decision_time, 60),
        "temp_120m_ago": _lookback_value(weather, "temp_current_clean", decision_time, 120),
        "temp_change_30m": row.get("temp_change_30m"),
        "temp_change_60m": row.get("temp_change_60m"),
        "temp_volatility_60m": row.get("temp_volatility_60m"),
        "temp_acceleration_60m": row.get("temp_acceleration_60m"),
        "rh_change_60m": row.get("rh_change_60m"),
        "dew_point_change_60m": row.get("dew_point_change_60m"),
        "dew_point_spread_change_60m": row.get("dew_point_spread_change_60m"),
        "time_since_max": row.get("time_since_max"),
    }
    observation_status = build_observation_buffer_status(
        weather_buffer,
        decision_timestamp=decision_time,
        values=buffer_values,
        stale_after_minutes=stale_after,
    )

    return jsonable(
        {
            "status_contract_version": "phase2a.v1",
            "numeric_policy": "legacy_compatible",
            "status_policy": "truthful",
            "decision_timestamp": decision_time,
            "mode": mode,
            "weather_input_status": weather_status,
            "wind_input_status": wind_status,
            "forecast_input_status": forecast_status,
            "observation_buffer_status": observation_status,
        }
    )


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
