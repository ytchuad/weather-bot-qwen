# features/model_2a_source_adapters.py
"""Model 2A source adapters.

Converts raw historical/live data into canonical schemas for:
- Weather observations (temp, RH, pressure, dew point)
- Wind observations (wind speed/direction by station)
- Forecast (issue time, max/min temp)
"""

import pandas as pd
import numpy as np

STATION_GROUP_MAP = {
    "參考": "ref",
    "離岸及高地": "offshore_highland",
    "維多利亞港": "victoria_harbour",
}

VICTORIA_HARBOUR_STATIONS = ["京士柏", "啟德", "九龍天星碼頭"]

SPECIAL_STATIONS = {
    "wind_kings_park_current": "京士柏",
    "wind_kai_tak_current": "啟德",
}

HKT = "Asia/Hong_Kong"


def _quality_flag_mask(values, index: pd.Index) -> pd.Series:
    """Return a null-safe boolean quality-flag mask aligned to ``index``."""
    if values is None:
        return pd.Series(False, index=index, dtype=bool)

    if isinstance(values, pd.Series):
        values = values.reindex(index)
    else:
        values = pd.Series(values, index=index)

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    # Quality flags are booleans in the canonical contract.  Numeric values
    # are accepted for already-encoded inputs; non-numeric anomalies are
    # treated as false rather than raising during source standardization.
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0).ne(0)


def _quality_flag_bit(values, index: pd.Index, bit: int) -> pd.Series:
    """Encode one quality flag without relying on pandas Series bit shifts."""
    return _quality_flag_mask(values, index).astype("int64") * (1 << bit)


def standardize_weather_obs(
    df: pd.DataFrame,
    source_type: str,
    valid_temp_min: float = 0,
    valid_temp_max: float = 40,
) -> pd.DataFrame:
    """Convert raw weather observations to canonical schema.

    Required raw columns (by convention):
        timestamp, temp_current [, rh_current, dew_point_current, pressure_current]

    Returns canonical DataFrame with:
        source_system, source_mode, available_time, timestamp, station_id,
        temp_current, rh_current, dew_point_current, pressure_current,
        temp_current_clean, temp_anomaly_flag, temp_spike_flag, wind_missing_flag,
        data_quality_flags
    """
    result = df.copy()
    result["source_system"] = "hko_obs"
    result["source_mode"] = source_type
    result["station_id"] = result.get("station_id", "hko_hq")

    if "available_time" not in result.columns:
        if "timestamp" in result.columns:
            result["available_time"] = result["timestamp"]
        else:
            result["timestamp"] = pd.NaT
            result["available_time"] = pd.NaT

    for col in ["rh_current", "dew_point_current", "pressure_current"]:
        if col not in result.columns:
            result[col] = np.nan

    temp_col = "temp_current"
    if temp_col in result.columns:
        result["temp_current_raw"] = pd.to_numeric(
            result[temp_col], errors="coerce"
        )
        result["temp_current_clean"] = result["temp_current_raw"].where(
            result["temp_current_raw"].between(valid_temp_min, valid_temp_max)
        )
        anomaly_mask = _quality_flag_mask(
            result.get("temp_anomaly_flag"), result.index
        )
        result["temp_current_clean"] = result["temp_current_clean"].where(
            ~anomaly_mask, np.nan
        )
    else:
        result["temp_current_raw"] = np.nan
        result["temp_current_clean"] = np.nan

    if "temp_anomaly_flag" not in result.columns:
        result["temp_anomaly_flag"] = False

    if "temp_spike_flag" not in result.columns:
        result["temp_spike_flag"] = _detect_temp_spikes(result)

    if "wind_missing_flag" not in result.columns:
        result["wind_missing_flag"] = True

    result["data_quality_flags"] = (
        _quality_flag_bit(result.get("temp_anomaly_flag"), result.index, 0)
        + _quality_flag_bit(result.get("temp_spike_flag"), result.index, 1)
        + _quality_flag_bit(result.get("wind_missing_flag"), result.index, 2)
    ).astype("int64")

    canonical_cols = [
        "source_system", "source_mode", "available_time", "timestamp",
        "station_id", "temp_current_raw", "temp_current_clean",
        "rh_current", "dew_point_current", "pressure_current",
        "temp_anomaly_flag", "temp_spike_flag", "wind_missing_flag",
        "data_quality_flags",
    ]
    for col in canonical_cols:
        if col not in result.columns:
            result[col] = np.nan

    return result[canonical_cols]


def _detect_temp_spikes(df: pd.DataFrame) -> pd.Series:
    """Detect temperature spikes: abs change >= 5 in 1 minute."""
    if "temp_current_clean" not in df.columns or "timestamp" not in df.columns:
        return pd.Series(False, index=df.index)

    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["date"] = df_sorted["timestamp"].dt.date
    temp_change_1m_abs = (
        df_sorted.groupby("date")["temp_current_clean"].diff().abs()
    )
    spike = temp_change_1m_abs >= 5
    return spike.reindex(df.index, fill_value=False)


def standardize_wind_obs(
    df: pd.DataFrame,
    source_type: str,
) -> pd.DataFrame:
    """Convert raw wind observations to canonical schema.

    Required raw columns:
        timestamp, station, station_type, wind_speed
        [, wind_direction]

    Returns canonical DataFrame with:
        source_system, source_mode, available_time, timestamp, station_id,
        wind_speed, wind_direction, wind_missing_flag, wind_anomaly_flag,
        station_group, data_quality_flags
    """
    result = df.copy()
    result["source_system"] = "wind_obs"
    result["source_mode"] = source_type

    if "available_time" not in result.columns:
        if "timestamp" in result.columns:
            result["available_time"] = result["timestamp"]
        else:
            result["timestamp"] = pd.NaT
            result["available_time"] = pd.NaT

    if "station_id" not in result.columns:
        result["station_id"] = result.get("station", "unknown")

    if "station_type" in result.columns:
        result["station_group"] = result["station_type"].map(STATION_GROUP_MAP)
        vic_harbour_mask = result["station_id"].isin(VICTORIA_HARBOUR_STATIONS)
        result.loc[vic_harbour_mask, "station_group"] = "victoria_harbour"
        default_mask = result["station_group"].isna() & (
            result["station_type"].astype("string").str.lower()
            == "victoria harbour"
        )
        result.loc[default_mask, "station_group"] = "victoria_harbour"
    else:
        result["station_group"] = np.nan

    if "wind_speed" not in result.columns:
        result["wind_speed"] = np.nan

    if "wind_direction" not in result.columns:
        result["wind_direction"] = np.nan

    if "wind_missing_flag" not in result.columns:
        result["wind_missing_flag"] = result["wind_speed"].isna()

    if "wind_anomaly_flag" not in result.columns:
        result["wind_anomaly_flag"] = False

    result["data_quality_flags"] = (
        _quality_flag_bit(result.get("wind_missing_flag"), result.index, 0)
        + _quality_flag_bit(result.get("wind_anomaly_flag"), result.index, 1)
    ).astype("int64")

    canonical_cols = [
        "source_system", "source_mode", "available_time", "timestamp",
        "station_id", "wind_speed", "wind_direction",
        "wind_missing_flag", "wind_anomaly_flag", "station_group",
        "data_quality_flags",
    ]
    for col in canonical_cols:
        if col not in result.columns:
            result[col] = np.nan

    return result[canonical_cols]


def standardize_forecast(
    df: pd.DataFrame,
    source_type: str,
) -> pd.DataFrame:
    """Convert raw forecast data to canonical schema.

    Required raw columns:
        forecast_issue_datetime, target_date,
        forecast_max_temp, forecast_min_temp [, forecast_missing_flag]

    Rules:
        forecast_max_temp >= forecast_min_temp
        available_time == forecast_issue_datetime
        forecast_lead_days = target_date - forecast_issue_datetime

    Returns canonical DataFrame with:
        source_system, source_mode, available_time, timestamp,
        forecast_issue_datetime, forecast_max_temp, forecast_min_temp,
        forecast_missing_flag, data_quality_flags
    """
    result = df.copy()
    result["source_system"] = "hko_forecast"
    result["source_mode"] = source_type

    if "available_time" not in result.columns:
        if "forecast_issue_datetime" in result.columns:
            result["available_time"] = result["forecast_issue_datetime"]
        elif "timestamp" in result.columns:
            result["available_time"] = result["timestamp"]
        else:
            result["available_time"] = pd.NaT

    if "timestamp" not in result.columns:
        result["timestamp"] = result["available_time"]

    for col in ["forecast_max_temp", "forecast_min_temp"]:
        if col not in result.columns:
            result[col] = np.nan

    forecast_max = result["forecast_max_temp"].astype(float)
    forecast_min = result["forecast_min_temp"].astype(float)
    invalid = forecast_max < forecast_min
    if invalid.any():
        result.loc[invalid, "forecast_max_temp"] = np.nan
        result.loc[invalid, "forecast_min_temp"] = np.nan

    if "forecast_issue_datetime" not in result.columns:
        result["forecast_issue_datetime"] = result["available_time"]

    if "forecast_source" not in result.columns:
        result["forecast_source"] = result["source_system"]

    if "forecast_missing_flag" not in result.columns:
        result["forecast_missing_flag"] = (
            result["forecast_max_temp"].isna() | result["forecast_min_temp"].isna()
        )

    result["forecast_lead_days"] = np.nan
    if "target_date" in result.columns and "forecast_issue_datetime" in result.columns:
        td = pd.to_datetime(result["target_date"]).dt.normalize()
        fi = pd.to_datetime(result["forecast_issue_datetime"]).dt.normalize()
        result["forecast_lead_days"] = (td - fi).dt.days

    result["data_quality_flags"] = result["forecast_missing_flag"].astype(int)

    canonical_cols = [
        "source_system", "source_mode", "available_time", "timestamp",
        "forecast_issue_datetime", "forecast_max_temp",
        "forecast_min_temp", "target_date", "forecast_source",
        "forecast_missing_flag",
        "forecast_lead_days", "data_quality_flags",
    ]
    for col in canonical_cols:
        if col not in result.columns:
            result[col] = np.nan

    return result[canonical_cols]
