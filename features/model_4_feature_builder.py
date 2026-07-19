# features/model_4_feature_builder.py
"""Model 4 live inference feature builder.

Extracts forecast rain probability + humidity features from raw HKO daily
forecast data, using the segment-based parser.

Usage:
    from model_4_feature_builder import build_forecast_features_m4

    # Raw forecast data (from API or daily parquet)
    fc_df = pd.read_parquet("data/hk_daily_forecast/daily_forecast_clean.parquet")
    # Get latest forecast for today
    latest = fc_df.sort_values("forecast_datetime").iloc[-1:]

    features = build_forecast_features_m4(latest)
    # Returns dict with keys:
    #   forecast_rain_prob_morning, forecast_rain_prob_afternoon,
    #   forecast_rain_prob_overall, forecast_rain_prob_missing,
    #   forecast_rain_prob_label, forecast_min_rh, forecast_max_rh, forecast_rh_range
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from forecast_rain_prob_parser import (
    map_weather_desc_to_ordinal,
    resolve_and_parse,
)

# Default feature values when forecast is unavailable
_DEFAULT_FEATURES: Dict[str, float] = {
    "forecast_rain_prob_morning": 0.0,
    "forecast_rain_prob_afternoon": 0.0,
    "forecast_rain_prob_overall": 0.0,
    "forecast_rain_prob_missing": 1.0,
    "forecast_rain_prob_label": 0.0,
    "forecast_min_rh": np.nan,
    "forecast_max_rh": np.nan,
    "forecast_rh_range": np.nan,
}


def build_forecast_features_m4(
    forecast_df: pd.DataFrame,
) -> Dict[str, float]:
    """Build Model 4 forecast features from raw HKO forecast data.

    Args:
        forecast_df: DataFrame with at least one row containing
            'forecast_rain_prob', 'forecast_weather_desc',
            'forecast_min_rh', 'forecast_max_rh'.
            If multiple rows, uses the LAST row (most recent forecast).

    Returns:
        dict of 8 feature values. Returns _DEFAULT_FEATURES if forecast_df
        is empty or all required columns are missing.
    """
    if forecast_df is None or len(forecast_df) == 0:
        return _DEFAULT_FEATURES.copy()

    # Use the last row (most recent forecast issue)
    row = forecast_df.iloc[-1]

    # Parse rain probability description
    rain_prob_col = forecast_df["forecast_rain_prob"].iloc[-1:]
    weather_desc_col = forecast_df["forecast_weather_desc"].iloc[-1:]
    parsed = resolve_and_parse(rain_prob_col, weather_desc_col)
    parsed_row = parsed.iloc[0] if len(parsed) > 0 else None

    if parsed_row is not None:
        morning = float(parsed_row.get("forecast_rain_prob_morning", 0.0))
        afternoon = float(parsed_row.get("forecast_rain_prob_afternoon", 0.0))
        overall = float(parsed_row.get("forecast_rain_prob_overall", 0.0))
        missing = float(parsed_row.get("forecast_rain_prob_missing", 1.0))
    else:
        morning = 0.0
        afternoon = 0.0
        overall = 0.0
        missing = 1.0

    # Extract probability label (低→1 … 高→5)
    label = map_weather_desc_to_ordinal(
        row.get("forecast_weather_desc", np.nan)
    )
    if label is None or label == 0:
        label = map_weather_desc_to_ordinal(
            row.get("forecast_rain_prob", np.nan)
        )
    label = float(label) if label is not None else 0.0

    # Extract RH values
    min_rh = _safe_float(row.get("forecast_min_rh", np.nan))
    max_rh = _safe_float(row.get("forecast_max_rh", np.nan))
    rh_range = max_rh - min_rh if not (np.isnan(min_rh) or np.isnan(max_rh)) else np.nan

    return {
        "forecast_rain_prob_morning": morning,
        "forecast_rain_prob_afternoon": afternoon,
        "forecast_rain_prob_overall": overall,
        "forecast_rain_prob_missing": missing,
        "forecast_rain_prob_label": label,
        "forecast_min_rh": min_rh,
        "forecast_max_rh": max_rh,
        "forecast_rh_range": rh_range,
    }


def _safe_float(val: object) -> float:
    """Convert value to float, returning NaN on failure."""
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return np.nan
