# features/model_3b_feature_builder.py
"""Model 3B feature builder = Model 3A features + 9 rainfall features.

Used for live inference. Wraps ``build_model_3a_features`` from the 3A builder
and appends 9 rainfall features (same set as Model 2B).

Feature names must EXACTLY match models/intraday_minute_ai_model_3b/feature_list.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.model_3a_feature_builder import build_model_3a_features


def build_model_3b_features(
    decision_time: pd.Timestamp,
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    spec: dict,
    mode: str,
    temp_buffer: list | None = None,
    rh_buffer: list | None = None,
    # Rainfall features (same 9 as Model 2B)
    rainfall_60m: float = 0.0,
    rainfall_120m: float = 0.0,
    has_recent_rainfall_obs: int = 0,
    rain_intensity_max_120m: float = 0.0,
    rain_cooling_60m: float = 0.0,
    rain_after_max_flag: int = 0,
    post_peak_rain_flag: int = 0,
    rain_data_gap_flag: int = 0,
    rainfall_data_age_minutes: float = 0.0,
) -> pd.DataFrame:
    """Build Model 3B feature vector for a single decision_time (59 features).

    50 features from Model 3A (2A v2 + 5 trend) + 9 rainfall features.

    Parameters
    ----------
    temp_buffer : list, optional
        Minute-resolution temperature buffer (passed to 3A builder).
    rh_buffer : list, optional
        Minute-resolution RH buffer (passed to 3A builder).
    rainfall_60m/120m/... : float/int, optional
        Observed rainfall features. Defaults to 0 so Model 3B degrades to
        Model 3A when rainfall data is unavailable.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 59 feature columns, indexed by decision_time.
    """
    # Step 1: Get the 50 3A features
    df = build_model_3a_features(
        decision_time, weather_canonical, wind_canonical,
        forecast_canonical, spec, mode,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
    )

    # Step 2: Append 9 rainfall features
    rain_features = {
        "rainfall_60m": rainfall_60m,
        "rainfall_120m": rainfall_120m,
        "has_recent_rainfall_obs": has_recent_rainfall_obs,
        "rain_intensity_max_120m": rain_intensity_max_120m,
        "rain_cooling_60m": rain_cooling_60m,
        "rain_after_max_flag": rain_after_max_flag,
        "post_peak_rain_flag": post_peak_rain_flag,
        "rain_data_gap_flag": rain_data_gap_flag,
        "rainfall_data_age_minutes": rainfall_data_age_minutes,
    }

    for k, v in rain_features.items():
        df[k] = v

    return df
