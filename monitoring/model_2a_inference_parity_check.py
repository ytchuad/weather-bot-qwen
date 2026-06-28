# monitoring/model_2a_inference_parity_check.py
"""Model 2A specific inference parity check.

Replays logged inference rows through the same feature builder
and compares live vs replay feature values.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from monitoring.inference_parity_check_base import run_parity_check
from features.model_2a_source_adapters import (
    standardize_weather_obs,
    standardize_wind_obs,
    standardize_forecast,
)
from features.model_2a_feature_builder import build_model_2a_features

logger = logging.getLogger(__name__)


def run_model_2a_parity_check(
    inference_log_path: str = "logs/model_2a_inference_log.parquet",
    raw_weather: pd.DataFrame = None,
    raw_wind: pd.DataFrame = None,
    raw_forecast: pd.DataFrame = None,
    model_spec_path: str = "config/model_2a_feature_spec.yaml",
    output_path: str = "reports",
) -> dict:
    """Run Model 2A specific parity check.

    Args:
        inference_log_path: Path to inference log parquet.
        raw_weather: Raw weather data for replay.
        raw_wind: Raw wind data for replay.
        raw_forecast: Raw forecast data for replay.
        model_spec_path: Path to model spec.
        output_path: Output directory for reports.

    Returns:
        Summary dict of parity results.
    """
    raw_sources = {}
    if raw_weather is not None:
        raw_sources["weather_obs"] = raw_weather
    if raw_wind is not None:
        raw_sources["wind_obs"] = raw_wind
    if raw_forecast is not None:
        raw_sources["forecast"] = raw_forecast

    def source_adapter_fn(df, source_name, source_type):
        if source_name == "weather_obs":
            return standardize_weather_obs(df, source_type)
        elif source_name == "wind_obs":
            return standardize_wind_obs(df, source_type)
        elif source_name == "forecast":
            return standardize_forecast(df, source_type)
        return df

    def feature_builder_fn(decision_time, canonical_sources, spec, mode):
        weather = canonical_sources.get("weather_obs", pd.DataFrame())
        wind = canonical_sources.get("wind_obs", pd.DataFrame())
        forecast = canonical_sources.get("forecast", pd.DataFrame())
        return build_model_2a_features(
            decision_time=decision_time,
            weather_canonical=weather,
            wind_canonical=wind,
            forecast_canonical=forecast,
            spec=spec,
            mode=mode,
        )

    summary = run_parity_check(
        inference_log_path=inference_log_path,
        raw_sources_dict=raw_sources,
        model_spec_path=model_spec_path,
        output_path=output_path,
        source_adapter_fn=source_adapter_fn,
        feature_builder_fn=feature_builder_fn,
        model_name="model_2a",
    )

    return summary
