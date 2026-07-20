# monitoring/model_2a_inference_parity_check.py
"""Model 2A specific inference parity check.

Replays logged inference rows through the same feature builder
and compares live vs replay feature values.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from monitoring.inference_parity_check_base import run_parity_check
from inference.model_2a_adapters import (
    Model2AVersionError,
    get_model_2a_adapter,
)
from features.model_2a_source_adapters import (
    standardize_weather_obs,
    standardize_wind_obs,
    standardize_forecast,
)

logger = logging.getLogger(__name__)


def run_model_2a_parity_check(
    inference_log_path: str = "logs/model_2a_inference_log.parquet",
    raw_weather: pd.DataFrame = None,
    raw_wind: pd.DataFrame = None,
    raw_forecast: pd.DataFrame = None,
    model_spec_path: Optional[str] = None,
    output_path: str = "reports",
    model_version: Optional[str] = None,
) -> dict:
    """Run Model 2A specific parity check.

    Args:
        inference_log_path: Path to inference log parquet.
        raw_weather: Raw weather data for replay.
        raw_wind: Raw wind data for replay.
        raw_forecast: Raw forecast data for replay.
        model_spec_path: Path to model spec.
        output_path: Output directory for reports.
        model_version: Explicitly selected ``v1`` or ``v2``.

    Returns:
        Summary dict of parity results.
    """
    adapter = get_model_2a_adapter(
        model_version=model_version,
        model_spec_path=model_spec_path,
    )
    lineage = adapter.validate_lineage()
    if not adapter.supports_realtime:
        raise Model2AVersionError(
            "Model 2A v1 parity replay is deprecated/unsupported because no "
            "v1-specific feature builder is available."
        )

    _validate_inference_log_version(inference_log_path, adapter.model_version)
    spec_path = str(adapter.feature_spec_path)

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
        return adapter.feature_builder(
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
        model_spec_path=spec_path,
        output_path=output_path,
        source_adapter_fn=source_adapter_fn,
        feature_builder_fn=feature_builder_fn,
        model_name=f"model_2a_{adapter.model_version}",
        run_metadata={
            "model_version": adapter.model_version,
            "feature_version": adapter.feature_version,
            "spec_path": lineage["spec_path"],
            "artifact_directory": lineage["artifact_directory"],
            "artifact_identity": lineage["artifact_identity"],
        },
    )

    return summary


def _validate_inference_log_version(inference_log_path: str, model_version: str) -> None:
    """Reject missing or mixed version metadata before parity replay."""
    log_path = Path(inference_log_path)
    if not log_path.exists():
        return
    log = pd.read_parquet(log_path)
    if "model_version" not in log.columns:
        raise Model2AVersionError(
            "Model 2A parity log has no model_version metadata; version is ambiguous."
        )
    versions = set(log["model_version"].dropna().astype(str))
    if versions != {model_version}:
        raise Model2AVersionError(
            f"Model 2A parity log versions {sorted(versions)!r} do not match "
            f"the explicitly selected {model_version!r}."
        )
