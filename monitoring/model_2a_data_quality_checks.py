# monitoring/model_2a_data_quality_checks.py
"""Model 2A data quality monitoring.

Runs schema, range, freshness, missingness, and source availability
checks on Model 2A canonical sources.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from monitoring.data_quality_checks_base import run_data_quality_checks
from inference.model_2a_adapters import get_model_2a_adapter

logger = logging.getLogger(__name__)


def run_model_2a_data_quality_checks(
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    model_spec_path: Optional[str] = None,
    output_path: str = "reports",
    model_version: Optional[str] = None,
) -> pd.DataFrame:
    """Run Model 2A specific data quality checks.

    Checks cover:
    - Temperature: valid range [0, 40], spike detection, anomaly flags
    - Wind: valid range [0, 150], station coverage, missing flags
    - Forecast: max >= min, age within limits, issue hour bounds
    - Freshness: all sources within configured max age
    - Missingness: per-source thresholds

    Args:
        weather_canonical: Canonical weather obs DataFrame.
        wind_canonical: Canonical wind obs DataFrame.
        forecast_canonical: Canonical forecast DataFrame.
        model_spec_path: Path to model spec YAML.
        output_path: Directory for output reports.
        model_version: Explicitly selected ``v1`` or ``v2``.

    Returns:
        DataFrame with data quality check results.
    """
    adapter = get_model_2a_adapter(
        model_version=model_version,
        model_spec_path=model_spec_path,
    )
    lineage = adapter.validate_lineage()
    spec = adapter.load_spec()

    canonical_sources = {}
    if weather_canonical is not None:
        canonical_sources["weather_obs"] = weather_canonical
    if wind_canonical is not None:
        canonical_sources["wind_obs"] = wind_canonical
    if forecast_canonical is not None:
        canonical_sources["forecast"] = forecast_canonical

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = run_data_quality_checks(
        canonical_sources=canonical_sources,
        spec=spec,
        output_path=str(output_dir),
        model_name=f"model_2a_{adapter.model_version}",
        run_metadata={
            "model_version": adapter.model_version,
            "feature_version": adapter.feature_version,
            "spec_path": lineage["spec_path"],
            "artifact_directory": lineage["artifact_directory"],
            "artifact_identity": lineage["artifact_identity"],
        },
    )

    _run_model_2a_specific_checks(
        weather_canonical, wind_canonical, forecast_canonical,
        spec, output_dir, adapter.model_version, lineage,
    )

    return checks


def _run_model_2a_specific_checks(
    weather: pd.DataFrame,
    wind: pd.DataFrame,
    forecast: pd.DataFrame,
    spec: dict,
    output_dir: Path,
    model_version: str,
    lineage: dict,
) -> list:
    """Run Model 2A specific quality checks beyond the generic ones."""
    specific_checks = []
    temp_cleaning = spec.get("temperature_cleaning", {})
    valid_min = temp_cleaning.get("valid_min_temp", 0)
    valid_max = temp_cleaning.get("valid_max_temp", 40)

    # Temperature cleaning check
    if weather is not None and "temp_current_raw" in weather.columns:
        raw = weather["temp_current_raw"].dropna()
        out_of_range = ((raw < valid_min) | (raw > valid_max)).sum()
        specific_checks.append({
            "check": "temperature_cleaning",
            "status": "pass" if out_of_range == 0 else "warn",
            "detail": f"{out_of_range} values outside [{valid_min}, {valid_max}]",
        })

    # Spike detection check
    if weather is not None and "temp_spike_flag" in weather.columns:
        spikes = weather["temp_spike_flag"].sum()
        specific_checks.append({
            "check": "temp_spike_detection",
            "status": "warn" if spikes > 0 else "pass",
            "detail": f"{spikes} temperature spikes detected (change >= 5 in 1 min)",
        })

    # Wind station coverage
    if wind is not None and "station_group" in wind.columns:
        coverage = wind["station_group"].value_counts().to_dict()
        ref_present = "ref" in coverage or "urban" in coverage
        expected_wind_group = (
            "offshore_highland" if model_version == "v2" else "highland"
        )
        expected_wind_group_present = expected_wind_group in coverage
        missing_groups = []
        if not ref_present:
            missing_groups.append("ref/urban")
        if not expected_wind_group_present:
            missing_groups.append(expected_wind_group)
        specific_checks.append({
            "check": "wind_station_coverage",
            "status": "pass" if not missing_groups else "warn",
            "detail": (
                f"Model {model_version} station groups: {coverage}, "
                f"missing: {missing_groups if missing_groups else 'none'}"
            ),
        })

    # Forecast max >= min check
    if forecast is not None and "forecast_max_temp" in forecast.columns and "forecast_min_temp" in forecast.columns:
        invalid = (forecast["forecast_max_temp"].dropna() < forecast["forecast_min_temp"].dropna()).sum()
        specific_checks.append({
            "check": "forecast_consistency",
            "status": "pass" if invalid == 0 else "fail",
            "detail": f"{invalid} forecasts with max_temp < min_temp",
        })

    # Data freshness detail
    now = pd.Timestamp.now()
    max_age = spec.get("data_quality_rules", {}).get("max_data_age_minutes", 30)
    for source_name, df in [("weather_obs", weather), ("wind_obs", wind), ("forecast", forecast)]:
        if df is not None and "available_time" in df.columns and len(df) > 0:
            latest = df["available_time"].max()
            age = (now - pd.Timestamp(latest)).total_seconds() / 60
            if age > max_age:
                specific_checks.append({
                    "check": f"freshness_{source_name}",
                    "status": "fail",
                    "detail": f"Latest data {age:.0f} min old (max {max_age} min)",
                })

    specific_df = pd.DataFrame(specific_checks)
    if len(specific_df) > 0:
        specific_df.insert(0, "model_version", model_version)
        specific_df.insert(1, "spec_path", lineage["spec_path"])
        specific_df.insert(2, "artifact_directory", lineage["artifact_directory"])
        specific_df.insert(3, "artifact_identity", lineage["artifact_identity"])
        specific_path = output_dir / f"model_2a_{model_version}_specific_dq_checks.csv"
        specific_df.to_csv(specific_path, index=False)

    return specific_checks
