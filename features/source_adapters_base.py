# features/source_adapters_base.py
import pandas as pd
import numpy as np
from typing import Optional


def standardize_source(
    df: pd.DataFrame,
    source_type: str,
    source_system: str = "unknown",
    expected_fields: Optional[list] = None,
) -> pd.DataFrame:
    """Generic source adapter that converts raw data into canonical schema.

    Args:
        df: Raw source DataFrame.
        source_type: One of 'historical', 'live', 'replay'.
        source_system: Name identifying the source system.
        expected_fields: Required canonical fields to ensure exist in output.

    Returns:
        DataFrame with canonical schema:
        source_system, source_mode, available_time, timestamp,
        station_id, value, data_quality_flags
    """
    if source_type not in ("historical", "live", "replay"):
        raise ValueError(f"Invalid source_type: {source_type}")

    result = df.copy()
    result["source_system"] = source_system
    result["source_mode"] = source_type

    if "available_time" not in result.columns:
        if "timestamp" in result.columns:
            result["available_time"] = result["timestamp"]
        else:
            raise ValueError(
                "available_time not found and cannot be derived. "
                "Model spec must define derivation rule."
            )

    if "timestamp" not in result.columns:
        result["timestamp"] = result["available_time"]

    if "station_id" not in result.columns:
        result["station_id"] = "default"

    if "value" not in result.columns:
        result["value"] = np.nan

    if "data_quality_flags" not in result.columns:
        result["data_quality_flags"] = 0

    if "source_systems" not in result.columns:
        result["source_systems"] = source_system

    if "source_timestamps" not in result.columns:
        result["source_timestamps"] = None

    if "source_available_times" not in result.columns:
        result["source_available_times"] = None

    canonical_cols = [
        "source_system",
        "source_mode",
        "available_time",
        "timestamp",
        "station_id",
        "value",
        "data_quality_flags",
    ]
    if expected_fields:
        for field in expected_fields:
            if field not in result.columns:
                result[field] = np.nan
        canonical_cols = list(dict.fromkeys(canonical_cols + expected_fields))

    for col in canonical_cols:
        if col not in result.columns:
            result[col] = np.nan

    return result[canonical_cols]


def validate_canonical_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the canonical schema."""
    required = [
        "source_system",
        "source_mode",
        "available_time",
        "timestamp",
        "station_id",
        "value",
        "data_quality_flags",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical schema missing fields: {missing}")
    valid_modes = {"historical", "live", "replay"}
    invalid_modes = set(df["source_mode"].dropna().unique()) - valid_modes
    if invalid_modes:
        raise ValueError(f"Invalid source_mode values: {invalid_modes}")
