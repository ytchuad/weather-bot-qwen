# features/feature_schema.py
"""Central feature schema loader.

Single source of truth for feature lists used across:
- training scripts
- inference scripts
- validation scripts
- promotion scripts
- smoke tests
- audit scripts
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_PATH = Path("config/feature_schema.json")


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_list(model_type: str) -> List[str]:
    schema = load_schema()
    baseline = list(schema["baseline_features"])

    if model_type == "baseline":
        return baseline

    if model_type == "rain_aware":
        return (
            baseline
            + list(schema["rain_observed_features"])
            + list(schema["rain_interaction_features"])
            + list(schema["metadata_features"])
        )

    if model_type == "rain_aware_nowcast":
        return (
            baseline
            + list(schema["rain_observed_features"])
            + list(schema["rain_interaction_features"])
            + list(schema["rain_nowcast_features"])
            + list(schema["metadata_features"])
        )

    raise ValueError(f"Unknown model_type: {model_type}. Use 'baseline', 'rain_aware', or 'rain_aware_nowcast'.")


def get_target_columns() -> List[str]:
    return list(load_schema()["target_columns"])


def get_forbidden_live_features() -> List[str]:
    return list(load_schema()["forbidden_live_features"])


def get_rain_feature_columns() -> List[str]:
    schema = load_schema()
    return (
        list(schema["rain_observed_features"])
        + list(schema["rain_interaction_features"])
        + list(schema.get("rain_nowcast_features", []))
    )
