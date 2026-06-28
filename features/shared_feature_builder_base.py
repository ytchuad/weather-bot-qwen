# features/shared_feature_builder_base.py
import pandas as pd
import numpy as np
from typing import Optional


def build_features(
    decision_time: pd.Timestamp,
    canonical_sources: dict[str, pd.DataFrame],
    spec: dict,
    mode: str,
) -> pd.DataFrame:
    """Generic shared feature builder contract.

    All models must expose a function with this signature, used by:
    1. Historical feature-store build
    2. Real-time inference
    3. Replay parity check

    Args:
        decision_time: The decision time for which features are built.
        canonical_sources: Dict mapping source name -> canonical DataFrame.
        spec: Model-specific specification dict.
        mode: One of 'historical', 'live', 'replay'.

    Returns:
        DataFrame with one row of features indexed by decision_time.
    """
    raise NotImplementedError(
        "Each model must implement its own build_features() following this contract."
    )


def apply_availability_filter(
    df: pd.DataFrame,
    decision_time: pd.Timestamp,
    time_col: str = "available_time",
) -> pd.DataFrame:
    """Apply the core availability rule: available_time <= decision_time.

    This enforces that no future data leaks into feature computation.
    """
    if time_col not in df.columns:
        raise ValueError(
            f"Cannot apply availability filter: '{time_col}' not found. "
            "Use timestamp only when available_time is unavailable and "
            "explicitly derived from model spec."
        )
    return df[df[time_col] <= decision_time].copy()


def validate_feature_vector(
    feature_df: pd.DataFrame,
    feature_list_path: str,
) -> tuple[pd.DataFrame, list]:
    """Validate that feature_df contains all trained features.

    Loads the feature list from training time and ensures exact match.

    Args:
        feature_df: DataFrame with computed features.
        feature_list_path: Path to saved feature_list.json from training.

    Returns:
        Tuple of (X, FEATURE_COLS) where X is the feature matrix with
        exactly the training features in the correct order.

    Raises:
        ValueError: If any trained features are missing from feature_df.
    """
    import json

    with open(feature_list_path, "r") as f:
        FEATURE_COLS = json.load(f)

    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = feature_df[FEATURE_COLS]
    return X, FEATURE_COLS
