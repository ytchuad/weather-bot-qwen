# features/model_3a_feature_builder.py
"""Model 3A feature builder = Model 2A v2 features + 5 trend-relation features.

Used for live inference. Wraps ``build_model_2a_features`` from the 2A builder
and appends 5 feature columns that distinguish sustained trends from temporary noise
by comparing short-term changes against long-term background.

Feature names must EXACTLY match models/intraday_minute_ml_model_3a/feature_list.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.model_2a_feature_builder import build_model_2a_features


def build_model_3a_features(
    decision_time: pd.Timestamp,
    weather_canonical: pd.DataFrame,
    wind_canonical: pd.DataFrame,
    forecast_canonical: pd.DataFrame,
    spec: dict,
    mode: str,
    temp_buffer: list | None = None,
    rh_buffer: list | None = None,
) -> pd.DataFrame:
    """Build Model 3A feature vector for a single decision_time (50 features).

    45 features from Model 2A v2 + 5 trend-relation features computed from the
    minute-level temperature buffer.

    Parameters
    ----------
    temp_buffer : list, optional
        Minute-resolution temperature readings for the current day.
        Used to compute the 5 trend-relation features.
    rh_buffer : list, optional
        Minute-resolution RH readings (not used for trend features but passed
        through to the 2A builder for consistency).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 50 feature columns, indexed by decision_time.
    """
    # Step 1: Get the 45 2A v2 features
    df = build_model_2a_features(
        decision_time, weather_canonical, wind_canonical,
        forecast_canonical, spec, mode,
    )

    # Step 2: Compute 5 trend-relation features from the temp buffer
    trend_features = _compute_trend_features_live(temp_buffer, decision_time)

    # Append to the DataFrame
    for k, v in trend_features.items():
        df[k] = v

    return df


def _compute_trend_features_live(
    temp_buffer: list | None,
    decision_time: pd.Timestamp,
) -> dict[str, float]:
    """Compute the 5 trend-relation features from minute-level temp_buffer.

    All features distinguish noise from sustained trends.
    """
    features: dict[str, float] = {}

    if not temp_buffer or len(temp_buffer) < 60:
        # Not enough data — return neutral defaults
        return {
            "temp_direction_alignment": 0.0,
            "temp_short_long_ratio": 1.0,
            "temp_volatility_ratio_60m_360m": 1.0,
            "temp_reversal_count_120m": 0.0,
            "temp_direction_persistence_60m": 0.5,
        }

    arr = np.array(list(temp_buffer), dtype=float)
    idx = len(arr) - 1  # current position

    # Guard: index bounds
    idx_10 = max(0, idx - 10)
    idx_30 = max(0, idx - 30)
    idx_60 = max(0, idx - 60)
    idx_120 = max(0, idx - 120)
    idx_240 = max(0, idx - 240)
    idx_360 = max(0, idx - 360)

    temp_now = arr[idx]

    # Δ10m and Δ60m for direction alignment
    delta_10 = temp_now - arr[idx_10]
    delta_60 = temp_now - arr[idx_60]

    # 1. Direction alignment: sign(Δ10m) × sign(Δ60m)
    features["temp_direction_alignment"] = float(
        np.sign(delta_10) * np.sign(delta_60)
    )

    # 2. Short-long ratio: |Δ30m| / max(|Δ240m|, 0.01), clipped [0, 10]
    delta_30 = temp_now - arr[idx_30]
    delta_240 = temp_now - arr[idx_240]
    ratio = abs(delta_30) / max(abs(delta_240), 0.01)
    features["temp_short_long_ratio"] = float(min(ratio, 10.0))

    # 3. Volatility ratio: σ(last 60 min) / max(σ(last 360 min), 0.01), clipped [0, 10]
    vol_60 = float(np.std(arr[idx_60:idx + 1], ddof=1)) if (idx - idx_60) >= 2 else 0.0
    vol_360 = float(np.std(arr[idx_360:idx + 1], ddof=1)) if (idx - idx_360) >= 2 else 0.0
    vol_ratio = vol_60 / max(vol_360, 0.01)
    features["temp_volatility_ratio_60m_360m"] = float(min(vol_ratio, 10.0))

    # 4. Reversal count over 120 min: count sign changes in 10-min diffs
    # Sample at 10-min intervals: positions [idx, idx-10, idx-20, ..., idx-120]
    sample_steps = list(range(0, 121, 10))
    sampled = [arr[max(0, idx - s)] for s in sample_steps]
    diffs = [sampled[i] - sampled[i + 1] for i in range(len(sampled) - 1)]
    signs = np.sign(diffs)
    rev_count = sum(
        1 for i in range(1, len(signs))
        if signs[i] * signs[i - 1] < 0
    )
    features["temp_reversal_count_120m"] = float(rev_count)

    # 5. Direction persistence: fraction of last 6 10-min diffs with same sign as Δ10m
    last_sign = np.sign(delta_10)
    if last_sign == 0:
        features["temp_direction_persistence_60m"] = 0.5
    else:
        # Examine last 6 10-min diffs [idx, idx-10, ..., idx-50]
        persistence_steps = list(range(0, 61, 10))
        p_sampled = [arr[max(0, idx - s)] for s in persistence_steps]
        p_diffs = [p_sampled[i] - p_sampled[i + 1] for i in range(len(p_sampled) - 1)]
        same_count = sum(1 for d in p_diffs if np.sign(d) == last_sign)
        features["temp_direction_persistence_60m"] = same_count / max(len(p_diffs), 1)

    return features
