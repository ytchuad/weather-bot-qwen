"""
build_model_3b_feature_store.py

Build Model 3B feature store = Model 2B feature store + 5 trend-relation features.
Reads existing 2B store, computes new features, appends columns, saves.

Output: data/model_3b_feature_store.parquet
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TWO_B_STORE = Path("data/model_2b_feature_store.parquet")
OUTPUT_PATH = Path("data/model_3b_feature_store.parquet")


def compute_trend_relation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 trend-relation features that distinguish noise from sustained trends.

    All features are computed per target_date using 10-min decision steps.
    """
    print("  Computing trend-relation features...")

    df = df.sort_values(["target_date", "decision_time"]).reset_index(drop=True)

    # 1-step = 10 min; 6-step = 60 min; 12-step = 120 min; 24-step = 240 min; 36-step = 360 min
    dt1 = df.groupby("target_date")["temp_current"].diff(1)      # 10-min change
    dt6 = df.groupby("target_date")["temp_current"].diff(6)      # 60-min change
    dt24 = df.groupby("target_date")["temp_current"].diff(24)    # 240-min change

    # 1. Direction alignment: sign(Δ10m) × sign(Δ60m)
    align = np.sign(dt1) * np.sign(dt6)
    df["temp_direction_alignment"] = align.fillna(0).astype(float)

    # 2. Short-long ratio: |Δ30m| / max(|Δ240m|, 0.01), clamped to [0, 10]
    dt3 = df.groupby("target_date")["temp_current"].diff(3)  # 30-min change
    ratio = dt3.abs() / np.maximum(dt24.abs(), 0.01)
    ratio = ratio.clip(upper=10.0)
    df["temp_short_long_ratio"] = ratio.fillna(1.0).astype(float)

    # 3. Volatility ratio: σ60m / σ360m, clamped to [0, 10]
    vol6 = df.groupby("target_date")["temp_current"].transform(
        lambda x: x.rolling(6, min_periods=3).std()
    )
    vol36 = df.groupby("target_date")["temp_current"].transform(
        lambda x: x.rolling(36, min_periods=6).std()
    )
    vol_ratio = vol6 / np.maximum(vol36, 0.01)
    vol_ratio = vol_ratio.clip(upper=10.0)
    df["temp_volatility_ratio_60m_360m"] = vol_ratio.fillna(1.0).astype(float)

    # 4. Reversal count over 120 min: number of sign changes in 10-min diffs over 12 steps
    def _rev_count(x):
        d = x.diff().dropna()
        if len(d) < 3:
            return 0.0
        rev = ((d.iloc[1:].values * d.iloc[:-1].values) < 0).sum()
        return float(rev)

    df["temp_reversal_count_120m"] = (
        df.groupby("target_date")["temp_current"]
        .transform(lambda x: x.rolling(12, min_periods=3).apply(_rev_count, raw=False))
        .fillna(0)
    )

    # 5. Direction persistence: fraction of last 6 10-min diffs with same sign as most recent
    def _persistence(x):
        d = x.diff().dropna()
        if len(d) < 2:
            return 0.5
        last_dir = np.sign(d.iloc[-1])
        if last_dir == 0:
            return 0.5
        same = (np.sign(d.tail(6).values) * last_dir > 0).mean()
        return float(same)

    df["temp_direction_persistence_60m"] = (
        df.groupby("target_date")["temp_current"]
        .transform(lambda x: x.rolling(6, min_periods=2).apply(_persistence, raw=False))
        .fillna(0.5)
    )

    print(f"  Columns added: temp_direction_alignment, temp_short_long_ratio, "
          f"temp_volatility_ratio_60m_360m, temp_reversal_count_120m, temp_direction_persistence_60m")
    return df


def main():
    print("=" * 60)
    print("  Model 3B Feature Store Builder")
    print("=" * 60)

    print("\n=== Loading Model 2B feature store ===")
    df = pd.read_parquet(TWO_B_STORE)
    print(f"  Shape: {df.shape}")
    print(f"  target_date: {df['target_date'].min()} ~ {df['target_date'].max()}")

    df = compute_trend_relation_features(df)

    print(f"\n=== Saving Model 3B feature store ===")
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
