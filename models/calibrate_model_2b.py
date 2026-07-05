"""
calibrate_model_2b.py

Residual-based interval calibration for Model 2B.
Computes empirical p10/p90 residuals by rain regime.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path("models/intraday_minute_ai_model_2b")
OOT_PATH = MODEL_DIR / "oot_predictions.parquet"
STORE_PATH = Path("data/model_2b_feature_store.parquet")
OUTPUT_PATH = MODEL_DIR / "calibration_residuals.json"


def load_oot_with_rain():
    """Load OOT predictions and overlay rain features from store."""
    df = pd.read_parquet(OOT_PATH)
    store = pd.read_parquet(STORE_PATH)
    store["target_date"] = pd.to_datetime(store["target_date"])
    store["decision_time"] = pd.to_datetime(store["decision_time"])

    rain_cols = [c for c in store.columns if c.startswith("rain")
                 or c in ("drop_from_max", "time_since_max", "hour")]
    rain_cols = [c for c in rain_cols if c in store.columns and c not in df.columns]

    if rain_cols:
        df = df.merge(
            store[["target_date", "decision_time"] + rain_cols],
            on=["target_date", "decision_time"],
            how="left",
        )
    return df


def main():
    print("=" * 60)
    print("  Model 2B Calibration Check")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_oot_with_rain()
    print(f"Loaded {len(df):,} OOT rows")

    # Residual = actual_remaining_upside - pred_q50
    df["residual"] = df["remaining_upside"] - df["upside_q50"]

    valid = df[df["actual_high_today"] == df["actual_high_today"]]

    regimes = {
        "no_rain": valid["rainfall_60m"] == 0,
        "rainfall_60m_gt_0": valid["rainfall_60m"] > 0,
        "heavy_recent_rain": valid["rainfall_60m"] >= 10,
        "post_peak_rain": valid["post_peak_rain_flag"] == 1,
    }

    if "has_recent_rainfall_obs" in valid.columns:
        regimes["has_recent_rain"] = valid["has_recent_rainfall_obs"] == 1

    # 06-12 rain regime
    regimes["06-12_rain"] = (
        (valid["hour"] >= 6) & (valid["hour"] < 12)
        & (valid["rainfall_60m"] > 0)
    )

    calibration = {}
    for name, mask in regimes.items():
        sub = valid[mask]
        n = len(sub)
        if n < 10:
            print(f"  {name}: too few samples ({n}), skipping")
            continue

        residuals = sub["residual"].values
        p10 = float(np.percentile(residuals, 10))
        p50 = float(np.percentile(residuals, 50))
        p90 = float(np.percentile(residuals, 90))
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))

        # Proposed calibrated intervals
        base_q50 = sub["upside_q50"].values
        cali_q10 = float(np.mean(base_q50 + p10))
        cali_q90 = float(np.mean(base_q50 + p90))

        calibration[name] = {
            "n_rows": n,
            "n_dates": int(sub["target_date"].nunique()),
            "residual_p10": round(p10, 4),
            "residual_p50": round(p50, 4),
            "residual_p90": round(p90, 4),
            "residual_mean": round(mean_res, 4),
            "residual_std": round(std_res, 4),
            "suggested_cali_q10_offset": round(p10, 4),
            "suggested_cali_q90_offset": round(p90, 4),
        }
        print(f"  {name:>20s}: n={n:>6,d}  "
              f"p10={p10:+.4f}  p50={p50:+.4f}  p90={p90:+.4f}  "
              f"mean={mean_res:+.4f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nCalibration residuals saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
