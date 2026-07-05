"""
evaluate_model_2b.py

Compare Model 2A v2, Model 2B full, and Model 2B restricted on the same OOT period.
Outputs CSV reports to reports/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_2A_V2_OOT = Path("models/intraday_minute_ml_model_2a_v2/oot_predictions.parquet")
MODEL_2B_FEATURE_STORE = Path("data/model_2b_feature_store.parquet")
MODEL_2B_OOT = Path("models/intraday_minute_ai_model_2b/oot_predictions.parquet")
MODEL_2B_RESTRICTED_OOT = Path("models/intraday_minute_ai_model_2b_restricted/oot_predictions.parquet")
MODEL_2B_RESTRICTED_OOT = Path("models/intraday_minute_ai_model_2b_restricted/oot_predictions.parquet")

ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]


def compute_metrics(sub):
    """Compute all metrics for a given subset."""
    n = len(sub)
    if n == 0:
        return {}

    actual = sub["remaining_upside"].values
    actual_tmax = sub["actual_high_today"].values
    q50 = sub["upside_q50"].values
    q50_tmax = sub["pred_tmax_q50"].values
    q10 = sub["upside_q10"].values
    q90 = sub["upside_q90"].values

    mae_up = float(np.nanmean(np.abs(actual - q50)))
    mae_tx = float(np.nanmean(np.abs(actual_tmax - q50_tmax)))
    rmse = float(np.sqrt(np.nanmean((actual - q50) ** 2)))
    bias = float(np.nanmean(q50 - actual))
    q50_breach = float(np.nanmean(actual > q50))

    inside = (actual >= q10) & (actual <= q90)
    cov80 = float(np.nanmean(inside))
    piw = float((q90 - q10).mean())

    pr_auc = float(average_precision_score(sub["is_upside_zero"], sub["zero_proba"]))
    prec = float((sub["zero_pred"] & sub["is_upside_zero"]).sum() / max(sub["zero_pred"].sum(), 1))
    rec = float((sub["zero_pred"] & sub["is_upside_zero"]).sum() / max(sub["is_upside_zero"].sum(), 1))
    f1 = float(2 * prec * rec / max(prec + rec, 1e-9))

    return {
        "n_rows": n,
        "n_dates": int(sub["target_date"].nunique()),
        "mae_upside": round(mae_up, 4),
        "mae_pred_tmax": round(mae_tx, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "q50_breach": round(q50_breach, 4),
        "cov80": round(cov80, 4),
        "piw": round(piw, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def load_oot_with_rain(model_oot_path, rain_store_path=None):
    """Load OOT predictions and overlay rain columns from feature store."""
    df = pd.read_parquet(model_oot_path)
    if rain_store_path is not None and rain_store_path.exists():
        store = pd.read_parquet(rain_store_path)
        store["target_date"] = pd.to_datetime(store["target_date"])
        store["decision_time"] = pd.to_datetime(store["decision_time"])
        rain_cols = [c for c in store.columns if (
            c.startswith("rain") or c.startswith("post_peak")
            or c.startswith("rain_after") or c.startswith("heavy_recent")
            or c in ("drop_from_max", "time_since_max", "hour", "has_recent_rainfall_obs",
                     "morning_peak_rain_flag")
        )]
        rain_cols = [c for c in rain_cols if c in store.columns and c not in df.columns]
        if rain_cols:
            df = df.merge(
                store[["target_date", "decision_time"] + rain_cols],
                on=["target_date", "decision_time"],
                how="left",
            )
    return df


def evaluate_by_regime(df):
    """Evaluate on all rain regime slices."""
    results = {}
    valid = df[df["actual_high_today"] == df["actual_high_today"]]

    rain_col = "rainfall_60m" if "rainfall_60m" in df.columns else "has_recent_rainfall_obs"

    slices = {
        "ALL": valid,
        "no_rain": valid[valid[rain_col] == 0],
    }

    if "has_recent_rainfall_obs" in df.columns:
        slices["recent_rain"] = valid[valid["has_recent_rainfall_obs"] == 1]
    if "rainfall_60m" in df.columns:
        slices["heavy_recent_rain"] = valid[valid["rainfall_60m"] >= 10]
    if "post_peak_rain_flag" in df.columns:
        slices["post_peak_rain"] = valid[valid["post_peak_rain_flag"] == 1]
    if "rain_after_max_flag" in df.columns:
        slices["rain_after_max"] = valid[valid["rain_after_max_flag"] == 1]

    # Hourly rain slices
    if "has_recent_rainfall_obs" in df.columns:
        for lo, hi, lb in [(6, 9, "06-09_rain"), (9, 12, "09-12_rain"),
                           (12, 15, "12-15_rain"), (15, 18, "15-18_rain")]:
            mask = (
                (valid["hour"] >= lo) & (valid["hour"] < hi)
                & (valid["has_recent_rainfall_obs"] == 1)
            )
            slices[lb] = valid[mask]

    for name, sub in slices.items():
        if len(sub) == 0:
            continue
        results[name] = compute_metrics(sub)

    return results


def main():
    print("=" * 60)
    print("  Model 2B vs Model 2A v2 Evaluation")
    print("=" * 60)

    # Always use Model 2B feature store as the rain source (has all rain columns)
    rain_source = MODEL_2B_FEATURE_STORE

    # Load Model 2A v2 OOT predictions
    print("\nLoading Model 2A v2 OOT...")
    df_2a = load_oot_with_rain(MODEL_2A_V2_OOT, rain_source)
    print(f"  {len(df_2a):,} rows")

    # Load Model 2B full OOT
    print("Loading Model 2B full OOT...")
    df_2b = load_oot_with_rain(MODEL_2B_OOT, rain_source)
    print(f"  {len(df_2b):,} rows")

    # Load Model 2B restricted OOT
    print("Loading Model 2B restricted OOT...")
    df_2b_r = load_oot_with_rain(MODEL_2B_RESTRICTED_OOT, rain_source)
    print(f"  {len(df_2b_r):,} rows")

    # Filter to common target_date range
    common_dates = set(df_2a["target_date"].unique())
    common_dates &= set(df_2b["target_date"].unique())
    common_dates &= set(df_2b_r["target_date"].unique())

    df_2a = df_2a[df_2a["target_date"].isin(common_dates)]
    df_2b = df_2b[df_2b["target_date"].isin(common_dates)]
    df_2b_r = df_2b_r[df_2b_r["target_date"].isin(common_dates)]
    print(f"  Common target dates: {len(common_dates)}")

    # Evaluate each model
    print("\n=== Evaluating Model 2A v2 ===")
    results_2a = evaluate_by_regime(df_2a)

    print("\n=== Evaluating Model 2B full ===")
    results_2b = evaluate_by_regime(df_2b)

    print("\n=== Evaluating Model 2B restricted ===")
    results_2b_r = evaluate_by_regime(df_2b_r)

    # Build comparison table
    rows = []
    all_slices = sorted(set(list(results_2a.keys()) + list(results_2b.keys()) + list(results_2b_r.keys())))

    for s in all_slices:
        for model_name, results in [("2A_v2", results_2a), ("2B_full", results_2b), ("2B_restricted", results_2b_r)]:
            r = results.get(s, {})
            if r:
                row = {"slice": s, "model": model_name, **r}
                rows.append(row)

    df_out = pd.DataFrame(rows)
    cols = ["slice", "model", "n_rows", "n_dates", "mae_upside", "mae_pred_tmax",
            "rmse", "bias", "q50_breach", "cov80", "piw", "pr_auc", "precision",
            "recall", "f1"]
    cols = [c for c in cols if c in df_out.columns]
    df_out = df_out[cols]

    # Save summary
    summary_path = REPORTS_DIR / "model_2b_vs_2a_v2_oot_summary.csv"
    df_out.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print("\n--- OOT Summary ---")
    print(f"{'Slice':<20s} {'Model':<15s} {'n_rows':>8s} {'MAE_up':>8s} {'MAE_tx':>8s} "
          f"{'RMSE':>8s} {'bias':>8s} {'q50_br':>8s} {'PR-AUC':>8s} {'F1':>8s}")
    print("-" * 100)
    for s in all_slices:
        for mn in ["2A_v2", "2B_full", "2B_restricted"]:
            r = df_out[(df_out["slice"] == s) & (df_out["model"] == mn)]
            if len(r) == 0:
                continue
            row = r.iloc[0]
            print(f"{s:<20s} {mn:<15s} {int(row['n_rows']):>8,} {row['mae_upside']:>8.4f} "
                  f"{row['mae_pred_tmax']:>8.4f} {row['rmse']:>8.4f} {row['bias']:>+8.4f} "
                  f"{row['q50_breach']:>8.4f} {row['pr_auc']:>8.4f} {row['f1']:>8.4f}")

    # Rain regime breakdown
    rain_rows = [r for r in rows if r["slice"] in [
        "recent_rain", "heavy_recent_rain", "post_peak_rain", "rain_after_max",
        "no_rain", "06-09_rain", "09-12_rain", "12-15_rain", "15-18_rain"]]
    if rain_rows:
        df_rain = pd.DataFrame(rain_rows)
        rain_path = REPORTS_DIR / "model_2b_rain_regime_breakdown.csv"
        df_rain.to_csv(rain_path, index=False)
        print(f"\nRain regime breakdown saved to {rain_path}")

    # Restricted comparison
    restricted_rows = [r for r in rows if r["model"] == "2B_restricted"]
    if restricted_rows:
        df_res = pd.DataFrame(restricted_rows)
        res_path = REPORTS_DIR / "model_2b_restricted_comparison.csv"
        df_res.to_csv(res_path, index=False)
        print(f"Restricted comparison saved to {res_path}")

    # Acceptance check
    print("\n--- Acceptance Check ---")
    no_rain_m2a = results_2a.get("no_rain", {}).get("mae_upside", None)
    no_rain_m2b = results_2b.get("no_rain", {}).get("mae_upside", None)
    rain_m2a = results_2a.get("recent_rain", {}).get("mae_upside", None)
    rain_m2b = results_2b.get("recent_rain", {}).get("mae_upside", None)
    bias_m2a = results_2a.get("recent_rain", {}).get("bias", None)
    bias_m2b = results_2b.get("recent_rain", {}).get("bias", None)
    q50_rain_m2a = results_2a.get("recent_rain", {}).get("q50_breach", None)
    q50_rain_m2b = results_2b.get("recent_rain", {}).get("q50_breach", None)

    if rain_m2a is not None and rain_m2b is not None:
        mae_improve = rain_m2a - rain_m2b
        print(f"  Rain MAE improvement: {mae_improve:+.4f} (target >= +0.05)")
    if bias_m2a is not None and bias_m2b is not None:
        bias_reduce = abs(bias_m2a) - abs(bias_m2b)
        print(f"  Rain bias reduction: {bias_reduce:+.4f} (target >= +0.05)")
    if q50_rain_m2a is not None and q50_rain_m2b is not None:
        q50_improve = q50_rain_m2a - q50_rain_m2b
        print(f"  Rain q50 breach improvement: {q50_improve:+.4f} (target >= +0.02)")
    if no_rain_m2a is not None and no_rain_m2b is not None:
        no_rain_degrade = no_rain_m2b - no_rain_m2a
        print(f"  No-rain MAE degradation: {no_rain_degrade:+.4f} (threshold <= +0.02)")

    print(f"\nDone. Reports in {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
