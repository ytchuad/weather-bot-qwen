"""
Model 2A comprehensive validation script.
Outputs 7 deliverable files to reports/.
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from sklearn.metrics import average_precision_score

REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

FEATURE_STORE = Path("data/model_2a_feature_store.parquet")
OOT_PRED = Path("models/intraday_minute_ml_model_2a/oot_predictions.parquet")
MODEL_DIR = Path("models/intraday_minute_ml_model_2a")
ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

BUCKETS = [
    (6, 9, "06-09"),
    (9, 12, "09-12"),
    (12, 15, "12-15"),
    (15, 18, "15-18"),
    (18, 24, "18-24"),
]

FEATURE_SETS = {
    "2A-0": [
        "temp_current", "rh_current", "max_so_far", "min_so_far",
        "range_so_far", "drop_from_max", "time_since_max",
        "minutes_since_midnight", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_morning", "is_afternoon", "is_evening",
        "obs_data_age_minutes",
    ],
    "2A-1": [
        "temp_current", "rh_current", "max_so_far", "min_so_far",
        "range_so_far", "drop_from_max", "time_since_max",
        "temp_change_30m", "temp_change_60m",
        "temp_slope_30m", "temp_slope_60m", "temp_acceleration_60m",
        "temp_volatility_60m", "rh_change_60m",
        "dew_point_change_60m", "dew_point_spread_change_60m",
        "pressure_change_60m", "pressure_change_180m",
        "minutes_since_midnight", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_morning", "is_afternoon", "is_evening",
        "obs_data_age_minutes",
    ],
    "2A-2": [
        "temp_current", "rh_current", "max_so_far", "min_so_far",
        "range_so_far", "drop_from_max", "time_since_max",
        "temp_change_30m", "temp_change_60m",
        "temp_slope_30m", "temp_slope_60m", "temp_acceleration_60m",
        "temp_volatility_60m", "rh_change_60m",
        "dew_point_change_60m", "dew_point_spread_change_60m",
        "pressure_change_60m", "pressure_change_180m",
        "forecast_min_temp", "forecast_max_temp", "forecast_range",
        "forecast_gap_from_max_so_far",
        "forecast_age_minutes", "forecast_lead_days",
        "minutes_since_midnight", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_morning", "is_afternoon", "is_evening",
        "obs_data_age_minutes",
    ],
}

CORE_WIND = [
    "wind_ref_mean", "wind_ref_max",
    "wind_victoria_harbour_mean", "wind_victoria_harbour_max",
    "wind_highland_mean", "wind_highland_max",
    "wind_all_change_60m",
    "wind_data_age_minutes",
]

FULL = [
    "temp_current", "rh_current", "pressure_current",
    "dew_point_current", "dew_point_spread",
    "max_so_far", "min_so_far", "range_so_far",
    "drop_from_max", "time_since_max",
    "temp_change_30m", "temp_change_60m",
    "temp_slope_30m", "temp_slope_60m",
    "temp_acceleration_60m", "temp_volatility_60m",
    "rh_change_60m", "dew_point_change_60m",
    "dew_point_spread_change_60m",
    "pressure_change_60m", "pressure_change_180m",
    "forecast_min_temp", "forecast_max_temp", "forecast_range",
    "forecast_gap_from_max_so_far",
    "forecast_age_minutes", "forecast_lead_days",
    "wind_ref_mean", "wind_ref_max",
    "wind_victoria_harbour_mean", "wind_victoria_harbour_max",
    "wind_highland_mean", "wind_highland_max",
    "wind_all_change_60m", "wind_kings_park_current",
    "minutes_since_midnight",
    "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening",
    "obs_data_age_minutes", "wind_data_age_minutes",
]

FEATURE_SETS["2A-3"] = CORE_WIND
FEATURE_SETS["2A-4"] = FULL

# ──────────────────────────────────────────────
# Step 1: Row universe check
# ──────────────────────────────────────────────
def step1_row_universe(fs):
    lines = []
    lines.append("=" * 60)
    lines.append("STEP 1: Decision-time Row Universe")
    lines.append("=" * 60)
    hr = fs["decision_time"].dt.hour.value_counts().sort_index()
    lines.append("Hour distribution:")
    for h, c in hr.items():
        lines.append(f"  {h:02d}:00  {c:>8,}")
    lines.append(f"")
    lines.append(f"Min decision_time: {fs['decision_time'].min()}")
    lines.append(f"Max decision_time: {fs['decision_time'].max()}")
    lines.append(f"Total rows: {len(fs)}")
    lines.append(f"")
    unusual = hr.index[~hr.index.isin(range(6, 24))]
    if len(unusual) > 0:
        lines.append(f"UNUSUAL HOURS: {list(unusual)}")
        lines.append(f"  -> Need to check decision calendar or stale feature store")
    else:
        lines.append(f"OK: All hours in 06-23 range")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Step 2: Actual high alignment
# ──────────────────────────────────────────────
def step2_actual_high(fs):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("STEP 2: Full-minute actual_high_today alignment")
    lines.append("=" * 60)
    raw_files = glob.glob("data/hk_weather_raw/*_temperature.parquet")
    raw = pd.concat([pd.read_parquet(f) for f in raw_files])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["date"] = raw["timestamp"].dt.date
    raw_daily = raw.groupby("date")["value"].max().rename("raw_actual_high")
    fs_daily = fs.groupby("target_date")["actual_high_today"].max().rename("fs_actual_high")
    cmp = pd.concat([raw_daily, fs_daily], axis=1).dropna()
    cmp["diff"] = cmp["fs_actual_high"] - cmp["raw_actual_high"]
    lines.append(cmp["diff"].describe().to_string())
    lines.append(f"")
    bad = cmp[cmp["diff"].abs() > 0.05]
    lines.append(f"Days with diff > 0.05°C: {len(bad)} / {len(cmp)} ({100*len(bad)/len(cmp):.1f}%)")
    if len(bad) > 0:
        lines.append(f"Worst 30:")
        for idx, r in bad.head(30).iterrows():
            lines.append(f"  {idx}  raw={r['raw_actual_high']:.1f}  fs={r['fs_actual_high']:.1f}  diff={r['diff']:+.1f}")
    return "\n".join(lines), cmp


# ──────────────────────────────────────────────
# Step 3: OOT bucket evaluation
# ──────────────────────────────────────────────
def bucket_metrics(sub):
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
    inside = (actual >= q10) & (actual <= q90)
    cov80 = float(np.nanmean(inside))
    piw = float((q90 - q10).mean())
    bias = float(np.nanmean(q50 - actual))
    q90_br = float(np.nanmean(actual > q90))
    q10_br = float(np.nanmean(actual < q10))

    n_dates = sub["target_date"].nunique() if "target_date" in sub.columns else 0
    pr_auc = float(average_precision_score(sub["is_upside_zero"], sub["zero_proba"]))
    prec = float((sub["zero_pred"] & sub["is_upside_zero"]).sum() / max(sub["zero_pred"].sum(), 1))
    rec = float((sub["zero_pred"] & sub["is_upside_zero"]).sum() / max(sub["is_upside_zero"].sum(), 1))
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return dict(
        n_rows=n, n_dates=n_dates,
        MAE_up=mae_up, MAE_tx=mae_tx,
        cov80=cov80, PIW=piw, bias=bias,
        q90_br=q90_br, q10_br=q10_br,
        pr_auc=pr_auc, precision=prec, recall=rec, f1=f1,
    )


def step3_oot_eval(oot):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("STEP 3: OOT Bucket Report (06:00-23:50 only)")
    lines.append("=" * 60)
    if "hour" in oot.columns:
        oot = oot[oot["hour"].between(6, 23)].copy()
    keys = ["n_rows", "n_dates", "MAE_up", "MAE_tx", "cov80", "PIW",
            "bias", "q90_br", "q10_br", "pr_auc", "precision", "recall", "f1"]
    header = f"{'bucket':>8s}" + "".join(f"{k:>12s}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    rows = {}
    for lo, hi, lb in BUCKETS:
        mask = (oot["hour"] >= lo) & (oot["hour"] < hi)
        if mask.any():
            m = bucket_metrics(oot[mask])
            rows[lb] = m
            vals = "".join(f"{m[k]:>12.4f}" if isinstance(m.get(k), float) else f"{m.get(k, 0):>12,}" for k in keys)
            lines.append(f"{lb:>8s}{vals}")
    m_all = bucket_metrics(oot)
    rows["ALL"] = m_all
    vals = "".join(f"{m_all[k]:>12.4f}" if isinstance(m_all.get(k), float) else f"{m_all.get(k, 0):>12,}" for k in keys)
    lines.append(f"{'ALL':>8s}{vals}")
    return "\n".join(lines), rows


# ──────────────────────────────────────────────
# Step 4: Classifier subset evaluation
# ──────────────────────────────────────────────
def step4_classifier_subsets(oot):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("STEP 4: Classifier Subset Analysis")
    lines.append("=" * 60)
    if "hour" in oot.columns:
        oot = oot[oot["hour"].between(6, 23)].copy()
    subsets = {
        "ALL (06-23)": oot,
        "06-12": oot[oot["hour"].between(6, 11)],
        "06-15": oot[oot["hour"].between(6, 14)],
        "Excl 18-24": oot[~oot["hour"].between(18, 23)],
    }
    keys = ["n_rows", "pr_auc", "precision", "recall", "f1", "best_thr", "pos_rate"]
    header = f"{'subset':>15s}" + "".join(f"{k:>12s}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        pr_auc = average_precision_score(sub["is_upside_zero"], sub["zero_proba"])
        y_true = sub["is_upside_zero"].values
        y_prob = sub["zero_proba"].values
        pos_rate = y_true.mean()
        # Find best threshold on THIS subset
        from sklearn.metrics import precision_recall_curve
        precs, recs, thrs = precision_recall_curve(y_true, y_prob)
        f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-9)
        best_i = np.argmax(f1s)
        best_thr = thrs[best_i]
        best_f1 = f1s[best_i]
        pred = (y_prob >= best_thr).astype(int)
        prec_val = (pred & y_true).sum() / max(pred.sum(), 1)
        rec_val = (pred & y_true).sum() / max(y_true.sum(), 1)
        vals = f"{len(sub):>12,}{pr_auc:>12.4f}{prec_val:>12.4f}{rec_val:>12.4f}{best_f1:>12.4f}{best_thr:>12.4f}{pos_rate:>12.4f}"
        lines.append(f"{name:>15s}{vals}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Step 5: Quantile calibration check
# ──────────────────────────────────────────────
def step5_calibration(oot):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("STEP 5: Quantile Calibration Check")
    lines.append("=" * 60)
    if "hour" in oot.columns:
        oot = oot[oot["hour"].between(6, 23)].copy()
    keys = ["n_rows", "cov80", "PIW", "q10_br", "q90_br", "action"]
    header = f"{'bucket':>8s}" + "".join(f"{k:>12s}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for lo, hi, lb in BUCKETS:
        mask = (oot["hour"] >= lo) & (oot["hour"] < hi)
        if not mask.any():
            continue
        sub = oot[mask]
        actual = sub["remaining_upside"].values
        q10 = sub["upside_q10"].values
        q50 = sub["upside_q50"].values
        q90 = sub["upside_q90"].values
        inside = (actual >= q10) & (actual <= q90)
        cov80 = np.nanmean(inside)
        piw = (q90 - q10).mean()
        q10_br = np.nanmean(actual < q10)
        q90_br = np.nanmean(actual > q90)
        act = "keep" if abs(cov80 - 0.80) < 0.03 else ("narrow" if cov80 > 0.83 else "widen")
        lines.append(f"{lb:>8s}{len(sub):>12,}{cov80:>12.4f}{piw:>12.4f}{q10_br:>12.4f}{q90_br:>12.4f}{act:>12s}")
    m = bucket_metrics(oot)
    lines.append(f"{'ALL':>8s}{m['n_rows']:>12,}{m['cov80']:>12.4f}{m['PIW']:>12.4f}{m['q10_br']:>12.4f}{m['q90_br']:>12.4f}{'tune':>12s}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Step 7: High drop-from-max diagnostic
# ──────────────────────────────────────────────
def step7_high_drop(oot, fs):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("STEP 7: High Drop-from-Max Diagnostic")
    lines.append("=" * 60)
    if "hour" in oot.columns:
        oot = oot[oot["hour"].between(6, 23)].copy()
    # Merge drop_from_max
    merge_cols = ["decision_time", "drop_from_max", "max_so_far", "actual_high_today"]
    if "drop_from_max" not in oot.columns:
        oot = oot.merge(fs[merge_cols], on="decision_time", how="left", suffixes=("", "_fs"))
    subsets = {
        "drop>=2": oot[oot["drop_from_max"] >= 2],
        "drop>=3": oot[oot["drop_from_max"] >= 3],
        "drop>=5": oot[oot["drop_from_max"] >= 5],
        "forecast_gap<=0": oot[oot["forecast_gap_from_max_so_far"] <= 0] if "forecast_gap_from_max_so_far" in oot.columns else oot.iloc[:0],
        "forecast_gap>2": oot[oot["forecast_gap_from_max_so_far"] > 2] if "forecast_gap_from_max_so_far" in oot.columns else oot.iloc[:0],
    }
    keys = ["n_rows", "MAE_up", "bias", "cov80", "PIW", "q10_br", "q90_br", "mean_pred_q50", "mean_actual"]
    header = f"{'subset':>25s}" + "".join(f"{k:>13s}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        actual = sub["remaining_upside"].values
        q50 = sub["upside_q50"].values
        q10 = sub["upside_q10"].values
        q90 = sub["upside_q90"].values
        inside = (actual >= q10) & (actual <= q90)
        vals = f"{len(sub):>13,}"
        vals += f"{np.nanmean(np.abs(actual - q50)):>13.4f}"
        vals += f"{np.nanmean(q50 - actual):>13.4f}"
        vals += f"{np.nanmean(inside):>13.4f}"
        vals += f"{(q90 - q10).mean():>13.4f}"
        vals += f"{np.nanmean(actual < q10):>13.4f}"
        vals += f"{np.nanmean(actual > q90):>13.4f}"
        vals += f"{np.nanmean(q50):>13.4f}"
        vals += f"{np.nanmean(actual):>13.4f}"
        lines.append(f"{name:>25s}{vals}")
    # Max-reached subset
    if "actual_high_today" in oot.columns and "max_so_far" in oot.columns:
        reached = oot[(oot["drop_from_max"] >= 5) & (oot["actual_high_today"] == oot["max_so_far"])]
        lines.append(f"")
        lines.append(f"Max reached, drop>=5: n={len(reached):,}")
        if len(reached) > 0:
            lines.append(f"  mean pred upside_q50: {reached['upside_q50'].mean():.4f}")
            lines.append(f"  mean actual: {reached['remaining_upside'].mean():.4f}")
            lines.append(f"  is_upside_zero%: {reached['is_upside_zero'].mean():.1%}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading feature store...")
    fs = pd.read_parquet(FEATURE_STORE)
    fs["decision_time"] = pd.to_datetime(fs["decision_time"])
    fs["target_date"] = fs["decision_time"].dt.date
    cols_for_oot = ["decision_time", "target_date", "hour", "drop_from_max",
                     "max_so_far", "actual_high_today", "remaining_upside",
                     "is_upside_zero",
                     "forecast_gap_from_max_so_far"]
    fs_for_oot = fs[cols_for_oot].copy()

    print("Loading OOT predictions...")
    oot = pd.read_parquet(OOT_PRED)
    oot["hour"] = pd.to_datetime(oot["decision_time"]).dt.hour
    # Merge target_date from feature store
    oot = oot.drop(columns=["target_date"], errors="ignore")
    oot = oot.merge(fs_for_oot, on="decision_time", how="left", suffixes=("", "_fs"))
    oot.loc[:, "hour"] = oot["hour_fs"].fillna(oot["hour"]).fillna(-1).astype(int)

    # Step 1
    r1 = step1_row_universe(fs)
    with open(REPORTS / "model_2a_row_universe_check.txt", "w") as f:
        f.write(r1)
    print(r1)

    # Step 2
    r2, cmp = step2_actual_high(fs)
    with open(REPORTS / "model_2a_actual_high_validation.txt", "w") as f:
        f.write(r2)
    cmp.to_csv(REPORTS / "model_2a_actual_high_validation.csv")
    print(r2)

    # Step 3
    r3, bucket_rows = step3_oot_eval(oot)
    with open(REPORTS / "model_2a_oot_bucket_report.txt", "w") as f:
        f.write(r3)
    pd.DataFrame(bucket_rows).T.to_csv(REPORTS / "model_2a_oot_bucket_report.csv")
    print(r3)

    # Step 4
    r4 = step4_classifier_subsets(oot)
    with open(REPORTS / "model_2a_classifier_subset_report.txt", "w") as f:
        f.write(r4)
    print(r4)

    # Step 5
    r5 = step5_calibration(oot)
    with open(REPORTS / "model_2a_quantile_calibration_report.txt", "w") as f:
        f.write(r5)
    print(r5)

    # Step 7
    r7 = step7_high_drop(oot, fs)
    with open(REPORTS / "model_2a_high_drop_diagnostic.txt", "w") as f:
        f.write(r7)
    print(r7)

    print("\nAll reports saved to reports/")

if __name__ == "__main__":
    main()
