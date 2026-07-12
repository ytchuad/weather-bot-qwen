"""Model E Morning Tmin — hour-block calibration + classifier diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

OOT_PATH = Path("models/intraday_minute_ml_model_e_morning_tmin/oot_predictions.parquet")
FEATURE_PATH = Path("data/intraday_minute_ml_features_tmin_e.parquet")
CALIB_PATH = Path("models/intraday_minute_ml_model_e_morning_tmin/morning_calibration.json")

HOUR_BUCKETS = [(0, 2, "00-02"), (2, 4, "02-04"), (4, 6, "04-06"), (6, 8, "06-08")]


def pr_metrics(y_true, y_prob):
    mask = ~np.isnan(y_prob)
    y_t, y_p = y_true[mask], y_prob[mask]
    if y_t.sum() == 0:
        return 0.0, 1.0, 1.0, 1.0
    pr_auc = average_precision_score(y_t, y_p)
    precisions, recalls, thrs = precision_recall_curve(y_t, y_p)
    f1 = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1)
    return pr_auc, precisions[best_idx], recalls[best_idx], f1[best_idx]


def load_data():
    oot = pd.read_parquet(OOT_PATH)
    oot["target_date"] = pd.to_datetime(oot["target_date"])

    feat = pd.read_parquet(FEATURE_PATH)
    feat = feat.drop_duplicates(subset=["as_of_datetime_hkt"], keep="first")

    merged = oot.merge(
        feat[["as_of_datetime_hkt", "rh_current", "dew_point_c", "temp_current", "month"]],
        on="as_of_datetime_hkt", how="left",
    )

    merged["dew_point_spread"] = merged["temp_current"].fillna(merged["min_so_far_1m"]) - merged["dew_point_c"]

    merged["_season"] = "other"
    merged.loc[merged["month"].isin([12, 1, 2]), "_season"] = "winter"
    merged.loc[merged["month"].isin([6, 7, 8]), "_season"] = "summer"

    return merged


def compute_calibration(df):
    print("=" * 70)
    print("HOUR-BLOCK CALIBRATION")
    print("=" * 70)
    calib = {}
    for lo, hi, label in HOUR_BUCKETS:
        sub = df[(df["hour"] >= lo) & (df["hour"] < hi)]
        valid = sub.dropna(subset=["remaining_morning_downside"])
        if len(valid) < 50:
            continue
        residual = valid["remaining_morning_downside"].values - valid["downside_q50"].values
        p10_r = float(np.percentile(residual, 10))
        p90_r = float(np.percentile(residual, 90))
        calib[label] = {"p10": round(p10_r, 6), "p90": round(p90_r, 6), "n": int(len(valid))}
        print(f"  {label}: n={len(valid):,}  p10={p10_r:.4f}  p90={p90_r:.4f}")

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(calib, indent=2))
    print(f"  Saved to {CALIB_PATH}")

    print()
    print("CALIBRATED METRICS BY BUCKET")
    header = f"{'Bucket':>8s} {'n':>7s} {'COV80_orig':>10s} {'COV80_cal':>10s} {'PIW_orig':>9s} {'PIW_cal':>9s}"
    print(header)
    print("-" * len(header))
    for lo, hi, label in HOUR_BUCKETS:
        sub = df[(df["hour"] >= lo) & (df["hour"] < hi)]
        valid = sub.dropna(subset=["remaining_morning_downside"])
        if len(valid) < 50:
            continue
        actual = valid["remaining_morning_downside"].values
        q50 = valid["downside_q50"].values
        q10_orig = valid["downside_q10"].values
        q90_orig = valid["downside_q90"].values

        if label in calib:
            p10_r = calib[label]["p10"]
            p90_r = calib[label]["p90"]
            q10_cal = np.maximum(0.0, q50 + p10_r)
            q90_cal = np.maximum(q10_cal, q50 + p90_r)
        else:
            q10_cal, q90_cal = q10_orig, q90_orig

        cov80_orig = float(np.nanmean((actual >= q10_orig) & (actual <= q90_orig)))
        cov80_cal = float(np.nanmean((actual >= q10_cal) & (actual <= q90_cal)))
        piw_orig = float((q90_orig - q10_orig).mean())
        piw_cal = float((q90_cal - q10_cal).mean())
        print(f"  {label:>8s} {len(valid):>7,d} {cov80_orig:>9.1%} {cov80_cal:>9.1%} {piw_orig:>8.3f} {piw_cal:>8.3f}")

    return calib


def diagnostics_morning_low_reached(df):
    print()
    print("=" * 70)
    print("CLASSIFIER: morning_low_reached — BY HOUR")
    print("=" * 70)
    header = f"{'Buckets':>8s} {'n':>7s} {'pos':>7s} {'pos_rate':>9s} {'PR-AUC':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>5s}"
    print(header)
    print("-" * len(header))
    for lo, hi, label in HOUR_BUCKETS:
        sub = df[(df["hour"] >= lo) & (df["hour"] < hi)]
        if len(sub) == 0:
            continue
        y_true = sub["morning_low_reached"].values
        y_prob = sub["morning_low_reached_proba"].values
        pr_auc, prec, rec, f1 = pr_metrics(y_true, y_prob)
        pos_rate = y_true.mean()
        print(f"  {label:>8s} {len(sub):>7,d} {y_true.sum():>7,d} {pos_rate:>8.4f} {pr_auc:>7.4f} {prec:>6.4f} {rec:>7.4f} {f1:>5.4f}")
    y_true = df["morning_low_reached"].values
    y_prob = df["morning_low_reached_proba"].values
    pr_auc, prec, rec, f1 = pr_metrics(y_true, y_prob)
    pos_rate = y_true.mean()
    print(f"  {'ALL':>8s} {len(df):>7,d} {y_true.sum():>7,d} {pos_rate:>8.4f} {pr_auc:>7.4f} {prec:>6.4f} {rec:>7.4f} {f1:>5.4f}")


def diagnostics_survives_day(df):
    print()
    print("=" * 70)
    print("CLASSIFIER: morning_low_survives_day — BY REGIME")
    print("=" * 70)

    regimes = [
        ("ALL", slice(None)),
        ("No rain", df["rainfall_60m"] <= 0),
        ("Rain", df["rainfall_60m"] > 0),
        ("00-06 rain", (df["rainfall_60m"] > 0) & (df["hour"] >= 0) & (df["hour"] < 6)),
        ("High RH (>=80%)", df["rh_current"] >= 80),
        ("Low dew spread", df["dew_point_spread"] <= 2.0),
        ("Winter (DJF)", df["_season"] == "winter"),
        ("Summer (JJA)", df["_season"] == "summer"),
    ]

    header = f"{'Regime':>18s} {'n':>7s} {'pos':>7s} {'pos_rate':>9s} {'PR-AUC':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>5s}"
    print(header)
    print("-" * len(header))
    for name, mask in regimes:
        if isinstance(mask, slice):
            sub = df
        else:
            sub = df[mask]
        if len(sub) < 50:
            continue
        y_true = sub["morning_low_survives_day"].values
        y_prob = sub["morning_low_survives_day_proba"].values
        pr_auc, prec, rec, f1 = pr_metrics(y_true, y_prob)
        pos_rate = y_true.mean()
        print(f"  {name:>18s} {len(sub):>7,d} {y_true.sum():>7,d} {pos_rate:>8.4f} {pr_auc:>7.4f} {prec:>6.4f} {rec:>7.4f} {f1:>5.4f}")


def main():
    df = load_data()
    print(f"Loaded {len(df):,} OOT rows")
    print()

    compute_calibration(df)
    diagnostics_morning_low_reached(df)
    diagnostics_survives_day(df)

    print()
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
