"""Calibration experiment: per-regime interval calibration for Model B.

Goal: For each rain regime group, find a multiplicative factor f such that
  calibrated_q10 = q50 - f * (q50 - q10)
  calibrated_q90 = q50 + f * (q90 - q50)
yields COV80 ≈ 80% for that group.

Key constraint on rain rows: keep q50 unchanged, only adjust q10/q90 interval width.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/intraday_minute_ml_features.parquet")
RAIN_PATH = Path("data/hko_rainfall_15min_features.parquet")
MODEL_B_DIR = Path("models/intraday_minute_ml_model_b")

ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]
VALID_END = "2025-06-11"

RAIN_HEAVY_THRESHOLD = 5.0
DROP_FROM_MAX_THRESHOLD = 0.5
POST_PEAK_MINUTES_MIN = 30
POST_PEAK_MINUTES_MAX = 240

BASE_FEATURE_COLS = [
    "temp_current", "rh_current", "max_so_far_1m", "min_so_far_1m",
    "range_so_far_1m", "time_since_max_1m", "time_since_min_1m",
    "drop_from_max_1m", "rise_from_min_1m", "temp_change_5m", "temp_change_15m",
    "temp_change_30m", "temp_change_60m", "temp_acceleration_30m", "temp_std_30m",
    "temp_std_60m", "rh_change_15m", "rh_change_30m", "rh_change_60m",
    "rh_mean_30m", "rh_mean_60m", "rh_std_60m", "temp_x_rh", "dew_point_c",
    "dew_point_spread", "hour", "minute", "minutes_since_midnight", "month",
    "day_of_year", "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening", "is_night",
]

RAIN_FEATURE_COLS = [
    "rainfall_60m", "rainfall_120m",
    "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
    "rain_cooling_60m", "rain_cooling_120m",
    "post_peak_rain_flag", "morning_peak_rain_flag",
]

FEATURE_COLS = BASE_FEATURE_COLS + RAIN_FEATURE_COLS


def load_oot_data():
    logger.info("Loading minute features...")
    df = pd.read_parquet(DATA_PATH)
    df = df[df["minute"] % 5 == 0].reset_index(drop=True)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("as_of_datetime_hkt").reset_index(drop=True)

    oot = df[df["target_date"] >= VALID_END].copy()
    logger.info(f"OOT rows (pre-rain-merge): {len(oot):,}")

    rain = pd.read_parquet(RAIN_PATH)
    rain["datetime"] = pd.to_datetime(rain["datetime"])
    if rain["datetime"].dt.tz is None:
        rain = rain.assign(datetime=rain["datetime"].dt.tz_localize("Asia/Hong_Kong"))

    oot["_join_dt"] = oot["as_of_datetime_hkt"]
    rain_ff = rain[["datetime", "rainfall_60m", "rainfall_120m"]].sort_values("datetime")

    oot = pd.merge_asof(
        oot.sort_values("_join_dt"),
        rain_ff,
        left_on="_join_dt",
        right_on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )

    oot["rainfall_60m_missing_flag"] = oot["datetime"].isna().astype(int)
    oot["rainfall_120m_missing_flag"] = oot["datetime"].isna().astype(int)
    oot["rainfall_60m"] = oot["rainfall_60m"].fillna(0.0)
    oot["rainfall_120m"] = oot["rainfall_120m"].fillna(0.0)

    oot["rain_cooling_60m"] = np.where(
        oot["rainfall_60m"] > 0, np.maximum(0, -oot["temp_change_60m"]), 0.0)
    oot["rain_cooling_120m"] = np.where(
        oot["rainfall_120m"] > 0, np.maximum(0, -oot["temp_change_60m"]), 0.0)

    cond_pp = (
        (oot["rainfall_60m"] > RAIN_HEAVY_THRESHOLD)
        & (oot["drop_from_max_1m"] >= DROP_FROM_MAX_THRESHOLD)
        & (oot["time_since_max_1m"] >= POST_PEAK_MINUTES_MIN)
        & (oot["time_since_max_1m"] <= POST_PEAK_MINUTES_MAX)
    )
    oot["post_peak_rain_flag"] = cond_pp.astype(int)
    oot["morning_peak_rain_flag"] = (
        cond_pp & (oot["hour"] >= 9) & (oot["hour"] <= 14)
    ).astype(int)

    oot = oot.drop(columns=["_join_dt", "datetime"])

    for c in FEATURE_COLS:
        if c in df.columns:
            oot[c] = oot[c].fillna(0)

    logger.info(f"OOT rows after merge: {len(oot):,}")
    return oot


def load_models():
    models = {}
    for a in ALPHAS:
        key = f"upside_q{int(a*100)}"
        models[key] = lgb.Booster(model_file=str(MODEL_B_DIR / f"{key}.txt"))
    clf = lgb.Booster(model_file=str(MODEL_B_DIR / "upside_zero.txt"))
    with open(MODEL_B_DIR / "best_threshold.json") as f:
        thr = json.load(f)["upside_zero_threshold"]
    with open(MODEL_B_DIR / "feature_list.json") as f:
        feature_cols = json.load(f)
    return models, clf, thr, feature_cols


def predict_oot(models, clf, thr, feature_cols, df_oot):
    X = df_oot[feature_cols]
    df_out = df_oot[["target_date", "as_of_datetime_hkt", "max_so_far_1m",
                      "remaining_upside", "is_upside_zero", "official_tmax",
                      "hour", "rainfall_60m", "rainfall_120m",
                      "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
                      "rain_cooling_60m", "rain_cooling_120m",
                      "post_peak_rain_flag", "morning_peak_rain_flag"]].copy()

    for a in ALPHAS:
        key = f"upside_q{int(a*100)}"
        df_out[f"upside_q{int(a*100)}"] = models[key].predict(X)
    q50 = df_out["upside_q50"].values
    q10 = df_out["upside_q10"].values
    q90 = df_out["upside_q90"].values

    # enforce monotonicity
    qs = np.column_stack([df_out[f"upside_q{int(a*100)}"].values for a in ALPHAS])
    qs.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        df_out[f"upside_q{int(a*100)}"] = qs[:, i]

    for a in ALPHAS:
        df_out[f"pred_tmax_q{int(a*100)}"] = df_out["max_so_far_1m"] + df_out[f"upside_q{int(a*100)}"]

    zero_proba = clf.predict(X)
    df_out["zero_proba"] = zero_proba
    df_out["zero_pred"] = (zero_proba >= thr).astype(int)

    return df_out


def find_calibration_quantiles(actual, q50):
    """Return the empirical p10 and p90 of residuals (actual - q50).
    calibrated_q10 = q50 + residual_p10
    calibrated_q90 = q50 + residual_p90
    This guarantees 80% coverage on the calibration set.
    """
    residuals = actual - q50
    valid = ~np.isnan(residuals)
    r = residuals[valid]
    p10 = np.percentile(r, 10)
    p90 = np.percentile(r, 90)
    return p10, p90


def compute_coverage(actual, q10_cal, q50, q90_cal):
    inside = (actual >= q10_cal) & (actual <= q90_cal)
    return np.nanmean(inside), np.nanmean(np.abs(q50 - actual)), np.nanmean(q90_cal - q10_cal)


def main():
    logger.info("=" * 60)
    logger.info("Model B Calibration Experiment")
    logger.info("=" * 60)

    oot = load_oot_data()
    models, clf, thr, feature_cols = load_models()
    df = predict_oot(models, clf, thr, feature_cols, oot)

    valid = df[df["official_tmax"] == df["official_tmax"].copy()]
    logger.info(f"Valid OOT rows: {len(valid):,}")

    remaining = valid["remaining_upside"].values
    q10 = valid["upside_q10"].values
    q50 = valid["upside_q50"].values
    q90 = valid["upside_q90"].values

    # Define regime groups
    groups = {
        "ALL": pd.Series(True, index=valid.index),
        "rainfall_60m==0": valid["rainfall_60m"] == 0,
        "rainfall_60m>0": valid["rainfall_60m"] > 0,
        "rainfall_60m>5 (heavy)": valid["rainfall_60m"] > 5,
        "rainfall_120m>0": valid["rainfall_120m"] > 0,
        "post_peak_rain_flag=1": valid["post_peak_rain_flag"] == 1,
        "morning_peak_rain_flag=1": valid["morning_peak_rain_flag"] == 1,
        "rainfall_60m_missing=1": valid["rainfall_60m_missing_flag"] == 1,
    }

    results = []
    for name, mask in groups.items():
        n = mask.sum()
        if n < 10:
            continue
        g_actual = remaining[mask]
        g_q10 = q10[mask]
        g_q50 = q50[mask]
        g_q90 = q90[mask]

        # Current metrics
        current_cov = np.nanmean((g_actual >= g_q10) & (g_actual <= g_q90))
        current_mae = np.nanmean(np.abs(g_q50 - g_actual))
        current_piw = np.nanmean(g_q90 - g_q10)
        current_bias = np.nanmean(g_q50 - g_actual)

        # Find residual quantiles for calibration
        p10, p90 = find_calibration_quantiles(g_actual, g_q50)
        cal_q10 = g_q50 + p10
        cal_q90 = g_q50 + p90
        cal_cov, cal_mae, cal_piw = compute_coverage(g_actual, cal_q10, g_q50, cal_q90)

        results.append({
            "group": name, "n": n,
            "current_cov": current_cov, "current_mae": current_mae,
            "current_piw": current_piw, "current_bias": current_bias,
            "residual_p10": p10, "residual_p90": p90,
            "cal_cov": cal_cov, "cal_mae": cal_mae, "cal_piw": cal_piw,
        })

    # Print table
    print()
    print(f"{'Group':>30s}  {'n':>7s}  {'cur_cov':>7s}  {'cur_MAE':>7s}  "
          f"{'cur_PIW':>7s}  {'cur_bias':>7s}  {'res_p10':>7s}  "
          f"{'res_p90':>7s}  {'cal_cov':>7s}  {'cal_PIW':>7s}")
    print("  " + "-" * 110)
    for r in results:
        print(f"{r['group']:>30s}  {r['n']:>7,d}  {r['current_cov']:>7.4f}  "
              f"{r['current_mae']:>7.3f}  {r['current_piw']:>7.3f}  "
              f"{r['current_bias']:>+7.4f}  {r['residual_p10']:>7.4f}  "
              f"{r['residual_p90']:>7.4f}  {r['cal_cov']:>7.4f}  "
              f"{r['cal_piw']:>7.3f}")

    # Per-time-bucket breakdown for rain rows
    print()
    print("=" * 60)
    print("Rain rows (rainfall_60m>0) — per-hour bucket breakdown")
    print("=" * 60)
    rain_mask = valid["rainfall_60m"] > 0
    rain_df = valid[rain_mask].copy()
    rain_df["hour_bucket"] = pd.cut(rain_df["hour"], bins=[0, 6, 12, 18, 24],
                                     labels=["00-06", "06-12", "12-18", "18-24"], right=False)

    r_actual = rain_df["remaining_upside"].values
    r_q10 = rain_df["upside_q10"].values
    r_q50 = rain_df["upside_q50"].values
    r_q90 = rain_df["upside_q90"].values

    print(f"{'Bucket':>8s}  {'n':>6s}  {'cur_cov':>7s}  {'cur_MAE':>7s}  "
          f"{'cur_PIW':>7s}  {'cur_bias':>7s}  {'res_p10':>7s}  "
          f"{'res_p90':>7s}  {'cal_cov':>7s}  {'cal_PIW':>7s}")
    print("  " + "-" * 90)
    for label in ["00-06", "06-12", "12-18", "18-24"]:
        m = rain_df["hour_bucket"] == label
        n = m.sum()
        if n < 5:
            continue
        g_actual = r_actual[m]
        g_q10 = r_q10[m]
        g_q50 = r_q50[m]
        g_q90 = r_q90[m]
        cur_cov = np.nanmean((g_actual >= g_q10) & (g_actual <= g_q90))
        cur_mae = np.nanmean(np.abs(g_q50 - g_actual))
        cur_piw = np.nanmean(g_q90 - g_q10)
        cur_bias = np.nanmean(g_q50 - g_actual)
        p10, p90 = find_calibration_quantiles(g_actual, g_q50)
        cal_q10 = g_q50 + p10
        cal_q90 = g_q50 + p90
        cal_cov, _, cal_piw = compute_coverage(g_actual, cal_q10, g_q50, cal_q90)
        print(f"{label:>8s}  {n:>6,d}  {cur_cov:>7.4f}  {cur_mae:>7.3f}  "
              f"{cur_piw:>7.3f}  {cur_bias:>+7.4f}  {p10:>7.4f}  "
              f"{p90:>7.4f}  {cal_cov:>7.4f}  {cal_piw:>7.3f}")

    # Fine-grained: hour + rain regime calibration
    print()
    print("=" * 60)
    print("Fine-grained calibration: hour + rain/no-rain regime")
    print("=" * 60)
    valid_df = valid.copy()
    valid_df["hour_bucket"] = pd.cut(valid_df["hour"], bins=[0, 6, 12, 18, 24],
                                      labels=["00-06", "06-12", "12-18", "18-24"], right=False)
    valid_df["is_rain"] = valid_df["rainfall_60m"] > 0
    print(f"{'Hour':>8s}  {'Rain':>6s}  {'n':>7s}  {'cur_cov':>7s}  {'cur_bias':>7s}  "
          f"{'res_p10':>7s}  {'res_p90':>7s}  {'cal_cov':>7s}  {'cal_PIW':>7s}")
    print("  " + "-" * 80)
    for hour_label in ["00-06", "06-12", "12-18", "18-24"]:
        for rain_val, rain_label in [(False, "dry"), (True, "rain")]:
            m = (valid_df["hour_bucket"] == hour_label) & (valid_df["is_rain"] == rain_val)
            n = m.sum()
            if n < 10:
                continue
            g_actual = remaining[m]
            g_q50 = q50[m]
            cur_cov = np.nanmean((g_actual >= q10[m]) & (g_actual <= q90[m]))
            cur_bias = np.nanmean(g_q50 - g_actual)
            p10, p90 = find_calibration_quantiles(g_actual, g_q50)
            cal_q10 = g_q50 + p10
            cal_q90 = g_q50 + p90
            cal_cov, _, cal_piw = compute_coverage(g_actual, cal_q10, g_q50, cal_q90)
            print(f"{hour_label:>8s}  {rain_label:>6s}  {n:>7,d}  {cur_cov:>7.4f}  "
                  f"{cur_bias:>+7.4f}  {p10:>7.4f}  {p90:>7.4f}  "
                  f"{cal_cov:>7.4f}  {cal_piw:>7.3f}")

    # Save calibration factors as residual quantiles per hour+rain regime
    cal_data = {}
    for r in results:
        cal_data[r["group"]] = {"p10": r["residual_p10"], "p90": r["residual_p90"]}
    with open(MODEL_B_DIR / "calibration_residuals.json", "w") as f:
        json.dump(cal_data, f, indent=2)
    logger.info(f"Calibration residuals saved to {MODEL_B_DIR / 'calibration_residuals.json'}")

    # Save hierarchical calibration: per-hour + rain regime
    MIN_N = 500
    hour_bins = {"00-06": (0, 6), "06-12": (6, 12), "12-18": (12, 18), "18-24": (18, 24)}
    hierarchical = {}
    for hour_label, (h_lo, h_hi) in hour_bins.items():
        for rain_val, rain_key in [(False, "dry"), (True, "rain")]:
            m = (valid["hour"] >= h_lo) & (valid["hour"] < h_hi) & (valid["rainfall_60m"] > 0) == rain_val
            n = m.sum()
            if n >= 10:
                g_actual = remaining[m]
                g_q50 = q50[m]
                p10, p90 = find_calibration_quantiles(g_actual, g_q50)
                hierarchical[f"{hour_label}_{rain_key}"] = {
                    "n": int(n), "p10": p10, "p90": p90
                }
            else:
                hierarchical[f"{hour_label}_{rain_key}"] = {
                    "n": int(n), "p10": None, "p90": None
                }

    # Fallback: broad rain calibration (hour-agnostic)
    rain_all = valid["rainfall_60m"] > 0
    g_actual = remaining[rain_all]
    g_q50 = q50[rain_all]
    p10, p90 = find_calibration_quantiles(g_actual, g_q50)
    hierarchical["rain_fallback"] = {
        "n": int(rain_all.sum()), "p10": p10, "p90": p90
    }

    with open(MODEL_B_DIR / "calibration_hierarchical.json", "w") as f:
        json.dump(hierarchical, f, indent=2)
    logger.info(f"Hierarchical calibration saved to {MODEL_B_DIR / 'calibration_hierarchical.json'}")

    print()
    print("=" * 60)
    print("Hierarchical calibration (hour + rain regime)")
    print("=" * 60)
    print(f"{'Key':>20s}  {'n':>6s}  {'p10':>7s}  {'p90':>7s}")
    print("  " + "-" * 45)
    for key, v in sorted(hierarchical.items()):
        p10_str = f"{v['p10']:.4f}" if v['p10'] is not None else "  N/A  "
        p90_str = f"{v['p90']:.4f}" if v['p90'] is not None else "  N/A  "
        valid_flag = " *" if (v['n'] >= MIN_N and v['p10'] is not None) else "  "
        print(f"{key:>20s}  {v['n']:>6,d}  {p10_str:>7s}  {p90_str:>7s}{valid_flag}")


if __name__ == "__main__":
    main()
