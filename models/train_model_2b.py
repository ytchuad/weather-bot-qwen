"""
Train Model 2B — observed-rainfall extension of Model 2A v2.

LightGBM quantile regression (remaining_upside) + binary classifier (is_upside_zero).
Two variants:
  Variant 1 (full):    Uses full history from Model 2A v2.
  Variant 2 (restricted): Uses rows with target_date >= 2023-06-01 only.

Outputs:
  models/intraday_minute_ai_model_2b/          (Variant 1)
  models/intraday_minute_ai_model_2b_restricted/ (Variant 2)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_STORE_PATH = Path("data/model_2b_feature_store.parquet")
MODEL_DIR_FULL = Path("models/intraday_minute_ai_model_2b")
MODEL_DIR_RESTRICTED = Path("models/intraday_minute_ai_model_2b_restricted")

TRAIN_END = "2024-06-11"
VALID_END = "2025-06-11"
RAIN_START = "2023-06-01"

ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

LGB_PARAMS = dict(
    max_depth=6,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_data_in_leaf=500,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
)

# Model 2A v2 features (45)
BASE_FEATURES = [
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
    "wind_offshore_highland_mean", "wind_offshore_highland_max",
    "wind_all_change_60m", "wind_kings_park_current",
    "minutes_since_midnight",
    "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening",
    "obs_data_age_minutes", "wind_data_age_minutes",
]

# Minimum Model 2B rainfall features
RAIN_FEATURES = [
    "rainfall_60m",
    "rainfall_120m",
    "has_recent_rainfall_obs",
    "rain_intensity_max_120m",
    "rain_cooling_60m",
    "rain_after_max_flag",
    "post_peak_rain_flag",
    "rain_data_gap_flag",
    "rainfall_data_age_minutes",
]

FEATURE_COLS = BASE_FEATURES + RAIN_FEATURES


def enforce_monotonicity(preds_dict):
    preds_matrix = np.column_stack([preds_dict[f"q{int(a*100)}"] for a in ALPHAS])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        preds_dict[f"q{int(a*100)}"] = preds_matrix[:, i]
    return preds_dict


def load_and_prepare():
    logger.info("Loading Model 2B feature store...")
    df = pd.read_parquet(FEATURE_STORE_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("decision_time").reset_index(drop=True)
    return df


def time_split(df):
    train = df[df["target_date"] < TRAIN_END].copy()
    valid = df[(df["target_date"] >= TRAIN_END) & (df["target_date"] < VALID_END)].copy()
    oot = df[df["target_date"] >= VALID_END].copy()
    logger.info(f"Split: train={len(train):,}  valid={len(valid):,}  oot={len(oot):,}")
    return train, valid, oot


def fill_feature_nulls(df):
    for c in FEATURE_COLS:
        if c in df.columns:
            df.loc[:, c] = df[c].fillna(0)
    return df


def train_quantile_models(X_train, y_train, X_valid, y_valid, model_dir):
    models = {}
    for alpha in ALPHAS:
        logger.info(f"Training upside quantile alpha={alpha}")
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        key = f"upside_q{int(alpha*100)}"
        model.booster_.save_model(str(model_dir / f"{key}.txt"))
        logger.info(f"  {key}: best_iter={model.best_iteration_}")
        models[key] = model
    return models


def train_classifier(X_train, y_train, X_valid, y_valid, model_dir):
    logger.info("Training upside_zero classifier")
    model = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(model_dir / "upside_zero.txt"))
    logger.info(f"  upside_zero: best_iter={model.best_iteration_}")
    return model


def tune_threshold(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_valid, proba)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    logger.info(f"Best classifier threshold={best_thr:.4f}  F1={f1_scores[best_idx]:.4f}")
    return float(best_thr)


def oot_predict(quantile_models, clf, df_oot, best_thr):
    X = df_oot[FEATURE_COLS].fillna(0)
    df_out = df_oot[["target_date", "decision_time", "max_so_far",
                      "remaining_upside", "is_upside_zero", "actual_high_today",
                      "hour"]].copy()

    # Add rain columns for slice evaluation
    for c in ["rainfall_60m", "rainfall_120m", "has_recent_rainfall_obs",
              "post_peak_rain_flag", "rain_after_max_flag", "rain_data_gap_flag",
              "heavy_recent_rain_flag", "drop_from_max", "time_since_max"]:
        if c in df_oot.columns:
            df_out[c] = df_oot[c].values

    preds = {}
    for a in ALPHAS:
        key = f"upside_q{int(a*100)}"
        preds[f"q{int(a*100)}"] = quantile_models[key].predict(X)
    preds = enforce_monotonicity(preds)

    for a in ALPHAS:
        df_out[f"upside_q{int(a*100)}"] = preds[f"q{int(a*100)}"]
        df_out[f"pred_tmax_q{int(a*100)}"] = df_out["max_so_far"] + preds[f"q{int(a*100)}"]

    zero_proba = clf.predict_proba(X)[:, 1]
    df_out["zero_proba"] = zero_proba
    df_out["zero_pred"] = (zero_proba >= best_thr).astype(int)

    return df_out


def bucket_metrics(sub, label):
    n = len(sub)
    if n == 0:
        return {}
    actual = sub["remaining_upside"].values
    actual_tmax = sub["actual_high_today"].values
    q50 = sub["upside_q50"].values
    q50_tmax = sub["pred_tmax_q50"].values
    q10 = sub["upside_q10"].values
    q90 = sub["upside_q90"].values

    mae_up = np.nanmean(np.abs(actual - q50))
    mae_tx = np.nanmean(np.abs(actual_tmax - q50_tmax))
    rmse = np.sqrt(np.nanmean((actual - q50) ** 2))
    bias = np.nanmean(q50 - actual)
    q50_breach = np.nanmean(actual > q50)

    return {
        "label": label,
        "n_rows": n,
        "n_dates": int(sub["target_date"].nunique()),
        "mae_upside": round(mae_up, 4),
        "mae_pred_tmax": round(mae_tx, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "q50_breach": round(q50_breach, 4),
    }


def evaluate(df, label):
    n = len(df)
    if n == 0:
        logger.warning(f"Empty set: {label}")
        return {}

    results = {}

    logger.info(f"\n=== {label} (n={n:,}) ===")

    pr_auc = average_precision_score(df["is_upside_zero"], df["zero_proba"])
    prec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["zero_pred"].sum(), 1)
    rec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["is_upside_zero"].sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    logger.info(f"  Classifier: PR-AUC={pr_auc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    results["classifier"] = {
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    # Hourly buckets
    valid_mask = df["actual_high_today"] == df["actual_high_today"]
    for lo, hi, lb in [(0, 6, "00-06"), (6, 9, "06-09"), (9, 12, "09-12"),
                        (12, 15, "12-15"), (15, 18, "15-18"), (18, 24, "18-24")]:
        mask = (df["hour"] >= lo) & (df["hour"] < hi) & valid_mask
        if mask.any():
            m = bucket_metrics(df[mask], lb)
            results[lb] = m

    results["ALL"] = bucket_metrics(df[valid_mask], "ALL")
    return results


def evaluate_by_rain_regime(df, label):
    """Evaluate on rain-specific slices."""
    results = {}
    if "rainfall_60m" not in df.columns:
        return results

    valid = df[df["actual_high_today"] == df["actual_high_today"]]

    slices = {
        "ALL": slice(None),
        "no_rain": (valid["rainfall_60m"] == 0),
        "recent_rain": (valid["has_recent_rainfall_obs"] == 1),
        "heavy_recent_rain": (valid["rainfall_60m"] >= 10),
        "post_peak_rain": (valid["post_peak_rain_flag"] == 1),
        "rain_after_max": (valid["rain_after_max_flag"] == 1),
    }
    for lo, hi, lb in [(6, 9, "06-09_rain"), (9, 12, "09-12_rain"),
                        (12, 15, "12-15_rain"), (15, 18, "15-18_rain")]:
        mask = (valid["hour"] >= lo) & (valid["hour"] < hi) & (valid["has_recent_rainfall_obs"] == 1)
        slices[lb] = mask

    for name, mask in slices.items():
        if name == "ALL":
            sub = valid
        else:
            sub = valid[mask]
        if len(sub) == 0:
            continue
        m = bucket_metrics(sub, name)
        results[name] = m

    return results


def train_variant(df, model_dir, variant_name, restrict=False):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING {variant_name}")
    logger.info(f"{'=' * 60}")

    model_dir.mkdir(parents=True, exist_ok=True)

    if restrict:
        before = len(df)
        df = df[df["target_date"] >= RAIN_START].copy()
        logger.info(f"Restricted to >= {RAIN_START}: {len(df):,} rows (from {before:,})")

    train, valid, oot = time_split(df)

    for name, split in [("train", train), ("valid", valid), ("oot", oot)]:
        split[FEATURE_COLS] = fill_feature_nulls(split[FEATURE_COLS])

    X_train = train[FEATURE_COLS]
    y_train = train["remaining_upside"]
    y_train_zero = train["is_upside_zero"]

    X_valid = valid[FEATURE_COLS]
    y_valid = valid["remaining_upside"]
    y_valid_zero = valid["is_upside_zero"]

    logger.info("Training quantile models...")
    quantile_models = train_quantile_models(X_train, y_train, X_valid, y_valid, model_dir)

    logger.info("Training classifier...")
    clf = train_classifier(X_train, y_train_zero, X_valid, y_valid_zero, model_dir)
    best_thr = tune_threshold(clf, X_valid, y_valid_zero)

    logger.info("OOT predictions...")
    df_oot = oot_predict(quantile_models, clf, oot, best_thr)

    eval_results = evaluate(df_oot, f"OOT {variant_name}")
    rain_results = evaluate_by_rain_regime(df_oot, f"OOT {variant_name}")

    # Save feature list
    with open(model_dir / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(model_dir / "best_threshold.json", "w") as f:
        json.dump({"upside_zero_threshold": best_thr}, f)

    oot_out = model_dir / "oot_predictions.parquet"
    df_oot.to_parquet(oot_out, index=False)
    logger.info(f"OOT saved to {oot_out}")

    return df_oot, eval_results, rain_results


def main():
    df = load_and_prepare()

    # Variant 1: full history
    df_oot_full, eval_full, rain_full = train_variant(
        df, MODEL_DIR_FULL, "Model 2B full", restrict=False
    )

    # Variant 2: restricted to >= 2023-06-01
    df_oot_restricted, eval_restricted, rain_restricted = train_variant(
        df, MODEL_DIR_RESTRICTED, "Model 2B restricted", restrict=True
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("Training complete!")
    logger.info(f"  Full model: {MODEL_DIR_FULL}")
    logger.info(f"  Restricted model: {MODEL_DIR_RESTRICTED}")
    logger.info(f"  Features: {len(FEATURE_COLS)} ({len(BASE_FEATURES)} base + {len(RAIN_FEATURES)} rain)")


if __name__ == "__main__":
    main()
