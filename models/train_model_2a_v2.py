"""
Train Model 2A v2 — Core baseline with minute obs + forecast + wind features.

LightGBM quantile regression (remaining_upside) + binary classifier (is_upside_zero).
Output: models/intraday_minute_ml_model_2a_v2/{upside_q*.txt, upside_zero.txt, ...}

Changes from v1:
- wind_offshore_highland_mean/max replaces wind_highland_mean/max
- wind_offshore_highland represents the merged group (offshore + highland + 離岸及高地)
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

FEATURE_STORE_PATH = Path("data/model_2a_feature_store.parquet")
MODEL_DIR = Path("models/intraday_minute_ml_model_2a_v2")

TRAIN_END = "2024-06-11"
VALID_END = "2025-06-11"

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

MODEL_2A_MIN_FEATURES = [
    # current state (5)
    "temp_current", "rh_current", "pressure_current",
    "dew_point_current", "dew_point_spread",
    # anchor (5)
    "max_so_far", "min_so_far", "range_so_far",
    "drop_from_max", "time_since_max",
    # temperature trend (6)
    "temp_change_30m", "temp_change_60m",
    "temp_slope_30m", "temp_slope_60m",
    "temp_acceleration_60m", "temp_volatility_60m",
    # moisture / pressure (5)
    "rh_change_60m", "dew_point_change_60m",
    "dew_point_spread_change_60m",
    "pressure_change_60m", "pressure_change_180m",
    # forecast (6)
    "forecast_min_temp", "forecast_max_temp", "forecast_range",
    "forecast_gap_from_max_so_far",
    "forecast_age_minutes", "forecast_lead_days",
    # wind (8) - v2: offshore_highland replaces highland
    "wind_ref_mean", "wind_ref_max",
    "wind_victoria_harbour_mean", "wind_victoria_harbour_max",
    "wind_offshore_highland_mean", "wind_offshore_highland_max",
    "wind_all_change_60m", "wind_kings_park_current",
    # time (7)
    "minutes_since_midnight",
    "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening",
    # freshness (2)
    "obs_data_age_minutes", "wind_data_age_minutes",
]

FEATURE_COLS = MODEL_2A_MIN_FEATURES


def enforce_monotonicity(preds_dict):
    preds_matrix = np.column_stack([preds_dict[f"q{int(a*100)}"] for a in ALPHAS])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        preds_dict[f"q{int(a*100)}"] = preds_matrix[:, i]
    return preds_dict


def load_and_prepare():
    logger.info("Loading Model 2A v2 feature store...")
    df = pd.read_parquet(FEATURE_STORE_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    df['target_date'] = pd.to_datetime(df['target_date'])
    df = df.sort_values('decision_time').reset_index(drop=True)
    return df


def time_split(df):
    train = df[df['target_date'] < TRAIN_END].copy()
    valid = df[(df['target_date'] >= TRAIN_END) & (df['target_date'] < VALID_END)].copy()
    oot = df[df['target_date'] >= VALID_END].copy()
    logger.info(f"Split: train={len(train):,}  valid={len(valid):,}  oot={len(oot):,}")
    return train, valid, oot


def fill_feature_nulls(df):
    for c in FEATURE_COLS:
        if c in df.columns:
            df.loc[:, c] = df[c].fillna(0)
    return df


def train_quantile_models(X_train, y_train, X_valid, y_valid):
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
        model.booster_.save_model(str(MODEL_DIR / f"{key}.txt"))
        logger.info(f"  {key}: best_iter={model.best_iteration_}")
        models[key] = model
    return models


def train_classifier(X_train, y_train, X_valid, y_valid):
    logger.info("Training upside_zero classifier")
    model = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(MODEL_DIR / "upside_zero.txt"))
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
        return
    actual = sub["remaining_upside"].values
    actual_tmax = sub["actual_high_today"].values
    q50 = sub["upside_q50"].values
    q50_tmax = sub["pred_tmax_q50"].values
    q10 = sub["upside_q10"].values
    q90 = sub["upside_q90"].values

    mae_up = np.nanmean(np.abs(actual - q50))
    mae_tx = np.nanmean(np.abs(actual_tmax - q50_tmax))
    inside = (actual >= q10) & (actual <= q90)
    cov80 = np.nanmean(inside)
    piw = (q90 - q10).mean()
    bias_q50 = np.nanmean(q50 - actual)
    q90_br = np.nanmean(actual > q90)
    q10_br = np.nanmean(actual < q10)

    logger.info(
        f"  [{label:>6s}]  "
        f"n={n:>7,}  "
        f"MAE_up={mae_up:.3f}  "
        f"MAE_tx={mae_tx:.3f}  "
        f"cov80={cov80:.4f}  "
        f"PIW={piw:.3f}  "
        f"bias={bias_q50:+.4f}  "
        f"q90_br={q90_br:.4f}  "
        f"q10_br={q10_br:.4f}"
    )


def evaluate(df, label):
    n = len(df)
    if n == 0:
        logger.warning(f"Empty set: {label}")
        return

    logger.info(f"\n=== {label} (n={n:,}) ===")
    logger.info(f"  {'bucket':>6s}  {'n_rows':>7s}  {'MAE_up':>6s}  {'MAE_tx':>6s}  "
                f"{'cov80':>6s}  {'PIW':>5s}  {'bias':>7s}  {'q90_br':>7s}  {'q10_br':>7s}")
    logger.info("  " + "-" * 75)

    pr_auc = average_precision_score(df["is_upside_zero"], df["zero_proba"])
    prec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["zero_pred"].sum(), 1)
    rec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["is_upside_zero"].sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    logger.info(f"  Classifier: PR-AUC={pr_auc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    valid_mask = df["actual_high_today"] == df["actual_high_today"]
    for lo, hi, lb in [(0, 6, "00-06"), (6, 9, "06-09"), (9, 12, "09-12"),
                        (12, 15, "12-15"), (15, 18, "15-18"), (18, 24, "18-24")]:
        mask = (df["hour"] >= lo) & (df["hour"] < hi) & valid_mask
        if mask.any():
            bucket_metrics(df[mask], lb)

    bucket_metrics(df[valid_mask], "ALL")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()
    train, valid, oot = time_split(df)
    del df

    for name, split in [("train", train), ("valid", valid), ("oot", oot)]:
        split[FEATURE_COLS] = fill_feature_nulls(split[FEATURE_COLS])

    X_train = train[FEATURE_COLS]
    y_train = train["remaining_upside"]
    y_train_zero = train["is_upside_zero"]

    X_valid = valid[FEATURE_COLS]
    y_valid = valid["remaining_upside"]
    y_valid_zero = valid["is_upside_zero"]

    logger.info("=" * 50)
    logger.info("TRAINING QUANTILE MODELS (Model 2A v2)")
    logger.info("=" * 50)
    quantile_models = train_quantile_models(X_train, y_train, X_valid, y_valid)

    logger.info("=" * 50)
    logger.info("TRAINING CLASSIFIER (Model 2A v2)")
    logger.info("=" * 50)
    clf = train_classifier(X_train, y_train_zero, X_valid, y_valid_zero)
    best_thr = tune_threshold(clf, X_valid, y_valid_zero)

    logger.info("=" * 50)
    logger.info("OOT PREDICTIONS (Model 2A v2)")
    logger.info("=" * 50)
    df_oot = oot_predict(quantile_models, clf, oot, best_thr)
    evaluate(df_oot, "OOT Overall")

    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(MODEL_DIR / "best_threshold.json", "w") as f:
        json.dump({"upside_zero_threshold": best_thr}, f)

    oot_out = MODEL_DIR / "oot_predictions.parquet"
    df_oot.to_parquet(oot_out, index=False)
    logger.info(f"OOT saved to {oot_out}")

    logger.info(f"🎉 Model 2A v2 training complete! Features: {len(FEATURE_COLS)}")


if __name__ == "__main__":
    main()