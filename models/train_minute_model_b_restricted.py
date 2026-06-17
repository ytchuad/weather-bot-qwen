"""Train B_restricted — Model B restricted to >= 2023-06-01 (no pre-2023 zero-filled rows).

Control experiment: same features as Model B (38 temp+RH + 8 rainfall history), but
only rows with as_of_datetime_hkt >= 2023-06-01 (dropping pre-2023 zero-filled rows).
This isolates whether the rainfall features add value within the rainfall-available
period, controlling for training data size (same as A_restricted at 108K rows).

Compare: A_restricted (38 temp+RH features) vs B_restricted (38+8 rainfall features),
both trained on the same 108K-row training set (>= 2023-06-01).

LightGBM quantile regression (remaining_upside) + binary classifier (is_upside_zero).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/intraday_minute_ml_features.parquet")
RAIN_PATH = Path("data/hko_rainfall_15min_features.parquet")
MODEL_DIR = Path("models/intraday_minute_ml_b_restricted")

ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]
TRAIN_END = "2024-06-11"
VALID_END = "2025-06-11"

RAIN_HEAVY_THRESHOLD = 5.0
DROP_FROM_MAX_THRESHOLD = 0.5
POST_PEAK_MINUTES_MIN = 30
POST_PEAK_MINUTES_MAX = 240

LGB_PARAMS = dict(
    max_depth=6,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_data_in_leaf=300,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

BASE_FEATURE_COLS = [
    "temp_current",
    "rh_current",
    "max_so_far_1m",
    "min_so_far_1m",
    "range_so_far_1m",
    "time_since_max_1m",
    "time_since_min_1m",
    "drop_from_max_1m",
    "rise_from_min_1m",
    "temp_change_5m",
    "temp_change_15m",
    "temp_change_30m",
    "temp_change_60m",
    "temp_acceleration_30m",
    "temp_std_30m",
    "temp_std_60m",
    "rh_change_15m",
    "rh_change_30m",
    "rh_change_60m",
    "rh_mean_30m",
    "rh_mean_60m",
    "rh_std_60m",
    "temp_x_rh",
    "dew_point_c",
    "dew_point_spread",
    "hour",
    "minute",
    "minutes_since_midnight",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "is_night",
]

RAIN_FEATURE_COLS = [
    "rainfall_60m",
    "rainfall_120m",
    "rainfall_60m_missing_flag",
    "rainfall_120m_missing_flag",
    "rain_cooling_60m",
    "rain_cooling_120m",
    "post_peak_rain_flag",
    "morning_peak_rain_flag",
]

FEATURE_COLS = BASE_FEATURE_COLS + RAIN_FEATURE_COLS


def enforce_monotonicity(preds_dict):
    preds_matrix = np.column_stack([preds_dict[f"q{int(a*100)}"] for a in ALPHAS])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        preds_dict[f"q{int(a*100)}"] = preds_matrix[:, i]
    return preds_dict


def load_and_merge():
    logger.info("Loading minute features...")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    # 5-min deterministic grid
    before = len(df)
    df = df[df["minute"] % 5 == 0].reset_index(drop=True)
    logger.info(f"5-min grid: {len(df):,} rows (kept {len(df)/before*100:.0f}%)")

    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("as_of_datetime_hkt").reset_index(drop=True)

    # Restrict to rainfall data period (same as A_restricted)
    before_filter = len(df)
    df = df[df["as_of_datetime_hkt"] >= "2023-06-01"].reset_index(drop=True)
    logger.info(f"Restricted to >= 2023-06-01: {len(df):,} rows")

    # Load and merge rainfall features
    logger.info("Loading rainfall features...")
    rain = pd.read_parquet(RAIN_PATH)
    rain["datetime"] = pd.to_datetime(rain["datetime"])
    if rain["datetime"].dt.tz is None:
        rain = rain.assign(datetime=rain["datetime"].dt.tz_localize("Asia/Hong_Kong"))

    rain_start = rain["datetime"].min()
    rain_end = rain["datetime"].max()
    logger.info(f"Rainfall range: {rain_start} to {rain_end}")

    df["_join_dt"] = df["as_of_datetime_hkt"]
    rain_ff = rain[["datetime", "rainfall_60m", "rainfall_120m"]].sort_values("datetime")

    df = pd.merge_asof(
        df.sort_values("_join_dt"),
        rain_ff,
        left_on="_join_dt",
        right_on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )

    df["rainfall_60m_missing_flag"] = df["datetime"].isna().astype(int)
    df["rainfall_120m_missing_flag"] = df["datetime"].isna().astype(int)

    df["rainfall_60m"] = df["rainfall_60m"].fillna(0.0)
    df["rainfall_120m"] = df["rainfall_120m"].fillna(0.0)

    df["rain_cooling_60m"] = np.where(
        df["rainfall_60m"] > 0,
        np.maximum(0, -df["temp_change_60m"]),
        0.0,
    )
    df["rain_cooling_120m"] = np.where(
        df["rainfall_120m"] > 0,
        np.maximum(0, -df["temp_change_60m"]),
        0.0,
    )

    condition_post_peak = (
        (df["rainfall_60m"] > RAIN_HEAVY_THRESHOLD)
        & (df["drop_from_max_1m"] >= DROP_FROM_MAX_THRESHOLD)
        & (df["time_since_max_1m"] >= POST_PEAK_MINUTES_MIN)
        & (df["time_since_max_1m"] <= POST_PEAK_MINUTES_MAX)
    )
    df["post_peak_rain_flag"] = condition_post_peak.astype(int)
    df["morning_peak_rain_flag"] = (
        condition_post_peak & (df["hour"] >= 9) & (df["hour"] <= 14)
    ).astype(int)

    df = df.drop(columns=["_join_dt", "datetime"])

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    logger.info(f"Final dataset: {len(df):,} rows, rain_missing={df['rainfall_60m_missing_flag'].mean():.1%}")

    return df


def time_split(df):
    train = df[df["target_date"] < TRAIN_END].copy()
    valid = df[
        (df["target_date"] >= TRAIN_END) & (df["target_date"] < VALID_END)
    ].copy()
    oot = df[df["target_date"] >= VALID_END].copy()
    logger.info(
        f"Split: train={len(train):,}  valid={len(valid):,}  oot={len(oot):,}"
    )
    return train, valid, oot


def train_quantile_models(X_train, y_train, X_valid, y_valid):
    models = {}
    for alpha in ALPHAS:
        logger.info(f"Training upside quantile alpha={alpha}")
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, **LGB_PARAMS
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        key = f"upside_q{int(alpha*100)}"
        model.booster_.save_model(str(MODEL_DIR / f"{key}.txt"))
        logger.info(
            f"  {key}: best_iter={model.best_iteration_}, "
            f"valid_loss={model.best_score_['valid_0'][list(model.best_score_['valid_0'])[0]]:.4f}"
        )
        models[key] = model
    return models


def train_classifier(X_train, y_train, X_valid, y_valid):
    logger.info("Training upside_zero classifier")
    model = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(MODEL_DIR / "upside_zero.txt"))
    logger.info(
        f"  upside_zero: best_iter={model.best_iteration_}, "
        f"valid_loss={model.best_score_['valid_0'][list(model.best_score_['valid_0'])[0]]:.4f}"
    )
    return model


def tune_threshold(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_valid, proba)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    logger.info(
        f"Best classifier threshold={best_thr:.4f}  F1={f1_scores[best_idx]:.4f}  "
        f"Precision={precisions[best_idx]:.4f}  Recall={recalls[best_idx]:.4f}"
    )
    return float(best_thr)


def oot_predict(quantile_models, clf, df_oot, best_thr):
    X = df_oot[FEATURE_COLS]
    df_out = df_oot[["target_date", "as_of_datetime_hkt", "max_so_far_1m",
                      "remaining_upside", "is_upside_zero", "official_tmax",
                      "hour", "rainfall_60m"]].copy()

    preds = {}
    for a in ALPHAS:
        key = f"upside_q{int(a*100)}"
        preds[f"q{int(a*100)}"] = quantile_models[key].predict(X)
    preds = enforce_monotonicity(preds)

    for a in ALPHAS:
        df_out[f"upside_q{int(a*100)}"] = preds[f"q{int(a*100)}"]
        df_out[f"pred_tmax_q{int(a*100)}"] = df_out["max_so_far_1m"] + preds[f"q{int(a*100)}"]

    zero_proba = clf.predict_proba(X)[:, 1]
    df_out["zero_proba"] = zero_proba
    df_out["zero_pred"] = (zero_proba >= best_thr).astype(int)

    return df_out


def _bucket_metrics(sub, label):
    n = len(sub)
    n_dates = sub["target_date"].nunique() if "target_date" in sub.columns else 0
    if n == 0:
        return
    actual = sub["remaining_upside"].values
    actual_tmax = sub["official_tmax"].values
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
    q90_breach = np.nanmean(actual > q90)
    q10_breach = np.nanmean(actual < q10)

    logger.info(
        f"  [{label:>6s}]  "
        f"n={n:>7,}  "
        f"dates={n_dates:>5}  "
        f"MAE_up={mae_up:.3f}  "
        f"MAE_tx={mae_tx:.3f}  "
        f"cov80={cov80:.4f}  "
        f"PIW={piw:.3f}  "
        f"bias={bias_q50:+.4f}  "
        f"q90_br={q90_breach:.4f}  "
        f"q10_br={q10_breach:.4f}"
    )


def evaluate(df, label):
    n = len(df)
    if n == 0:
        logger.warning(f"Empty set: {label}")
        return

    logger.info(f"\n=== {label} (n={n:,}) ===")
    logger.info(
        f"  {'bucket':>6s}  {'n_rows':>7s}  {'dates':>5s}  "
        f"{'MAE_up':>6s}  {'MAE_tx':>6s}  {'cov80':>6s}  "
        f"{'PIW':>5s}  {'bias':>7s}  {'q90_br':>7s}  {'q10_br':>7s}"
    )
    logger.info("  " + "-" * 80)

    pr_auc = average_precision_score(df["is_upside_zero"], df["zero_proba"])
    prec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["zero_pred"].sum(), 1)
    rec = (df["zero_pred"] & df["is_upside_zero"]).sum() / max(df["is_upside_zero"].sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    logger.info(f"  Classifier: PR-AUC={pr_auc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    valid_mask = df["official_tmax"] == df["official_tmax"]
    bins = [(0, 6), (6, 12), (12, 18), (18, 24)]
    labels_bin = ["00-06", "06-12", "12-18", "18-24"]
    for (lo, hi), lb in zip(bins, labels_bin):
        mask = (df["hour"] >= lo) & (df["hour"] < hi) & valid_mask
        if mask.any():
            _bucket_metrics(df[mask], lb)

    _bucket_metrics(df[valid_mask], "ALL")


def evaluate_by_rain(df, label):
    if "rainfall_60m" not in df.columns:
        return
    valid = df[df["official_tmax"] == df["official_tmax"]]
    for rain_val, rain_label in [(0, "norain"), (1, "rain")]:
        mask = valid["rainfall_60m"] > 0 if rain_val else valid["rainfall_60m"] <= 0
        sub = valid[mask]
        if len(sub) > 0:
            _bucket_metrics(sub, rain_label)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge()
    train, valid, oot = time_split(df)
    del df

    X_train = train[FEATURE_COLS]
    y_train = train["remaining_upside"]
    y_train_zero = train["is_upside_zero"]

    X_valid = valid[FEATURE_COLS]
    y_valid = valid["remaining_upside"]
    y_valid_zero = valid["is_upside_zero"]

    logger.info("=" * 50)
    logger.info("TRAINING QUANTILE MODELS (B_restricted)")
    logger.info("=" * 50)
    quantile_models = train_quantile_models(
        X_train, y_train, X_valid, y_valid
    )

    logger.info("=" * 50)
    logger.info("TRAINING CLASSIFIER")
    logger.info("=" * 50)
    clf = train_classifier(
        X_train, y_train_zero, X_valid, y_valid_zero
    )

    best_thr = tune_threshold(clf, X_valid, y_valid_zero)

    logger.info("=" * 50)
    logger.info("OOT PREDICTIONS (B_restricted)")
    logger.info("=" * 50)
    df_oot = oot_predict(quantile_models, clf, oot, best_thr)

    evaluate(df_oot, "OOT Overall")
    evaluate_by_rain(df_oot, "OOT")

    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    with open(MODEL_DIR / "best_threshold.json", "w") as f:
        json.dump({"upside_zero_threshold": best_thr}, f)

    oot_out = MODEL_DIR / "oot_predictions.parquet"
    df_oot.to_parquet(oot_out, index=False)
    logger.info(f"OOT predictions saved to {oot_out}")

    logger.info("✅ B_restricted training complete")


if __name__ == "__main__":
    main()
