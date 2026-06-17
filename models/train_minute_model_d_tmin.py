"""Train Model D Tmin — 117 features (38 base + 8 rain + 37 nowcast + 34 cross-midnight/evening).
Stage 1: 3 auxiliary classifiers (will_make_new_low, is_downside_zero, tmin_timing_bucket)
Stage 2: 5 downside quantile regressors on ALL rows (no conditional subsetting)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/intraday_minute_ml_features_tmin_d.parquet")
RAIN_PATH = Path("data/hko_rainfall_15min_features.parquet")
NOWCAST_PATH = Path("data/features/rainfall_nowcast/rainfall_nowcast_station_features_wide_all.parquet")
NOWCAST_AGE_MINUTES_TOLERANCE = 60
MODEL_DIR = Path("models/intraday_minute_ml_model_d_tmin")

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
    "temp_current", "rh_current",
    "max_so_far_1m", "min_so_far_1m", "range_so_far_1m",
    "time_since_max_1m", "time_since_min_1m",
    "drop_from_max_1m", "rise_from_min_1m",
    "temp_change_5m", "temp_change_15m", "temp_change_30m", "temp_change_60m",
    "temp_acceleration_30m", "temp_std_30m", "temp_std_60m",
    "rh_change_15m", "rh_change_30m", "rh_change_60m",
    "rh_mean_30m", "rh_mean_60m", "rh_std_60m",
    "temp_x_rh", "dew_point_c", "dew_point_spread",
    "hour", "minute", "minutes_since_midnight",
    "month", "day_of_year",
    "month_sin", "month_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening", "is_night",
]

RAIN_FEATURE_COLS = [
    "rainfall_60m", "rainfall_120m",
    "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
    "rain_cooling_60m", "rain_cooling_120m",
    "post_peak_rain_flag", "morning_peak_rain_flag",
]

NOWCAST_FEATURE_COLS_RAW = [
    "rain_nc_nearest_mm_sum_30m", "rain_nc_nearest_mm_sum_60m",
    "rain_nc_nearest_mm_sum_90m", "rain_nc_nearest_mm_sum_120m",
    "rain_nc_mean_r5km_sum_30m", "rain_nc_mean_r5km_sum_60m",
    "rain_nc_mean_r5km_sum_90m", "rain_nc_mean_r5km_sum_120m",
    "rain_nc_max_r5km_sum_30m", "rain_nc_max_r5km_sum_60m",
    "rain_nc_max_r5km_sum_90m", "rain_nc_max_r5km_sum_120m",
    "rain_nc_min_r5km_sum_30m", "rain_nc_min_r5km_sum_60m",
    "rain_nc_min_r5km_sum_90m", "rain_nc_min_r5km_sum_120m",
    "rain_nc_p90_r5km_sum_30m", "rain_nc_p90_r5km_sum_60m",
    "rain_nc_p90_r5km_sum_90m", "rain_nc_p90_r5km_sum_120m",
    "rain_nc_area_gt0_r5km_sum_30m", "rain_nc_area_gt0_r5km_sum_60m",
    "rain_nc_area_gt0_r5km_sum_90m", "rain_nc_area_gt0_r5km_sum_120m",
    "rain_nc_area_gt5_r5km_sum_30m", "rain_nc_area_gt5_r5km_sum_60m",
    "rain_nc_area_gt5_r5km_sum_90m", "rain_nc_area_gt5_r5km_sum_120m",
    "rain_nc_sum_0_60m", "rain_nc_sum_0_120m",
    "rain_nc_front_loaded_ratio", "rain_nc_any_0_120m",
    "rain_nc_heavy_0_120m", "rain_nc_valid_horizon_count",
    "rain_nc_missing_flag",
]

NOWCAST_FEATURE_COLS_DERIVED = [
    "rain_nowcast_age_minutes", "rain_nowcast_missing_flag",
]

NOWCAST_FEATURE_COLS = NOWCAST_FEATURE_COLS_RAW + NOWCAST_FEATURE_COLS_DERIVED

MODEL_D_FEATURE_COLS = [
    "temp_change_180m_crossday", "temp_change_360m_crossday", "temp_change_720m_crossday",
    "temp_slope_360m_crossday", "temp_slope_720m_crossday",
    "temp_min_360m_crossday", "temp_min_720m_crossday",
    "temp_range_360m_crossday", "temp_std_360m_crossday",
    "dew_point_spread_min_360m", "rh_mean_360m", "rh_max_360m",
    "prev_18_temp", "prev_21_temp", "prev_2359_temp",
    "prev_evening_temp_change", "prev_evening_temp_min",
    "prev_evening_temp_range", "prev_evening_temp_slope",
    "prev_evening_rh_mean", "prev_evening_rh_max",
    "prev_evening_dew_point_mean",
    "prev_evening_rainfall_18_24", "prev_evening_rain_flag",
    "cooling_since_prev_18", "cooling_since_prev_21",
    "distance_to_prev_evening_min", "dew_point_floor_gap",
    "is_before_evening_cooling_window", "daytime_warming_so_far",
    "afternoon_temp_drop_60m", "afternoon_temp_drop_120m",
]

FEATURE_COLS = BASE_FEATURE_COLS + RAIN_FEATURE_COLS + NOWCAST_FEATURE_COLS + MODEL_D_FEATURE_COLS


def enforce_monotonicity(preds_dict):
    preds_matrix = np.column_stack([preds_dict[f"q{int(a*100)}"] for a in ALPHAS])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        preds_dict[f"q{int(a*100)}"] = preds_matrix[:, i]
    return preds_dict


def load_and_merge():
    logger.info("Loading Model D features...")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} cols")

    before = len(df)
    df = df[df["minute"] % 5 == 0].reset_index(drop=True)
    logger.info(f"5-min grid: {len(df):,} rows (kept {len(df)/before*100:.0f}%)")

    if not isinstance(df["target_date"].iloc[0], str):
        df["target_date"] = df["target_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("as_of_datetime_hkt").reset_index(drop=True)

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

    logger.info("Loading nowcast features...")
    nc = pd.read_parquet(NOWCAST_PATH)
    nc["issue_time"] = pd.to_datetime(nc["issue_time"])
    if nc["issue_time"].dt.tz is None:
        nc = nc.assign(issue_time=nc["issue_time"].dt.tz_localize("Asia/Hong_Kong"))

    nc_cols_keep = ["issue_time"] + NOWCAST_FEATURE_COLS_RAW
    nc = nc[nc_cols_keep].sort_values("issue_time")
    nc = nc.drop_duplicates(subset="issue_time", keep="last")

    nc_start = nc["issue_time"].min()
    nc_end = nc["issue_time"].max()
    logger.info(f"Nowcast range: {nc_start} to {nc_end}")
    logger.info(f"Nowcast features: {len(NOWCAST_FEATURE_COLS_RAW)} raw columns")

    nc_ff = nc.rename(columns={"issue_time": "_join_nc_dt"})
    nc_ff = nc_ff.sort_values("_join_nc_dt")

    df["_join_nc_dt"] = df["as_of_datetime_hkt"]
    df = pd.merge_asof(
        df.sort_values("_join_nc_dt"),
        nc_ff,
        left_on="_join_nc_dt",
        right_on="_join_nc_dt",
        direction="backward",
    )

    df["rain_nowcast_age_minutes"] = (
        (df["as_of_datetime_hkt"] - df["_join_nc_dt"]).dt.total_seconds() / 60
    ).fillna(-1).astype(int)
    df["rain_nowcast_missing_flag"] = (
        (df["_join_nc_dt"].isna())
        | (df["rain_nowcast_age_minutes"] > NOWCAST_AGE_MINUTES_TOLERANCE)
    ).astype(int)

    for c in NOWCAST_FEATURE_COLS_RAW:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.drop(columns=["_join_nc_dt"])

    all_features = BASE_FEATURE_COLS + RAIN_FEATURE_COLS + NOWCAST_FEATURE_COLS_RAW + NOWCAST_FEATURE_COLS_DERIVED + MODEL_D_FEATURE_COLS
    for c in all_features:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    logger.info(f"Final dataset: {len(df):,} rows")
    logger.info(f"  rain_missing={df['rainfall_60m_missing_flag'].mean():.1%}")
    logger.info(f"  nowcast_missing={df['rain_nowcast_missing_flag'].mean():.1%}")
    logger.info(f"  Feature count: {len(FEATURE_COLS)} ({len(BASE_FEATURE_COLS)} base + {len(RAIN_FEATURE_COLS)} rain + {len(NOWCAST_FEATURE_COLS)} nowcast + {len(MODEL_D_FEATURE_COLS)} model_d)")

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
        logger.info(f"Training downside quantile alpha={alpha}")
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, **LGB_PARAMS
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        key = f"downside_q{int(alpha*100)}"
        model.booster_.save_model(str(MODEL_DIR / f"{key}.txt"))
        logger.info(
            f"  {key}: best_iter={model.best_iteration_}, "
            f"valid_loss={model.best_score_['valid_0'][list(model.best_score_['valid_0'])[0]]:.4f}"
        )
        models[key] = model
    return models


def train_binary_classifier(X_train, y_train, X_valid, y_valid, name):
    logger.info(f"Training binary classifier: {name}")
    model = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(MODEL_DIR / f"{name}.txt"))
    logger.info(
        f"  {name}: best_iter={model.best_iteration_}, "
        f"valid_loss={model.best_score_['valid_0'][list(model.best_score_['valid_0'])[0]]:.4f}"
    )
    return model


def train_multiclass_classifier(X_train, y_train, X_valid, y_valid):
    logger.info("Training multiclass classifier: tmin_timing_bucket")
    params = {**LGB_PARAMS}
    valid_labels = y_valid.values.ravel()
    valid_labels = np.clip(valid_labels, 0, 2)
    y_valid_cln = pd.Series(valid_labels, index=y_valid.index)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid_cln)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(MODEL_DIR / "tmin_timing_clf.txt"))
    logger.info(
        f"  tmin_timing_clf: best_iter={model.best_iteration_}, "
        f"valid_loss={model.best_score_['valid_0'][list(model.best_score_['valid_0'])[0]]:.4f}"
    )
    return model


def tune_binary_threshold(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_valid, proba)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    return best_thr, f1_scores[best_idx], precisions[best_idx], recalls[best_idx]


def oot_predict(quantile_models, clf_dict, df_oot, thresholds):
    X = df_oot[FEATURE_COLS]
    df_out = df_oot[["target_date", "as_of_datetime_hkt", "min_so_far_1m",
                      "remaining_downside", "is_downside_zero", "official_tmin",
                      "hour", "rainfall_60m"]].copy()

    preds = {}
    for a in ALPHAS:
        key = f"downside_q{int(a*100)}"
        preds[f"q{int(a*100)}"] = quantile_models[key].predict(X)
    preds = enforce_monotonicity(preds)

    for a in ALPHAS:
        df_out[f"downside_q{int(a*100)}"] = preds[f"q{int(a*100)}"]
        df_out[f"pred_tmin_q{int(a*100)}"] = df_out["min_so_far_1m"] - preds[f"q{int(a*100)}"]

    df_out["will_make_new_low_proba"] = clf_dict["will_make_new_low_clf"].predict_proba(X)[:, 1]
    df_out["will_make_new_low_pred"] = (
        df_out["will_make_new_low_proba"] >= thresholds["will_make_new_low_clf"]
    ).astype(int)

    df_out["zero_proba"] = clf_dict["is_downside_zero_clf"].predict_proba(X)[:, 1]
    df_out["zero_pred"] = (
        df_out["zero_proba"] >= thresholds["is_downside_zero_clf"]
    ).astype(int)

    timing_proba = clf_dict["tmin_timing_clf"].predict_proba(X)
    df_out["tmin_timing_proba_0"] = timing_proba[:, 0]
    df_out["tmin_timing_proba_1"] = timing_proba[:, 1]
    df_out["tmin_timing_proba_2"] = timing_proba[:, 2]
    df_out["tmin_timing_pred"] = np.argmax(timing_proba, axis=1)

    return df_out


def _bucket_metrics(sub, label):
    n = len(sub)
    n_dates = sub["target_date"].nunique() if "target_date" in sub.columns else 0
    if n == 0:
        return
    actual = sub["remaining_downside"].values
    actual_tmin = sub["official_tmin"].values
    q50 = sub["downside_q50"].values
    q50_tmin = sub["pred_tmin_q50"].values
    q10 = sub["downside_q10"].values
    q90 = sub["downside_q90"].values

    mae_down = np.nanmean(np.abs(actual - q50))
    mae_tn = np.nanmean(np.abs(actual_tmin - q50_tmin))
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
        f"MAE_down={mae_down:.3f}  "
        f"MAE_tn={mae_tn:.3f}  "
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
        f"{'MAE_down':>8s}  {'MAE_tn':>6s}  {'cov80':>6s}  "
        f"{'PIW':>5s}  {'bias':>7s}  {'q90_br':>7s}  {'q10_br':>7s}"
    )
    logger.info("  " + "-" * 80)

    pr_auc = average_precision_score(df["is_downside_zero"], df["zero_proba"])
    prec = (df["zero_pred"] & df["is_downside_zero"]).sum() / max(df["zero_pred"].sum(), 1)
    rec = (df["zero_pred"] & df["is_downside_zero"]).sum() / max(df["is_downside_zero"].sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    logger.info(f"  is_downside_zero: PR-AUC={pr_auc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    new_low_actual = df.get("will_make_new_low_after_now", pd.Series(0, index=df.index))
    if new_low_actual.sum() > 0:
        pr_auc_nl = average_precision_score(new_low_actual, df["will_make_new_low_proba"])
        prec_nl = (df["will_make_new_low_pred"] & new_low_actual.astype(bool)).sum() / max(df["will_make_new_low_pred"].sum(), 1)
        rec_nl = (df["will_make_new_low_pred"] & new_low_actual.astype(bool)).sum() / max(new_low_actual.sum(), 1)
        f1_nl = 2 * prec_nl * rec_nl / max(prec_nl + rec_nl, 1e-9)
        logger.info(f"  will_make_new_low: PR-AUC={pr_auc_nl:.4f}  P={prec_nl:.4f}  R={rec_nl:.4f}  F1={f1_nl:.4f}")

    timing_actual = df.get("tmin_timing_bucket", pd.Series(-1, index=df.index))
    if (timing_actual >= 0).sum() > 0:
        acc = accuracy_score(timing_actual[timing_actual >= 0], df.loc[timing_actual >= 0, "tmin_timing_pred"])
        logger.info(f"  tmin_timing_bucket: accuracy={acc:.4f}")

    valid_mask = df["official_tmin"] == df["official_tmin"]
    bins = [(0, 6), (6, 12), (12, 18), (18, 24)]
    labels_bin = ["00-06", "06-12", "12-18", "18-24"]
    for (lo, hi), lb in zip(bins, labels_bin):
        mask = (df["hour"] >= lo) & (df["hour"] < hi) & valid_mask
        if mask.any():
            _bucket_metrics(df[mask], lb)

    _bucket_metrics(df[valid_mask], "ALL")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge()
    train, valid, oot = time_split(df)
    del df

    X_train = train[FEATURE_COLS]
    y_train = train["remaining_downside"]
    y_train_zero = train["is_downside_zero"]
    y_train_newlow = train["will_make_new_low_after_now"]
    y_train_timing = train["tmin_timing_bucket"].clip(0, 2)

    X_valid = valid[FEATURE_COLS]
    y_valid = valid["remaining_downside"]
    y_valid_zero = valid["is_downside_zero"]
    y_valid_newlow = valid["will_make_new_low_after_now"]
    y_valid_timing = valid["tmin_timing_bucket"].clip(0, 2)

    logger.info("=" * 60)
    logger.info("STAGE 1: TRAINING AUXILIARY CLASSIFIERS (Model D Tmin)")
    logger.info("=" * 60)

    will_clf = train_binary_classifier(
        X_train, y_train_newlow, X_valid, y_valid_newlow, "will_make_new_low_clf"
    )
    zero_clf = train_binary_classifier(
        X_train, y_train_zero, X_valid, y_valid_zero, "is_downside_zero_clf"
    )
    timing_clf = train_multiclass_classifier(
        X_train, y_train_timing, X_valid, y_valid_timing
    )

    thr_will, f1_w, prec_w, rec_w = tune_binary_threshold(will_clf, X_valid, y_valid_newlow)
    logger.info(f"  will_make_new_low threshold={thr_will:.4f} F1={f1_w:.4f} P={prec_w:.4f} R={rec_w:.4f}")

    thr_zero, f1_z, prec_z, rec_z = tune_binary_threshold(zero_clf, X_valid, y_valid_zero)
    logger.info(f"  is_downside_zero threshold={thr_zero:.4f} F1={f1_z:.4f} P={prec_z:.4f} R={rec_z:.4f}")

    thresholds = {
        "will_make_new_low_clf": thr_will,
        "is_downside_zero_clf": thr_zero,
    }

    logger.info("=" * 60)
    logger.info("STAGE 2: TRAINING QUANTILE MODELS (Model D Tmin, ALL rows)")
    logger.info("=" * 60)
    quantile_models = train_quantile_models(
        X_train, y_train, X_valid, y_valid
    )

    logger.info("=" * 60)
    logger.info("OOT PREDICTIONS (Model D Tmin)")
    logger.info("=" * 60)
    clf_dict = {
        "will_make_new_low_clf": will_clf,
        "is_downside_zero_clf": zero_clf,
        "tmin_timing_clf": timing_clf,
    }
    df_oot = oot_predict(quantile_models, clf_dict, oot, thresholds)

    evaluate(df_oot, "OOT Overall")

    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    with open(MODEL_DIR / "classifier_thresholds.json", "w") as f:
        json.dump(thresholds, f)

    oot_out = MODEL_DIR / "oot_predictions.parquet"
    df_oot.to_parquet(oot_out, index=False)
    logger.info(f"OOT predictions saved to {oot_out}")

    nc_missing = df_oot["rain_nowcast_missing_flag"].mean() if "rain_nowcast_missing_flag" in df_oot.columns else -1
    logger.info(f"Nowcast coverage: {99 - nc_missing*100:.1f}% rows have fresh nowcast")
    logger.info(f"Feature count: {len(FEATURE_COLS)} ({len(BASE_FEATURE_COLS)} base + {len(RAIN_FEATURE_COLS)} rain + {len(NOWCAST_FEATURE_COLS)} nowcast + {len(MODEL_D_FEATURE_COLS)} model_d)")

    logger.info(" Model D Tmin training complete")


if __name__ == "__main__":
    main()
