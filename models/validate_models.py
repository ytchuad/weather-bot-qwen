# models/validate_models.py
"""
Model validation report generator with rain regime analysis.
Produces MAE, Pinball Loss, coverage, and failure-mode-specific metrics by regime.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')
MODEL_DIR = Path('models/intraday_ml')
FEATURE_LIST_PATH = MODEL_DIR / 'feature_list.json'
REPORT_PATH_XLSX = Path('reports/model_validation_report.xlsx')
REPORT_PATH_JSON = Path('reports/model_validation_report.json')
REPORT_PATH_XLSX.parent.mkdir(parents=True, exist_ok=True)

MIN_SLICE_COUNT = 50

SLICES = {
    "all_data": lambda df: pd.Series(True, index=df.index),
    "rain_present": lambda df: df["rainfall_60m_filled"] > 0,
    "no_rain": lambda df: df["rainfall_60m_filled"] == 0,
    "thin_rain": lambda df: (df["rainfall_60m_filled"] > 0) & (df["rainfall_60m_filled"] <= 1),
    "heavy_rain": lambda df: df["rainfall_60m_filled"] > 5,
    "post_peak_rain": lambda df: df["post_peak_rain_flag"] == 1,
    "morning_peak_rain": lambda df: df.get("morning_peak_rain_flag", pd.Series(0, index=df.index)) == 1,
}


def load_models():
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f'upside_q{q}'] = lgb.Booster(model_file=str(MODEL_DIR / f'upside_q{q}.txt'))
        models[f'downside_q{q}'] = lgb.Booster(model_file=str(MODEL_DIR / f'downside_q{q}.txt'))
    models['upside_zero'] = lgb.Booster(model_file=str(MODEL_DIR / 'upside_zero.txt'))
    models['downside_zero'] = lgb.Booster(model_file=str(MODEL_DIR / 'downside_zero.txt'))
    return models


def pinball_loss(y_true, y_pred, tau):
    diff = y_true - y_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()


def compute_slice_metrics(actual, pred10, pred50, pred90):
    if len(actual) == 0:
        return {"count": 0}

    actual_zero = actual < 0.05
    false_negative = actual_zero & (pred50 > 0.5)

    result = {
        "count": int(len(actual)),
        "mae": float((actual - pred50).abs().mean()),
        "pinball_q10": float(pinball_loss(actual, pred10, 0.1)),
        "pinball_q90": float(pinball_loss(actual, pred90, 0.9)),
        "coverage_p50": float((actual < pred50).mean()),
        "false_negative_max_reached_count": int(false_negative.sum()),
        "false_negative_max_reached_rate": float(false_negative.mean()) if len(actual) else None,
        "avg_predicted_upside_when_actual_zero": float(pred50[actual_zero].mean()) if actual_zero.any() else None,
    }

    if result["count"] < MIN_SLICE_COUNT:
        result["warning"] = "small sample size; interpret this slice with caution."

    return result


def compute_all_slice_metrics(df, preds):
    slice_metrics = {}

    for slice_name, slice_fn in SLICES.items():
        mask = slice_fn(df)
        actual = df.loc[mask, "remaining_upside"]
        pred10 = preds["upside_q10"][mask]
        pred50 = preds["upside_q50"][mask]
        pred90 = preds["upside_q90"][mask]
        slice_metrics[slice_name] = compute_slice_metrics(actual, pred10, pred50, pred90)

    return slice_metrics


def generate_adoption_decision(slice_metrics, overall_mae, rain_mae_improvement):
    rain_present_metrics = slice_metrics.get("rain_present", {})
    heavy_rain_metrics = slice_metrics.get("heavy_rain", {})
    thin_rain_metrics = slice_metrics.get("thin_rain", {})

    do_not_replace = (
        overall_mae is None or
        rain_present_metrics.get("count", 0) < MIN_SLICE_COUNT or
        heavy_rain_metrics.get("count", 0) < MIN_SLICE_COUNT or
        thin_rain_metrics.get("count", 0) < MIN_SLICE_COUNT
    )

    if do_not_replace:
        return {
            "decision": "do_not_full_replace_yet",
            "reason": [
                f"rain-present MAE: {rain_present_metrics.get('mae', 'N/A'):.4f}" if isinstance(rain_present_metrics.get('mae'), float) else f"rain-present MAE: N/A",
                f"heavy-rain MAE: {heavy_rain_metrics.get('mae', 'N/A'):.4f}" if isinstance(heavy_rain_metrics.get('mae'), float) else f"heavy-rain MAE: N/A",
                f"thin-rain MAE: {thin_rain_metrics.get('mae', 'N/A'):.4f}" if isinstance(thin_rain_metrics.get('mae'), float) else f"thin-rain MAE: N/A",
            ],
            "required_before_promotion": [
                "Fix active feature list consistency.",
                "Rename vague rainfall warning.",
                "Establish forward test evidence.",
                "Run forward test in paper mode.",
                "Define promotion thresholds."
            ]
        }

    return {
        "decision": "regime_based_switch_candidate",
        "reason": [
            f"rain-aware model improves rain_present MAE." if rain_mae_improvement.get("rain_present", False) else f"rain-aware model may worsen rain_present MAE.",
            f"rain-aware model improves heavy_rain MAE." if rain_mae_improvement.get("heavy_rain", False) else f"rain-aware model may worsen heavy_rain MAE.",
            "Forward test evidence is pending before production adoption."
        ],
        "required_before_promotion": [
            "Establish forward test evidence.",
            "Run forward test in paper mode."
        ]
    }


def main():
    logger.info("Loading validation data...")
    df = pd.read_parquet(DATA_PATH)

    df_valid = df[df['datetime'] >= '2025-01-01'].copy()
    if len(df_valid) < 1000:
        split_idx = int(len(df) * 0.85)
        df_valid = df.iloc[split_idx:].copy()
        logger.warning("Using last 15% of data as validation set")

    logger.info(f"Validation set size: {len(df_valid)}")

    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_cols = json.load(f)

    models = load_models()
    X_valid = df_valid[feature_cols].fillna(0)

    y_upside = df_valid['remaining_upside'].values
    y_downside = df_valid['remaining_downside'].values
    y_upside_zero = df_valid['is_upside_zero'].values
    y_downside_zero = df_valid['is_downside_zero'].values
    hour_valid = df_valid['hour'].values

    preds = {}
    for q in [10, 25, 50, 75, 90]:
        preds[f'upside_q{q}'] = models[f'upside_q{q}'].predict(X_valid)
        preds[f'downside_q{q}'] = models[f'downside_q{q}'].predict(X_valid)

    for prefix in ['upside', 'downside']:
        q_keys = [f'{prefix}_q{q}' for q in [10, 25, 50, 75, 90]]
        stacked = np.column_stack([preds[k] for k in q_keys])
        stacked.sort(axis=1)
        for i, k in enumerate(q_keys):
            preds[k] = stacked[:, i]

    # Add prediction columns to dataframe for Excel output
    df_valid['pred_upside_q10'] = preds['upside_q10']
    df_valid['pred_upside_q50'] = preds['upside_q50']
    df_valid['pred_upside_q90'] = preds['upside_q90']

    metrics = {}
    for target_name, y, prefix in [('Tmax', y_upside, 'upside'), ('Tmin', y_downside, 'downside')]:
        for q in [10, 25, 50, 75, 90]:
            q_val = q / 100.0
            key = f'{prefix}_q{q}'
            mae = np.mean(np.abs(y - preds[key]))
            pinball = pinball_loss(y, preds[key], q_val)
            metrics[f'{target_name}_q{q}_MAE'] = mae
            metrics[f'{target_name}_q{q}_Pinball'] = pinball

        p10 = preds[f'{prefix}_q10']
        p90 = preds[f'{prefix}_q90']
        coverage = np.mean((y >= p10) & (y <= p90))
        interval_width = np.mean(p90 - p10)
        metrics[f'{target_name}_Coverage_80pct'] = coverage
        metrics[f'{target_name}_Interval_Width_80pct'] = interval_width

    prob_upside_zero = models['upside_zero'].predict(X_valid)
    prob_downside_zero = models['downside_zero'].predict(X_valid)
    from sklearn.metrics import brier_score_loss
    metrics['Tmax_Brier_Zero'] = brier_score_loss(y_upside_zero, prob_upside_zero)
    metrics['Tmin_Brier_Zero'] = brier_score_loss(y_downside_zero, prob_downside_zero)

    logger.info("Overall metrics computed")

    by_hour_list = []
    for h in range(24):
        mask = hour_valid == h
        if mask.sum() < 10:
            continue
        row = {'Hour': h, 'Count': mask.sum()}
        for target_name, y, prefix in [('Tmax', y_upside, 'upside'), ('Tmin', y_downside, 'downside')]:
            y_h = y[mask]
            p50_h = preds[f'{prefix}_q50'][mask]
            row[f'{target_name}_MAE_q50'] = np.mean(np.abs(y_h - p50_h))
            row[f'{target_name}_Pinball_q50'] = pinball_loss(y_h, p50_h, 0.5)
            p10_h = preds[f'{prefix}_q10'][mask]
            p90_h = preds[f'{prefix}_q90'][mask]
            row[f'{target_name}_Coverage_80'] = np.mean((y_h >= p10_h) & (y_h <= p90_h))
            row[f'{target_name}_Interval_Width'] = np.mean(p90_h - p10_h)
        by_hour_list.append(row)

    df_by_hour = pd.DataFrame(by_hour_list).set_index('Hour')

    slice_metrics = compute_all_slice_metrics(df_valid, preds)

    overall_mae = slice_metrics.get("all_data", {}).get("mae")
    rain_mae_improvement = {
        "rain_present": slice_metrics.get("rain_present", {}).get("mae", 0) < overall_mae if overall_mae else False,
        "heavy_rain": slice_metrics.get("heavy_rain", {}).get("mae", 0) < overall_mae if overall_mae else False,
        "thin_rain": slice_metrics.get("thin_rain", {}).get("mae", 0) < overall_mae if overall_mae else False,
    }
    adoption_decision = generate_adoption_decision(slice_metrics, overall_mae, rain_mae_improvement)

    with open(REPORT_PATH_JSON, 'w') as f:
        json.dump({
            "overall_metrics": metrics,
            "slice_metrics": slice_metrics,
            "adoption_decision": adoption_decision,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"JSON report saved to {REPORT_PATH_JSON}")

    with pd.ExcelWriter(REPORT_PATH_XLSX) as writer:
        pd.Series(metrics, name='Overall').to_excel(writer, sheet_name='summary')

        df_slices = pd.DataFrame(slice_metrics).T
        df_slices.to_excel(writer, sheet_name='by_regime')

        df_by_hour.to_excel(writer, sheet_name='by_hour')

        rain_present_mask = df_valid["rainfall_60m_filled"] > 0
        rain_present_df = df_valid[rain_present_mask]
        rain_present_df[["remaining_upside", "pred_upside_q50", "pred_upside_q10", "pred_upside_q90"]].describe().to_excel(
            writer, sheet_name='rain_present_summary')

        heavy_rain_mask = df_valid["rainfall_60m_filled"] > 5
        heavy_rain_df = df_valid[heavy_rain_mask]
        heavy_rain_df[["remaining_upside", "pred_upside_q50", "pred_upside_q10", "pred_upside_q90"]].describe().to_excel(
            writer, sheet_name='heavy_rain_summary')

        thin_rain_mask = (df_valid["rainfall_60m_filled"] > 0) & (df_valid["rainfall_60m_filled"] <= 1)
        thin_rain_df = df_valid[thin_rain_mask]
        thin_rain_df[["remaining_upside", "pred_upside_q50", "pred_upside_q10", "pred_upside_q90"]].describe().to_excel(
            writer, sheet_name='thin_rain_summary')

        if "post_peak_rain_flag" in df_valid.columns:
            post_peak_mask = df_valid["post_peak_rain_flag"] == 1
            post_peak_df = df_valid[post_peak_mask]
            post_peak_df[["remaining_upside", "pred_upside_q50"]].describe().to_excel(
                writer, sheet_name='post_peak_rain')

        if "morning_peak_rain_flag" in df_valid.columns:
            morning_peak_rain_mask = df_valid["morning_peak_rain_flag"] == 1
            morning_peak_rain_df = df_valid[morning_peak_rain_mask]
            morning_peak_rain_df[["remaining_upside", "pred_upside_q50"]].describe().to_excel(
                writer, sheet_name='morning_peak_rain')

        pd.DataFrame([adoption_decision]).to_excel(writer, sheet_name='adoption_decision')

    logger.info(f"Validation report saved to {REPORT_PATH_XLSX}")
    logger.info(f"Overall Tmax Coverage 80%: {metrics.get('Tmax_Coverage_80pct', np.nan):.3f}")
    logger.info(f"Overall Tmin Coverage 80%: {metrics.get('Tmin_Coverage_80pct', np.nan):.3f}")


if __name__ == '__main__':
    main()