# monitoring/model_2a_daily_shadow_eval.py
"""Model 2A daily shadow evaluation.

After actual high temperature is known, evaluates prediction performance.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from monitoring.daily_shadow_eval_base import run_daily_shadow_eval

logger = logging.getLogger(__name__)


def run_model_2a_shadow_eval(
    inference_log_path: str = "logs/model_2a_inference_log.parquet",
    actual_high_source: str = "data/hko_tmax_historical.parquet",
    model_spec_path: str = "config/model_2a_feature_spec.yaml",
    output_path: str = "reports",
) -> pd.DataFrame:
    """Run Model 2A shadow evaluation against actual daily high temperatures.

    Target definition: remaining_upside = max(actual_high_today - max_so_far, 0)

    Args:
        inference_log_path: Path to inference log parquet.
        actual_high_source: Path to actual high temperature data.
        model_spec_path: Path to model spec YAML.
        output_path: Directory for output reports.

    Returns:
        DataFrame with shadow evaluation metrics.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(model_spec_path, "r") as f:
        spec = yaml.safe_load(f)

    log_path = Path(inference_log_path)
    if not log_path.exists():
        logger.error(f"Model 2A inference log not found: {inference_log_path}")
        return pd.DataFrame()

    inference_log = pd.read_parquet(log_path)
    model_name = spec.get("model_name", "model_2a")

    actual = _load_actual_highs(actual_high_source)
    if actual is None or len(actual) == 0:
        logger.warning("No actual high temperature data for shadow evaluation")
        return pd.DataFrame()

    if "decision_time" in inference_log.columns:
        inference_log["decision_time"] = pd.to_datetime(inference_log["decision_time"])
        inference_log["target_date"] = inference_log["decision_time"].dt.date

    predictions = _extract_model_2a_predictions(inference_log)
    if predictions.empty:
        logger.warning("No predictions extracted from inference log")
        return pd.DataFrame()

    if "target_date" in actual.columns:
        actual["target_date"] = pd.to_datetime(actual["target_date"]).dt.date

    merged = predictions.merge(actual, on="target_date", how="inner")
    if len(merged) == 0:
        logger.warning("No matching target dates")
        return pd.DataFrame()

    metrics = _compute_model_2a_metrics(merged, spec)

    metrics_path = output_dir / f"{model_name}_live_shadow_metrics.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Model 2A shadow metrics written to {metrics_path}")

    return metrics_df


def _load_actual_highs(source: str) -> pd.DataFrame:
    """Load actual daily high temperature data."""
    source_path = Path(source)
    if source_path.exists():
        if source_path.suffix == ".parquet":
            df = pd.read_parquet(source_path)
            return df
        elif source_path.suffix == ".csv":
            df = pd.read_csv(source_path)
            return df
    logger.warning(f"Actual high source not found: {source}")
    return None


def _extract_model_2a_predictions(inference_log: pd.DataFrame) -> pd.DataFrame:
    """Extract Model 2A predictions and feature values from log."""
    rows = []
    for _, row in inference_log.iterrows():
        entry = {
            "target_date": row.get("target_date") or (
                pd.Timestamp(row["decision_time"]).date()
                if "decision_time" in row else None
            ),
            "decision_time": row.get("decision_time"),
        }

        for q in ["q10", "q25", "q50", "q75", "q90"]:
            for prefix in ["pred_tmax_", "upside_"]:
                key = f"{prefix}{q}"
                if key in row:
                    entry[key] = row[key]

        for feat in ["max_so_far", "temp_current", "min_so_far"]:
            if feat in row:
                entry[feat] = row[feat]

        rows.append(entry)

    result = pd.DataFrame(rows) if rows else pd.DataFrame()
    if "decision_time" in result.columns:
        result = result.sort_values("decision_time")
    return result


def _compute_model_2a_metrics(
    merged: pd.DataFrame,
    spec: dict,
) -> dict:
    """Compute Model 2A specific shadow evaluation metrics.

    Target: remaining_upside = max(actual_high_today - max_so_far, 0)
    """
    n_predictions = len(merged)
    n_dates = merged["target_date"].nunique() if "target_date" in merged.columns else 0

    metrics = {
        "model_name": spec.get("model_name", "model_2a"),
        "model_version": spec.get("model_version", "v1"),
        "n_predictions": n_predictions,
        "n_target_dates": n_dates,
        "n_predictions_per_date": round(n_predictions / n_dates, 1) if n_dates > 0 else 0,
    }

    actual_col = _find_actual_col(merged)
    if actual_col is None or "pred_tmax_q50" not in merged.columns:
        logger.warning(f"Cannot compute metrics: missing actual column or pred_tmax_q50")
        return metrics

    actual = merged[actual_col].values.astype(float)
    pred_q50 = merged["pred_tmax_q50"].values.astype(float)
    max_so_far = merged.get("max_so_far",
                            merged.get("temp_current", pd.Series([np.nan] * len(merged)))).values.astype(float)

    valid = ~(np.isnan(actual) | np.isnan(pred_q50))
    actual_v = actual[valid]
    pred_v = pred_q50[valid]
    max_so_far_v = max_so_far[valid] if len(max_so_far) == len(actual) else np.array([np.nan] * len(actual))[valid]

    if len(actual_v) == 0:
        return metrics

    # Core target: remaining upside
    actual_upside = np.maximum(actual_v - max_so_far_v, 0) if len(max_so_far_v) == len(actual_v) else np.zeros_like(actual_v)
    pred_upside = merged.get("upside_q50", pred_v - max_so_far_v).values.astype(float)[valid] \
        if "upside_q50" in merged.columns else actual_upside.copy()

    errors = actual_v - pred_v
    upside_errors = actual_upside - pred_upside[:len(actual_upside)]

    metrics.update({
        "n_valid": len(actual_v),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias_upside": float(np.mean(upside_errors)),
        "mae_upside": float(np.mean(np.abs(upside_errors))),
        "rmse_upside": float(np.sqrt(np.mean(upside_errors ** 2))),
        "mean_actual": float(np.mean(actual_v)),
        "mean_predicted": float(np.mean(pred_v)),
        "prediction_range": float(np.ptp(pred_v)) if len(pred_v) > 1 else 0,
    })

    # Quantile-specific metrics
    for q in ["q10", "q25", "q75", "q90"]:
        col = f"pred_tmax_{q}"
        if col in merged.columns:
            q_pred = merged[col].values.astype(float)[valid]
            q_errors = actual_v - q_pred
            valid_q = ~np.isnan(q_errors)
            if valid_q.sum() > 0:
                metrics[f"mae_{q}"] = float(np.mean(np.abs(q_errors[valid_q])))
                metrics[f"bias_{q}"] = float(np.mean(q_errors[valid_q]))

    # Breach rate (prediction interval coverage)
    if "pred_tmax_q10" in merged.columns and "pred_tmax_q90" in merged.columns:
        q10 = merged["pred_tmax_q10"].values.astype(float)[valid]
        q90 = merged["pred_tmax_q90"].values.astype(float)[valid]
        covered = ((actual_v >= q10) & (actual_v <= q90))
        metrics["pi80_coverage"] = float(covered.mean())
        metrics["pi80_width"] = float(np.mean(q90 - q10))

    # False alarm analysis (predicted upside when none occurred)
    if len(actual_upside) > 0 and len(pred_upside[:len(actual_upside)]) > 0:
        false_alarm = (pred_upside[:len(actual_upside)] > 0.5) & (actual_upside < 0.5)
        hit = (pred_upside[:len(actual_upside)] > 0.5) & (actual_upside >= 0.5)
        metrics["false_alarm_rate"] = float(false_alarm.sum() / max((pred_upside[:len(actual_upside)] > 0.5).sum(), 1))
        metrics["hit_rate"] = float(hit.sum() / max((actual_upside >= 0.5).sum(), 1))

    return metrics


def _find_actual_col(df: pd.DataFrame) -> str:
    """Find the actual high temperature column."""
    candidates = ["actual_high_today", "actual_tmax", "actual_max_temp", "observed_max_temp"]
    for c in candidates:
        if c in df.columns:
            return c
    return None
