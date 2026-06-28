# monitoring/daily_shadow_eval_base.py
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def run_daily_shadow_eval(
    inference_log_path: str,
    actual_outcome_source: str,
    model_spec_path: str,
    output_path: str,
    target_extractor_fn=None,
    metric_fn=None,
) -> pd.DataFrame:
    """Generic daily shadow evaluation after actual outcome is known.

    Each model spec must define its own realized target and metrics.

    Args:
        inference_log_path: Path to inference log parquet.
        actual_outcome_source: Path or data source with actual outcomes.
        model_spec_path: Path to model spec YAML.
        output_path: Directory for output reports.
        target_extractor_fn: Function to extract actual target from outcomes.
        metric_fn: Custom metric function if needed.

    Returns:
        DataFrame with shadow evaluation metrics.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_spec_path, "r") as f:
        spec = yaml.safe_load(f)

    log_path = Path(inference_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Inference log not found: {inference_log_path}")

    inference_log = pd.read_parquet(log_path)
    model_name = spec.get("model_name", "unknown")

    actual_outcomes = _load_actual_outcomes(actual_outcome_source)
    if actual_outcomes is None or len(actual_outcomes) == 0:
        logger.warning("No actual outcomes available for shadow evaluation")
        return pd.DataFrame()

    if "decision_time" in inference_log.columns:
        inference_log["decision_time"] = pd.to_datetime(inference_log["decision_time"])
        inference_log["target_date"] = inference_log["decision_time"].dt.date
    else:
        logger.error("inference_log missing decision_time column")
        return pd.DataFrame()

    predictions = _extract_predictions(inference_log, spec)
    if predictions.empty:
        logger.warning("No predictions extracted from inference log")
        return pd.DataFrame()

    if "target_date" in actual_outcomes.columns:
        actual_outcomes["target_date"] = pd.to_datetime(actual_outcomes["target_date"]).dt.date

    merged = predictions.merge(
        actual_outcomes, on="target_date", how="inner", suffixes=("_pred", "_actual")
    )
    if len(merged) == 0:
        logger.warning("No matching target dates between predictions and actual outcomes")
        return pd.DataFrame()

    metrics = _compute_generic_metrics(merged, spec)

    metrics_path = output_dir / f"{model_name}_live_shadow_metrics.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Shadow evaluation metrics written to {metrics_path}")

    return metrics_df


def _load_actual_outcomes(source: str) -> pd.DataFrame:
    """Load actual outcomes from file path or source identifier."""
    source_path = Path(source)
    if source_path.exists():
        if source_path.suffix == ".parquet":
            return pd.read_parquet(source_path)
        elif source_path.suffix == ".csv":
            return pd.read_csv(source_path)
    logger.warning(f"Actual outcome source not found: {source}")
    return None


def _extract_predictions(
    inference_log: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    """Extract predictions from inference log rows."""
    rows = []
    for _, row in inference_log.iterrows():
        pred = row.get("prediction_payload", {})
        if isinstance(pred, str):
            import json
            pred = json.loads(pred)
        if not isinstance(pred, dict):
            continue

        entry = {
            "target_date": row.get("target_date") or (
                pd.Timestamp(row["decision_time"]).date()
                if "decision_time" in row else None
            ),
            "decision_time": row.get("decision_time"),
        }

        for q in ["q10", "q25", "q50", "q75", "q90"]:
            pred_key = f"pred_tmax_{q}"
            if pred_key in pred:
                entry[pred_key] = pred[pred_key]
            upside_key = f"upside_{q}"
            if upside_key in pred:
                entry[upside_key] = pred[upside_key]

        rows.append(entry)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_generic_metrics(
    merged: pd.DataFrame,
    spec: dict,
) -> dict:
    """Compute generic shadow evaluation metrics."""
    n_predictions = len(merged)
    metrics = {
        "model_name": spec.get("model_name", "unknown"),
        "model_version": spec.get("model_version", "unknown"),
        "n_predictions": n_predictions,
        "n_target_dates": merged["target_date"].nunique() if "target_date" in merged.columns else 0,
    }

    actual_col = _find_actual_col(merged)
    pred_col = "pred_tmax_q50" if "pred_tmax_q50" in merged.columns else None

    if actual_col and pred_col and n_predictions > 0:
        actual = merged[actual_col].values.astype(float)
        predicted = merged[pred_col].values.astype(float)

        valid = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[valid]
        predicted = predicted[valid]

        if len(actual) > 0:
            errors = actual - predicted
            metrics["bias"] = np.mean(errors)
            metrics["mae"] = np.mean(np.abs(errors))
            metrics["rmse"] = np.sqrt(np.mean(errors ** 2))
            metrics["n_valid"] = len(actual)

    return metrics


def _find_actual_col(df: pd.DataFrame) -> str:
    """Find the actual outcome column in the merged dataframe."""
    candidates = ["actual_high_today", "actual_tmax", "actual_max_temp",
                  "observed_max_temp", "actual_tmin", "actual_min_temp"]
    for c in candidates:
        if c in df.columns:
            return c
    return None
