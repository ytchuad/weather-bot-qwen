# inference/model_2a_realtime_inference.py
"""Model 2A real-time inference implementation.

Uses the generic framework from inference/realtime_inference_base.py
with Model 2A specific adapters, feature builder, and guardrails.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

from inference.realtime_inference_base import (
    load_model_spec,
    validate_active_hours,
    validate_decision_grid,
)
from features.model_2a_source_adapters import (
    standardize_weather_obs,
    standardize_wind_obs,
    standardize_forecast,
)
from features.model_2a_feature_builder import build_model_2a_features

logger = logging.getLogger(__name__)

HKT = timedelta(hours=8)
MODEL_DIR = Path("models/intraday_minute_ml_model_2a")


def run_model_2a_inference(
    decision_time: pd.Timestamp,
    raw_weather: pd.DataFrame,
    raw_wind: pd.DataFrame,
    raw_forecast: pd.DataFrame,
    model_spec_path: str = "config/model_2a_feature_spec.yaml",
) -> dict:
    """Run Model 2A real-time inference end-to-end.

    Args:
        decision_time: Decision time for inference.
        raw_weather: Raw weather observation data.
        raw_wind: Raw wind observation data.
        raw_forecast: Raw forecast data.
        model_spec_path: Path to model spec YAML.

    Returns:
        Prediction payload dict with warnings and metadata.
    """
    spec = load_model_spec(model_spec_path)

    if not validate_active_hours(decision_time, spec):
        return {
            "error": f"Decision time {decision_time} outside active hours",
            "warning_flags": ["outside_active_hours"],
            "prediction": None,
        }

    if not validate_decision_grid(decision_time, spec):
        return {
            "error": f"Decision time {decision_time} not on {spec['decision_grid_minutes']}-min grid",
            "warning_flags": ["off_grid"],
            "prediction": None,
        }

    temp_cleaning = spec.get("temperature_cleaning", {})
    valid_min = temp_cleaning.get("valid_min_temp", 0)
    valid_max = temp_cleaning.get("valid_max_temp", 40)

    weather_canonical = standardize_weather_obs(
        raw_weather, "live", valid_temp_min=valid_min, valid_temp_max=valid_max
    )
    wind_canonical = standardize_wind_obs(raw_wind, "live")
    forecast_canonical = standardize_forecast(raw_forecast, "live")

    features = build_model_2a_features(
        decision_time=decision_time,
        weather_canonical=weather_canonical,
        wind_canonical=wind_canonical,
        forecast_canonical=forecast_canonical,
        spec=spec,
        mode="live",
    )

    model_cache = _load_model_2a(spec)
    feature_cols = model_cache["feature_cols"]

    missing_feats = [c for c in feature_cols if c not in features.columns]
    if missing_feats:
        raise ValueError(f"Missing required features: {missing_feats}")

    X = features[feature_cols]

    upside_q10 = model_cache["upside_q10"].predict(X)[0]
    upside_q25 = model_cache["upside_q25"].predict(X)[0]
    upside_q50 = model_cache["upside_q50"].predict(X)[0]
    upside_q75 = model_cache["upside_q75"].predict(X)[0]
    upside_q90 = model_cache["upside_q90"].predict(X)[0]

    max_so_far = features["max_so_far"].iloc[0] if "max_so_far" in features.columns else np.nan

    quantiles = sorted([upside_q10, upside_q25, upside_q50, upside_q75, upside_q90])
    upside_q10, upside_q25, upside_q50, upside_q75, upside_q90 = quantiles

    # CRITICAL: pred_tmax_qXX = max_so_far + upside_qXX (NOT temp_current + upside_qXX)
    pred_tmax_q10 = max_so_far + upside_q10
    pred_tmax_q25 = max_so_far + upside_q25
    pred_tmax_q50 = max_so_far + upside_q50
    pred_tmax_q75 = max_so_far + upside_q75
    pred_tmax_q90 = max_so_far + upside_q90

    zero_prob = _predict_zero_prob(model_cache, X, feature_cols)
    hour = decision_time.hour

    guardrail_flags = _apply_guardrails(
        pred_tmax_q50, max_so_far, upside_q50, hour, spec
    )

    stop_conditions = _evaluate_stop_conditions(
        features, guardrail_flags, spec
    )

    prediction = {
        "model_name": spec["model_name"],
        "model_version": spec["model_version"],
        "decision_time": decision_time,
        "run_timestamp": datetime.now(),
        "upside_q10": float(upside_q10),
        "upside_q25": float(upside_q25),
        "upside_q50": float(upside_q50),
        "upside_q75": float(upside_q75),
        "upside_q90": float(upside_q90),
        "pred_tmax_q10": float(pred_tmax_q10),
        "pred_tmax_q25": float(pred_tmax_q25),
        "pred_tmax_q50": float(pred_tmax_q50),
        "pred_tmax_q75": float(pred_tmax_q75),
        "pred_tmax_q90": float(pred_tmax_q90),
        "zero_prob": zero_prob,
        "classifier_reliable_window": int(hour >= spec.get("classifier_usage", {}).get("reliable_window_start_hour", 15)),
        "late_day_unreasonable_upside_flag": guardrail_flags.get("late_day_unreasonable_upside_flag", 0),
        "temp_anomaly_flag": bool(features.get("temp_current", pd.Series([np.nan])).iloc[0] != features.get("temp_current", pd.Series([np.nan])).iloc[0]) if False else False,
        "temp_spike_flag": guardrail_flags.get("temp_spike_flag", False),
        "any_source_missing_flag": False,
        "wind_missing_flag": False,
        "forecast_missing_flag": False,
        "warning_flags": stop_conditions,
        "guardrail_violation": len([k for k, v in guardrail_flags.items() if v]) > 0,
    }

    _update_missing_flags(prediction, weather_canonical, wind_canonical, forecast_canonical)

    _write_inference_log(prediction, spec)

    return prediction


def _load_model_2a(spec: dict) -> dict:
    """Load Model 2A model artifacts."""
    import lightgbm as lgb

    model_dir = Path(spec.get("feature_list_path", str(MODEL_DIR))).parent
    fl_path = model_dir / "feature_list.json"

    if not fl_path.exists():
        fl_path = MODEL_DIR / "feature_list.json"
        model_dir = MODEL_DIR

    with open(fl_path, "r") as f:
        feature_cols = json.load(f)

    cache = {"feature_cols": feature_cols}
    for q in [10, 25, 50, 75, 90]:
        f_path = model_dir / f"upside_q{q}.txt"
        if f_path.exists():
            cache[f"upside_q{q}"] = lgb.Booster(model_file=str(f_path))

    zero_path = model_dir / "upside_zero.txt"
    cache["upside_zero"] = lgb.Booster(model_file=str(zero_path)) if zero_path.exists() else None

    best_threshold_path = model_dir / "best_threshold.json"
    if cache["upside_zero"] is not None and best_threshold_path.exists():
        with open(best_threshold_path) as f:
            cache["upside_zero_threshold"] = json.load(f).get("upside_zero_threshold", 0.5)
    else:
        cache["upside_zero_threshold"] = 0.5

    return cache


def _predict_zero_prob(model_cache: dict, X: pd.DataFrame, feature_cols: list) -> float:
    """Predict probability that max has already been reached."""
    if model_cache.get("upside_zero") is None:
        return 0.0
    try:
        clf_features = model_cache["upside_zero"].feature_name()
        missing_clf = [c for c in clf_features if c not in X.columns]
        if missing_clf:
            clf_cols = [c for c in clf_features if c in X.columns]
            return float(model_cache["upside_zero"].predict(X[clf_cols])[0])
        return float(model_cache["upside_zero"].predict(X[clf_features])[0])
    except Exception:
        import math
        try:
            prob = model_cache["upside_zero"].predict(X, pred_contrib=False)[0]
            prob = 1.0 / (1.0 + math.exp(-prob)) if isinstance(prob, float) else 0.0
            th = model_cache.get("upside_zero_threshold", 0.5)
            return 1.0 if prob > th else 0.0
        except Exception:
            return 0.0


def _apply_guardrails(
    pred_tmax_q50: float,
    max_so_far: float,
    upside_q50: float,
    hour: int,
    spec: dict,
) -> dict:
    """Apply Model 2A specific guardrails."""
    flags = {}

    formula_check = abs(pred_tmax_q50 - (max_so_far + upside_q50)) < 1e-6
    flags["formula_check_failed"] = not formula_check

    late_day = spec.get("late_day_guardrail", {})
    late_start = late_day.get("start_hour", 18)
    max_allowed = late_day.get("max_allowed_q50_above_max_so_far", 0.5)

    late_day_flag = int(hour >= late_start and pred_tmax_q50 > max_so_far + max_allowed)
    flags["late_day_unreasonable_upside_flag"] = late_day_flag
    flags["guardrail_violation"] = late_day_flag > 0 or not formula_check

    return flags


def _evaluate_stop_conditions(
    features: pd.DataFrame,
    guardrail_flags: dict,
    spec: dict,
) -> list:
    """Evaluate Model 2A stop conditions."""
    triggered = []

    temp_current = features.get("temp_current", pd.Series([np.nan])).iloc[0]
    if pd.isna(temp_current):
        triggered.append("temp_current_missing")

    hour = features.index[0].hour if hasattr(features.index, "dtype") else 0

    if guardrail_flags.get("late_day_unreasonable_upside_flag", 0) > 0:
        triggered.append("late_day_unreasonable_upside")

    if guardrail_flags.get("formula_check_failed", False):
        triggered.append("prediction_formula_check_fails")

    if guardrail_flags.get("guardrail_violation", False):
        triggered.append("guardrail_violation")

    return triggered


def _update_missing_flags(
    prediction: dict,
    weather: pd.DataFrame,
    wind: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    """Update missing/flags in prediction based on canonical source availability."""
    if weather is not None and len(weather) > 0:
        temp_missing = weather["temp_current_clean"].isna().all() if "temp_current_clean" in weather.columns else True
        prediction["temp_anomaly_flag"] = bool(
            weather["temp_anomaly_flag"].any() if "temp_anomaly_flag" in weather.columns else False
        )
        prediction["temp_spike_flag"] = bool(
            weather["temp_spike_flag"].any() if "temp_spike_flag" in weather.columns else False
        )
    else:
        prediction["any_source_missing_flag"] = True

    if wind is None or len(wind) == 0:
        prediction["wind_missing_flag"] = True
        prediction["any_source_missing_flag"] = True

    if forecast is None or len(forecast) == 0:
        prediction["forecast_missing_flag"] = True
        prediction["any_source_missing_flag"] = True


def _write_inference_log(prediction: dict, spec: dict) -> None:
    """Write inference log row to parquet file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    model_name = spec.get("model_name", "model_2a")
    log_path = log_dir / f"{model_name}_inference_log.parquet"

    log_row = {
        "model_name": model_name,
        "model_version": prediction.get("model_version"),
        "decision_time": prediction.get("decision_time"),
        "run_timestamp": prediction.get("run_timestamp"),
        "source_mode": "live",
        "source_systems": ["weather_obs", "wind_obs", "forecast"],
        "upside_q10": prediction.get("upside_q10"),
        "upside_q25": prediction.get("upside_q25"),
        "upside_q50": prediction.get("upside_q50"),
        "upside_q75": prediction.get("upside_q75"),
        "upside_q90": prediction.get("upside_q90"),
        "pred_tmax_q10": prediction.get("pred_tmax_q10"),
        "pred_tmax_q25": prediction.get("pred_tmax_q25"),
        "pred_tmax_q50": prediction.get("pred_tmax_q50"),
        "pred_tmax_q75": prediction.get("pred_tmax_q75"),
        "pred_tmax_q90": prediction.get("pred_tmax_q90"),
        "zero_prob": prediction.get("zero_prob"),
        "classifier_reliable_window": prediction.get("classifier_reliable_window"),
        "late_day_unreasonable_upside_flag": prediction.get("late_day_unreasonable_upside_flag"),
        "any_source_missing_flag": prediction.get("any_source_missing_flag"),
        "wind_missing_flag": prediction.get("wind_missing_flag"),
        "forecast_missing_flag": prediction.get("forecast_missing_flag"),
        "temp_anomaly_flag": prediction.get("temp_anomaly_flag"),
        "temp_spike_flag": prediction.get("temp_spike_flag"),
        "feature_parity_status": "pending",
        "guardrail_flags": {
            "late_day_unreasonable_upside_flag": prediction.get("late_day_unreasonable_upside_flag"),
            "guardrail_violation": prediction.get("guardrail_violation"),
        },
    }

    new_row_df = pd.DataFrame([log_row])

    if log_path.exists():
        existing = pd.read_parquet(log_path)
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_parquet(log_path, index=False)
