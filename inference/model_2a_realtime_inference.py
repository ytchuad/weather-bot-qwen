# inference/model_2a_realtime_inference.py
"""Model 2A real-time inference implementation.

Uses the generic framework from inference/realtime_inference_base.py
with Model 2A specific adapters, feature builder, and guardrails.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from inference.realtime_inference_base import (
    load_model_spec,
    validate_active_hours,
    validate_decision_grid,
)
from inference.model_2a_adapters import (
    Model2AAdapter,
    get_model_2a_adapter,
)
from features.model_2a_source_adapters import (
    standardize_weather_obs,
    standardize_wind_obs,
    standardize_forecast,
)
from features.model_2a_feature_builder import build_model_2a_input_status
from features.input_status import jsonable, serialize_status

logger = logging.getLogger(__name__)

HKT = timedelta(hours=8)


def run_model_2a_inference(
    decision_time: pd.Timestamp,
    raw_weather: pd.DataFrame,
    raw_wind: pd.DataFrame,
    raw_forecast: pd.DataFrame,
    model_spec_path: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Run Model 2A real-time inference end-to-end.

    Args:
        decision_time: Decision time for inference.
        raw_weather: Raw weather observation data.
        raw_wind: Raw wind observation data.
        raw_forecast: Raw forecast data.
        model_spec_path: Optional path to the selected model spec YAML.
        model_version: Explicitly selected ``v1`` or ``v2``.

    Returns:
        Prediction payload dict with warnings and metadata.
    """
    adapter = get_model_2a_adapter(
        model_version=model_version,
        model_spec_path=model_spec_path,
    )
    lineage = adapter.validate_lineage()
    spec = load_model_spec(str(adapter.feature_spec_path))

    lineage_metadata = {
        "model_version": adapter.model_version,
        "feature_version": adapter.feature_version,
        "spec_path": lineage["spec_path"],
        "feature_spec_path": lineage["feature_spec_path"],
        "feature_spec": lineage["feature_spec_path"],
        "artifact_directory": lineage["artifact_directory"],
        "artifact_identity": lineage["artifact_identity"],
        "feature_list_path": lineage["feature_list_path"],
    }

    if not adapter.supports_realtime:
        return {
            **lineage_metadata,
            "error": (
                "Model 2A v1 realtime inference is deprecated/unsupported: "
                "the original v1 highland source semantics cannot be "
                "reconstructed safely."
            ),
            "warning_flags": ["unsupported_model_version"],
            "prediction": None,
        }

    if not validate_active_hours(decision_time, spec):
        return {
            **lineage_metadata,
            "error": f"Decision time {decision_time} outside active hours",
            "warning_flags": ["outside_active_hours"],
            "prediction": None,
        }

    if not validate_decision_grid(decision_time, spec):
        return {
            **lineage_metadata,
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

    features = adapter.feature_builder(
        decision_time=decision_time,
        weather_canonical=weather_canonical,
        wind_canonical=wind_canonical,
        forecast_canonical=forecast_canonical,
        spec=spec,
        mode="live",
    )

    model_cache = _load_model_2a(adapter, spec, lineage=lineage)
    feature_cols = model_cache["feature_cols"]

    missing_feats = [c for c in feature_cols if c not in features.columns]
    if missing_feats:
        raise ValueError(f"Missing required features: {missing_feats}")

    X = features[feature_cols]

    try:
        input_status = build_model_2a_input_status(
            decision_time=decision_time,
            weather_canonical=weather_canonical,
            wind_canonical=wind_canonical,
            forecast_canonical=forecast_canonical,
            spec=spec,
            mode="live",
        )
    except Exception as status_error:
        logger.warning("Model 2A input status build failed: %s", status_error)
        input_status = {
            "status_contract_version": "phase2a.v1",
            "numeric_policy": "legacy_compatible",
            "status_policy": "truthful",
            "decision_timestamp": decision_time,
            "status_build_error": str(status_error),
        }

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

    temp_anomaly_flag, temp_spike_flag, any_source_missing_flag, wind_missing_flag, forecast_missing_flag = \
        _update_missing_flags_from_canonical(
            weather_canonical, wind_canonical, forecast_canonical
        )

    prediction = {
        **lineage_metadata,
        "model_name": spec["model_name"],
        "decision_time": decision_time,
        "run_timestamp": datetime.now(),
        "source_mode": "live",
        "source_systems": ["weather_obs", "wind_obs", "forecast"],
        "source_timestamps": {
            "weather_obs": weather_canonical["timestamp"].iloc[-1] if len(weather_canonical) > 0 and "timestamp" in weather_canonical.columns else None,
            "wind_obs": wind_canonical["timestamp"].iloc[-1] if len(wind_canonical) > 0 and "timestamp" in wind_canonical.columns else None,
            "forecast": forecast_canonical["timestamp"].iloc[-1] if len(forecast_canonical) > 0 and "timestamp" in forecast_canonical.columns else None,
        },
        "source_available_times": {
            "weather_obs": weather_canonical["available_time"].iloc[-1] if len(weather_canonical) > 0 and "available_time" in weather_canonical.columns else None,
            "wind_obs": wind_canonical["available_time"].iloc[-1] if len(wind_canonical) > 0 and "available_time" in wind_canonical.columns else None,
            "forecast": forecast_canonical["available_time"].iloc[-1] if len(forecast_canonical) > 0 and "available_time" in forecast_canonical.columns else None,
        },
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
        "temp_current": float(features["temp_current"].iloc[0]) if "temp_current" in features.columns else None,
        "rh_current": float(features["rh_current"].iloc[0]) if "rh_current" in features.columns else None,
        "max_so_far": float(features["max_so_far"].iloc[0]) if "max_so_far" in features.columns else None,
        "min_so_far": float(features["min_so_far"].iloc[0]) if "min_so_far" in features.columns else None,
        "forecast_max_temp": float(features["forecast_max_temp"].iloc[0]) if "forecast_max_temp" in features.columns else None,
        "forecast_min_temp": float(features["forecast_min_temp"].iloc[0]) if "forecast_min_temp" in features.columns else None,
        "temp_anomaly_flag": temp_anomaly_flag,
        "temp_spike_flag": temp_spike_flag,
        "any_source_missing_flag": any_source_missing_flag,
        "wind_missing_flag": wind_missing_flag,
        "forecast_missing_flag": forecast_missing_flag,
        "warning_flags": stop_conditions,
        "guardrail_violation": len([k for k, v in guardrail_flags.items() if v]) > 0,
        "feature_parity_status": "pending",
        "decision_timestamp": decision_time,
        "input_status": jsonable(input_status),
        "weather_input_status": jsonable(input_status.get("weather_input_status", {})),
        "wind_input_status": jsonable(input_status.get("wind_input_status", {})),
        "forecast_input_status": jsonable(input_status.get("forecast_input_status", {})),
        "observation_buffer_status": jsonable(input_status.get("observation_buffer_status", {})),
        "feature_values": jsonable(features.iloc[0].to_dict()),
        "numeric_features": jsonable(features.iloc[0].to_dict()),
    }

    _write_inference_log(prediction, spec)

    return prediction


def _load_model_2a(
    adapter: Model2AAdapter,
    spec: dict,
    lineage: Optional[dict] = None,
) -> dict:
    """Load only the artifacts proven by the selected version adapter."""
    import lightgbm as lgb

    lineage = lineage or adapter.validate_lineage(spec=spec)
    model_dir = Path(lineage["artifact_directory"])
    feature_cols = list(lineage["ordered_feature_names"])

    cache = {
        "feature_cols": feature_cols,
        "model_version": adapter.model_version,
        "spec_path": lineage["spec_path"],
        "artifact_directory": lineage["artifact_directory"],
        "artifact_identity": lineage["artifact_identity"],
    }
    for q in [10, 25, 50, 75, 90]:
        f_path = model_dir / f"upside_q{q}.txt"
        cache[f"upside_q{q}"] = lgb.Booster(model_file=str(f_path))

    zero_path = model_dir / "upside_zero.txt"
    cache["upside_zero"] = lgb.Booster(model_file=str(zero_path))
    cache["upside_zero_threshold"] = lineage["threshold_metadata"]["value"]

    return cache


def _predict_zero_prob(model_cache: dict, X: pd.DataFrame, feature_cols: list) -> float:
    """Predict probability that max has already been reached.

    LightGBM binary classifier predict() returns probability by default.
    Falls back to sigmoid(raw_score) if raw scores are returned.
    """
    if model_cache.get("upside_zero") is None:
        return 0.0
    try:
        clf_features = model_cache["upside_zero"].feature_name()
        missing_clf = [c for c in clf_features if c not in X.columns]
        if missing_clf:
            clf_cols = [c for c in clf_features if c in X.columns]
            X_clf = X[clf_cols]
        else:
            X_clf = X[clf_features]

        raw = model_cache["upside_zero"].predict(X_clf, raw_score=False)
        prob = float(raw[0])
        if 0.0 <= prob <= 1.0:
            return prob
        import math
        prob = 1.0 / (1.0 + math.exp(-prob))
        return prob
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

    if guardrail_flags.get("late_day_unreasonable_upside_flag", 0) > 0:
        triggered.append("late_day_unreasonable_upside")

    if guardrail_flags.get("formula_check_failed", False):
        triggered.append("prediction_formula_check_fails")

    if guardrail_flags.get("guardrail_violation", False):
        triggered.append("guardrail_violation")

    return triggered


def _update_missing_flags_from_canonical(
    weather: pd.DataFrame,
    wind: pd.DataFrame,
    forecast: pd.DataFrame,
) -> tuple[bool, bool, bool, bool, bool]:
    """Check canonical sources for missing data flags and return them."""
    w_missing = weather is None or len(weather) == 0
    w_anomaly = False
    w_spike = False
    if not w_missing:
        w_anomaly = bool(weather["temp_anomaly_flag"].any()) if "temp_anomaly_flag" in weather.columns else False
        w_spike = bool(weather["temp_spike_flag"].any()) if "temp_spike_flag" in weather.columns else False

    wnd_missing = wind is None or len(wind) == 0
    fc_missing = forecast is None or len(forecast) == 0

    return w_anomaly, w_spike, w_missing or wnd_missing or fc_missing, wnd_missing, fc_missing


def _write_inference_log(prediction: dict, spec: dict) -> None:
    """Write inference log row to parquet file.

    Log fields must match inference_log_schema in model spec.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    model_name = spec.get("model_name", "model_2a")
    log_path = log_dir / f"{model_name}_inference_log.parquet"

    log_fields = spec.get("inference_log_schema", {}).get("fields", [])
    log_row = {}
    for field in log_fields:
        if field == "feature_parity_status":
            log_row[field] = "pending"
        elif field == "source_systems":
            log_row[field] = str(prediction.get("source_systems", []))
        elif field == "source_timestamps":
            log_row[field] = str(prediction.get("source_timestamps", {}))
        elif field == "source_available_times":
            log_row[field] = str(prediction.get("source_available_times", {}))
        elif field == "guardrail_flags":
            log_row[field] = str(prediction.get("guardrail_flags", {}))
        else:
            log_row[field] = prediction.get(field)

    # Supplemental Phase 2A diagnostics are serialized as clean JSON strings
    # so nested status maps never enter the model feature vector or an object
    # dtype parquet column.
    for field in (
        "input_status",
        "weather_input_status",
        "wind_input_status",
        "forecast_input_status",
        "observation_buffer_status",
        "feature_values",
        "numeric_features",
        "source_timestamps",
        "source_available_times",
        "source_systems",
    ):
        if field in prediction:
            log_row[field] = serialize_status(prediction.get(field))
    for field in (
        "decision_timestamp",
        "feature_spec",
        "feature_spec_path",
        "artifact_identity",
        "model_version",
        "feature_version",
    ):
        if field in prediction:
            log_row[field] = jsonable(prediction.get(field))

    new_row_df = pd.DataFrame([log_row])

    if log_path.exists():
        existing = pd.read_parquet(log_path)
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_parquet(log_path, index=False)
