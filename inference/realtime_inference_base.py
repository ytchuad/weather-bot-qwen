# inference/realtime_inference_base.py
import pandas as pd
import numpy as np
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

HKT = timedelta(hours=8)


def load_model_spec(spec_path: str) -> dict:
    """Load and validate a model specification YAML file."""
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Model spec not found: {spec_path}")

    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    required_keys = [
        "model_name", "model_version", "feature_versions",
        "feature_list_path", "active_hours", "decision_grid_minutes",
        "availability_rule", "canonical_sources", "feature_groups",
        "feature_tolerances", "data_quality_rules",
        "inference_log_schema", "guardrails", "stop_conditions",
    ]
    missing = [k for k in required_keys if k not in spec]
    if missing:
        raise ValueError(f"Model spec missing required keys: {missing}")

    if spec.get("model_name") is None:
        raise ValueError("model_name must not be null in spec")
    if spec.get("model_version") is None:
        raise ValueError("model_version must not be null in spec")

    return spec


def validate_active_hours(
    decision_time: pd.Timestamp,
    spec: dict,
) -> bool:
    """Check if decision_time falls within configured active hours."""
    active = spec.get("active_hours", {})
    start_str = active.get("start", "00:00")
    end_str = active.get("end", "23:59")
    start_h, start_m = map(int, start_str.split(":"))
    end_h, end_m = map(int, end_str.split(":"))

    t_minutes = decision_time.hour * 60 + decision_time.minute
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    if end_minutes < start_minutes:
        return t_minutes >= start_minutes or t_minutes <= end_minutes
    return start_minutes <= t_minutes <= end_minutes


def validate_decision_grid(
    decision_time: pd.Timestamp,
    spec: dict,
) -> bool:
    """Check that decision_time aligns to the configured grid."""
    grid_minutes = spec.get("decision_grid_minutes", 10)
    t_minutes = decision_time.hour * 60 + decision_time.minute
    return t_minutes % grid_minutes == 0


def run_realtime_inference(
    decision_time: pd.Timestamp,
    raw_sources_dict: dict[str, pd.DataFrame],
    model_spec_path: str,
    source_adapter_fn=None,
    feature_builder_fn=None,
    model_loader_fn=None,
    prediction_fn=None,
    guardrail_fn=None,
    log_writer_fn=None,
) -> dict:
    """Generic real-time inference flow.

    Steps:
    1. Load model-specific spec
    2. Validate active hours and decision grid
    3. Standardize all raw sources to canonical schemas
    4. Build features using shared feature builder
    5. Load model and trained feature list
    6. Score model
    7. Apply model-specific prediction formula
    8. Apply model-specific calibration if configured
    9. Apply guardrails
    10. Write inference log
    11. Return prediction payload with warning flags
    """
    spec = load_model_spec(model_spec_path)

    if not validate_active_hours(decision_time, spec):
        return {"error": f"Decision time {decision_time} outside active hours", "warning_flags": ["outside_active_hours"]}

    if not validate_decision_grid(decision_time, spec):
        return {"error": f"Decision time {decision_time} not aligned to {spec['decision_grid_minutes']}-min grid", "warning_flags": ["off_grid"]}

    canonical_sources = {}
    source_systems = []
    source_timestamps = {}
    source_available_times = {}

    if source_adapter_fn is not None:
        for source_name, raw_df in raw_sources_dict.items():
            canonical = source_adapter_fn(raw_df, source_name, "live")
            canonical_sources[source_name] = canonical
            source_systems.append(source_name)
            if "timestamp" in canonical.columns and len(canonical) > 0:
                source_timestamps[source_name] = canonical["timestamp"].iloc[0]
            if "available_time" in canonical.columns and len(canonical) > 0:
                source_available_times[source_name] = canonical["available_time"].iloc[0]
    else:
        canonical_sources = raw_sources_dict

    if feature_builder_fn is not None:
        feature_df = feature_builder_fn(
            decision_time=decision_time,
            canonical_sources=canonical_sources,
            spec=spec,
            mode="live",
        )
    else:
        feature_df = pd.DataFrame({"dummy_feature": [0.0]}, index=[decision_time])

    if model_loader_fn is not None:
        model, feature_cols = model_loader_fn(spec)
    else:
        model, feature_cols = None, list(feature_df.columns)

    if prediction_fn is not None:
        prediction_payload = prediction_fn(
            model=model,
            feature_df=feature_df,
            feature_cols=feature_cols,
            spec=spec,
        )
    else:
        prediction_payload = {"raw_prediction": 0.0}

    if guardrail_fn is not None:
        guardrail_flags = guardrail_fn(
            prediction_payload=prediction_payload,
            feature_df=feature_df,
            decision_time=decision_time,
            spec=spec,
        )
    else:
        guardrail_flags = {}

    stop_condition_status = _evaluate_stop_conditions(
        spec, guardrail_flags, prediction_payload, feature_df
    )

    result = {
        "model_name": spec["model_name"],
        "model_version": spec["model_version"],
        "decision_time": decision_time,
        "run_timestamp": datetime.now(),
        "source_mode": "live",
        "source_systems": source_systems,
        "source_timestamps": source_timestamps,
        "source_available_times": source_available_times,
        "feature_values": feature_df.to_dict(orient="records")[0] if len(feature_df) > 0 else {},
        "prediction_payload": prediction_payload,
        "guardrail_flags": guardrail_flags,
        "data_quality_flags": {},
        "stop_condition_status": stop_condition_status,
        "inference_parity_status": "pending",
        "warning_flags": [],
    }

    if stop_condition_status:
        result["warning_flags"].extend(stop_condition_status)

    if log_writer_fn is not None:
        log_writer_fn(result, spec)

    return result


def _evaluate_stop_conditions(
    spec: dict,
    guardrail_flags: dict,
    prediction_payload: dict,
    feature_df: pd.DataFrame,
) -> list:
    """Evaluate generic stop conditions based on spec."""
    conditions = spec.get("stop_conditions", [])
    triggered = []

    for condition in conditions:
        if condition == "feature_parity_pass_rate_below_threshold":
            threshold = spec.get("feature_tolerances", {}).get(
                "feature_parity_pass_rate_threshold", 0.95
            )
            if guardrail_flags.get("feature_parity_pass_rate", 1.0) < threshold:
                triggered.append("feature_parity_pass_rate_below_threshold")

        if condition == "temp_current_missing_or_anomalous":
            if feature_df.get("temp_current", pd.Series([np.nan])).isna().any():
                triggered.append("temp_current_missing_or_anomalous")

        if condition == "wind_all_change_30m_ago_missing":
            if "wind_all_change_30m_ago" in feature_df.columns:
                if feature_df["wind_all_change_30m_ago"].isna().any():
                    triggered.append("wind_all_change_30m_ago_missing")

        if condition == "source_data_stale":
            if guardrail_flags.get("source_stale_flag", False):
                triggered.append("source_data_stale")

        if condition == "feature_outside_allowed_range":
            if guardrail_flags.get("feature_range_violation", False):
                triggered.append("feature_outside_allowed_range")

        if condition == "prediction_formula_check_fails":
            if guardrail_flags.get("formula_check_failed", False):
                triggered.append("prediction_formula_check_fails")

        if condition == "guardrail_violation":
            if guardrail_flags.get("guardrail_violation", False):
                triggered.append("guardrail_violation")

        if condition == "raw_data_spike":
            if guardrail_flags.get("temp_spike_flag", False):
                triggered.append("raw_data_spike")

    return triggered
