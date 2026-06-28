# monitoring/inference_parity_check_base.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_parity_check(
    inference_log_path: str,
    raw_sources_dict: dict[str, pd.DataFrame],
    model_spec_path: str,
    output_path: str,
    source_adapter_fn=None,
    feature_builder_fn=None,
    model_name: str = "unknown",
) -> dict:
    """Generic replay parity check.

    For each logged inference row:
    1. Read decision_time
    2. Rebuild features using replay sources and same shared feature builder
    3. Compare logged live feature values to replay feature values
    4. Apply tolerances defined in model spec
    5. Output feature-level pass/fail

    Args:
        inference_log_path: Path to the inference log parquet file.
        raw_sources_dict: Dict of raw source DataFrames for replay.
        model_spec_path: Path to model spec YAML.
        output_path: Directory for output reports.
        source_adapter_fn: Optional source adapter function.
        feature_builder_fn: Shared feature builder function.
        model_name: Model name for report filenames.

    Returns:
        Summary dict of parity check results.
    """
    import yaml

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_spec_path, "r") as f:
        spec = yaml.safe_load(f)

    log_path = Path(inference_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Inference log not found: {inference_log_path}")

    inference_log = pd.read_parquet(log_path)
    tolerances = spec.get("feature_tolerances", {})
    model_name = spec.get("model_name", model_name)

    canonical_sources = {}
    if source_adapter_fn is not None:
        for source_name, raw_df in raw_sources_dict.items():
            canonical_sources[source_name] = source_adapter_fn(
                raw_df, source_name, "replay"
            )
    else:
        canonical_sources = raw_sources_dict

    all_comparisons = []
    n_rows = len(inference_log)
    n_comparisons = 0
    total_passes = 0
    feature_passes = {}
    feature_counts = {}
    max_abs_diff = 0.0
    missing_feature_count = 0
    total_feature_checks = 0
    spike_detection_count = 0

    for idx, row in inference_log.iterrows():
        decision_time = row.get("decision_time")
        if decision_time is None:
            continue

        decision_time = pd.Timestamp(decision_time)

        if feature_builder_fn is not None:
            replay_features = feature_builder_fn(
                decision_time=decision_time,
                canonical_sources=canonical_sources,
                spec=spec,
                mode="replay",
            )
        else:
            continue

        live_features = row.get("feature_values", {})
        if isinstance(live_features, str):
            import json
            live_features = json.loads(live_features)

        if isinstance(live_features, dict):
            for fname, live_val in live_features.items():
                if fname not in replay_features.columns:
                    missing_feature_count += 1
                    continue

                replay_val = replay_features[fname].iloc[0] if len(replay_features) > 0 else np.nan

                total_feature_checks += 1
                fname_passes = feature_passes.get(fname, 0)
                fname_count = feature_counts.get(fname, 0)

                live_num = _to_float(live_val)
                replay_num = _to_float(replay_val)

                if np.isnan(live_num) and np.isnan(replay_num):
                    is_pass = True
                    abs_diff = 0.0
                    rel_diff = 0.0
                elif np.isnan(live_num) or np.isnan(replay_num):
                    is_pass = False
                    abs_diff = np.nan
                    rel_diff = np.nan
                else:
                    abs_diff = abs(live_num - replay_num)
                    rel_diff = abs_diff / (abs(replay_num) + 1e-10)
                    tol = tolerances.get(fname, 0.1)
                    is_pass = abs_diff <= tol

                if is_pass:
                    feature_passes[fname] = fname_passes + 1
                    total_passes += 1
                feature_counts[fname] = fname_count + 1
                n_comparisons += 1

                if not np.isnan(abs_diff) and abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff

                if not is_pass and abs_diff >= 5.0:
                    spike_detection_count += 1

                all_comparisons.append({
                    "decision_time": decision_time,
                    "feature_name": fname,
                    "live_value": live_num,
                    "replay_value": replay_num,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "pass_flag": is_pass,
                })

    report_df = pd.DataFrame(all_comparisons)
    report_path = output_dir / f"{model_name}_feature_parity_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"Parity report written to {report_path}")

    overall_pass_rate = total_passes / n_comparisons if n_comparisons > 0 else 1.0
    pass_rate_by_feature = {
        fname: (feature_passes.get(fname, 0) / feature_counts.get(fname, 1))
        for fname in feature_counts
    }
    missing_feature_rate = missing_feature_count / total_feature_checks if total_feature_checks > 0 else 0.0

    summary = {
        "n_inference_rows": n_rows,
        "n_feature_comparisons": n_comparisons,
        "overall_pass_rate": overall_pass_rate,
        "pass_rate_by_feature": pass_rate_by_feature,
        "max_absolute_diff": max_abs_diff,
        "missing_feature_rate": missing_feature_rate,
        "spike_detection_rate": spike_detection_count / n_comparisons if n_comparisons > 0 else 0.0,
        "guardrail_violation_count": int(inference_log["guardrail_flags"].apply(
            lambda x: _count_violations(x) if isinstance(x, dict) else 0
        ).sum()) if "guardrail_flags" in inference_log.columns else 0,
    }

    summary_path = output_dir / f"{model_name}_feature_parity_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Feature Parity Summary - {model_name}\n")
        f.write(f"{'=' * 60}\n")
        for key, val in summary.items():
            if isinstance(val, dict):
                f.write(f"\n{key}:\n")
                for k, v in val.items():
                    f.write(f"  {k}: {v:.4f}\n")
            elif isinstance(val, float):
                f.write(f"{key}: {val:.4f}\n")
            else:
                f.write(f"{key}: {val}\n")

    logger.info(f"Parity summary written to {summary_path}")
    return summary


def _to_float(val):
    try:
        return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _count_violations(flags):
    if isinstance(flags, dict):
        return sum(1 for v in flags.values() if v)
    return 0
