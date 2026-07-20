# monitoring/data_quality_checks_base.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_data_quality_checks(
    canonical_sources: dict[str, pd.DataFrame],
    spec: dict,
    output_path: str,
    model_name: str = "unknown",
    run_metadata: dict = None,
) -> pd.DataFrame:
    """Generic data quality and drift monitoring checks.

    Checks:
    - schema checks
    - range checks
    - freshness checks
    - missingness checks
    - timestep checks
    - source availability checks
    - distribution drift checks (concept only)
    - baseline divergence checks (concept only)

    Args:
        canonical_sources: Dict of canonical DataFrames by source name.
        spec: Model spec dict with data_quality_rules.
        output_path: Directory for output report.
        model_name: Model name for report filename.

    Returns:
        DataFrame with data quality check results.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dq_rules = spec.get("data_quality_rules", {})
    all_checks = []

    # Schema checks
    all_checks.append(_check_schema(canonical_sources, dq_rules))

    # Range checks
    for source_name, df in canonical_sources.items():
        source_rules = dq_rules.get(source_name, {})
        all_checks.extend(_check_range(df, source_name, source_rules))

    # Freshness checks
    all_checks.append(_check_freshness(canonical_sources, dq_rules))

    # Missingness checks
    for source_name, df in canonical_sources.items():
        source_rules = dq_rules.get(source_name, {})
        all_checks.extend(_check_missingness(df, source_name, source_rules))

    # Timestep checks
    for source_name, df in canonical_sources.items():
        all_checks.extend(_check_timesteps(df, source_name))

    # Source availability checks
    all_checks.append(_check_source_availability(canonical_sources, spec))

    run_metadata = dict(run_metadata or {})
    if run_metadata:
        all_checks = [{**check, **run_metadata} for check in all_checks]

    checks_df = pd.DataFrame(all_checks)
    report_path = output_dir / f"{model_name}_data_quality_report.csv"
    checks_df.to_csv(report_path, index=False)
    logger.info(f"Data quality report written to {report_path}")

    return checks_df


def _check_schema(
    canonical_sources: dict[str, pd.DataFrame],
    dq_rules: dict,
) -> dict:
    """Check that all canonical sources have required schema fields."""
    required_fields = [
        "source_system", "source_mode", "available_time",
        "timestamp", "station_id", "value", "data_quality_flags",
    ]
    results = {"check": "schema", "status": "pass", "detail": ""}
    issues = []

    for source_name, df in canonical_sources.items():
        missing = [c for c in required_fields if c not in df.columns]
        if missing:
            issues.append(f"{source_name}: missing {missing}")
            results["status"] = "fail"

    if issues:
        results["detail"] = "; ".join(issues)
    else:
        results["detail"] = "All sources have required schema"

    return results


def _check_range(
    df: pd.DataFrame, source_name: str, rules: dict
) -> list[dict]:
    """Check numeric fields against configured valid ranges."""
    checks = []
    range_rules = rules.get("range_checks", {})

    for col_name, allowed_range in range_rules.items():
        if col_name not in df.columns:
            checks.append({
                "check": f"range_{source_name}_{col_name}",
                "status": "skip",
                "detail": f"Column '{col_name}' not found in {source_name}",
            })
            continue

        col = df[col_name].dropna()
        if len(col) == 0:
            checks.append({
                "check": f"range_{source_name}_{col_name}",
                "status": "skip",
                "detail": f"Column '{col_name}' has no valid data in {source_name}",
            })
            continue

        if isinstance(allowed_range, dict):
            lo = allowed_range.get("min", -np.inf)
            hi = allowed_range.get("max", np.inf)
        elif isinstance(allowed_range, (list, tuple)) and len(allowed_range) == 2:
            lo, hi = allowed_range
        else:
            lo, hi = -np.inf, np.inf

        out_of_range = ((col < lo) | (col > hi)).sum()
        pct = out_of_range / len(col) * 100

        status = "pass" if pct == 0 else ("warn" if pct < 5 else "fail")
        checks.append({
            "check": f"range_{source_name}_{col_name}",
            "status": status,
            "detail": f"{out_of_range}/{len(col)} ({pct:.1f}%) out of range [{lo}, {hi}]",
        })

    return checks


def _check_freshness(
    canonical_sources: dict[str, pd.DataFrame],
    dq_rules: dict,
) -> dict:
    """Check freshness of data sources based on available_time."""
    now = pd.Timestamp.now()
    max_age_minutes = dq_rules.get("max_data_age_minutes", 60)
    issues = []

    for source_name, df in canonical_sources.items():
        if "available_time" not in df.columns or len(df) == 0:
            issues.append(f"{source_name}: no available_time data")
            continue
        latest = df["available_time"].max()
        age_minutes = (now - pd.Timestamp(latest)).total_seconds() / 60
        if age_minutes > max_age_minutes:
            issues.append(f"{source_name}: {age_minutes:.0f}min old (max {max_age_minutes}min)")

    return {
        "check": "freshness",
        "status": "pass" if not issues else "fail",
        "detail": "; ".join(issues) if issues else f"All sources within {max_age_minutes}min freshness",
    }


def _check_missingness(
    df: pd.DataFrame, source_name: str, rules: dict
) -> list[dict]:
    """Check missingness rates against configured thresholds."""
    checks = []
    max_missing_pct = rules.get("max_missing_pct", 20)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100 if len(df) > 0 else 100

        if missing_pct > max_missing_pct:
            checks.append({
                "check": f"missing_{source_name}_{col}",
                "status": "fail",
                "detail": f"{missing_count}/{len(df)} ({missing_pct:.1f}%) missing",
            })

    return checks


def _check_timesteps(
    df: pd.DataFrame, source_name: str
) -> list[dict]:
    """Check for irregular timesteps in time series data."""
    checks = []
    time_cols = [c for c in ["available_time", "timestamp"] if c in df.columns]
    if not time_cols:
        return checks

    ts = df[time_cols[0]].dropna().sort_values()
    if len(ts) < 2:
        return checks

    diffs = ts.diff().dropna()
    if len(diffs) == 0:
        return checks

    diffs_minutes = diffs.dt.total_seconds() / 60
    irregular = (diffs_minutes > 15).sum()
    gap_count = (diffs_minutes > 60).sum()

    status = "pass" if irregular == 0 else ("warn" if irregular < len(diffs) * 0.05 else "fail")
    checks.append({
        "check": f"timestep_{source_name}",
        "status": status,
        "detail": f"{irregular} irregular steps, {gap_count} gaps > 60min",
    })

    return checks


def _check_source_availability(
    canonical_sources: dict[str, pd.DataFrame],
    spec: dict,
) -> dict:
    """Check that all configured canonical sources are present."""
    required_sources = spec.get("canonical_sources", [])
    available = list(canonical_sources.keys())
    missing = [s for s in required_sources if s not in available]

    return {
        "check": "source_availability",
        "status": "fail" if missing else "pass",
        "detail": f"Required: {required_sources}, available: {available}"
        + (f", missing: {missing}" if missing else ""),
    }
