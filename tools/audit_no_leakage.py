# tools/audit_no_leakage.py
"""
Data leakage audit for ML training datasets.

Checks:
  1. Rainfall point-in-time: rainfall_datetime <= datetime
  2. Rainfall freshness: rainfall_data_age_minutes >= 0
  3. No future rain leakage: has_recent_rainfall_obs=1 implies rainfall_datetime <= datetime
  4. No forbidden rainfall feature names (daily totals, future info, etc.)
  5. Target columns are not used as features
  6. Basic dataset integrity (required columns exist, no NaN in targets)

Usage:
    python tools/audit_no_leakage.py
    python tools/audit_no_leakage.py --dataset data/intraday_ml_train.parquet
"""
import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORT_PATH = Path('reports/leakage_audit_report.json')

# Forbidden feature name patterns (indicate leakage)
FORBIDDEN_RAINFALL_PATTERNS = [
    'daily_total_rainfall',
    'actual_rainfall_today',
    'actual_rainfall_tomorrow',
    'full_day_max_rainfall',
    'rainfall_after_snapshot',
    'rain_end_time',
    'total_event_rainfall',
    'actual_future_rain',
    'full_day_rainfall',
]

# Allowed rainfall feature names (explicit whitelist)
ALLOWED_RAINFALL_FEATURES = [
    'rainfall_interval_15m',
    'rainfall_60m',
    'rainfall_60m_filled',
    'rainfall_60m_missing_flag',
    'rainfall_120m',
    'rainfall_120m_filled',
    'rainfall_120m_missing_flag',
    'rainfall_30m',
    'rainfall_30m_filled',
    'rainfall_30m_missing_flag',
    'rainfall_max_30m',
    'rainfall_max_60m',
    'rainfall_data_age_minutes',
    'has_rainfall_data',
    'has_recent_rainfall_obs',
    'rain_cooling_60m',
    'rain_cooling_120m',
    'rain_cooling_30m',
    'post_peak_rain_flag',
    'morning_peak_then_rain_flag',
    'morning_peak_rain_flag',
    'rain_data_gap_flag',
    'rainfall_datetime',
    # Nowcast features (forecast available at issue_time, point-in-time safe)
    'rain_nc_sum_0_60m',
    'rain_nc_sum_0_120m',
    'rain_nc_any_0_120m',
    'rain_nc_front_loaded_ratio',
    'rain_nc_heavy_0_120m',
    'rain_nc_valid_horizon_count',
    'rain_nc_missing_flag',
    'rain_nc_nearest_mm_sum_30m',
    'rain_nc_nearest_mm_sum_60m',
    'rain_nc_nearest_mm_sum_90m',
    'rain_nc_nearest_mm_sum_120m',
    'rain_nc_mean_r5km_sum_30m',
    'rain_nc_mean_r5km_sum_60m',
    'rain_nc_mean_r5km_sum_90m',
    'rain_nc_mean_r5km_sum_120m',
    'rain_nc_max_r5km_sum_30m',
    'rain_nc_max_r5km_sum_60m',
    'rain_nc_max_r5km_sum_90m',
    'rain_nc_max_r5km_sum_120m',
    'rain_nc_min_r5km_sum_30m',
    'rain_nc_min_r5km_sum_60m',
    'rain_nc_min_r5km_sum_90m',
    'rain_nc_min_r5km_sum_120m',
    'rain_nc_p90_r5km_sum_30m',
    'rain_nc_p90_r5km_sum_60m',
    'rain_nc_p90_r5km_sum_90m',
    'rain_nc_p90_r5km_sum_120m',
    'rain_nc_area_gt0_r5km_sum_30m',
    'rain_nc_area_gt0_r5km_sum_60m',
    'rain_nc_area_gt0_r5km_sum_90m',
    'rain_nc_area_gt0_r5km_sum_120m',
    'rain_nc_area_gt5_r5km_sum_30m',
    'rain_nc_area_gt5_r5km_sum_60m',
    'rain_nc_area_gt5_r5km_sum_90m',
    'rain_nc_area_gt5_r5km_sum_120m',
    'rain_nowcast_age_minutes',
    'rain_nowcast_missing_flag',
]

# Vague rainfall column names that should be renamed (not just unallowed)
VAGUE_RAINFALL_COLUMNS = ['rainfall']  # bare 'rainfall' is ambiguous

# Target columns that must NOT appear as features
TARGET_COLUMNS = ['remaining_upside', 'remaining_downside', 'is_upside_zero', 'is_downside_zero']


def check_nowcast_point_in_time(df) -> dict:
    """Verify issue_time <= datetime for all rows with nowcast data (no future leakage)."""
    result = {'check': 'nowcast_point_in_time', 'status': 'pass', 'count': 0}

    if 'issue_time' not in df.columns or 'datetime' not in df.columns:
        result['status'] = 'skip'
        result['detail'] = 'issue_time or datetime column not found'
        return result

    # Only check rows where nowcast data was matched
    matched = df[df['rain_nowcast_missing_flag'] == 0] if 'rain_nowcast_missing_flag' in df.columns else df
    if len(matched) == 0:
        result['status'] = 'skip'
        result['detail'] = 'no matched nowcast rows'
        return result

    bad = matched[matched['issue_time'] > matched['datetime']]
    result['count'] = len(bad)

    if len(bad) > 0:
        result['status'] = 'fail'
        result['detail'] = f'{len(bad)} rows have issue_time > datetime (future nowcast leak)'
        logger.error("LEAKAGE: %s", result['detail'])
    else:
        logger.info("OK: All issue_time <= datetime")

    return result


def check_nowcast_freshness(df) -> dict:
    """Verify rain_nowcast_age_minutes >= 0 for all matched rows."""
    result = {'check': 'nowcast_freshness', 'status': 'pass', 'count': 0}

    if 'rain_nowcast_age_minutes' not in df.columns:
        result['status'] = 'skip'
        result['detail'] = 'rain_nowcast_age_minutes not found'
        return result

    matched = df[df['rain_nowcast_missing_flag'] == 0] if 'rain_nowcast_missing_flag' in df.columns else df
    bad = matched[matched['rain_nowcast_age_minutes'] < 0]
    result['count'] = len(bad)

    if len(bad) > 0:
        result['status'] = 'fail'
        result['detail'] = f'{len(bad)} rows have negative rain_nowcast_age_minutes'
        logger.error("LEAKAGE: %s", result['detail'])
    else:
        logger.info("OK: All rain_nowcast_age_minutes >= 0")

    return result


def check_no_valid_time_leakage(df) -> dict:
    """Ensure valid_time is not used as data availability time (would leak future)."""
    result = {'check': 'no_valid_time_leakage', 'status': 'pass', 'count': 0}

    valid_time_cols = [c for c in df.columns if 'valid_time' in c.lower()]
    if valid_time_cols:
        result['status'] = 'warn'
        result['detail'] = f'valid_time columns found: {valid_time_cols} — ensure not used as availability time'
        logger.warning("VALID_TIME columns: %s — verify these are not used for point-in-time checks", valid_time_cols)
    else:
        logger.info("OK: No valid_time columns found")

    return result


def check_rainfall_point_in_time(df) -> dict:
    """Verify rainfall_datetime <= datetime for all matched rows."""
    result = {'check': 'rainfall_point_in_time', 'status': 'pass', 'count': 0}

    if 'rainfall_datetime' not in df.columns or 'datetime' not in df.columns:
        result['status'] = 'skip'
        result['detail'] = 'rainfall_datetime or datetime column not found'
        return result

    # Only check rows where rainfall data was matched
    matched = df[df['has_rainfall_data'] == 1] if 'has_rainfall_data' in df.columns else df

    if len(matched) == 0:
        result['status'] = 'skip'
        result['detail'] = 'no matched rainfall rows'
        return result

    bad = matched[matched['rainfall_datetime'] > matched['datetime']]
    result['count'] = len(bad)

    if len(bad) > 0:
        result['status'] = 'fail'
        result['detail'] = f'{len(bad)} rows have rainfall_datetime > datetime (future leak)'
        logger.error("LEAKAGE: %s", result['detail'])
    else:
        logger.info("OK: All rainfall_datetime <= datetime")

    return result


def check_rainfall_freshness(df) -> dict:
    """Verify rainfall_data_age_minutes >= 0 for all matched rows."""
    result = {'check': 'rainfall_freshness', 'status': 'pass', 'count': 0}

    if 'rainfall_data_age_minutes' not in df.columns:
        result['status'] = 'skip'
        result['detail'] = 'rainfall_data_age_minutes not found'
        return result

    matched = df[df['has_rainfall_data'] == 1] if 'has_rainfall_data' in df.columns else df
    bad = matched[matched['rainfall_data_age_minutes'] < 0]
    result['count'] = len(bad)

    if len(bad) > 0:
        result['status'] = 'fail'
        result['detail'] = f'{len(bad)} rows have negative rainfall_data_age_minutes'
        logger.error("LEAKAGE: %s", result['detail'])
    else:
        logger.info("OK: All rainfall_data_age_minutes >= 0")

    return result


def check_no_future_rain_in_recent_flag(df) -> dict:
    """Verify has_recent_rainfall_obs=1 only when rainfall_datetime <= datetime."""
    result = {'check': 'no_future_rain_in_recent_flag', 'status': 'pass', 'count': 0}

    if 'has_recent_rainfall_obs' not in df.columns:
        result['status'] = 'skip'
        result['detail'] = 'has_recent_rainfall_obs not found'
        return result

    recent = df[df['has_recent_rainfall_obs'] == 1]
    if len(recent) == 0:
        result['status'] = 'skip'
        result['detail'] = 'no rows with has_recent_rainfall_obs=1'
        return result

    if 'rainfall_datetime' in recent.columns and 'datetime' in recent.columns:
        bad = recent[recent['rainfall_datetime'] > recent['datetime']]
        result['count'] = len(bad)
        if len(bad) > 0:
            result['status'] = 'fail'
            result['detail'] = f'{len(bad)} recent-rain rows use future rainfall data'
            logger.error("LEAKAGE: %s", result['detail'])
        else:
            logger.info("OK: All recent-rain flags use point-in-time data")
    else:
        result['status'] = 'skip'
        result['detail'] = 'cannot verify without rainfall_datetime and datetime columns'

    return result


def check_forbidden_rainfall_features(df) -> dict:
    """Flag any feature columns containing forbidden leakage patterns."""
    result = {'check': 'forbidden_rainfall_features', 'status': 'pass', 'forbidden_found': []}

    for col in df.columns:
        for pattern in FORBIDDEN_RAINFALL_PATTERNS:
            if pattern in col.lower():
                result['forbidden_found'].append(col)
                logger.error("FORBIDDEN FEATURE: %s (matches pattern: %s)", col, pattern)

    if result['forbidden_found']:
        result['status'] = 'fail'
        result['detail'] = f'Found forbidden features: {result["forbidden_found"]}'
    else:
        logger.info("OK: No forbidden rainfall feature names found")

    return result


def check_unallowed_rainfall_features(df) -> dict:
    """Flag rainfall-related columns that are not in the allowed whitelist."""
    result = {'check': 'unallowed_rainfall_features', 'status': 'pass', 'unallowed_found': []}

    rain_cols = [c for c in df.columns if 'rain' in c.lower()]
    for col in rain_cols:
        if col not in ALLOWED_RAINFALL_FEATURES:
            result['unallowed_found'].append(col)
            logger.warning("UNALLOWED rainfall feature: %s (not in whitelist)", col)

    if result['unallowed_found']:
        result['status'] = 'warn'
        result['detail'] = f'Unallowed features: {result["unallowed_found"]}'
    else:
        logger.info("OK: All rainfall features are in the allowed whitelist")

    return result


def check_target_not_in_features(df) -> dict:
    """Verify target columns are not used as feature columns."""
    result = {'check': 'target_not_in_features', 'status': 'pass', 'targets_as_features': []}

    # Check all candidate model feature lists
    feature_list_paths = [
        Path('models/intraday_ml/active/feature_list.json'),
        Path('models/intraday_ml/feature_list.json'),
    ]
    # Also check rain nowcast candidate dirs
    candidate_dir = Path('models/intraday_ml_rain_nowcast_candidate')
    if candidate_dir.exists():
        for ts_dir in sorted(candidate_dir.iterdir()):
            fl = ts_dir / 'C_rain_aware_nowcast' / 'feature_list.json'
            if fl.exists():
                feature_list_paths.append(fl)

    found_any = False
    for fl_path in feature_list_paths:
        if fl_path.exists():
            found_any = True
            with open(fl_path, 'r') as f:
                feature_list = json.load(f)
            for target in TARGET_COLUMNS:
                if target in feature_list:
                    result['targets_as_features'].append(f"{target} (in {fl_path})")
                    logger.error("TARGET LEAKAGE: %s is in feature list %s!", target, fl_path)

    if not found_any:
        result['status'] = 'skip'
        result['detail'] = 'No feature_list.json found to check'
        return result

    if result['targets_as_features']:
        result['status'] = 'fail'
        result['detail'] = f'Targets used as features: {result["targets_as_features"]}'
    else:
        logger.info("OK: No target columns found in any feature list")

    return result


def check_dataset_integrity(df) -> dict:
    """Basic dataset integrity checks."""
    result = {'check': 'dataset_integrity', 'status': 'pass', 'issues': []}

    # Check required columns
    required = ['datetime', 'date', 'temp', 'tmax', 'tmin',
                'max_so_far', 'min_so_far', 'remaining_upside', 'remaining_downside']
    missing = [c for c in required if c not in df.columns]
    if missing:
        result['issues'].append(f'Missing required columns: {missing}')
        result['status'] = 'fail'

    # Check NaN in targets
    for target in ['remaining_upside', 'remaining_downside']:
        if target in df.columns:
            nan_count = df[target].isna().sum()
            if nan_count > 0:
                result['issues'].append(f'{target} has {nan_count} NaN values')
                result['status'] = 'fail'

    if result['status'] == 'pass':
        logger.info("OK: Dataset integrity checks passed")
    else:
        for issue in result['issues']:
            logger.error("INTEGRITY: %s", issue)

    return result


def check_vague_rainfall_columns(df) -> dict:
    """Flag vague rainfall column names that should be renamed for clarity."""
    result = {'check': 'vague_rainfall_column_names', 'status': 'pass', 'found': []}

    for col in VAGUE_RAINFALL_COLUMNS:
        if col in df.columns:
            result['found'].append(col)
            logger.warning("VAGUE rainfall column: '%s' — rename to describe time semantics", col)

    if result['found']:
        result['status'] = 'warn'
        result['detail'] = f"Vague column names found: {result['found']}"
    else:
        logger.info("OK: No vague rainfall column names")

    return result


def main():
    parser = argparse.ArgumentParser(description='Audit ML datasets for data leakage.')
    parser.add_argument('--dataset', default='data/intraday_ml_train.parquet',
                        help='Path to dataset parquet file')
    args = parser.parse_args()

    import pandas as pd

    logger.info("=" * 60)
    logger.info("Leakage Audit — Starting")
    logger.info("Dataset: %s", args.dataset)
    logger.info("=" * 60)

    df = pd.read_parquet(args.dataset)
    logger.info("Dataset shape: %s", df.shape)

    results = []

    # Run all checks
    checks = [
        check_rainfall_point_in_time,
        check_rainfall_freshness,
        check_no_future_rain_in_recent_flag,
        check_nowcast_point_in_time,
        check_nowcast_freshness,
        check_no_valid_time_leakage,
        check_forbidden_rainfall_features,
        check_vague_rainfall_columns,
        check_unallowed_rainfall_features,
        check_target_not_in_features,
        check_dataset_integrity,
    ]

    for check_fn in checks:
        logger.info("\n--- %s ---", check_fn.__name__)
        result = check_fn(df)
        results.append(result)

    # Summary
    failures = [r for r in results if r.get('status') == 'fail']
    warnings = [r for r in results if r.get('status') == 'warn']
    passes = [r for r in results if r.get('status') == 'pass']

    logger.info("\n" + "=" * 60)
    logger.info("Audit Summary: %d pass, %d warn, %d fail", len(passes), len(warnings), len(failures))
    logger.info("=" * 60)

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'dataset_shape': list(df.shape),
        'results': results,
        'summary': {
            'pass': len(passes),
            'warn': len(warnings),
            'fail': len(failures),
            'overall': 'pass' if not failures else 'fail',
        }
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", REPORT_PATH)

    if failures:
        logger.error("AUDIT FAILED — %d check(s) failed", len(failures))
        for r in failures:
            logger.error("  FAIL: %s — %s", r['check'], r.get('detail', ''))
    else:
        logger.info("AUDIT PASSED — all checks clean")


if __name__ == '__main__':
    main()
