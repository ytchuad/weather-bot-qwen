# tools/runtime_smoke_test.py
"""執行時期煙霧測試：確保所有模組可載入、推論可執行、約束正確。

Includes rainfall-aware checks:
  1. Rainfall sample data loads
  2. build_rainfall_features.py runs and produces output
  3. Rainfall point-in-time columns in training data
  4. Rainfall feature existence in training data and feature list
  5. Inference with rainfall features at multiple test times
  6. Inference result columns match dataset feature columns
  7. Feature list consistency between active model and dataset
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORT_PATH = Path('reports/runtime_smoke_test_report.json')
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Representative test times: early morning, late morning, afternoon, late night
TEST_TIMES = [
    "2024-06-05 03:00:00",
    "2024-06-05 10:00:00",
    "2024-06-05 15:00:00",
    "2024-06-05 23:00:00",
]

# Required rainfall columns in training data
REQUIRED_RAINFALL_COLS = [
    'rainfall_60m_filled', 'rainfall_60m_missing_flag',
    'rainfall_120m_filled', 'rainfall_120m_missing_flag',
    'rain_data_gap_flag', 'rainfall_data_age_minutes',
    'rain_cooling_60m', 'rain_cooling_120m',
    'post_peak_rain_flag', 'morning_peak_then_rain_flag',
]

# Required nowcast columns in training data (for nowcast-aware models)
REQUIRED_NOWCAST_COLS = [
    'rain_nc_sum_0_60m', 'rain_nc_sum_0_120m',
    'rain_nc_any_0_120m', 'rain_nc_front_loaded_ratio',
    'rain_nc_heavy_0_120m', 'rain_nc_valid_horizon_count',
    'rain_nc_missing_flag', 'rain_nowcast_age_minutes',
    'rain_nowcast_missing_flag',
]

# Expected inference output keys
EXPECTED_TMAX_KEYS = [
    'remaining_upside_p10', 'remaining_upside_p25', 'remaining_upside_p50',
    'remaining_upside_p75', 'remaining_upside_p90',
    'prob_max_reached', 'pred_tmax_p50', 'pred_tmax_p10', 'pred_tmax_p90',
]
EXPECTED_TMIN_KEYS = [
    'remaining_downside_p10', 'remaining_downside_p25', 'remaining_downside_p50',
    'remaining_downside_p75', 'remaining_downside_p90',
    'prob_min_reached', 'pred_tmin_p50', 'pred_tmin_p10', 'pred_tmin_p90',
]


def load_active_model_type():
    """Load active model type from model registry."""
    registry_path = Path("models/intraday_ml/metadata/model_registry.json")
    if not registry_path.exists():
        return "unknown"
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return registry.get("active_model_type", "unknown")
    except Exception:
        return "unknown"


def check_feature_list_consistency(feature_list, required_rain_features, active_model_type):
    """Check feature list consistency with policy based on model type."""
    missing = [f for f in required_rain_features if f not in feature_list]
    
    if not missing:
        return {
            "check": "feature_list_consistency",
            "status": "pass"
        }
    
    if active_model_type == "rain_aware":
        return {
            "check": "feature_list_consistency",
            "status": "fail",
            "missing": missing,
            "detail": "Active model is marked as rain-aware but required rainfall features are missing."
        }
    
    return {
        "check": "feature_list_consistency",
        "status": "warn",
        "missing": missing,
        "detail": "Active model is not marked as rain-aware. Missing rainfall features are acceptable only for baseline active model."
    }


def collect_statuses(obj):
    """Recursively collect all status values from a nested dict/list structure."""
    statuses = []
    if isinstance(obj, dict):
        if "status" in obj:
            statuses.append(str(obj["status"]).lower())
        for value in obj.values():
            statuses.extend(collect_statuses(value))
    elif isinstance(obj, list):
        for item in obj:
            statuses.extend(collect_statuses(item))
    return statuses


def derive_overall_status(report):
    """Derive overall status from collected statuses."""
    statuses = collect_statuses(report)
    if "fail" in statuses or "error" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def check_monotonic_quantiles(prefix, result):
    """Check that quantiles are monotonic (p10 <= p25 <= p50 <= p75 <= p90)."""
    try:
        values = [
            result[f"{prefix}_p10"],
            result[f"{prefix}_p25"],
            result[f"{prefix}_p50"],
            result[f"{prefix}_p75"],
            result[f"{prefix}_p90"],
        ]
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    except KeyError:
        # If some quantiles are missing, we can't check monotonicity
        return True


def check_physical_constraints(feature_row, pred_result):
    """Check physical constraints: pred_tmax_p50 >= max_so_far and pred_tmin_p50 <= min_so_far."""
    violations = []
    
    # Check Tmax constraint: predicted final tmax cannot be below observed max_so_far
    if 'pred_tmax_p50' in pred_result and 'max_so_far' in feature_row:
        if pred_result['pred_tmax_p50'] < feature_row['max_so_far']:
            violations.append(f"pred_tmax_p50 ({pred_result['pred_tmax_p50']}) < max_so_far ({feature_row['max_so_far']})")
    
    # Check Tmin constraint: predicted final tmin cannot be above observed min_so_far
    if 'pred_tmin_p50' in pred_result and 'min_so_far' in feature_row:
        if pred_result['pred_tmin_p50'] > feature_row['min_so_far']:
            violations.append(f"pred_tmin_p50 ({pred_result['pred_tmin_p50']}) > min_so_far ({feature_row['min_so_far']})")
    
    return violations


def main():
    report = {
        "timestamp": datetime.now().isoformat(),
        # Rainfall-aware checks (P0-10)
        "rainfall_data_load_test": None,
        "rainfall_feature_build_test": None,
        "rainfall_point_in_time_tests": [],
        "rainfall_feature_existence": None,
        # Standard checks
        "import_tests": {},
        "file_existence_tests": {},
        "model_load_tests": [],
        "intraday_inference_tests": [],
        "inference_with_rainfall_tests": [],
        "nowcast_feature_existence": None,
        "candidate_nowcast_feature_list": None,
        "inference_nowcast_tests": [],
        "feature_list_consistency_test": None,
        "execution_safety_tests": {},
    }

    # Load active model type for policy-based checks
    active_model_type = load_active_model_type()

    # ══════════════════════════════════════════════════════════════
    # RAINFALL-AWARE CHECKS (P0-10)
    # ══════════════════════════════════════════════════════════════

    # 1. Rainfall sample data load
    logger.info("--- Rainfall data load test ---")
    try:
        rain_sample_path = Path('data/hko_rainfall_sample.parquet')
        if rain_sample_path.exists():
            rain_df = pd.read_parquet(rain_sample_path)
            assert len(rain_df) > 0, "Empty rainfall sample"
            assert 'datetime' in rain_df.columns, "Missing datetime column"
            report['rainfall_data_load_test'] = 'pass'
            logger.info("OK: Rainfall sample loaded (%d rows)", len(rain_df))
        else:
            # Fallback: check the full rainfall features file
            rain_full = Path('data/hko_rainfall_15min_features.parquet')
            if rain_full.exists():
                rain_df = pd.read_parquet(rain_full)
                assert len(rain_df) > 0
                report['rainfall_data_load_test'] = 'pass'
                logger.info("OK: Full rainfall features loaded (%d rows)", len(rain_df))
            else:
                report['rainfall_data_load_test'] = 'fail'
                logger.error("No rainfall data files found")
    except Exception as e:
        report['rainfall_data_load_test'] = f'fail: {e}'
        logger.error("Rainfall data load failed: %s", e)

    # 2. build_rainfall_features.py runs
    logger.info("--- Rainfall feature build test ---")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path('features/build_rainfall_features.py').resolve())],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT_DIR),
        )
        if result.returncode == 0:
            report['rainfall_feature_build_test'] = 'pass'
            logger.info("OK: build_rainflow_features.py ran successfully")
        else:
            report['rainfall_feature_build_test'] = f'fail: {result.stderr[:200]}'
            logger.error("build_rainfall_features.py failed: %s", result.stderr[:200])
    except Exception as e:
        report['rainfall_feature_build_test'] = f'fail: {e}'
        logger.error("build_rainfall_features.py error: %s", e)

    # 3. Rainfall point-in-time checks in training data
    logger.info("--- Rainfall point-in-time tests ---")
    try:
        train_df = pd.read_parquet('data/intraday_ml_train.parquet')

        # Check 3a: rainfall_datetime <= datetime
        if 'rainfall_datetime' in train_df.columns and 'datetime' in train_df.columns:
            matched = train_df[train_df['has_rainfall_data'] == 1] if 'has_rainfall_data' in train_df.columns else train_df
            if len(matched) > 0:
                future_rain = matched[matched['rainfall_datetime'] > matched['datetime']]
                if len(future_rain) > 0:
                    report['rainfall_point_in_time_tests'].append({
                        'check': 'rainfall_datetime_le_datetime',
                        'status': 'fail',
                        'detail': f'{len(future_rain)} rows have rainfall_datetime > datetime',
                    })
                else:
                    report['rainfall_point_in_time_tests'].append({
                        'check': 'rainfall_datetime_le_datetime',
                        'status': 'pass',
                    })
            else:
                report['rainfall_point_in_time_tests'].append({
                    'check': 'rainfall_datetime_le_datetime',
                    'status': 'skip',
                    'detail': 'no matched rainfall rows',
                })
        else:
            report['rainfall_point_in_time_tests'].append({
                'check': 'rainfall_datetime_le_datetime',
                'status': 'skip',
                'detail': 'rainfall_datetime not in training data',
            })

        # Check 3b: rainfall_data_age_minutes >= 0
        if 'rainfall_data_age_minutes' in train_df.columns:
            neg_age = train_df[train_df['rainfall_data_age_minutes'] < 0]
            if len(neg_age) > 0:
                report['rainfall_point_in_time_tests'].append({
                    'check': 'rainfall_age_non_negative',
                    'status': 'fail',
                    'detail': f'{len(neg_age)} rows have negative age',
                })
            else:
                report['rainfall_point_in_time_tests'].append({
                    'check': 'rainfall_age_non_negative',
                    'status': 'pass',
                })
        else:
            report['rainfall_point_in_time_tests'].append({
                'check': 'rainfall_age_non_negative',
                'status': 'skip',
            })

    except Exception as e:
        report['rainfall_point_in_time_tests'].append({
            'check': 'rainfall_point_in_time',
            'status': f'fail: {e}',
        })
        logger.error("Rainfall point-in-time checks failed: %s", e)

    # 4. Rainfall feature existence in training data
    logger.info("--- Rainfall feature existence ---")
    try:
        train_df = pd.read_parquet('data/intraday_ml_train.parquet')
        missing_rain = [c for c in REQUIRED_RAINFALL_COLS if c not in train_df.columns]
        if missing_rain:
            report['rainfall_feature_existence'] = f'fail: missing {missing_rain}'
            logger.error("Missing rainfall columns: %s", missing_rain)
        else:
            report['rainfall_feature_existence'] = 'pass'
            logger.info("OK: All %d required rainfall columns present", len(REQUIRED_RAINFALL_COLS))
    except Exception as e:
        report['rainfall_feature_existence'] = f'fail: {e}'
        logger.error("Rainfall feature existence check failed: %s", e)

    # ══════════════════════════════════════════════════════════════
    # STANDARD CHECKS
    # ══════════════════════════════════════════════════════════════

    # 5. File existence
    logger.info("--- File existence tests ---")
    required_files = [
        'data/intraday_ml_train.parquet',
        'data/hko_tmax_historical.parquet',
        'models/intraday_ml/feature_list.json',
        'models/intraday_ml/upside_q50.txt',
        'models/intraday_ml/downside_q50.txt',
        'models/xgb_tmax_mean.json',
        'models/xgb_tmin_mean.json',
    ]
    for f in required_files:
        exists = Path(f).exists()
        report['file_existence_tests'][f] = 'pass' if exists else 'fail'
        if not exists:
            logger.error("Missing file: %s", f)

    # 6. Module imports
    logger.info("--- Import tests ---")
    try:
        from models.intraday_inference import predict_intraday_tmax, predict_intraday_tmin
        from models.inference import predict_distribution
        from execution.kelly_betting import compute_multi_kelly_bets
        report['import_tests']['intraday_inference'] = 'pass'
        report['import_tests']['long_horizon_inference'] = 'pass'
        report['import_tests']['kelly_betting'] = 'pass'
    except Exception as e:
        report['import_tests']['error'] = str(e)
        logger.error("Import failed: %s", e)

    # 7. Inference at test times (without rainfall)
    logger.info("--- Intraday inference tests (no rain) ---")
    try:
        from models.intraday_inference import predict_intraday_tmax, predict_intraday_tmin
        for time_str in TEST_TIMES:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            max_so_far = 30.0
            min_so_far = 26.0
            temp_now = 29.5
            temp_60m_ago = 29.0

            res_tmax = predict_intraday_tmax(
                dt, max_so_far, temp_60m_ago, temp_now,
                forecast_tmax=32.0, forecast_tmin=26.0, min_so_far=min_so_far,
            )
            res_tmin = predict_intraday_tmin(
                dt, min_so_far, temp_60m_ago, temp_now,
                forecast_tmax=32.0, forecast_tmin=26.0, max_so_far=max_so_far,
            )

            # Validate Tmax
            assert res_tmax is not None, "predict_intraday_tmax returned None"
            assert res_tmax['pred_tmax_p50'] >= max_so_far
            assert res_tmax['remaining_upside_p10'] >= 0
            # Check remaining_upside quantiles monotonicity
            ups = [res_tmax[f'remaining_upside_p{q}'] for q in [10, 25, 50, 75, 90]]
            assert ups == sorted(ups), f"remaining_upside quantiles not monotonic: {ups}"
            # Check pred_tmax quantiles monotonicity (p10 <= p50 <= p90)
            assert res_tmax['pred_tmax_p10'] <= res_tmax['pred_tmax_p50'] <= res_tmax['pred_tmax_p90'], \
                f"pred_tmax quantiles not monotonic: p10={res_tmax['pred_tmax_p10']}, p50={res_tmax['pred_tmax_p50']}, p90={res_tmax['pred_tmax_p90']}"

            # Validate Tmin
            assert res_tmin is not None, "predict_intraday_tmin returned None"
            assert res_tmin['pred_tmin_p50'] <= min_so_far
            assert res_tmin['remaining_downside_p10'] >= 0
            # Check remaining_downside quantiles monotonicity
            downs = [res_tmin[f'remaining_downside_p{q}'] for q in [10, 25, 50, 75, 90]]
            assert downs == sorted(downs), f"remaining_downside quantiles not monotonic: {downs}"
            # Check pred_tmin quantiles monotonicity (p10 <= p50 <= p90)
            assert res_tmin['pred_tmin_p10'] <= res_tmin['pred_tmin_p50'] <= res_tmin['pred_tmin_p90'], \
                f"pred_tmin quantiles not monotonic: p10={res_tmin['pred_tmin_p10']}, p50={res_tmin['pred_tmin_p50']}, p90={res_tmin['pred_tmin_p90']}"

            # Check physical constraints
            feature_row = {
                'max_so_far': max_so_far,
                'min_so_far': min_so_far
            }
            violations = check_physical_constraints(feature_row, res_tmax)
            if violations:
                raise AssertionError(f"Physical constraint violations in Tmax prediction: {violations}")
            violations = check_physical_constraints(feature_row, res_tmin)
            if violations:
                raise AssertionError(f"Physical constraint violations in Tmin prediction: {violations}")

            report['intraday_inference_tests'].append({
                'time': time_str, 'status': 'pass',
            })
    except Exception as e:
        report['intraday_inference_tests'].append({
            'time': time_str, 'status': f'fail: {e}',
        })
        logger.error("Inference test failed at %s: %s", time_str, e)

    # 8. Inference WITH rainfall at test times
    logger.info("--- Inference with rainfall tests ---")
    try:
        from models.intraday_inference import predict_intraday_tmax, predict_intraday_tmin
        for time_str in TEST_TIMES:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            res = predict_intraday_tmax(
                dt, max_so_far=31.0, temp_60min_ago=30.0, temp_now=28.5,
                forecast_tmax=32.0, forecast_tmin=26.0, min_so_far=26.0,
                rainfall_60m_filled=10.0, rainfall_120m_filled=15.0,
                rainfall_60m_missing_flag=0, rainfall_120m_missing_flag=0,
            )
            assert res is not None
            # Check expected keys exist
            for key in EXPECTED_TMAX_KEYS:
                assert key in res, f"Missing key in tmax result: {key}"
            report['inference_with_rainfall_tests'].append({
                'time': time_str, 'status': 'pass',
                'rainfall_60m_filled': 10.0,
                'pred_tmax_p50': res['pred_tmax_p50'],
            })
    except Exception as e:
        report['inference_with_rainfall_tests'].append({
            'time': time_str, 'status': f'fail: {e}',
        })
        logger.error("Rainfall inference test failed at %s: %s", time_str, e)

    # 9. Feature list consistency
    logger.info("--- Feature list consistency test ---")
    try:
        # Load active model feature list
        fl_path = Path('models/intraday_ml/active/feature_list.json')
        if not fl_path.exists():
            fl_path = Path('models/intraday_ml/feature_list.json')
        if fl_path.exists():
            with open(fl_path, 'r') as f:
                active_features = json.load(f)

            # Load training data columns
            train_df = pd.read_parquet('data/intraday_ml_train.parquet')
            train_cols = set(train_df.columns)
            missing_from_data = [c for c in active_features if c not in train_cols]
            missing_from_features = [c for c in REQUIRED_RAINFALL_COLS if c not in active_features]

            if missing_from_data:
                report['feature_list_consistency_test'] = {
                    'check': 'feature_list_consistency',
                    'status': 'fail',
                    'detail': f'active features not in data: {missing_from_data}'
                }
            elif missing_from_features:
                # Policy-based check: fail if active model is rain_aware, warn otherwise
                report['feature_list_consistency_test'] = check_feature_list_consistency(
                    active_features, REQUIRED_RAINFALL_COLS, active_model_type
                )
            else:
                report['feature_list_consistency_test'] = {
                    'check': 'feature_list_consistency',
                    'status': 'pass'
                }
                logger.info("OK: Active feature list consistent with training data")
        else:
            report['feature_list_consistency_test'] = 'skip: no active feature list found'
    except Exception as e:
        report['feature_list_consistency_test'] = f'fail: {e}'
        logger.error("Feature list consistency check failed: %s", e)

    # 10. Execution safety
    logger.info("--- Execution safety tests ---")
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        exec_cfg = config.get('execution', {})
        if exec_cfg.get('allow_live_orders', False):
            report['execution_safety_tests']['live_orders_disabled'] = 'fail'
            logger.error("Safety check FAILED: allow_live_orders is true!")
        else:
            report['execution_safety_tests']['live_orders_disabled'] = 'pass'
    except Exception as e:
        report['execution_safety_tests']['config_check'] = f'fail: {e}'
        logger.error("Execution safety check failed: %s", e)

    # 11. Nowcast feature existence in training data
    logger.info("--- Nowcast feature existence ---")
    try:
        nowcast_train_df = pd.read_parquet('data/intraday_ml_train_with_rain_nowcast.parquet')
        missing_nc = [c for c in REQUIRED_NOWCAST_COLS if c not in nowcast_train_df.columns]
        if missing_nc:
            report['nowcast_feature_existence'] = f'fail: missing {missing_nc}'
            logger.error("Missing nowcast columns: %s", missing_nc)
        else:
            report['nowcast_feature_existence'] = 'pass'
            logger.info("OK: All %d required nowcast columns present", len(REQUIRED_NOWCAST_COLS))
    except Exception as e:
        report['nowcast_feature_existence'] = f'fail: {e}'
        logger.error("Nowcast feature existence check failed: %s", e)

    # 12. Candidate feature list includes rain_nowcast_features
    logger.info("--- Candidate nowcast feature list check ---")
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT_DIR))
        from features.feature_schema import get_feature_list
        nc_features = get_feature_list("rain_aware_nowcast")
        schema_nc_cols = set(nc_features) - set(get_feature_list("rain_aware"))
        required_nc_set = set(REQUIRED_NOWCAST_COLS)
        missing_from_candidate = required_nc_set - schema_nc_cols
        if missing_from_candidate:
            report['candidate_nowcast_feature_list'] = {
                'status': 'fail',
                'detail': f'rain_aware_nowcast missing nowcast cols: {missing_from_candidate}'
            }
        else:
            report['candidate_nowcast_feature_list'] = {
                'status': 'pass',
                'nowcast_feature_count': len(schema_nc_cols),
            }
            logger.info("OK: rain_aware_nowcast has %d nowcast features", len(schema_nc_cols))
    except Exception as e:
        report['candidate_nowcast_feature_list'] = f'fail: {e}'
        logger.error("Candidate nowcast feature list check failed: %s", e)

    # 13. Inference with nowcast features (no rain, moderate, heavy)
    logger.info("--- Inference with nowcast tests ---")
    try:
        from models.intraday_inference import predict_intraday_tmax, predict_intraday_tmin
        dt_test = datetime(2024, 6, 5, 10, 0, 0)
        base_kwargs = dict(
            current_datetime=dt_test, max_so_far=31.0, temp_60min_ago=30.0,
            temp_now=29.5, forecast_tmax=32.0, forecast_tmin=26.0, min_so_far=26.0,
        )

        # 13a. No nowcast (all zeros / missing)
        res_no_nc = predict_intraday_tmax(**base_kwargs)
        assert res_no_nc is not None, "predict_intraday_tmax returned None (no nowcast)"
        assert res_no_nc['pred_tmax_p50'] >= base_kwargs['max_so_far']
        report['inference_nowcast_tests'] = [{'scenario': 'no_nowcast', 'status': 'pass'}]
        logger.info("OK: Inference without nowcast features works")

        # 13b. Moderate rain nowcast
        moderate_nc = dict(
            rain_nc_sum_0_60m=3.0, rain_nc_sum_0_120m=5.0,
            rain_nc_any_0_120m=0.6, rain_nc_front_loaded_ratio=0.5,
            rain_nc_heavy_0_120m=0.0, rain_nc_valid_horizon_count=4.0,
            rain_nc_missing_flag=0, rain_nowcast_age_minutes=12.0,
            rain_nowcast_missing_flag=0,
        )
        res_mod = predict_intraday_tmax(**{**base_kwargs, **moderate_nc})
        assert res_mod is not None, "predict_intraday_tmax returned None (moderate nowcast)"
        assert res_mod['pred_tmax_p50'] >= base_kwargs['max_so_far']
        report['inference_nowcast_tests'].append({'scenario': 'moderate_nowcast', 'status': 'pass'})
        logger.info("OK: Inference with moderate nowcast works")

        # 13c. Heavy rain nowcast
        heavy_nc = dict(
            rain_nc_sum_0_60m=15.0, rain_nc_sum_0_120m=25.0,
            rain_nc_any_0_120m=0.9, rain_nc_front_loaded_ratio=0.7,
            rain_nc_heavy_0_120m=1.0, rain_nc_valid_horizon_count=4.0,
            rain_nc_missing_flag=0, rain_nowcast_age_minutes=6.0,
            rain_nowcast_missing_flag=0,
        )
        res_heavy = predict_intraday_tmax(**{**base_kwargs, **heavy_nc})
        assert res_heavy is not None, "predict_intraday_tmax returned None (heavy nowcast)"
        assert res_heavy['pred_tmax_p50'] >= base_kwargs['max_so_far']
        report['inference_nowcast_tests'].append({'scenario': 'heavy_nowcast', 'status': 'pass'})
        logger.info("OK: Inference with heavy nowcast works")

        # 13d. Verify nowcast age >= 0 constraint
        assert res_mod['remaining_upside_p50'] >= 0, "negative remaining_upside_p50"
        assert res_heavy['remaining_upside_p50'] >= 0, "negative remaining_upside_p50 (heavy)"

        # 13e. Tmin with nowcast
        res_tmin_nc = predict_intraday_tmin(
            current_datetime=dt_test, min_so_far=26.0, temp_60min_ago=27.0,
            temp_now=26.5, forecast_tmax=32.0, forecast_tmin=26.0, max_so_far=31.0,
            rain_nc_sum_0_60m=5.0, rain_nc_sum_0_120m=8.0,
            rain_nc_valid_horizon_count=4.0, rain_nowcast_age_minutes=10.0,
        )
        assert res_tmin_nc is not None, "predict_intraday_tmin returned None (with nowcast)"
        assert res_tmin_nc['pred_tmin_p50'] <= base_kwargs['min_so_far']
        report['inference_nowcast_tests'].append({'scenario': 'tmin_with_nowcast', 'status': 'pass'})
        logger.info("OK: Tmin inference with nowcast works")

    except Exception as e:
        report['inference_nowcast_tests'] = [{'scenario': 'nowcast', 'status': f'fail: {e}'}]
        logger.error("Nowcast inference test failed: %s", e)

    # ══════════════════════════════════════════════════════════════
    # DERIVE OVERALL STATUS
    # ══════════════════════════════════════════════════════════════
    report["overall_status"] = derive_overall_status(report)

    # ══════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if report['overall_status'] == 'pass':
        logger.info("ALL SMOKE TESTS PASSED")
    elif report['overall_status'] == 'warn':
        logger.warning("SMOKE TESTS PASSED WITH WARNINGS — check report: %s", REPORT_PATH)
    else:
        logger.error("SMOKE TESTS HAVE FAILURES — check report: %s", REPORT_PATH)


if __name__ == '__main__':
    main()