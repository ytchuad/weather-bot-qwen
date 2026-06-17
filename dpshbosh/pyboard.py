"""Manual multi-model testing dashboard helpers.

Provides model discovery, inference across all models, comparison tables,
gate matrix, and single-strategy paper apply — for the Manual Testing Mode.
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

_HKT_OFFSET = __import__('datetime', fromlist=[]).timedelta(hours=8)
def hkt_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + _HKT_OFFSET

# ── Candidate discovery paths ────────────────────────────────────
CANDIDATE_PRIMARY = Path('models/intraday_ai_rain_nowcast_candidate')
CANDIDATE_FALLBACK = Path('models/intraday_ml_rain_nowcast_candidate')

PRODUCTION_MODEL_DIRS = {
    'baseline_intraday': Path('models/intraday_ml'),
    'rain_nowcast': Path('models/intraday_ml_rain_nowcast'),
}


def _resolve_candidate_base() -> Path:
    """Return the candidate base directory, trying primary then fallback."""
    if CANDIDATE_PRIMARY.exists():
        return CANDIDATE_PRIMARY
    if CANDIDATE_FALLBACK.exists():
        return CANDIDATE_FALLBACK
    return CANDIDATE_PRIMARY  # return the desired path even if absent


def get_latest_candidate_run(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the latest timestamped candidate run directory."""
    if base_dir is None:
        base_dir = _resolve_candidate_base()
    if not base_dir.exists():
        return None
    runs = sorted(base_dir.iterdir(), key=lambda x: x.name, reverse=True)
    return runs[0] if runs else None


def discover_candidate_models(run_dir: Optional[Path] = None) -> dict:
    """Discover candidate sub-models (A_baseline, B_rain_observed, C_rain_aware_nowcast).

    Returns dict of model_name -> {dir: Path, feature_list: list, label: str}.
    """
    if run_dir is None:
        run_dir = get_latest_candidate_run()
    if run_dir is None:
        return {}
    candidates = {}
    LABEL_MAP = {
        'A_baseline': 'A: Baseline (candidate)',
        'B_rain_observed': 'B: Rain Observed (candidate)',
        'C_rain_aware_nowcast': 'C: Rain-Aware Nowcast (candidate)',
    }
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        fl_path = sub / 'feature_list.json'
        if not fl_path.exists():
            continue
        with open(fl_path, 'r') as f:
            feature_list = json.load(f)
        candidates[sub.name] = {
            'dir': sub,
            'feature_list': feature_list,
            'label': LABEL_MAP.get(sub.name, sub.name),
        }
    return candidates


def discover_all_models() -> dict:
    """Discover all available models: production intraday + candidates.

    Returns dict of model_key -> model_info dict with keys:
        dir, feature_list, label, model_type ('production'|'candidate')
    """
    models = {}
    # Production models
    for key, path in PRODUCTION_MODEL_DIRS.items():
        fl_path = path / 'feature_list.json'
        if path.exists() and fl_path.exists():
            with open(fl_path, 'r') as f:
                feature_list = json.load(f)
            models[key] = {
                'dir': path,
                'feature_list': feature_list,
                'label': 'Baseline Intraday' if key == 'baseline_intraday' else 'Rain Nowcast',
                'model_type': 'production',
            }
    # Candidate models
    candidates = discover_candidate_models()
    for key, info in candidates.items():
        info['model_type'] = 'candidate'
        models[key] = info
    return models


def load_model_inference(model_dir: Path):
    """Load LightGBM models from a directory.

    Returns dict of upside_q10..q90, downside_q10..q90, upside_zero, downside_zero.
    """
    import lightgbm as lgb
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f'upside_q{q}'] = lgb.Booster(model_file=str(model_dir / f'upside_q{q}.txt'))
        models[f'downside_q{q}'] = lgb.Booster(model_file=str(model_dir / f'downside_q{q}.txt'))
    for clf_name in ['upside_zero', 'downside_zero']:
        clf_path = model_dir / f'{clf_name}.txt'
        if clf_path.exists():
            models[clf_name] = lgb.Booster(model_file=str(clf_path))
        else:
            models[clf_name] = None
    return models


def predict_remaining(model_key: str, features: dict, models_info: dict) -> dict:
    """Run prediction for one model given current features.

    Returns dict: remaining_upside_p10/p50/p90, remaining_downside_p10/...,
                  pred_tmax_p50 (max_so_far + upside_p50), probs (derived).
    """
    from models.intraday_inference import _build_features, combine_with_prior

    _ALL_FEATURE_KEYS = [
        'temp', 'max_so_far', 'temp_change_30min', 'temp_change_60min',
        'hour', 'minute', 'minutes_since_midnight', 'month', 'day_of_year',
        'day_sin', 'day_cos', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'max_bucket', 'time_since_max_so_far', 'forecast_tmax', 'forecast_tmin',
        'range_so_far', 'temp_change_120min',
        'rainfall_60m', 'rain_cooling_60m', 'rainfall_120m', 'rain_cooling_30m',
        'drop_from_max', 'post_peak_rain_flag', 'morning_peak_rain_flag',
        # rain nowcast features (point-in-time safe)
        'rain_nc_sum_0_60m', 'rain_nc_sum_0_120m', 'rain_nc_any_0_120m',
        'rain_nc_front_loaded_ratio', 'rain_nc_heavy_0_120m',
        'rain_nc_valid_horizon_count', 'rain_nc_missing_flag',
        'rain_nowcast_age_minutes', 'rain_nowcast_missing_flag',
        'rain_nc_nearest_mm_sum_30m', 'rain_nc_nearest_mm_sum_60m',
        'rain_nc_nearest_mm_sum_90m', 'rain_nc_nearest_mm_sum_120m',
        'rain_nc_mean_r5km_sum_30m', 'rain_nc_mean_r5km_sum_60m',
        'rain_nc_mean_r5km_sum_90m', 'rain_nc_mean_r5km_sum_120m',
        'rain_nc_max_r5km_sum_30m', 'rain_nc_max_r5km_sum_60m',
        'rain_nc_max_r5km_sum_90m', 'rain_nc_max_r5km_sum_120m',
        'rain_nc_min_r5km_sum_30m', 'rain_nc_min_r5km_sum_60m',
        'rain_nc_min_r5km_sum_90m', 'rain_nc_min_r5km_sum_120m',
        'rain_nc_p90_r5km_sum_30m', 'rain_nc_p90_r5km_sum_60m',
        'rain_nc_p90_r5km_sum_90m', 'rain_nc_p90_r5km_sum_120m',
        'rain_nc_area_gt0_r5km_sum_30m', 'rain_nc_area_gt0_r5km_sum_60m',
        'rain_nc_area_gt0_r5km_sum_90m', 'rain_nc_area_gt0_r5km_sum_120m',
        'rain_nc_area_gt5_r5km_sum_30m', 'rain_nc_area_gt5_r5km_sum_60m',
        'rain_nc_area_gt5_r5km_sum_90m', 'rain_nc_area_gt5_r5km_sum_120m',
        'rainfall_60m_filled', 'rainfall_120m_filled', 'rainfall_30m_filled',
        'rain_cooling_120m', 'morning_peak_then_rain_flag',
        'rain_data_gap_flag', 'rainfall_data_age_minutes',
    ]
    safe_features = {k: features.get(k, 0.0) for k in _ALL_FEATURE_KEYS}
    feature_cols = models_info['feature_list']

    X_vals = {k: safe_features.get(k, 0.0) for k in feature_cols}
    import pandas as pd
    X = pd.DataFrame([X_vals], columns=feature_cols)

    models = load_model_inference(models_info['dir'])

    upside = {}
    for q in [10, 25, 50, 75, 90]:
        upside[f'q{q}'] = models[f'upside_q{q}'].predict(X)[0]
    downside = {}
    for q in [10, 25, 50, 75, 90]:
        downside[f'q{q}'] = models[f'downside_q{q}'].predict(X)[0]

    # Sort upside/downside for monotonicity
    up_keys = sorted(upside.keys())
    up_vals = sorted([upside[k] for k in up_keys])
    upside = {k: v for k, v in zip(up_keys, up_vals)}
    dn_keys = sorted(downside.keys())
    dn_vals = sorted([downside[k] for k in dn_keys])
    downside = {k: v for k, v in zip(dn_keys, dn_vals)}

    max_so_far = features.get('max_so_far', 30.0)
    pred_tmax_p50 = max_so_far + max(upside.get('q50', 0), 0)

    result = {
        'remaining_upside_p10': max(0, upside.get('q10', 0)),
        'remaining_upside_p50': max(0, upside.get('q50', 0)),
        'remaining_upside_p90': max(0, upside.get('q90', 0)),
        'remaining_downside_p10': max(0, downside.get('q10', 0)),
        'remaining_downside_p50': max(0, downside.get('q50', 0)),
        'remaining_downside_p90': max(0, downside.get('q90', 0)),
        'pred_tmax_p50': pred_tmax_p50,
        'model_key': model_key,
    }
    return result


def predict_all_models(features: dict) -> dict:
    """Run prediction across all discovered models.

    Returns dict of model_key -> prediction dict.
    """
    all_models = discover_all_models()
    results = {}
    for key, info in all_models.items():
        try:
            pred = predict_remaining(key, features, info)
            pred['label'] = info['label']
            pred['model_type'] = info['model_type']
            results[key] = pred
        except Exception as e:
            logger.warning("Model %s prediction failed: %s", key, e)
            results[key] = {'error': str(e), 'model_key': key, 'label': info['label'], 'model_type': info['model_type']}
    return results


def build_comparison_table(all_predictions: dict, markets: list = None) -> 'pd.DataFrame':
    """Build a model comparison DataFrame from predictions.

    Columns: model_key, label, model_type, pred_tmax_p50, upside_p50, upside_p10, upside_p90,
             downside_p50, status
    """
    import pandas as pd
    rows = []
    for key, pred in all_predictions.items():
        if 'error' in pred:
            rows.append({
                'model_key': key,
                'label': pred.get('label', key),
                'model_type': pred.get('model_type', ''),
                'pred_tmax_p50': None,
                'remaining_upside_p50': None,
                'status': '❌ error',
            })
        else:
            rows.append({
                'model_key': key,
                'label': pred.get('label', key),
                'model_type': pred.get('model_type', ''),
                'pred_tmax_p50': pred.get('pred_tmax_p50'),
                'remaining_upside_p50': pred.get('remaining_upside_p50'),
                'remaining_upside_p10': pred.get('remaining_upside_p10'),
                'remaining_upside_p90': pred.get('remaining_upside_p90'),
                'remaining_downside_p50': pred.get('remaining_downside_p50'),
                'status': '✅',
            })
    return pd.DataFrame(rows)


def compute_gate_matrix(all_predictions: dict, market_state: dict = None,
                        probs_by_model: dict = None, config: dict = None) -> 'pd.DataFrame':
    """Compute per-model gate pass/fail matrix.

    Checks: data_available, model_ready, has_edge, bucket_parsed, rain_gate

    Returns DataFrame: rows = model_key, columns = gate names.
    """
    from execution.strategy_gate import (
        check_data_gate, check_model_gate, check_rain_gate, validate_bucket_parsing,
        REASON_PASS_ALL
    )
    import pandas as pd
    gate_names = ['data_available', 'bucket_parsed', 'model_ready', 'has_edge', 'rain_gate']
    rows = []
    for key, pred in all_predictions.items():
        row = {'model_key': key, 'label': pred.get('label', key)}
        err = pred.get('error')
        if err:
            for g in gate_names:
                row[g] = f'❌ {err[:30]}'
            rows.append(row)
            continue

        # Simulate minimal probs for gate checks if not provided
        probs_for_gate = {}
        if probs_by_model and key in probs_by_model:
            probs_for_gate = probs_by_model[key]
        prices_for_gate = {}
        if market_state:
            prices_for_gate = market_state.get('prices_dict', {})

        fake_features = {'dummy': 1} if market_state else None
        fake_market = {'buckets': [{'name': 'dummy'}], 'prices_dict': {'dummy': 0.5}} if not market_state else market_state

        data_gate = check_data_gate(fake_features, fake_market)
        row['data_available'] = '✅' if data_gate.passed else f'❌ {data_gate.reason_code}'

        bp_gate = validate_bucket_parsing(fake_market)
        row['bucket_parsed'] = '✅' if bp_gate.passed else f'❌ {bp_gate.reason_code}'

        if probs_for_gate and prices_for_gate:
            model_gate = check_model_gate(probs_for_gate, prices_for_gate)
            row['model_ready'] = '✅' if model_gate.passed else f'❌ {model_gate.reason_code}'
            row['has_edge'] = '✅' if model_gate.passed else f'❌ {model_gate.reason_code}'
        else:
            row['model_ready'] = '⏳ (no probs)'
            row['has_edge'] = '⏳ (no probs)'

        rain_gate = check_rain_gate(None)
        row['rain_gate'] = '✅' if rain_gate.passed else f'❌ {rain_gate.reason_code}'

        rows.append(row)

    return pd.DataFrame(rows)


def load_paper_snapshot() -> dict:
    """Load current paper trading state (positions + metadata).

    Returns dict with keys: positions (dict), timestamp, slug (from market).
    """
    from execution.portfolio_reconciler import load_positions
    pos = load_positions() or {}
    return {
        'positions': pos,
        'timestamp': hkt_now(),
    }


def paper_trade_snapshot(markets: list = None, slug: str = '') -> dict:
    """Get full paper trade snapshot including current positions and market state."""
    from execution.portfolio_reconciler import load_positions
    positions = load_positions() or {}
    try:
        from execution.rebalancer import fetch_market_state
        market_state = fetch_market_state(slug) if slug and fetch_market_state else None
    except Exception as e:
        logger.warning("Cannot fetch market state: %s", e)
        market_state = None
    prices_dict = {m['name']: m['price_yes'] for m in markets} if markets else {}
    if market_state:
        prices_dict = market_state.get('prices_dict', prices_dict)
    pnl_by_account = {}
    try:
        from execution.rebalancer import calculate_pnl
        for pid in positions:
            pnl_by_account[pid] = {}
            for sl in positions[pid]:
                pnl_by_account[pid][sl] = {}
                for sk in positions[pid][sl]:
                    slug_pos = positions[pid][sl][sk]
                    records, pnl, cost = calculate_pnl({sl: slug_pos}, prices_dict)
                    pnl_by_account[pid][sl][sk] = {'records': records, 'pnl': pnl, 'cost': cost}
    except Exception as e:
        logger.warning("Cannot calculate PnL: %s", e)
    return {
        'positions': positions,
        'market_state': market_state,
        'prices_dict': prices_dict,
        'pnl_by_account': pnl_by_account,
        'timestamp': hkt_now(),
    }


def apply_paper_strategy(strategy_key: str, target_positions: dict,
                         slug: str, prices_dict: dict,
                         mock_slippage: bool = True,
                         portfolio_id: str = "weather_main") -> dict:
    """Apply target positions for exactly ONE paper strategy.

    Uses update_paper_positions with the given strategy_key.
    Does NOT affect other strategies' positions.

    Returns result dict with status and details.
    """
    try:
        from execution.rebalancer import update_paper_positions
    except Exception as e:
        return {'status': 'failed', 'strategy_key': strategy_key, 'error': str(e)}
    old_positions = load_paper_snapshot()['positions']
    strategy_context = {
        'strategy_key': strategy_key,
        'strategy_version': 'manual',
        'scheduler_source': 'manual',
        'selected_model': strategy_key,
    }
    success = update_paper_positions(
        target_positions, portfolio_id, slug, strategy_key, prices_dict, strategy_context
    )
    new_positions = load_paper_snapshot()['positions']
    return {
        'status': 'applied' if success else 'failed',
        'strategy_key': strategy_key,
        'slug': slug,
        'buckets_updated': list(target_positions.keys()),
        'old_bucket_count': len(old_positions.get(portfolio_id, {}).get(slug, {}).get(strategy_key, {})),
        'new_bucket_count': len(new_positions.get(portfolio_id, {}).get(slug, {}).get(strategy_key, {})),
    }


def run_orders_for_strategy(strategy_key: str, target_probs: dict,
                             prices_dict: dict, token_ids_dict: dict,
                             capital: float, slug: str,
                             mock_slippage: bool = True) -> dict:
    """Run generate_orders_from_probs and apply to a single strategy.

    Effectively a manual version of run_single_strategy_cycle but
    using the rebalancer's generate_orders_from_probs directly.
    """
    try:
        from execution.rebalancer import generate_orders_from_probs
    except Exception as e:
        return {'status': 'failed', 'strategy_key': strategy_key, 'error': str(e)}
    orders = generate_orders_from_probs(
        target_probs, prices_dict, token_ids_dict, capital,
        mock_slippage
    )
    if orders:
        return apply_paper_strategy(strategy_key, orders, slug, prices_dict, mock_slippage)
    return {'status': 'no_trades', 'strategy_key': strategy_key, 'buckets_updated': []}
