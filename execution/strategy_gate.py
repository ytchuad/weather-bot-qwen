# execution/strategy_gate.py
# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATED — use the new modular gates package instead:
#
#   from execution.gates import (
#       GateInput, GateOutput, GateConfig, GatePipeline,
#       time_gate, regime_edge_gate, confidence_gate,
#       boundary_gate, drawdown_gate, slippage_gate, exposure_gate,
#       kelly_sizer, rain_uncertainty_sizer, model_confidence_sizer,
#       ...
#   )
#
# The new package provides the same gate logic as composable functions
# with a standard (GateInput, GateConfig) -> GateOutput interface,
# organized into entry/exit/sizing/rebalance pipelines.
#
# This module is kept as a compatibility shim — all existing imports
# still work.  New code should import from execution.gates.
# ═══════════════════════════════════════════════════════════════════════════

import warnings as _warnings
_warnings.warn(
    "execution.strategy_gate is deprecated. Use execution.gates instead.\n"
    "  from execution.gates import GateInput, GateOutput, GatePipeline, gate_fns",
    DeprecationWarning, stacklevel=2,
)

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import copy
import logging
import re as _re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

ENTRY_EDGE_THRESHOLD = 0.04
EXIT_EDGE_THRESHOLD = 0.015
MAX_PER_BUCKET = 0.15
TOTAL_MAX = 0.50

REASON_PASS_ALL = 'PASS_ALL'
REASON_DATA_INSUFFICIENT = 'DATA_INSUFFICIENT'
REASON_MODEL_UNCERTAIN = 'MODEL_UNCERTAIN'
REASON_NO_EDGE = 'NO_EDGE'
REASON_RAIN_EXCLUDED = 'RAIN_EXCLUDED'
REASON_ENTRY_SIGNAL = 'ENTRY_SIGNAL'
REASON_EXIT_SIGNAL = 'EXIT_SIGNAL'
REASON_HOLD = 'HOLD'
REASON_CAPACITY_EXCEEDED = 'CAPACITY_EXCEEDED'
REASON_BUCKET_PARSE_FAIL = 'BUCKET_PARSE_FAIL'
REASON_EDGE_TOO_SMALL = 'EDGE_TOO_SMALL'
REASON_QTY_BELOW_MIN = 'QTY_BELOW_MIN'
REASON_EXPOSURE_LIMIT = 'EXPOSURE_LIMIT'
REASON_CHURN_COOLDOWN = 'CHURN_COOLDOWN'
REASON_RAIN_NOWCAST_STALE = 'RAIN_NOWCAST_STALE'
REASON_HARD_FAILURE = 'HARD_FAILURE'
REASON_TARGET_ZERO = 'TARGET_ZERO'
REASON_EDGE_FALLS_BELOW_MIN = 'EDGE_FALLS_BELOW_MIN'
REASON_MODEL_REVERSED = 'MODEL_REVERSED'
REASON_LOSS_THRESHOLD = 'LOSS_THRESHOLD'
REASON_RISK_LIMIT = 'RISK_LIMIT'
REASON_FEATURE_MISSING = 'FEATURE_MISSING'
REASON_PREDICTION_FAILED = 'PREDICTION_FAILED'
REASON_NOWCAST_AGE_EXCEEDED = 'NOWCAST_AGE_EXCEEDED'
REASON_STALE_FORECAST = 'STALE_FORECAST'
REASON_STALE_MARKET_PRICE = 'STALE_MARKET_PRICE'
REASON_NO_TRADE = 'NO_TRADE'

REASON_EDGE_REVERSED = 'EDGE_REVERSED'
REASON_PROFIT_TAKE = 'PROFIT_TAKE'
REASON_RAIN_EMERGENCY = 'RAIN_EMERGENCY'
REASON_STD_SPIKE = 'STD_SPIKE'
REASON_REDUCE_50_PCT = 'REDUCE_50_PCT'
REASON_HOLD_UNTIL_EXPIRY = 'HOLD_UNTIL_EXPIRY'
REASON_LIQUIDITY_INSUFFICIENT = 'LIQUIDITY_INSUFFICIENT'
REASON_TIME_WINDOW_CLOSED = 'TIME_WINDOW_CLOSED'


@dataclass
class GateResult:
    passed: bool
    reason_code: str
    detail: str = ''
    probs_filtered: Optional[dict] = None
    decisions: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


def check_data_gate(features, market_state) -> GateResult:
    if features is None or market_state is None:
        return GateResult(False, REASON_DATA_INSUFFICIENT, 'features or market_state is None')
    if not market_state.get('buckets'):
        return GateResult(False, REASON_DATA_INSUFFICIENT, 'market_state has no buckets')
    if not market_state.get('prices_dict'):
        return GateResult(False, REASON_DATA_INSUFFICIENT, 'market_state has no prices_dict')
    return GateResult(True, REASON_PASS_ALL, 'data gate passed')


def check_model_gate(probs_dict: dict, prices_dict: dict, min_edge: float = 0.005, max_std: Optional[float] = None) -> GateResult:
    if not probs_dict:
        return GateResult(False, REASON_MODEL_UNCERTAIN, 'probs_dict is empty')
    if not prices_dict:
        return GateResult(False, REASON_MODEL_UNCERTAIN, 'prices_dict is empty')
    has_edge = False
    for k in probs_dict:
        edge_yes = probs_dict.get(k, 0) - prices_dict.get(k, 0)
        edge_no = prices_dict.get(k, 0) - probs_dict.get(k, 0)
        if edge_yes > min_edge or edge_no > min_edge:
            has_edge = True
            break
    if not has_edge:
        return GateResult(False, REASON_NO_EDGE, f'no bucket with edge > {min_edge}')
    return GateResult(True, REASON_PASS_ALL, 'model gate passed')


def check_rain_gate(rain_regime) -> GateResult:
    if rain_regime is None:
        return GateResult(True, REASON_PASS_ALL, 'rain regime not available, skip gate')
    if isinstance(rain_regime, dict):
        if rain_regime.get('exclude_trade', False):
            return GateResult(False, REASON_RAIN_EXCLUDED, f"rain exclusion: {rain_regime.get('reason', '')}")
        if rain_regime.get('confidence', 1.0) < 0.3:
            return GateResult(False, REASON_RAIN_EXCLUDED, 'rain nowcast confidence too low')
        if rain_regime.get('stale', False):
            return GateResult(False, REASON_RAIN_NOWCAST_STALE, 'rain nowcast data is stale')
    return GateResult(True, REASON_PASS_ALL, 'rain gate passed')


def validate_bucket_parsing(market_state: dict) -> GateResult:
    buckets = market_state.get('buckets', [])
    if not buckets:
        return GateResult(False, REASON_BUCKET_PARSE_FAIL, 'no buckets to validate')
    problems = []
    for i, b in enumerate(buckets):
        name = b.get('name', '')
        lower = b.get('lower')
        upper = b.get('upper')
        if lower is None or upper is None:
            problems.append(f"bucket[{i}] '{name}' missing lower/upper")
        elif lower >= upper:
            problems.append(f"bucket[{i}] '{name}' lower({lower}) >= upper({upper})")
    sorted_names = sorted([b['name'] for b in buckets if b.get('name')])
    if len(sorted_names) != len(set(sorted_names)):
        problems.append('duplicate bucket names found')
    labels = [b.get('name', '') for b in buckets]
    market_state['bucket_labels'] = labels
    market_state['bucket_parse_ok'] = len(problems) == 0
    if problems:
        detail = '; '.join(problems)
        return GateResult(False, REASON_BUCKET_PARSE_FAIL, detail, metadata={'problems': problems})
    return GateResult(True, REASON_PASS_ALL, 'bucket parsing valid')


def classify_rain_regime(signal_context: dict = None) -> str:
    if signal_context is None:
        return 'no_rain'
    stale = signal_context.get('nowcast_stale', True)
    missing = signal_context.get('nowcast_missing', True)
    if stale or missing:
        return 'stale_or_missing'
    rain_sum = signal_context.get('rain_nc_sum_0_120m', 0)
    if rain_sum <= 0:
        return 'no_rain'
    if rain_sum <= 3:
        return 'weak_rain'
    return 'moderate_or_heavy_rain'


def select_prediction_engine(rain_regime: str, signal_context: dict = None, explicit_model: str = None) -> dict:
    if explicit_model == 'rain_nowcast':
        return {
            'selected_model': 'rain_nowcast',
            'model_weight_baseline': 0.0,
            'model_weight_rain': 1.0,
            'reason_code': REASON_PASS_ALL
        }
    if explicit_model == 'baseline':
        return {
            'selected_model': 'baseline',
            'model_weight_baseline': 1.0,
            'model_weight_rain': 0.0,
            'reason_code': REASON_PASS_ALL
        }
    if rain_regime in ('stale_or_missing', 'no_rain'):
        return {
            'selected_model': 'baseline',
            'model_weight_baseline': 1.0,
            'model_weight_rain': 0.0,
            'reason_code': REASON_PASS_ALL
        }
    if signal_context and signal_context.get('rain_model_validated', False):
        return {
            'selected_model': 'rain_aware_ensemble',
            'model_weight_baseline': 0.5,
            'model_weight_rain': 0.5,
            'reason_code': REASON_PASS_ALL
        }
    return {
        'selected_model': 'baseline',
        'model_weight_baseline': 1.0,
        'model_weight_rain': 0.0,
        'reason_code': 'RAIN_MODEL_PAPER_ONLY'
    }


def evaluate_all_gates(features, market_state, probs_dict, prices_dict, rain_regime=None, config=None) -> GateResult:
    data_result = check_data_gate(features, market_state)
    if not data_result.passed:
        return data_result
    bucket_result = validate_bucket_parsing(market_state)
    if not bucket_result.passed:
        return bucket_result
    model_result = check_model_gate(probs_dict, prices_dict)
    if not model_result.passed:
        return model_result
    rain_result = check_rain_gate(rain_regime)
    if not rain_result.passed:
        return rain_result
    return GateResult(True, REASON_PASS_ALL, 'all gates passed')


def check_entry_gate_phase2(
    target_positions: dict,
    adjusted_bets: dict,
    current_positions: dict,
    model_key: str,
    slug: str,
    prices_dict: dict,
    capital: float = 1000.0,
    entry_edge_threshold: float = ENTRY_EDGE_THRESHOLD
) -> GateResult:
    approved = {}
    metadata = {'blocked': [], 'reasons': {}}
    market_pos = current_positions.get(model_key, {}).get(slug, {})

    for label, pos in list(target_positions.items()):
        bet = adjusted_bets.get(label, {})
        edge_after = bet.get('expected_edge', 0)
        qty = pos.get('quantity', 0)
        side = pos.get('side', 'YES')
        target_price = pos.get('target_price', bet.get('avg_fill_price', 0.5))

        if qty < 5.0:
            metadata['blocked'].append(label)
            metadata['reasons'][label] = REASON_QTY_BELOW_MIN
            del target_positions[label]
            continue

        if side == 'YES':
            price_yes = prices_dict.get(label, 0.5)
            edge_net = edge_after
        else:
            price_yes = prices_dict.get(label, 0.5)
            edge_net = (1.0 - price_yes) - target_price

        if edge_net < entry_edge_threshold:
            metadata['blocked'].append(label)
            metadata['reasons'][label] = REASON_EDGE_TOO_SMALL
            del target_positions[label]
            continue

        # check per-bucket exposure
        current_qty = market_pos.get(label, {}).get('quantity', 0)
        current_side = market_pos.get(label, {}).get('side')
        current_entry = market_pos.get(label, {}).get('entry_price', 0)
        new_exposure = 0.0
        if current_side == side and current_qty > 0:
            new_exposure = current_qty * current_entry + (qty - current_qty) * target_price
        else:
            new_exposure = qty * target_price
        bucket_exposure = new_exposure / capital if capital > 0 else 1.0
        if bucket_exposure > MAX_PER_BUCKET:
            metadata['blocked'].append(label)
            metadata['reasons'][label] = REASON_EXPOSURE_LIMIT
            del target_positions[label]
            continue

        approved[label] = pos

    if not approved:
        return GateResult(False, REASON_EDGE_TOO_SMALL, 'no positions passed entry gate phase 2', metadata=metadata)

    return GateResult(True, REASON_PASS_ALL, f'{len(approved)} positions approved', metadata=metadata)


def check_exit_gate(
    bucket: str,
    current_position: dict,
    target_qty: float,
    probs_dict: dict = None,
    prices_dict: dict = None
) -> GateResult:
    current_qty = current_position.get('quantity', 0)
    current_side = current_position.get('side')
    current_price = current_position.get('entry_price', 0)

    if current_qty <= 0:
        return GateResult(False, REASON_HOLD, 'no current position to exit')

    if target_qty < 0.1:
        return GateResult(True, REASON_TARGET_ZERO, 'target is zero', metadata={'exit_reason': REASON_TARGET_ZERO})

    if 0 < target_qty < 5.0:
        return GateResult(True, REASON_TARGET_ZERO, f'target qty {target_qty} below PM_MIN_QTY', metadata={'exit_reason': 'EXIT_QTY_BELOW_MIN'})

    if probs_dict and prices_dict:
        price_yes = prices_dict.get(bucket, 0.5)
        p_model = probs_dict.get(bucket, 0.5)
        if current_side == 'YES':
            edge = p_model - price_yes
        else:
            edge = (1.0 - p_model) - (1.0 - price_yes) if current_side == 'NO' else 0
        if edge < -EXIT_EDGE_THRESHOLD:
            return GateResult(True, REASON_MODEL_REVERSED, f'edge {edge:.4f} below exit threshold', metadata={'exit_reason': REASON_MODEL_REVERSED})

    return GateResult(False, REASON_HOLD, 'no exit condition triggered')


def evaluate_entry_exit_for_bucket(
    bucket: str,
    current_position: dict,
    target_signal: dict,
    probs_dict: dict = None,
    prices_dict: dict = None,
    adjusted_bets: dict = None
) -> GateResult:
    target_qty = target_signal.get('quantity', 0)
    target_side = target_signal.get('side')
    current_qty = current_position.get('quantity', 0)

    # Check exit first
    if target_qty < 0.1 or (0 < target_qty < 5.0 and current_qty > 0):
        return check_exit_gate(bucket, current_position, target_qty, probs_dict, prices_dict)

    # Check entry
    if current_qty <= 0 and target_qty >= 5.0:
        bet = (adjusted_bets or {}).get(bucket, {})
        edge = bet.get('expected_edge', 0)
        if edge < ENTRY_EDGE_THRESHOLD:
            return GateResult(False, REASON_EDGE_TOO_SMALL, f'entry edge {edge:.4f} < {ENTRY_EDGE_THRESHOLD}')
        return GateResult(True, REASON_ENTRY_SIGNAL, f'entry approved, edge={edge:.4f}', metadata={'entry_reason': REASON_ENTRY_SIGNAL})

    return GateResult(True, REASON_HOLD, 'hold current position')


def decide_entry_exit(current_positions: dict, target_positions: dict, model_key: str, slug: str) -> dict:
    decisions = {}
    current = current_positions.get(model_key, {}).get(slug, {})
    all_buckets = set(list(current.keys()) + list(target_positions.keys()))
    for bucket in all_buckets:
        has_current = bucket in current
        has_target = bucket in target_positions
        if has_target and not has_current:
            decisions[bucket] = {'decision': 'ENTRY', 'reason_code': REASON_ENTRY_SIGNAL}
        elif has_current and not has_target:
            decisions[bucket] = {'decision': 'EXIT', 'reason_code': REASON_EXIT_SIGNAL}
        elif has_current and has_target:
            decisions[bucket] = {'decision': 'HOLD', 'reason_code': REASON_HOLD}
        else:
            decisions[bucket] = {'decision': 'NO_TRADE', 'reason_code': REASON_HOLD}
    return decisions


# ═══════════════════════════════════════════════════════════════
# Refined Entry/Exit Gates (config-driven)
# ═══════════════════════════════════════════════════════════════

def load_strategy_config(strategy_key: str = None) -> dict:
    """Load global defaults + per-strategy overrides from paper_strategies.json."""
    import json
    cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'paper_strategies.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    defaults = raw.get('defaults', {})
    if strategy_key:
        strategy_def = raw.get('strategies', {}).get(strategy_key, {})
        override = strategy_def.get('override', {})
        # Deep-merge override into defaults
        merged = _deep_merge(defaults, override)
        return merged
    return defaults


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (override wins)."""
    result = {}
    all_keys = set(list(base.keys()) + list(override.keys()))
    for k in all_keys:
        if k in override and k in base:
            if isinstance(base[k], dict) and isinstance(override[k], dict):
                result[k] = _deep_merge(base[k], override[k])
            elif isinstance(base[k], list) and isinstance(override[k], list):
                result[k] = override[k]
            else:
                result[k] = override[k]
        elif k in override:
            result[k] = override[k]
        else:
            result[k] = base[k]
    return result


def select_minute_model(signal_context: dict, config: dict, model_cache_available: set = None) -> dict:
    """Select the best available minute model: Model G > Model F > Model C > B > A > baseline fallback.
    
    Returns dict with keys: selected_model, multiplier, source_label, reason_code.
    """
    if model_cache_available is None:
        model_cache_available = set()
    ms = config.get('entry', {}).get('model_selection', {})
    nowcast_fresh = not signal_context.get('nowcast_stale', True)
    nowcast_available = not signal_context.get('nowcast_missing', True)
    rainfall_available = signal_context.get('rainfall_available', False)

    # Model G selection (forecast_gap + max_so_far)
    if 'model_g' in model_cache_available:
        return {
            'selected_model': 'model_g',
            'multiplier': ms.get('model_g_multiplier', 1.0),
            'source_label': 'Model G (forecast_gap+max)',
            'reason_code': REASON_PASS_ALL,
        }
    # Model F selection (forecast_gap-based)
    if 'model_f' in model_cache_available:
        return {
            'selected_model': 'model_f',
            'multiplier': ms.get('model_f_multiplier', 1.0),
            'source_label': 'Model F (forecast_gap)',
            'reason_code': REASON_PASS_ALL,
        }
    if nowcast_fresh and nowcast_available and 'model_c' in model_cache_available:
        return {
            'selected_model': 'model_c',
            'multiplier': ms.get('model_c_multiplier', 1.0),
            'source_label': 'Model C fresh nowcast',
            'reason_code': REASON_PASS_ALL,
        }
    if rainfall_available and 'model_b' in model_cache_available:
        return {
            'selected_model': 'model_b',
            'multiplier': ms.get('model_b_fallback_multiplier', 0.7),
            'source_label': 'Model B rainfall fallback',
            'reason_code': REASON_PASS_ALL,
        }
    if 'model_a' in model_cache_available:
        return {
            'selected_model': 'model_a',
            'multiplier': ms.get('model_a_fallback_multiplier', 0.5),
            'source_label': 'Model A dry/no-nowcast fallback',
            'reason_code': REASON_PASS_ALL,
        }
    return {
        'selected_model': 'baseline',
        'multiplier': 0.5,
        'source_label': 'Baseline fallback (no minute model)',
        'reason_code': 'NO_MINUTE_MODEL',
    }


def get_entry_regime(dt_now, rain_regime: str = None, config: dict = None) -> str:
    """Classify current time/rain slot into a regime key for threshold lookup."""
    h = dt_now.hour
    if 8 <= h < 12:
        if rain_regime in ('moderate_or_heavy_rain', 'weak_rain'):
            return 'rain_08_12'
        return 'day_08_12'
    elif 12 <= h < 16:
        return 'slot_12_16'
    return 'slot_16_24'


def check_regime_edge_threshold(edge: float, regime: str, config: dict) -> dict:
    """Check if edge meets the regime-specific threshold.
    
    Returns dict: passes (bool), threshold (float), note (str).
    """
    thresholds = config.get('entry', {}).get('regime_thresholds', {})
    regime_cfg = thresholds.get(regime, {})
    min_edge = regime_cfg.get('min_edge')
    if min_edge is None:
        return {'passes': False, 'threshold': None, 'note': f'no entry in regime={regime}'}
    exposure_cap = regime_cfg.get('exposure_cap', 0.50)
    if edge >= min_edge:
        return {'passes': True, 'threshold': min_edge, 'exposure_cap': exposure_cap, 'note': 'edge meets regime threshold'}
    return {'passes': False, 'threshold': min_edge, 'exposure_cap': exposure_cap, 'note': f'edge {edge:.4f} < {min_edge}'}


def _parse_bucket_bounds(bucket_name: str):
    """Extract lower/upper bounds from a bucket name string.

    Handles: "32°C" -> (32, 33), "Below 26°C" -> (-inf, 27), "Above 36°C" -> (36, inf)
    Falls back to (-inf, inf) if no number found.
    """
    if not isinstance(bucket_name, str):
        return -float('inf'), float('inf')
    bn = bucket_name.lower()
    match = _re.search(r'([\d.]+)', bn)
    if not match:
        return -float('inf'), float('inf')
    val = float(match.group(1))
    if any(kw in bn for kw in ['below', 'lower', 'under']):
        return -float('inf'), val + 1.0
    if any(kw in bn for kw in ['higher', 'above', 'over']):
        return val, float('inf')
    return val, val + 1.0


def check_boundary_proximity(mean: float, std: float, bucket_lower: float, bucket_upper: float,
                              config: dict) -> dict:
    """Check if the predicted distribution is too close to a bucket boundary.
    
    Returns dict: passes (bool), multiplier (float), distance_raw (float), distance_std (float).
    """
    bp = config.get('entry', {}).get('boundary_proximity', {})
    min_dist = bp.get('min_standardized_distance', 0.5)
    agg_thresh = bp.get('aggressive_reduction_threshold', 0.3)
    agg_mult = bp.get('aggressive_reduction_multiplier', 0.5)

    # [關鍵修復 1] 使用絕對值計算距離，避免外部距離產生負數
    dist_raw = 999.0
    if bucket_lower != -float('inf'):
        dist_raw = min(dist_raw, abs(mean - bucket_lower))
    if bucket_upper != float('inf'):
        dist_raw = min(dist_raw, abs(bucket_upper - mean))

    dist_std = dist_raw / std if std > 0 else 999.0

    if dist_std < agg_thresh:
        # 距離極近，使用激進縮減倍率
        return {'passes': True, 'multiplier': agg_mult, 'distance_raw': dist_raw, 'distance_std': dist_std}
    if dist_std < min_dist:
        # 距離稍近，按比例縮減
        ratio = dist_std / min_dist
        return {'passes': True, 'multiplier': ratio, 'distance_raw': dist_raw, 'distance_std': dist_std}
    
    # 距離夠遠，不縮減
    return {'passes': True, 'multiplier': 1.0, 'distance_raw': dist_raw, 'distance_std': dist_std}


def check_probability_confidence(model_std: float, config: dict) -> bool:
    """Return True if model_std <= configured max."""
    max_std = config.get('entry', {}).get('probability_confidence', {}).get('max_model_std', 2.5)
    return model_std <= max_std


def get_model_confidence_multiplier(model_key: str, config: dict) -> float:
    """Get the sizing multiplier based on which model is active."""
    ms = config.get('entry', {}).get('model_selection', {})
    multipliers = {
        'model_g': ms.get('model_g_multiplier', 1.0),
        'model_f': ms.get('model_f_multiplier', 1.0),
        'model_c': ms.get('model_c_multiplier', 1.0),
        'model_b': ms.get('model_b_fallback_multiplier', 0.7),
        'model_a': ms.get('model_a_fallback_multiplier', 0.5),
    }
    return multipliers.get(model_key, 0.5)


def get_time_window_multiplier(hour: int, sizing_config: list) -> float:
    """Lookup time-window multiplier from config list of {hours: [lo, hi], value: float}."""
    for entry in sizing_config:
        lo, hi = entry['hours']
        if lo <= hour < hi:
            return entry['value']
    return 0.3


def get_rain_uncertainty_multiplier(rain_regime: str, rum_config: dict) -> float:
    """Lookup rain uncertainty multiplier from config dict."""
    return rum_config.get(rain_regime, 1.0)


def compute_position_size(kelly_size: float, model_key: str, dt_now, rain_regime: str,
                           distance_std: float, config: dict) -> float:
    """Unified position sizing: kelly × model_conf × time_window × rain_uncertainty × boundary."""
    sc = config.get('position_sizing', {})
    model_mult = get_model_confidence_multiplier(model_key, config)
    time_mult = get_time_window_multiplier(dt_now.hour, sc.get('time_window_multiplier', []))
    rain_mult = get_rain_uncertainty_multiplier(rain_regime, sc.get('rain_uncertainty_multiplier', {}))

    bp_cfg = config.get('entry', {}).get('boundary_proximity', {})
    min_std = bp_cfg.get('min_standardized_distance', 0.5)
    agg_thresh = bp_cfg.get('aggressive_reduction_threshold', 0.3)
    baseline = bp_cfg.get('aggressive_reduction_multiplier', 0.5)
    if distance_std < agg_thresh:
        boundary_mult = baseline
    elif distance_std < min_std:
        boundary_mult = max(baseline, distance_std / min_std)
    else:
        boundary_mult = 1.0

    return kelly_size * model_mult * time_mult * rain_mult * boundary_mult


def compute_drawdown_multiplier(drawdown_pct: float, config: dict) -> dict:
    """Check drawdown levels and return (action, multiplier).
    
    Returns dict: action (str), multiplier (float).
    """
    dd = config.get('exit', {}).get('drawdown', {})
    hard = dd.get('hard_flatten', -0.15)
    reduce = dd.get('reduce_risk', -0.075)
    stop = dd.get('stop_new_entries', -0.10)

    if drawdown_pct <= hard:
        return {'action': 'HARD_FLATTEN', 'multiplier': 0.0}
    if drawdown_pct <= reduce:
        ratio = max(0.0, drawdown_pct / reduce)
        return {'action': 'REDUCE_RISK', 'multiplier': ratio}
    if drawdown_pct <= stop:
        return {'action': 'STOP_ENTRIES', 'multiplier': 0.0}
    return {'action': 'NORMAL', 'multiplier': 1.0}


def compute_time_to_settlement_multiplier(hours_to_settlement: float, config: dict) -> float:
    """Linear taper for time-to-settlement de-risking."""
    t2s = config.get('exit', {}).get('time_to_settlement', {})
    strong_hours = t2s.get('strong_taper_hours', 2)
    taper_start = t2s.get('taper_start_hours', 4)
    strong_mult = t2s.get('strong_taper_multiplier', 0.3)

    if hours_to_settlement < strong_hours:
        return strong_mult
    if hours_to_settlement < taper_start:
        return hours_to_settlement / taper_start
    return 1.0


def check_model_output_consistency(preds_dict: dict, model_key: str, config: dict = None) -> dict:
    """Check model output for inconsistencies (nans, monotonicity breaks, extreme jumps)."""
    issues = []
    for k, v in preds_dict.items():
        if v is None or (isinstance(v, float) and (v != v)):
            issues.append(f'{k}=NaN')
    quantiles = [preds_dict.get(k) for k in ['remaining_upside_p10', 'remaining_upside_p25',
                                              'remaining_upside_p50', 'remaining_upside_p75',
                                              'remaining_upside_p90'] if k in preds_dict]
    if len(quantiles) >= 3:
        for i in range(len(quantiles) - 1):
            if quantiles[i] is not None and quantiles[i+1] is not None and quantiles[i] > quantiles[i+1]:
                issues.append(f'quantile non-monotonic at index {i}')
    return {'consistent': len(issues) == 0, 'issues': issues, 'model_key': model_key}


def detect_bucket_dominance_shift(probs_old: dict, probs_new: dict, threshold_pp: float = 5.0) -> dict:
    """Detect if the top bucket has changed or if probability shifted materially."""
    if not probs_old or not probs_new:
        return {'shifted': False}
    top_old = max(probs_old, key=probs_old.get)
    top_new = max(probs_new, key=probs_new.get)
    shifted = top_old != top_new
    pp_changes = {k: abs(probs_new.get(k, 0) - probs_old.get(k, 0)) * 100 for k in set(list(probs_old.keys()) + list(probs_new.keys()))}
    max_pp_change = max(pp_changes.values()) if pp_changes else 0
    return {'shifted': shifted, 'top_old': top_old, 'top_new': top_new, 'max_pp_change': max_pp_change}


def should_rebalance(current_positions: dict, target_positions: dict,
                      probs_old: dict, probs_new: dict, prices_dict: dict,
                      nowcast_regime_changed: bool, drawdown_pct: float,
                      hours_to_settlement: float, config: dict) -> dict:
    """Evaluate all 7 rebalance triggers and produce a decision."""
    rc = config.get('rebalance', {})
    triggers = []
    qty_deltas = {}
    for bucket, target in target_positions.items():
        current_qty = current_positions.get(bucket, {}).get('quantity', 0)
        target_qty = target.get('quantity', 0)
        delta = abs(target_qty - current_qty)
        if delta > 0:
            qty_deltas[bucket] = delta

    # 1. Target position delta
    if any(d > rc.get('min_qty_delta', 0.5) for d in qty_deltas.values()):
        triggers.append('QTY_DELTA')

    # 2. Edge crosses material threshold
    if prices_dict and probs_new:
        for bucket, price in prices_dict.items():
            edge = probs_new.get(bucket, 0) - price
            if abs(edge) > rc.get('material_edge_delta', 0.01):
                triggers.append('MATERIAL_EDGE')
                break

    # 3. Top bucket changed
    shift = detect_bucket_dominance_shift(probs_old, probs_new, rc.get('material_prob_pp', 5))
    if shift.get('shifted'):
        triggers.append('TOP_BUCKET_CHANGED')

    # 4. Probability confidence materially changed
    if shift.get('max_pp_change', 0) >= rc.get('material_prob_pp', 5):
        triggers.append('PROB_CONFIDENCE_CHANGE')
    # EV change check (approximate: prob * price)
    if probs_old and probs_new and prices_dict:
        for bucket in set(list(probs_old.keys()) + list(probs_new.keys())):
            ev_old = probs_old.get(bucket, 0) * prices_dict.get(bucket, 0)
            ev_new = probs_new.get(bucket, 0) * prices_dict.get(bucket, 0)
            if abs(ev_new - ev_old) * 100 >= rc.get('material_ev_pp', 2):
                triggers.append('MATERIAL_EV_CHANGE')
                break

    # 5. Nowcast regime changed
    if nowcast_regime_changed:
        triggers.append('NOWCAST_REGIME')

    # 6. Exposure exceeds risk limit
    total_exposure = sum(p.get('quantity', 0) * p.get('entry_price', 0) for p in current_positions.values())
    max_exposure = config.get('position_sizing', {}).get('total_max', 0.50) * 10000  # rough capital
    if total_exposure > max_exposure:
        triggers.append('EXPOSURE_LIMIT')

    # 7. Time-to-settlement de-risking
    if hours_to_settlement < config.get('exit', {}).get('time_to_settlement', {}).get('taper_start_hours', 4):
        triggers.append('T2S_DE_RISK')

    top_old = shift.get('top_old', '')
    top_new = shift.get('top_new', '')
    prob_pp_change = shift.get('max_pp_change', 0)
    is_stable = (top_old == top_new or not top_old) and prob_pp_change < 5

    # Drawdown override
    dd_result = compute_drawdown_multiplier(drawdown_pct, config)

    return {
        'should_rebalance': len(triggers) > 0 or dd_result['action'] != 'NORMAL',
        'triggers': triggers,
        'is_stable': is_stable,
        'skip_rebalance': (not is_stable and 'TOP_BUCKET_CHANGED' not in triggers) or dd_result['action'] == 'HARD_FLATTEN',
        'reduce_position': prob_pp_change > 10 or dd_result['action'] in ('REDUCE_RISK', 'HARD_FLATTEN'),
        'drawdown_action': dd_result['action'],
        'drawdown_multiplier': dd_result['multiplier'],
    }


def evaluate_refined_entry(bucket: str, model_prob: float, market_price: float, model_std: float,
                            dt_now, rain_regime: str, model_key: str, adjusted_bet: dict,
                            drawdown_pct: float, config: dict, post_mean: float = None) -> dict:
    """Full entry gate pipeline with all 7 conditions."""
    # 1. Time gate
    if dt_now.hour < config.get('entry', {}).get('min_hour', 8):
        return {'passes': False, 'reason': 'TIME_GATE', 'detail': f'before {config["entry"]["min_hour"]}:00'}

    # 2. Regime-based edge threshold
    regime = get_entry_regime(dt_now, rain_regime, config)
    regime_check = check_regime_edge_threshold(model_prob - market_price, regime, config)
    if not regime_check['passes']:
        return {'passes': False, 'reason': 'EDGE_TOO_LOW', 'detail': regime_check['note']}

    # 3. Probability confidence
    if not check_probability_confidence(model_std, config):
        return {'passes': False, 'reason': 'LOW_CONFIDENCE', 'detail': f'std={model_std:.2f}'}

    # 4. Boundary proximity
    bucket_lo, bucket_hi = _parse_bucket_bounds(bucket)
    mean_for_boundary = post_mean if post_mean is not None else model_prob
    bd_result = check_boundary_proximity(mean_for_boundary, model_std, bucket_lo, bucket_hi, config)
    
    # [關鍵修復 2] 不再硬性阻擋，而是讓其通過並返回縮減倍率
    # if not bd_result['passes']:
    #     return {'passes': False, 'reason': 'BOUNDARY_TOO_CLOSE', ...}

    # 5. Slippage-adjusted edge positive
    if adjusted_bet:
        slippage_pct = adjusted_bet.get('slippage_pct', 0)
        edge_after = (model_prob - market_price) - (slippage_pct / 100)
        if edge_after <= 0:
            return {'passes': False, 'reason': 'SLIPPAGE_EATS_EDGE', 'detail': f'edge_after={edge_after:.4f}'}

    # 6. Drawdown gate
    dd_result = compute_drawdown_multiplier(drawdown_pct, config)
    if dd_result['action'] in ('STOP_ENTRIES', 'HARD_FLATTEN'):
        return {'passes': False, 'reason': 'DRAWDOWN_STOP', 'detail': f'drawdown={drawdown_pct:.1%}'}

    # 7. Exposure limits (caller-level check)
    return {
        'passes': True, 'reason': 'PASS', 'detail': 'all entry gates passed',
        'regime': regime, 'exposure_cap': regime_check.get('exposure_cap', 0.50),
        'boundary_multiplier': bd_result.get('multiplier', 1.0),  # 這個倍率會在後續縮減下注量
        'distance_raw': bd_result.get('distance_raw', 999.0),
        'distance_std': bd_result.get('distance_std', 999.0),
    }


def evaluate_refined_exit(bucket: str, position: dict, model_prob: float, market_price: float,
                           model_std: float, dt_now, max_so_far: float = None, temp_now: float = None,
                           rain_regime: str = None, drawdown_pct: float = 0.0,
                           nowcast_stale: bool = False, data_missing: bool = False,
                           model_key: str = None, config: dict = None,
                           prob_top_bucket: float = 0.0, hours_to_settlement: float = 24.0) -> dict:
    """Full exit pipeline: forecast-driven + risk-driven + time-to-settlement."""
    side = position.get('side', 'YES')
    qty = position.get('quantity', 0)
    if qty <= 0:
        return {'action': 'HOLD', 'reasons': [], 'multiplier': 1.0, 'detail': 'no position'}

    ec = config.get('exit', {})
    if side == 'YES':
        edge = model_prob - market_price
    else:
        edge = market_price - model_prob

    reasons = []
    multiplier = 1.0

    # ── Extreme conviction hold ──
    conv_prob = ec.get('hold_conviction_prob', 0.98)
    if model_prob >= conv_prob or model_prob <= (1 - conv_prob):
        if hours_to_settlement <= ec.get('hold_max_hours', 6):
            return {'action': 'HOLD_CONVICTION', 'reasons': [], 'multiplier': 1.0,
                    'detail': f'extreme conviction prob={model_prob:.3f}'}

    # ── Forecast-driven exits ──
    # A1: Edge reversed
    reversal_thresh = ec.get('edge_reversal_threshold', -0.05)
    if edge < reversal_thresh:
        reasons.append('EDGE_REVERSED')
        multiplier = 0.0

    # A2: Edge disappeared
    if abs(edge) < 0.005 and not reasons:
        reasons.append('EDGE_DISAPPEARED')
        multiplier = 0.0

    # A3: Bucket probability below stop_prob
    stop_prob = ec.get('stop_prob', 0.05)
    if prob_top_bucket < stop_prob:
        reasons.append('PROB_BELOW_STOP')
        multiplier = min(multiplier, 0.3)

    # A4: Another bucket became dominant (checked at caller)
    # A5: Profit take
    if side == 'YES' and market_price > model_prob and multiplier > 0:
        reasons.append('PROFIT_TAKE')
        multiplier *= 0.5
    elif side == 'NO' and market_price < model_prob and multiplier > 0:
        reasons.append('PROFIT_TAKE')
        multiplier *= 0.5

    # A5: Model confidence deteriorated
    max_std = config.get('entry', {}).get('probability_confidence', {}).get('max_model_std', 2.5)
    if model_std > max_std * 1.5:
        reasons.append('CONFIDENCE_DROP')
        multiplier *= 0.5

    # ── Risk-driven exits ──
    # B1: Nowcast stale
    if nowcast_stale and model_key in ('model_c',):
        reasons.append('NOWCAST_STALE')
        multiplier *= 0.7
    elif nowcast_stale and model_key in ('rain_nowcast',):
        reasons.append('NOWCAST_STALE')
        multiplier *= 0.5

    # B2: Data feed missing
    if data_missing:
        reasons.append('DATA_MISSING')
        multiplier *= 0.5

    # B3: Rain emergency
    if rain_regime in ('weak_rain', 'moderate_or_heavy_rain') and max_so_far is not None and temp_now is not None:
        temp_drop = max_so_far - temp_now
        if temp_drop > 1.5:
            reasons.append('RAIN_EMERGENCY')
            multiplier *= 0.3

    # B4: Drawdown
    dd_result = compute_drawdown_multiplier(drawdown_pct, config)
    if dd_result['action'] == 'REDUCE_RISK':
        reasons.append('DRAWDOWN_REDUCE')
        multiplier *= dd_result['multiplier']
    elif dd_result['action'] == 'HARD_FLATTEN':
        return {'action': 'HARD_FLATTEN', 'reasons': ['DRAWDOWN_HARD'], 'multiplier': 0.0,
                'detail': f'drawdown={drawdown_pct:.1%}'}

    # ── Time-to-settlement de-risking ──
    t2s_mult = compute_time_to_settlement_multiplier(hours_to_settlement, config)
    if t2s_mult < 1.0:
        reasons.append('T2S_TAPER')
        multiplier *= t2s_mult

    if multiplier <= 0:
        return {'action': 'EXIT', 'reasons': reasons, 'multiplier': 0.0, 'detail': '; '.join(reasons)}
    if multiplier < 0.8:
        return {'action': 'REDUCE', 'reasons': reasons, 'multiplier': multiplier, 'detail': '; '.join(reasons)}
    if reasons:
        return {'action': 'HOLD_WITH_WARNING', 'reasons': reasons, 'multiplier': multiplier, 'detail': '; '.join(reasons)}
    return {'action': 'HOLD', 'reasons': [], 'multiplier': 1.0, 'detail': 'no exit triggered'}