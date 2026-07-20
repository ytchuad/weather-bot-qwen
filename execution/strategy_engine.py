# execution/strategy_engine.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

HKT_OFFSET = timedelta(hours=8)
ENHANCED_VERSION = "enhanced_v1"


def hkt_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + HKT_OFFSET


from execution.strategy_gate import (
    GateResult, REASON_PASS_ALL, REASON_EDGE_TOO_SMALL, REASON_HOLD,
    REASON_EXIT_SIGNAL, REASON_ENTRY_SIGNAL, REASON_RAIN_EXCLUDED,
    REASON_QTY_BELOW_MIN, REASON_EXPOSURE_LIMIT,
    load_strategy_config, select_minute_model, evaluate_refined_entry,
    evaluate_refined_exit, compute_position_size, compute_drawdown_multiplier,
    compute_time_to_settlement_multiplier, should_rebalance, get_entry_regime,
    check_boundary_proximity
)
from execution.portfolio_reconciler import PM_MIN_QTY
from execution.kelly_betting import compute_multi_kelly_bets
from execution.clob_slippage import apply_slippage_to_bets
from execution.clob_execution import (
    CLOBExecutionSnapshot,
    compute_depth_adjusted_bets,
    compute_sell_execution,
)

logger = logging.getLogger(__name__)

ENHANCED_ENTRY_EDGE = 0.03
ENHANCED_EXIT_EDGE_THRESHOLD = -0.05
HOLD_CONVICTION_PROB = 0.98
HOLD_CONVICTION_PRICE_HIGH = 0.95
HOLD_CONVICTION_PRICE_LOW = 0.05
HOLD_MAX_HOURS = 6
RAIN_EMERGENCY_TEMP_DROP = 1.5
STD_SPIKE_THRESHOLD = 3.0

MORNING_START = 8
AFTERNOON_START = 12
EVENING_START = 16
NIGHT_START = 20
DAY_END = 23

REASON_EDGE_REVERSED = 'EDGE_REVERSED'
REASON_PROFIT_TAKE = 'PROFIT_TAKE'
REASON_RAIN_EMERGENCY = 'RAIN_EMERGENCY'
REASON_STD_SPIKE = 'STD_SPIKE'
REASON_REDUCE_50_PCT = 'REDUCE_50_PCT'
REASON_HOLD_UNTIL_EXPIRY = 'HOLD_UNTIL_EXPIRY'
REASON_LIQUIDITY_INSUFFICIENT = 'LIQUIDITY_INSUFFICIENT'
REASON_TIME_WINDOW_CLOSED = 'TIME_WINDOW_CLOSED'
REASON_NO_EXECUTABLE_QUOTE = 'NO_EXECUTABLE_QUOTE'
REASON_PARTIAL_FILL = 'PARTIAL_FILL_REJECTED'


def _is_clob_mode(paper_execution_mode: str) -> bool:
    return paper_execution_mode == "clob_depth"


def _snapshot_for_position(
    execution_snapshots: dict,
    bucket: str,
    side: str,
) -> CLOBExecutionSnapshot | None:
    return (execution_snapshots.get(bucket) or {}).get(str(side).upper())


def _clob_reference_prices(
    execution_snapshots: dict | None,
) -> dict[str, float]:
    """Return YES-space CLOB references for rebalance diagnostics."""
    prices: dict[str, float] = {}
    for bucket, side_map in (execution_snapshots or {}).items():
        yes = side_map.get("YES") if isinstance(side_map, dict) else None
        no = side_map.get("NO") if isinstance(side_map, dict) else None
        if yes is not None and yes.best_ask is not None:
            prices[bucket] = yes.best_ask.price
        elif no is not None and no.best_ask is not None:
            prices[bucket] = 1.0 - no.best_ask.price
    return prices


def get_time_slot(dt_now: datetime) -> str:
    hour = dt_now.hour
    if MORNING_START <= hour < AFTERNOON_START:
        return 'morning'
    if AFTERNOON_START <= hour < EVENING_START:
        return 'afternoon'
    if EVENING_START <= hour < NIGHT_START:
        return 'evening'
    return 'night'


def get_time_based_exposure_limit(dt_now: datetime) -> float:
    slot = get_time_slot(dt_now)
    limits = {'morning': 0.50, 'afternoon': 0.30, 'evening': 0.10, 'night': 0.0}
    return limits[slot]


def get_confidence_multiplier(model_std: float) -> float:
    if model_std < 1.0:
        return 1.2
    if model_std < 2.0:
        return 1.0
    if model_std < 3.0:
        return 0.8
    return 0.6


def get_volatility_multiplier(recent_price_volatility: float) -> float:
    if recent_price_volatility < 0.02:
        return 1.0
    if recent_price_volatility < 0.05:
        return 0.8
    if recent_price_volatility < 0.10:
        return 0.6
    return 0.5


def get_effective_exposure_limit(
    dt_now: datetime,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    base_total_max: float = 0.50
) -> float:
    time_limit = get_time_based_exposure_limit(dt_now)
    confidence_mult = get_confidence_multiplier(model_std)
    volatility_mult = get_volatility_multiplier(recent_price_volatility)
    effective = min(time_limit, base_total_max) * confidence_mult * volatility_mult
    return max(0.0, effective)


def hours_to_resolution(dt_now: datetime) -> float:
    end_of_day = dt_now.replace(hour=23, minute=59, second=0, microsecond=0)
    delta = end_of_day - dt_now
    return max(0.0, delta.total_seconds() / 3600.0)


def is_extreme_conviction(model_prob: float, market_price: float) -> bool:
    is_extreme_yes = model_prob > HOLD_CONVICTION_PROB and market_price > HOLD_CONVICTION_PRICE_HIGH
    is_extreme_no = model_prob < (1.0 - HOLD_CONVICTION_PROB) and market_price < HOLD_CONVICTION_PRICE_LOW
    return is_extreme_yes or is_extreme_no


def check_enhanced_entry(
    bucket: str,
    model_prob: float,
    market_price: float,
    dt_now: datetime,
    adjusted_bet: dict = None
) -> GateResult:
    if dt_now.hour < MORNING_START:
        return GateResult(False, REASON_TIME_WINDOW_CLOSED, 'before 08:00, no entries allowed')

    slot = get_time_slot(dt_now)
    if slot in ('evening', 'night'):
        return GateResult(False, REASON_TIME_WINDOW_CLOSED, f'{slot}: no new entries')

    edge_yes = model_prob - market_price
    edge_no = market_price - model_prob
    if round(edge_yes, 4) <= ENHANCED_ENTRY_EDGE and round(edge_no, 4) <= ENHANCED_ENTRY_EDGE:
        return GateResult(False, REASON_EDGE_TOO_SMALL,
                          f'edge (yes={edge_yes:.3f}, no={edge_no:.3f}) <= {ENHANCED_ENTRY_EDGE}')

    if adjusted_bet is None:
        return GateResult(False, REASON_LIQUIDITY_INSUFFICIENT, 'no slippage data available')

    if not adjusted_bet.get('filled', False):
        return GateResult(False, REASON_LIQUIDITY_INSUFFICIENT,
                          f'order not fully filled: qty={adjusted_bet.get("adjusted_quantity",0)}')

    edge = edge_yes if edge_yes > ENHANCED_ENTRY_EDGE else edge_no
    slippage_pct = adjusted_bet.get('slippage_pct', 100)
    if slippage_pct > (edge * 100 / 2):
        return GateResult(False, REASON_LIQUIDITY_INSUFFICIENT,
                          f'slippage {slippage_pct:.1f}% > edge/2 ({edge*100/2:.1f}%)')

    action = 'BUY_YES' if edge_yes > ENHANCED_ENTRY_EDGE else 'BUY_NO'
    return GateResult(True, REASON_ENTRY_SIGNAL,
                      f'{action} approved: edge={edge:.3f}, slippage={slippage_pct:.1f}%')


def check_enhanced_exit(
    bucket: str,
    position: dict,
    model_prob: float,
    market_price: float,
    model_std: float,
    dt_now: datetime,
    max_so_far: float = None,
    temp_now: float = None,
    rain_regime: str = None
) -> GateResult:
    side = position.get('side', 'YES')
    qty = position.get('quantity', 0)
    if qty <= 0:
        return GateResult(False, REASON_HOLD, 'no position to exit')

    if side == 'YES':
        edge = model_prob - market_price
    else:
        edge = market_price - model_prob

    if is_extreme_conviction(model_prob, market_price):
        if hours_to_resolution(dt_now) <= HOLD_MAX_HOURS:
            return GateResult(True, REASON_HOLD_UNTIL_EXPIRY,
                              f'extreme conviction: prob={model_prob:.3f}, price={market_price:.3f}')
        else:
            return GateResult(True, REASON_EXIT_SIGNAL,
                              'extreme but >6h to expiry, exit to free capital')

    if round(edge, 4) < round(ENHANCED_EXIT_EDGE_THRESHOLD, 4):
        return GateResult(True, REASON_EDGE_REVERSED,
                          f'edge reversed: {edge:.3f} < {ENHANCED_EXIT_EDGE_THRESHOLD}')

    if side == 'YES' and market_price > model_prob:
        return GateResult(True, REASON_PROFIT_TAKE,
                          f'market ({market_price:.3f}) > model ({model_prob:.3f}), taking profit')

    if side == 'NO' and market_price < model_prob:
        return GateResult(True, REASON_PROFIT_TAKE,
                          f'market ({market_price:.3f}) < model ({model_prob:.3f}), taking profit')

    if max_so_far is not None and temp_now is not None and rain_regime in ('weak_rain', 'moderate_or_heavy_rain'):
        temp_drop = max_so_far - temp_now
        if temp_drop > RAIN_EMERGENCY_TEMP_DROP:
            return GateResult(True, REASON_RAIN_EMERGENCY,
                              f'rain + temp drop {temp_drop:.1f}°C > {RAIN_EMERGENCY_TEMP_DROP}°C')

    if model_std > STD_SPIKE_THRESHOLD:
        return GateResult(True, REASON_STD_SPIKE,
                          f'std {model_std:.1f} > {STD_SPIKE_THRESHOLD}, reducing 50%')

    return GateResult(False, REASON_HOLD, 'no exit condition triggered')


def compute_enhanced_orders(
    target_probs: dict,
    prices_dict: dict,
    token_ids_dict: dict,
    capital: float,
    mock_slippage: bool,
    dt_now: datetime,
    current_positions: dict,
    model_key: str,
    slug: str,
    temp_now: float = None,
    max_so_far: float = None,
    rain_regime: str = None,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    max_per_bucket: float = 0.15,
    execution_snapshots: dict | None = None,
    paper_execution_mode: str = "legacy_gamma_mock",
    partial_fill_policy: str = "fail_closed",
    gamma_reference_prices: dict | None = None,
) -> tuple:
    effective_limit = get_effective_exposure_limit(
        dt_now, model_std, recent_price_volatility, max_per_bucket * 3
    )
    total_max_limit = min(effective_limit, 0.50)

    clob_sizing = None
    if _is_clob_mode(paper_execution_mode):
        if execution_snapshots:
            clob_sizing = compute_depth_adjusted_bets(
                target_probs=target_probs,
                gamma_reference_prices=gamma_reference_prices or prices_dict,
                capital=capital,
                snapshots=execution_snapshots,
                max_per_bucket=max_per_bucket,
                total_max=total_max_limit,
                partial_fill_policy=partial_fill_policy,
            )
            adjusted_bets = clob_sizing.adjusted_bets
        else:
            adjusted_bets = {}
    else:
        bets = compute_multi_kelly_bets(
            target_probs, prices_dict, capital,
            max_per_bucket=max_per_bucket, total_max=total_max_limit
        )
        adjusted_bets = apply_slippage_to_bets(
            bets, token_ids_dict, prices_dict=prices_dict, mock_mode=mock_slippage
        )

    market_positions = current_positions.get(model_key, {}).get(slug, {})
    all_buckets = set(list(market_positions.keys()) + list(target_probs.keys()))
    target_positions = {}
    decisions = []
    exit_quotes = {}

    if _is_clob_mode(paper_execution_mode):
        for bucket, current_pos in market_positions.items():
            quantity = float(current_pos.get("quantity", 0.0))
            if quantity <= 0.0:
                continue
            snapshot = _snapshot_for_position(
                execution_snapshots or {}, bucket, current_pos.get("side", "YES")
            )
            exit_quotes[bucket] = (
                compute_sell_execution(snapshot, quantity, partial_fill_policy)
                if snapshot is not None else None
            )

    for bucket in sorted(all_buckets):
        model_prob = target_probs.get(bucket, 0.5)
        market_price = prices_dict.get(bucket, 0.5)
        current_pos = market_positions.get(bucket, {})
        bet = adjusted_bets.get(bucket, {})
        kel_qty = bet.get('adjusted_quantity', 0)

        if current_pos.get('quantity', 0) > 0:
            clob_exit = exit_quotes.get(bucket) if _is_clob_mode(paper_execution_mode) else None
            if _is_clob_mode(paper_execution_mode):
                if clob_exit is None or clob_exit.filled_shares <= 0:
                    decisions.append({
                        'bucket': bucket, 'action': 'BLOCKED',
                        'reason': REASON_NO_EXECUTABLE_QUOTE,
                        'detail': 'no valid CLOB bid depth for exit; residual retained'
                    })
                    continue
                if not clob_exit.is_full_fill and partial_fill_policy == 'fail_closed':
                    decisions.append({
                        'bucket': bucket, 'action': 'BLOCKED',
                        'reason': REASON_PARTIAL_FILL,
                        'detail': 'exit depth is insufficient under fail_closed policy'
                    })
                    continue
                exit_token_price = clob_exit.net_sell_vwap
                market_price = (
                    exit_token_price
                    if str(current_pos.get('side', 'YES')).upper() == 'YES'
                    else 1.0 - exit_token_price
                )
            exit_result = check_enhanced_exit(
                bucket, current_pos, model_prob, market_price,
                model_std, dt_now, max_so_far, temp_now, rain_regime
            )
            if exit_result.passed and exit_result.reason_code != REASON_HOLD_UNTIL_EXPIRY:
                if exit_result.reason_code == REASON_STD_SPIKE:
                    reduce_qty = current_pos['quantity'] * 0.5
                    if _is_clob_mode(paper_execution_mode):
                        reduce_qty = max(
                            0.0, float(current_pos['quantity']) - clob_exit.filled_shares
                        )
                    target_positions[bucket] = {
                        'side': current_pos['side'],
                        'quantity': round(reduce_qty, 2),
                        'target_price': (
                            clob_exit.net_sell_vwap
                            if _is_clob_mode(paper_execution_mode) else market_price
                        )
                    }
                    decisions.append({
                        'bucket': bucket, 'action': 'REDUCE',
                        'reason': REASON_STD_SPIKE,
                        'detail': f'halved to {reduce_qty:.1f} (std {model_std:.1f})'
                    })
                elif exit_result.reason_code == REASON_EXIT_SIGNAL:
                    target_positions[bucket] = {
                        'side': current_pos['side'],
                        'quantity': (
                            max(0.0, float(current_pos['quantity']) - clob_exit.filled_shares)
                            if _is_clob_mode(paper_execution_mode) else 0.0
                        ),
                        'target_price': (
                            clob_exit.net_sell_vwap
                            if _is_clob_mode(paper_execution_mode) else market_price
                        )
                    }
                    decisions.append({
                        'bucket': bucket, 'action': 'EXIT',
                        'reason': REASON_EXIT_SIGNAL,
                        'detail': 'exit to free capital (>6h from expiry)'
                    })
                else:
                    target_positions[bucket] = {
                        'side': current_pos['side'],
                        'quantity': (
                            max(0.0, float(current_pos['quantity']) - clob_exit.filled_shares)
                            if _is_clob_mode(paper_execution_mode) else 0.0
                        ),
                        'target_price': (
                            clob_exit.net_sell_vwap
                            if _is_clob_mode(paper_execution_mode) else market_price
                        )
                    }
                    decisions.append({
                        'bucket': bucket, 'action': 'EXIT',
                        'reason': exit_result.reason_code,
                        'detail': exit_result.detail
                    })
                continue

            if exit_result.reason_code == REASON_HOLD_UNTIL_EXPIRY:
                decisions.append({
                    'bucket': bucket, 'action': 'HOLD',
                    'reason': REASON_HOLD_UNTIL_EXPIRY,
                    'detail': exit_result.detail
                })
                target_positions[bucket] = current_pos
                continue

            decisions.append({
                'bucket': bucket, 'action': 'HOLD',
                'reason': REASON_HOLD,
                'detail': 'no exit condition triggered, keeping position'
            })
            continue
        else:
            kel_side_calc = 'YES' if bet.get('action') == 'BUY_YES' else 'NO'
            entry_qty = kel_qty

            if _is_clob_mode(paper_execution_mode):
                if not bet:
                    decisions.append({
                        'bucket': bucket, 'action': 'BLOCKED',
                        'reason': (
                            clob_sizing.rejected.get(bucket, REASON_NO_EXECUTABLE_QUOTE)
                            if clob_sizing else REASON_NO_EXECUTABLE_QUOTE
                        ),
                        'detail': 'no size-specific CLOB execution quote'
                    })
                    continue
                market_price = bet.get('execution_yes_price', market_price)

            if entry_qty < PM_MIN_QTY:
                decisions.append({
                    'bucket': bucket, 'action': 'NO_TRADE',
                    'reason': REASON_QTY_BELOW_MIN,
                    'detail': f'Kelly qty {entry_qty:.1f} < {PM_MIN_QTY}'
                })
                continue

            gate_bet = bet
            if (
                _is_clob_mode(paper_execution_mode)
                and partial_fill_policy in {"accept_partial", "reduce_to_available"}
                and bet.get("is_partial")
            ):
                # The target quantity is already reduced to the available
                # filled shares.  Let the explicit partial-fill policy carry
                # that reduced target through the entry gate.
                gate_bet = {**bet, "filled": True}
            entry_result = check_enhanced_entry(
                bucket, model_prob, market_price, dt_now, adjusted_bet=gate_bet
            )

            if entry_result.passed:
                target_positions[bucket] = {
                    'side': kel_side_calc,
                    'quantity': entry_qty,
                    'target_price': bet.get('avg_fill_price', market_price)
                }
                decisions.append({
                    'bucket': bucket, 'action': 'ENTRY',
                    'reason': REASON_ENTRY_SIGNAL,
                    'detail': entry_result.detail,
                    'requested_shares': bet.get('requested_shares'),
                    'depth_adjusted_vwap': bet.get('execution_price'),
                    'fee': bet.get('fee'),
                    'fill_ratio': bet.get('fill_ratio'),
                    'diagnostic_edge': bet.get('diagnostic_edge'),
                    'executable_edge_at_final_size': bet.get('executable_edge_at_final_size'),
                })
            else:
                decisions.append({
                    'bucket': bucket, 'action': 'BLOCKED',
                    'reason': entry_result.reason_code,
                    'detail': entry_result.detail
                })

    return target_positions, decisions


def run_enhanced_rebalance_cycle(
    slug: str,
    model_key: str,
    capital: float,
    mock_slippage: bool,
    target_probs: dict,
    prices_dict: dict,
    token_ids_dict: dict,
    dt_now: datetime = None,
    current_positions: dict = None,
    temp_now: float = None,
    max_so_far: float = None,
    rain_regime: str = None,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    max_per_bucket: float = 0.15,
    execution_snapshots: dict | None = None,
    paper_execution_mode: str = "legacy_gamma_mock",
    partial_fill_policy: str = "fail_closed",
    gamma_reference_prices: dict | None = None,
) -> dict:
    if dt_now is None:
        dt_now = hkt_now()
    if current_positions is None:
        from execution.portfolio_reconciler import load_positions
        current_positions = load_positions()

    target_positions, decisions = compute_enhanced_orders(
        target_probs=target_probs,
        prices_dict=prices_dict,
        token_ids_dict=token_ids_dict,
        capital=capital,
        mock_slippage=mock_slippage,
        dt_now=dt_now,
        current_positions=current_positions,
        model_key=model_key,
        slug=slug,
        temp_now=temp_now,
        max_so_far=max_so_far,
        rain_regime=rain_regime,
        model_std=model_std,
        recent_price_volatility=recent_price_volatility,
        max_per_bucket=max_per_bucket,
        execution_snapshots=execution_snapshots,
        paper_execution_mode=paper_execution_mode,
        partial_fill_policy=partial_fill_policy,
        gamma_reference_prices=gamma_reference_prices,
    )

    slot = get_time_slot(dt_now)
    effective_limit = get_effective_exposure_limit(
        dt_now, model_std, recent_price_volatility, max_per_bucket * 3
    )

    summary = {
        'version': ENHANCED_VERSION,
        'time_slot': slot,
        'effective_exposure_limit': effective_limit,
        'model_std': model_std,
        'volatility': recent_price_volatility,
        'target_positions': target_positions,
        'decisions': decisions,
        'total_entry_count': sum(1 for d in decisions if d['action'] == 'ENTRY'),
        'total_exit_count': sum(1 for d in decisions if d['action'] == 'EXIT' or d['action'] == 'REDUCE'),
        'total_hold_count': sum(1 for d in decisions if d['action'] == 'HOLD' or d['action'] == 'HOLD_UNTIL_EXPIRY'),
        'total_blocked_count': sum(1 for d in decisions if d['action'] in ('BLOCKED', 'NO_TRADE')),
        'execution_mode': paper_execution_mode,
    }

    return summary


# ═══════════════════════════════════════════════════════════════
# Config-driven strategy engine (v2)
# ═══════════════════════════════════════════════════════════════

def load_config_for_strategy(strategy_key: str = None) -> dict:
    """Load config from paper_strategies.json for a given strategy key."""
    return load_strategy_config(strategy_key)


def check_config_entry(
    bucket: str,
    model_prob: float,
    market_price: float,
    model_std: float,
    dt_now: datetime,
    rain_regime: str,
    model_key: str,
    adjusted_bet: dict = None,
    drawdown_pct: float = 0.0,
    config: dict = None,
    post_mean: float = None
) -> dict:
    """Config-driven entry gate. Returns dict with passes, reason, detail, multipliers."""
    if config is None:
        config = load_config_for_strategy()
    
    # [關鍵修正] 根據 action 調整 model_prob 和 market_price
    # 讓 evaluate_refined_entry 始終看到正向的 Edge
    action = adjusted_bet.get('action', 'BUY_YES') if adjusted_bet else 'BUY_YES'
    if action == 'BUY_NO':
        # 買 NO 時，將機率和價格反轉
        effective_prob = 1.0 - model_prob
        effective_price = 1.0 - market_price
    else:
        effective_prob = model_prob
        effective_price = market_price

    return evaluate_refined_entry(
        bucket, effective_prob, effective_price, model_std,
        dt_now, rain_regime, model_key, adjusted_bet, drawdown_pct, config, post_mean=post_mean
    )


def check_config_exit(
    bucket: str,
    position: dict,
    model_prob: float,
    market_price: float,
    model_std: float,
    dt_now: datetime,
    max_so_far: float = None,
    temp_now: float = None,
    rain_regime: str = None,
    drawdown_pct: float = 0.0,
    nowcast_stale: bool = False,
    data_missing: bool = False,
    model_key: str = None,
    config: dict = None,
    prob_top_bucket: float = 0.0,
    hours_to_settlement: float = 24.0
) -> dict:
    """Config-driven exit gate. Returns dict with action, reasons, multiplier."""
    if config is None:
        config = load_config_for_strategy()
    return evaluate_refined_exit(
        bucket, position, model_prob, market_price, model_std,
        dt_now, max_so_far, temp_now, rain_regime, drawdown_pct,
        nowcast_stale, data_missing, model_key, config,
        prob_top_bucket, hours_to_settlement
    )


def compute_config_orders(
    target_probs: dict,
    prices_dict: dict,
    token_ids_dict: dict,
    capital: float,
    mock_slippage: bool,
    dt_now: datetime,
    current_positions: dict,
    model_key: str,
    slug: str,
    config: dict = None,
    drawdown_pct: float = 0.0,
    temp_now: float = None,
    max_so_far: float = None,
    rain_regime: str = None,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    hours_to_settlement: float = 24.0,
    nowcast_stale: bool = False,
    data_missing: bool = False,
    probs_old: dict = None,
    probs_new: dict = None,
    post_mean: float = None,
    execution_snapshots: dict | None = None,
    paper_execution_mode: str = "legacy_gamma_mock",
    partial_fill_policy: str = "fail_closed",
    gamma_reference_prices: dict | None = None,
) -> tuple:
    """Config-driven order computation using strategy_gate v2 gates."""
    if config is None:
        config = load_config_for_strategy()
    if probs_old is None:
        probs_old = {}
    if probs_new is None:
        probs_new = target_probs

    sc = config.get('position_sizing', {})
    total_max = sc.get('total_max', 0.50)
    max_per_bucket = sc.get('max_per_bucket', 0.15)

    clob_sizing = None
    if _is_clob_mode(paper_execution_mode):
        if execution_snapshots:
            clob_sizing = compute_depth_adjusted_bets(
                target_probs=target_probs,
                gamma_reference_prices=gamma_reference_prices or prices_dict,
                capital=capital,
                snapshots=execution_snapshots,
                max_per_bucket=max_per_bucket,
                total_max=total_max,
                partial_fill_policy=partial_fill_policy,
            )
            adjusted_bets = clob_sizing.adjusted_bets
        else:
            adjusted_bets = {}
    else:
        bets = compute_multi_kelly_bets(
            target_probs, prices_dict, capital,
            max_per_bucket=max_per_bucket, total_max=total_max
        )
        adjusted_bets = apply_slippage_to_bets(
            bets, token_ids_dict, prices_dict=prices_dict, mock_mode=mock_slippage
        )

    market_positions = current_positions.get(model_key, {}).get(slug, {})
    all_buckets = set(list(market_positions.keys()) + list(target_probs.keys()))
    target_positions = {}
    decisions = []
    exit_quotes = {}

    if _is_clob_mode(paper_execution_mode):
        for bucket, current_pos in market_positions.items():
            quantity = float(current_pos.get("quantity", 0.0))
            if quantity <= 0.0:
                continue
            snapshot = _snapshot_for_position(
                execution_snapshots or {}, bucket, current_pos.get("side", "YES")
            )
            exit_quotes[bucket] = (
                compute_sell_execution(snapshot, quantity, partial_fill_policy)
                if snapshot is not None else None
            )

    # Pre-compute drawdown and T2S multipliers
    dd_result = compute_drawdown_multiplier(drawdown_pct, config)

    # Model selection
    if isinstance(rain_regime, str):
        has_rain = rain_regime != 'no_rain'
    elif isinstance(rain_regime, dict):
        has_rain = rain_regime.get('rain_nc_sum_0_120m', 0) > 0
    else:
        has_rain = False

    model_sel = select_minute_model(
        {'nowcast_stale': nowcast_stale, 'nowcast_missing': data_missing,
         'rainfall_available': has_rain},
        config, model_cache_available=set([model_key]) if model_key else set()
    )

    active_model = model_sel.get('selected_model', model_key)
    effective_post_mean = post_mean if post_mean is not None else temp_now

    for bucket in sorted(all_buckets):
        model_prob = target_probs.get(bucket, 0.5)
        market_price = prices_dict.get(bucket, 0.5)
        current_pos = market_positions.get(bucket, {})
        bet = adjusted_bets.get(bucket, {})
        kel_qty = bet.get('adjusted_quantity', bet.get('quantity', 0))

        if current_pos.get('quantity', 0) > 0:
            clob_exit = exit_quotes.get(bucket) if _is_clob_mode(paper_execution_mode) else None
            if _is_clob_mode(paper_execution_mode):
                if clob_exit is None or clob_exit.filled_shares <= 0:
                    decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                      'reason': REASON_NO_EXECUTABLE_QUOTE,
                                      'detail': 'no valid CLOB bid depth for exit; residual retained'})
                    continue
                if not clob_exit.is_full_fill and partial_fill_policy == 'fail_closed':
                    decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                      'reason': REASON_PARTIAL_FILL,
                                      'detail': 'exit depth is insufficient under fail_closed policy'})
                    continue
                exit_token_price = clob_exit.net_sell_vwap
                market_price = (
                    exit_token_price
                    if str(current_pos.get('side', 'YES')).upper() == 'YES'
                    else 1.0 - exit_token_price
                )
            # Evaluate exit
            exit_result = check_config_exit(
                bucket, current_pos, model_prob, market_price, model_std,
                dt_now, max_so_far, temp_now, rain_regime,
                drawdown_pct, nowcast_stale, data_missing,
                active_model, config, model_prob,
                hours_to_settlement
            )
            action = exit_result.get('action', 'HOLD')
            mult = exit_result.get('multiplier', 1.0)
            reasons = exit_result.get('reasons', [])

            if action == 'HARD_FLATTEN':
                target_positions[bucket] = {
                    'side': current_pos['side'],
                    'quantity': (
                        max(0.0, float(current_pos['quantity']) - clob_exit.filled_shares)
                        if _is_clob_mode(paper_execution_mode) else 0.0
                    ),
                    'target_price': (
                        clob_exit.net_sell_vwap
                        if _is_clob_mode(paper_execution_mode) else market_price
                    )
                }
                decisions.append({'bucket': bucket, 'action': 'EXIT', 'reason': 'DRAWDOWN_HARD',
                                  'detail': exit_result.get('detail', '')})
                continue
            if action == 'EXIT' or mult <= 0:
                target_positions[bucket] = {
                    'side': current_pos['side'],
                    'quantity': (
                        max(0.0, float(current_pos['quantity']) - clob_exit.filled_shares)
                        if _is_clob_mode(paper_execution_mode) else 0.0
                    ),
                    'target_price': (
                        clob_exit.net_sell_vwap
                        if _is_clob_mode(paper_execution_mode) else market_price
                    )
                }
                decisions.append({'bucket': bucket, 'action': 'EXIT', 'reason': ';'.join(reasons),
                                  'detail': exit_result.get('detail', '')})
                continue
            if action == 'REDUCE' and mult < 1.0:
                reduce_qty = current_pos['quantity'] * mult
                if _is_clob_mode(paper_execution_mode):
                    reduce_qty = max(
                        0.0, float(current_pos['quantity']) - clob_exit.filled_shares
                    )
                target_positions[bucket] = {
                    'side': current_pos['side'], 'quantity': round(reduce_qty, 2),
                    'target_price': (
                        clob_exit.net_sell_vwap
                        if _is_clob_mode(paper_execution_mode) else market_price
                    )
                }
                decisions.append({'bucket': bucket, 'action': 'REDUCE', 'reason': ';'.join(reasons),
                                  'detail': exit_result.get('detail', '')})
                continue
            # Hold
            decisions.append({'bucket': bucket, 'action': 'HOLD', 'reason': 'HOLD',
                              'detail': exit_result.get('detail', 'no exit triggered')})
            continue
        else:
            kel_side_calc = 'YES' if bet.get('action') == 'BUY_YES' else 'NO'
            entry_qty = kel_qty

            if _is_clob_mode(paper_execution_mode):
                if not bet:
                    decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                      'reason': (clob_sizing.rejected.get(bucket, REASON_NO_EXECUTABLE_QUOTE)
                                                if clob_sizing else REASON_NO_EXECUTABLE_QUOTE),
                                      'detail': 'no size-specific CLOB execution quote'})
                    continue
                market_price = bet.get('execution_yes_price', market_price)

            if entry_qty < sc.get('min_qty', 5.0):
                decisions.append({'bucket': bucket, 'action': 'NO_TRADE',
                                  'reason': 'QTY_TOO_SMALL',
                                  'detail': f'qty {entry_qty:.2f} < min {sc.get("min_qty", 5.0)} | bet={bet}'})
                continue

            # Drawdown gate for entries
            if dd_result['action'] in ('STOP_ENTRIES', 'HARD_FLATTEN'):
                decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                  'reason': 'DRAWDOWN_STOP',
                                  'detail': f'drawdown={drawdown_pct:.1%}'})
                continue

            entry_result = check_config_entry(
                bucket, model_prob, market_price, model_std,
                dt_now, rain_regime, active_model, bet,
                drawdown_pct, config, post_mean=effective_post_mean  # <--- 使用 effective_post_mean
            )
            if entry_result.get('passes', False):
                # Apply sizing multipliers
                boundary_mult = entry_result.get('boundary_multiplier', 1.0)
                distance_std = entry_result.get('distance_std', 999.0)
                final_qty = compute_position_size(
                    entry_qty, active_model, dt_now, rain_regime,
                    distance_std, config
                ) * boundary_mult * dd_result.get('multiplier', 1.0)
                final_qty = max(0, round(final_qty, 2))

                if final_qty >= sc.get('min_qty', 5.0):
                    target_positions[bucket] = {
                        'side': kel_side_calc, 'quantity': final_qty,
                        'target_price': bet.get('avg_fill_price', market_price)
                    }
                    decisions.append({'bucket': bucket, 'action': 'ENTRY',
                                      'reason': 'ENTRY_SIGNAL',
                                      'detail': entry_result.get('detail', ''),
                                      'requested_shares': bet.get('requested_shares'),
                                      'depth_adjusted_vwap': bet.get('execution_price'),
                                      'fee': bet.get('fee'),
                                      'fill_ratio': bet.get('fill_ratio'),
                                      'diagnostic_edge': bet.get('diagnostic_edge'),
                                      'executable_edge_at_final_size': bet.get('executable_edge_at_final_size')})
                else:
                    decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                      'reason': 'QTY_BELOW_MIN_AFTER_MULT',
                                      'detail': f'final_qty {final_qty:.1f} < min'})
            else:
                decisions.append({'bucket': bucket, 'action': 'BLOCKED',
                                  'reason': entry_result.get('reason', 'GATE_FAIL'),
                                  'detail': entry_result.get('detail', '')})

    return target_positions, decisions


def run_config_rebalance_cycle(
    slug: str,
    model_key: str,
    capital: float,
    mock_slippage: bool,
    target_probs: dict,
    prices_dict: dict,
    token_ids_dict: dict,
    config: dict = None,
    strategy_key: str = None,
    dt_now: datetime = None,
    current_positions: dict = None,
    temp_now: float = None,
    max_so_far: float = None,
    rain_regime: str = None,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    hours_to_settlement: float = 24.0,
    nowcast_stale: bool = False,
    data_missing: bool = False,
    drawdown_pct: float = 0.0,
    probs_old: dict = None,
    probs_new: dict = None,
    post_mean: float = None,
    execution_snapshots: dict | None = None,
    paper_execution_mode: str = "legacy_gamma_mock",
    partial_fill_policy: str = "fail_closed",
    gamma_reference_prices: dict | None = None,
) -> dict:
    """Full config-driven rebalance cycle."""
    if dt_now is None:
        dt_now = hkt_now()
    if current_positions is None:
        from execution.portfolio_reconciler import load_positions
        current_positions = load_positions()
    if config is None:
        config = load_config_for_strategy(strategy_key)

    target_positions, decisions = compute_config_orders(
        target_probs=target_probs,
        prices_dict=prices_dict,
        token_ids_dict=token_ids_dict,
        capital=capital,
        mock_slippage=mock_slippage,
        dt_now=dt_now,
        current_positions=current_positions,
        model_key=model_key,
        slug=slug,
        config=config,
        drawdown_pct=drawdown_pct,
        temp_now=temp_now,
        max_so_far=max_so_far,
        rain_regime=rain_regime,
        model_std=model_std,
        recent_price_volatility=recent_price_volatility,
        hours_to_settlement=hours_to_settlement,
        nowcast_stale=nowcast_stale,
        data_missing=data_missing,
        probs_old=probs_old,
        probs_new=probs_new,
        post_mean=post_mean,
        execution_snapshots=execution_snapshots,
        paper_execution_mode=paper_execution_mode,
        partial_fill_policy=partial_fill_policy,
        gamma_reference_prices=gamma_reference_prices,
    )

    # Rebalance decision
    rebalance_prices = prices_dict
    if _is_clob_mode(paper_execution_mode):
        rebalance_prices = _clob_reference_prices(execution_snapshots)
    rb_result = should_rebalance(
        current_positions.get(model_key, {}).get(slug, {}),
        target_positions,
        probs_old or {},
        probs_new or target_probs,
        rebalance_prices,
        nowcast_stale, drawdown_pct, hours_to_settlement, config
    )

    slot = get_time_slot(dt_now)
    dd_mult = compute_drawdown_multiplier(drawdown_pct, config)
    t2s_mult = compute_time_to_settlement_multiplier(hours_to_settlement, config)

    summary = {
        'version': 'config_v2',
        'time_slot': slot,
        'model_std': model_std,
        'volatility': recent_price_volatility,
        'target_positions': target_positions,
        'decisions': decisions,
        'rebalance_triggers': rb_result.get('triggers', []),
        'should_rebalance': rb_result.get('should_rebalance', True),
        'skip_rebalance': rb_result.get('skip_rebalance', False),
        'drawdown_action': dd_mult.get('action', 'NORMAL'),
        'drawdown_mult': dd_mult.get('multiplier', 1.0),
        't2s_mult': t2s_mult,
        'total_entry_count': sum(1 for d in decisions if d['action'] == 'ENTRY'),
        'total_exit_count': sum(1 for d in decisions if d['action'] == 'EXIT' or d['action'] == 'REDUCE'),
        'total_hold_count': sum(1 for d in decisions if d['action'] == 'HOLD'),
        'total_blocked_count': sum(1 for d in decisions if d['action'] in ('BLOCKED', 'NO_TRADE')),
        'execution_mode': paper_execution_mode,
        'depth_sizing': None,
    }

    return summary
