# execution/portfolio_reconciler.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import copy
import json
import logging
import uuid
import pandas as pd
from datetime import datetime, timedelta, timezone

_HKT_OFFSET = timedelta(hours=8)
def _hkt_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + _HKT_OFFSET
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

PM_MIN_QTY = 5.0
PM_QTY_STEP = 0.01
MIN_REBALANCE_QTY_DELTA = max(0.1, PM_MIN_QTY * 0.10)
MIN_REBALANCE_VALUE_DELTA = 2.0
POSITIONS_PATH = Path('data/current_positions.json')
AUDIT_LOG_PATH = Path('data/paper_trade_audit.parquet')

ZERO_POSITION = {'side': None, 'quantity': 0.0, 'target_price': 0.0}


@dataclass
class ReconciliationAction:
    bucket: str
    action: str
    side_before: Optional[str]
    qty_before: float
    entry_price_before: float
    side_after: Optional[str]
    qty_after: float
    target_price: float


@dataclass
class ReconciliationResult:
    portfolio_id: str
    slug: str
    strategy_key: str
    timestamp: str
    strategy_context: dict = field(default_factory=dict)
    before_snapshot: dict = field(default_factory=dict)
    after_snapshot: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    positions_updated: dict = field(default_factory=dict)
    run_id: str = ""
    preview: bool = False


def load_positions() -> dict:
    if POSITIONS_PATH.exists():
        try:
            with open(POSITIONS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_positions(positions: dict):
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def normalize_target_positions(target_positions: dict, current_snapshot: dict) -> dict:
    normalized = dict(target_positions)
    for bucket in current_snapshot:
        if bucket not in normalized:
            normalized[bucket] = dict(ZERO_POSITION)
    return normalized


def reconcile_positions(
    current_positions: dict,
    target_positions: dict,
    portfolio_id: str,
    slug: str,
    strategy_key: str,
    strategy_context: dict = None,
    preview: bool = False
) -> ReconciliationResult:
    updated = copy.deepcopy(current_positions) if current_positions else {}
    if portfolio_id not in updated:
        updated[portfolio_id] = {}
    if slug not in updated[portfolio_id]:
        updated[portfolio_id][slug] = {}
    if strategy_key not in updated[portfolio_id][slug]:
        updated[portfolio_id][slug][strategy_key] = {}

    before_snapshot = copy.deepcopy(updated[portfolio_id][slug][strategy_key])
    normalized_targets = normalize_target_positions(target_positions, before_snapshot)
    all_buckets = set(before_snapshot.keys()) | set(normalized_targets.keys())
    market_pos = updated[portfolio_id][slug][strategy_key]
    actions = []

    for bucket in sorted(all_buckets):
        target = normalized_targets.get(bucket, dict(ZERO_POSITION))
        current = before_snapshot.get(bucket, {'side': None, 'quantity': 0.0, 'entry_price': 0.0})

        target_qty = target.get('quantity', 0)
        target_side = target.get('side')
        target_price = target.get('target_price', target.get('avg_fill_price', 0.5))
        if target_side == 'NO':
            target_price = 1.0 - target_price
        current_qty = current.get('quantity', 0.0)
        current_side = current.get('side')
        current_price = current.get('entry_price', 0.0)
        if current_side == 'NO':
            current_price = 1.0 - current_price

        if target_qty < 0.1:
            if current_qty > 0:
                actions.append(ReconciliationAction(
                    bucket, 'EXIT_ZERO', current_side, current_qty, current_price,
                    None, 0, target_price
                ))
                market_pos.pop(bucket, None)
            continue

        if 0 < target_qty < PM_MIN_QTY:
            if current_qty > 0:
                actions.append(ReconciliationAction(
                    bucket, 'EXIT_DUST', current_side, current_qty, current_price,
                    None, 0, target_price
                ))
                market_pos.pop(bucket, None)
            continue

        if current_side == target_side and current_qty > 0:
            delta_qty = abs(target_qty - current_qty)
            delta_value = abs(target_qty * target_price - current_qty * current_price)
            if delta_qty < MIN_REBALANCE_QTY_DELTA and delta_value < MIN_REBALANCE_VALUE_DELTA:
                continue

            delta = target_qty - current_qty
            if abs(delta) < 0.1:
                continue
            action_type = 'INCREASE' if delta > 0 else 'DECREASE'
            if delta > 0:
                old_cost = current_qty * current_price
                new_cost = delta * target_price
                total_cost = old_cost + new_cost
                avg_price = total_cost / target_qty
                market_pos[bucket] = {'side': target_side, 'quantity': round(target_qty, 2), 'entry_price': round(avg_price, 6)}
            else:
                market_pos[bucket] = {'side': target_side, 'quantity': round(target_qty, 2), 'entry_price': current_price}
            actions.append(ReconciliationAction(
                bucket, action_type, current_side, current_qty, current_price,
                target_side, target_qty, target_price
            ))
        else:
            if current_qty > 0:
                actions.append(ReconciliationAction(
                    bucket, 'FLIP', current_side, current_qty, current_price,
                    target_side, target_qty, target_price
                ))
            else:
                actions.append(ReconciliationAction(
                    bucket, 'NEW', current_side, current_qty, current_price,
                    target_side, target_qty, target_price
                ))
            market_pos[bucket] = {'side': target_side, 'quantity': round(target_qty, 2), 'entry_price': target_price}

    after_snapshot = copy.deepcopy(market_pos)
    if not market_pos:
        if slug in updated.get(portfolio_id, {}):
            del updated[portfolio_id][slug]
    if not updated.get(portfolio_id):
        updated.pop(portfolio_id, None)

    run_id = generate_run_id()
    return ReconciliationResult(
        portfolio_id=portfolio_id,
        slug=slug,
        strategy_key=strategy_key,
        timestamp=_hkt_now().isoformat(),
        strategy_context=strategy_context or {},
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        actions=actions,
        positions_updated=updated,
        run_id=run_id,
        preview=preview,
    )


def build_audit_events(reconciliation_result: ReconciliationResult, prices_dict: dict = None) -> list:
    ctx = reconciliation_result.strategy_context or {}
    events = []
    for action in reconciliation_result.actions:
        fill = (ctx.get('execution_fill_by_bucket') or {}).get(action.bucket, {})
        events.append({
            'timestamp': reconciliation_result.timestamp,
            'run_id': reconciliation_result.run_id,
            'portfolio_id': reconciliation_result.portfolio_id,
            'strategy_key': reconciliation_result.strategy_key,
            'slug': reconciliation_result.slug,
            'strategy_version': ctx.get('strategy_version', ''),
            'selected_model': ctx.get('selected_model', ''),
            'scheduler_source': ctx.get('scheduler_source', 'manual'),
            'bucket': action.bucket,
            'action': action.action,
            'side_before': action.side_before,
            'qty_before': action.qty_before,
            'entry_price_before': action.entry_price_before,
            'side_after': action.side_after,
            'qty_after': action.qty_after,
            'target_price': action.target_price,
            'market_prices': json.dumps(prices_dict) if prices_dict else '{}',
            'entry_reason': ctx.get('entry_reason', ''),
            'exit_reason': ctx.get('exit_reason', ''),
            'edge_before_slippage': ctx.get('edge_before_slippage', None),
            'edge_after_slippage': ctx.get('edge_after_slippage', None),
            'rain_regime': ctx.get('rain_regime', ''),
            'rain_nc_sum_0_120m': ctx.get('rain_nc_sum_0_120m', None),
            'rain_nowcast_age_minutes': ctx.get('rain_nowcast_age_minutes', None),
            'data_quality_flags': ctx.get('data_quality_flags', ''),
            'reason_code': ctx.get('reason_code', ''),
            'execution_mode': ctx.get('execution_mode', ctx.get('paper_execution_mode', '')),
            'partial_fill_policy': ctx.get('partial_fill_policy', ''),
            'execution_status': fill.get('status', ''),
            'execution_side': fill.get('side', ''),
            'execution_filled_qty': fill.get('filled_qty', None),
            'execution_residual_shares': fill.get(
                'residual_shares', fill.get('residual_unfilled', None)
            ),
            'execution_gross_vwap': fill.get('avg_price', None),
            'execution_all_in_buy_vwap': fill.get('all_in_buy_vwap', None),
            'execution_net_sell_vwap': fill.get('net_sell_vwap', None),
            'execution_fee': fill.get('fee', None),
            'execution_fill_ratio': fill.get('fill_ratio', None),
            'execution_depth_levels_consumed': fill.get('depth_levels_consumed', None),
            'execution_is_partial': fill.get('is_partial', None),
            'execution_fill_detail': json.dumps(fill, ensure_ascii=False, default=str),
        })
    return events


def write_audit_log(audit_events: list):
    if not audit_events:
        return
    df_audit = pd.DataFrame(audit_events)
    if AUDIT_LOG_PATH.exists():
        try:
            existing = pd.read_parquet(AUDIT_LOG_PATH)
            df_audit = pd.concat([existing, df_audit], ignore_index=True)
        except Exception:
            pass
    df_audit.to_parquet(AUDIT_LOG_PATH, index=False)
