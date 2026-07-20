"""Paper-account mutation using one validated CLOB depth snapshot.

The legacy :class:`execution.paper_adapter.PaperAdapter` remains available for
``legacy_gamma_mock`` and shadow execution.  This adapter deliberately writes
the existing paper-trader SQLite schema directly so that the strategy quote,
fee formula and recorded fill all come from the same validated snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from pm_trader.engine import Engine
from execution.clob_execution import (
    CLOBExecutionSnapshot,
    compute_sell_execution,
    mark_to_market,
    walk_depth,
)
from execution.paper_adapter import PAPER_DATA_DIR
from execution.portfolio_reconciler import (
    PM_MIN_QTY,
    build_audit_events,
    load_positions,
    reconcile_positions,
    save_positions,
    write_audit_log,
)

logger = logging.getLogger(__name__)

_adapter_instance: ClobDepthPaperAdapter | None = None


class ClobDepthPaperAdapter:
    """Mutate paper balances/positions from validated CLOB fills only."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else PAPER_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._engine = Engine(self.data_dir)
        try:
            self._engine.get_account()
        except Exception:
            self._engine.init_account(balance=1_000_000.0)
        self.last_summary: dict[str, Any] = {}

    def close(self) -> None:
        self._engine.close()

    @staticmethod
    def _market_map(markets: list[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
        return {
            str(m.get("bucket")): m
            for m in (markets or [])
            if m.get("bucket")
        }

    @staticmethod
    def _snapshot_for(
        snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
        bucket: str,
        side: str,
    ) -> CLOBExecutionSnapshot | None:
        return (snapshots.get(bucket) or {}).get(str(side).upper())

    @staticmethod
    def _slippage_bps(snapshot: CLOBExecutionSnapshot, gross_vwap: float | None) -> float:
        midpoint = snapshot.midpoint
        if midpoint is None or midpoint <= 0 or gross_vwap is None:
            return 0.0
        return (gross_vwap - midpoint) / midpoint * 10_000.0

    def _apply_buy(
        self,
        market: Mapping[str, Any],
        side: str,
        delta: float,
        snapshot: CLOBExecutionSnapshot,
        partial_fill_policy: str,
        slug: str,
    ) -> dict[str, Any]:
        fill = walk_depth(snapshot, "BUY", delta)
        if fill.filled_shares <= 0:
            return {"status": "rejected", "reason": "no_liquidity", "fill": fill.to_dict()}
        if not fill.is_full_fill and partial_fill_policy == "fail_closed":
            return {
                "status": "rejected",
                "reason": "partial_fill_fail_closed",
                "fill": fill.to_dict(),
            }
        condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
        if not condition_id:
            return {"status": "rejected", "reason": "missing_condition_id", "fill": fill.to_dict()}
        outcome = side.lower()
        question = str(market.get("name") or market.get("bucket") or snapshot.bucket)
        existing = self._engine.db.get_position(condition_id, outcome)
        gross_cost = fill.gross_notional
        total_outflow = -fill.net_cash_flow
        account = self._engine.db.get_account()
        if account is None or total_outflow > account.cash + 1e-9:
            return {
                "status": "rejected",
                "reason": "insufficient_balance",
                "fill": fill.to_dict(),
            }
        self._engine.db.update_cash(account.cash - total_outflow)
        old_shares = existing.shares if existing else 0.0
        old_cost = existing.total_cost if existing else 0.0
        new_shares = old_shares + fill.filled_shares
        new_cost = old_cost + total_outflow
        self._engine.db.insert_trade(
            market_condition_id=condition_id,
            market_slug=str(market.get("slug") or slug),
            market_question=question,
            outcome=outcome,
            side="buy",
            order_type="fok" if fill.is_full_fill else "fak",
            avg_price=fill.gross_vwap or 0.0,
            amount_usd=gross_cost,
            shares=fill.filled_shares,
            fee_rate_bps=500,
            fee=fill.total_fee,
            slippage=self._slippage_bps(snapshot, fill.gross_vwap),
            levels_filled=fill.depth_levels_consumed,
            is_partial=not fill.is_full_fill,
        )
        self._engine.db.upsert_position(
            market_condition_id=condition_id,
            market_slug=str(market.get("slug") or slug),
            market_question=question,
            outcome=outcome,
            shares=new_shares,
            avg_entry_price=new_cost / new_shares if new_shares else 0.0,
            total_cost=new_cost,
            realized_pnl=existing.realized_pnl if existing else 0.0,
        )
        return {
            "status": "filled",
            "action": "BUY",
            "bucket": snapshot.bucket,
            "side": side,
            "filled_qty": fill.filled_shares,
            "residual_unfilled": fill.unfilled_shares,
            "avg_price": fill.gross_vwap,
            "all_in_buy_vwap": fill.all_in_buy_vwap,
            "gross_notional": fill.gross_notional,
            "cash_effect": fill.net_cash_flow,
            "fee": fill.total_fee,
            "fill_ratio": fill.fill_ratio,
            "depth_levels_consumed": fill.depth_levels_consumed,
            "is_partial": not fill.is_full_fill,
            "fill": fill.to_dict(),
        }

    def _apply_sell(
        self,
        market: Mapping[str, Any],
        side: str,
        delta: float,
        snapshot: CLOBExecutionSnapshot,
        partial_fill_policy: str,
        slug: str,
    ) -> dict[str, Any]:
        condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
        outcome = side.lower()
        existing = self._engine.db.get_position(condition_id, outcome) if condition_id else None
        if existing is None or existing.shares <= 0:
            return {"status": "rejected", "reason": "no_position"}
        requested = min(float(delta), existing.shares)
        fill = compute_sell_execution(snapshot, requested, partial_fill_policy)
        if fill.filled_shares <= 0:
            return {"status": "rejected", "reason": "no_liquidity", "fill": fill.to_dict()}
        if not condition_id:
            return {"status": "rejected", "reason": "missing_condition_id", "fill": fill.to_dict()}
        net_proceeds = fill.net_cash_flow
        account = self._engine.db.get_account()
        if account is None:
            return {"status": "rejected", "reason": "account_not_initialized", "fill": fill.to_dict()}
        self._engine.db.update_cash(account.cash + net_proceeds)
        sold_cost = existing.avg_entry_price * fill.filled_shares
        remaining_shares = max(0.0, existing.shares - fill.filled_shares)
        remaining_cost = max(0.0, existing.total_cost - sold_cost)
        realized_pnl = existing.realized_pnl + net_proceeds - sold_cost
        self._engine.db.insert_trade(
            market_condition_id=condition_id,
            market_slug=str(market.get("slug") or slug),
            market_question=str(market.get("name") or market.get("bucket") or snapshot.bucket),
            outcome=outcome,
            side="sell",
            order_type="fok" if fill.is_full_fill else "fak",
            avg_price=fill.gross_vwap or 0.0,
            amount_usd=fill.gross_notional,
            shares=fill.filled_shares,
            fee_rate_bps=500,
            fee=fill.total_fee,
            slippage=self._slippage_bps(snapshot, fill.gross_vwap),
            levels_filled=fill.depth_levels_consumed,
            is_partial=not fill.is_full_fill,
        )
        self._engine.db.upsert_position(
            market_condition_id=condition_id,
            market_slug=str(market.get("slug") or slug),
            market_question=str(market.get("name") or market.get("bucket") or snapshot.bucket),
            outcome=outcome,
            shares=remaining_shares,
            avg_entry_price=existing.avg_entry_price,
            total_cost=remaining_cost,
            realized_pnl=realized_pnl,
        )
        return {
            "status": "filled",
            "action": "SELL",
            "bucket": snapshot.bucket,
            "side": side,
            "filled_qty": fill.filled_shares,
            "residual_shares": remaining_shares,
            "avg_price": fill.gross_vwap,
            "net_sell_vwap": fill.net_sell_vwap,
            "gross_notional": fill.gross_notional,
            "cash_effect": fill.net_cash_flow,
            "fee": fill.total_fee,
            "fill_ratio": fill.fill_ratio,
            "depth_levels_consumed": fill.depth_levels_consumed,
            "is_partial": not fill.is_full_fill,
            "fill": fill.to_dict(),
        }

    def execute_target_positions(
        self,
        target_positions: Mapping[str, Mapping[str, Any]],
        portfolio_id: str,
        slug: str,
        strategy_key: str,
        prices_dict: Mapping[str, float] | None,
        strategy_context: Mapping[str, Any] | None = None,
        *,
        markets: list[Mapping[str, Any]],
        execution_snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
        partial_fill_policy: str = "fail_closed",
        persist_legacy_positions: bool = True,
    ) -> list[dict[str, Any]]:
        """Apply only fills supported by the supplied decision-time snapshots."""
        market_map = self._market_map(markets)
        ctx = dict(strategy_context or {})
        fills: list[dict[str, Any]] = []
        actual_targets: dict[str, dict[str, Any]] = {}
        for bucket, target in target_positions.items():
            market = market_map.get(bucket)
            side = str(target.get("side") or "").upper()
            if market is None or side not in {"YES", "NO"}:
                fills.append({"status": "rejected", "bucket": bucket, "reason": "market_or_side_missing"})
                continue
            snapshot = self._snapshot_for(execution_snapshots, bucket, side)
            if snapshot is None:
                fills.append({"status": "rejected", "bucket": bucket, "reason": "no_valid_executable_quote"})
                continue
            condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
            if not condition_id:
                fills.append({"status": "rejected", "bucket": bucket, "reason": "missing_condition_id"})
                continue
            outcome = side.lower()
            existing = self._engine.db.get_position(condition_id, outcome)
            current_qty = existing.shares if existing else 0.0
            target_qty = max(0.0, float(target.get("quantity", 0.0)))
            delta = target_qty - current_qty
            min_size = max(PM_MIN_QTY, snapshot.minimum_order_size)
            if abs(delta) < min_size:
                actual_targets[bucket] = {
                    "side": side,
                    "quantity": current_qty,
                    "target_price": target.get("target_price", 0.0),
                }
                continue
            if delta > 0:
                result = self._apply_buy(
                    market, side, delta, snapshot, partial_fill_policy, slug
                )
            else:
                result = self._apply_sell(
                    market, side, -delta, snapshot, partial_fill_policy, slug
                )
            result["bucket"] = bucket
            fills.append(result)
            updated = self._engine.db.get_position(condition_id, outcome)
            actual_targets[bucket] = {
                "side": side,
                "quantity": updated.shares if updated else 0.0,
                "target_price": target.get("target_price", 0.0),
            }

        self.last_summary = {
            "execution_mode": "clob_depth",
            "partial_fill_policy": partial_fill_policy,
            "fills": fills,
            "residual_positions": {
                bucket: target["quantity"]
                for bucket, target in actual_targets.items()
                if target.get("quantity", 0.0) > 0
            },
        }
        if persist_legacy_positions and actual_targets:
            ctx["execution_fill_by_bucket"] = {
                str(fill.get("bucket")): {
                    key: fill.get(key)
                    for key in (
                        "status", "action", "side", "filled_qty", "residual_shares",
                        "avg_price", "all_in_buy_vwap", "net_sell_vwap",
                        "gross_notional", "cash_effect", "fee", "fill_ratio",
                        "depth_levels_consumed", "is_partial", "residual_unfilled",
                    )
                    if key in fill
                }
                for fill in fills
                if fill.get("bucket")
            }
            ctx["execution_mode"] = "clob_depth"
            ctx["partial_fill_policy"] = partial_fill_policy
            all_positions = load_positions()
            existing_targets = (
                all_positions.get(portfolio_id, {})
                .get(slug, {})
                .get(strategy_key, {})
            )
            merged_targets = {
                bucket: dict(position)
                for bucket, position in existing_targets.items()
            }
            merged_targets.update(actual_targets)
            reconciliation = reconcile_positions(
                all_positions,
                merged_targets,
                portfolio_id,
                slug,
                strategy_key,
                strategy_context=ctx,
            )
            save_positions(reconciliation.positions_updated)
            write_audit_log(build_audit_events(reconciliation, dict(prices_dict or {})))
        return fills

    def mark_positions(
        self,
        markets: list[Mapping[str, Any]],
        execution_snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
    ) -> dict[str, dict[str, Any]]:
        """Return midpoint and immediate-liquidation marks without Gamma."""
        result: dict[str, dict[str, Any]] = {}
        for market in markets:
            bucket = str(market.get("bucket") or "")
            condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
            for side in ("YES", "NO"):
                position = self._engine.db.get_position(condition_id, side.lower()) if condition_id else None
                if position is None or position.shares <= 0:
                    continue
                snapshot = self._snapshot_for(execution_snapshots, bucket, side)
                if snapshot is not None:
                    result[f"{bucket}:{side}"] = mark_to_market(snapshot, position.shares)
        return result


def get_clob_depth_adapter() -> ClobDepthPaperAdapter:
    """Return the process-local CLOB-depth paper adapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ClobDepthPaperAdapter()
    return _adapter_instance
