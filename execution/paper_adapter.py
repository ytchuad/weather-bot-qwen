# execution/paper_adapter.py
"""Bridge between strategy engine and polymarket-paper-trader.

Replaces update_paper_positions() with real CLOB execution,
fee calculation, and paper-trader SQLite position tracking.

Usage:
    adapter = PaperAdapter()
    adapter.init_for_slug(slug, token_ids_dict=None)
    adapter.execute_target_positions(
        target_positions, portfolio_id, slug, strategy_key,
        prices_dict, strategy_context
    )
    pnl = adapter.get_portfolio_pnl()
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import pandas as pd
from pm_trader.engine import Engine
from pm_trader.models import InsufficientBalanceError, OrderRejectedError, MarketClosedError

from execution.portfolio_reconciler import (
    reconcile_positions, load_positions, save_positions,
    build_audit_events, write_audit_log, PM_MIN_QTY
)

logger = logging.getLogger(__name__)

PAPER_DATA_DIR = ROOT_DIR / "data" / "paper_trader"
GAMMA_API = "https://gamma-api.polymarket.com"

# Module-level singleton so multiple strategy runs share one Engine
_adapter_instance = None

def _get_adapter() -> "PaperAdapter":
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = PaperAdapter()
    return _adapter_instance


class PaperAdapter:
    """Adapter that uses polymarket-paper-trader for real CLOB execution
    while maintaining backward compatibility with current_positions.json."""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = PAPER_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        self._engine = Engine(data_dir)
        try:
            self._engine.get_account()
        except Exception:
            self._engine.init_account(balance=1_000_000.0)
        self._slug = None
        self._buckets = {}  # {bucket_title: {"condition_id": str, "slug": str}}

    # ── Market Resolution ──────────────────────────────────────

    def resolve_event_buckets(self, event_slug: str) -> dict:
        """Fetch sub-markets under an event slug from Gamma API.

        Returns {bucket_title: {"condition_id": str, "slug": str, ...}}
        """
        resp = requests.get(f"{GAMMA_API}/events?slug={event_slug}", timeout=10)
        events = resp.json()
        if not events:
            logger.warning("No event found for slug '%s'", event_slug)
            return {}

        markets = events[0].get("markets", [])
        self._buckets = {}
        for m in markets:
            title = m.get("groupItemTitle", "")
            if not title:
                continue
            cond_id = m.get("conditionId", "")
            sub_slug = m.get("slug", "")
            clob_ids_raw = m.get("clobTokenIds", "[]")
            try:
                clob_ids = json.loads(clob_ids_raw)
            except (json.JSONDecodeError, TypeError):
                clob_ids = []
            yes_token_id = clob_ids[0] if len(clob_ids) > 0 else ""
            no_token_id = clob_ids[1] if len(clob_ids) > 1 else ""
            self._buckets[title] = {
                "condition_id": cond_id,
                "slug": sub_slug,
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id,
            }
        logger.info("Resolved %d buckets for event '%s'", len(self._buckets), event_slug)
        return self._buckets

    def get_bucket_info(self, bucket_title: str) -> dict:
        return self._buckets.get(bucket_title, {})

    # ── Event Discovery ────────────────────────────────────────

    def discover_event_slug(
        self, lookahead_days: int = 3
    ) -> Optional[str]:
        """Auto-discover active Hong Kong temperature event slug.

        Tries the known slug pattern for today + next *lookahead_days*
        days via the Engine's API client (CLOB → Gamma fallback).
        Returns the first active non-closed slug, or None.
        """
        from datetime import datetime, timedelta, timezone
        hkt_offset = timedelta(hours=8)
        today = datetime.now(timezone.utc) + hkt_offset

        for offset in range(lookahead_days):
            dt = today + timedelta(days=offset)
            for prefix in ("highest", "lowest"):
                slug = (
                    f"{prefix}-temperature-in-hong-kong-on-"
                    f"{dt.strftime('%B').lower()}-{dt.day}-{dt.year}"
                )
                try:
                    market = self._engine.api.get_market(slug)
                    if market and not market.closed:
                        logger.info(
                            "Discovered active event: %s (slug=%s)",
                            market.question, slug
                        )
                        return slug
                except Exception:
                    continue
        logger.warning("No active HK temperature event found (lookahead=%d)", lookahead_days)
        return None

    # ── Market Resolution ──────────────────────────────────────

    def resolve_expired_markets(self) -> list:
        """Resolve all closed/expired markets via Engine.resolve_all().

        After resolution, realized PnL is recorded in the SQLite DB.
        Returns list of ResolveResult dicts.
        """
        results = self._engine.resolve_all()
        n = sum(1 for r in results if r.get("resolved"))
        logger.info("Resolved %d/%d expired markets", n, len(results))
        return results

    # ── Sell / Exit ────────────────────────────────────────────

    def sell_position(
        self, bucket: str, outcome: str, shares: float,
        order_type: str = "fok",
    ) -> dict:
        """Sell back shares of a position via Engine.sell().

        Parameters
        ----------
        bucket : str
            Bucket name (must exist in self._buckets).
        outcome : str
            "yes" or "no".
        shares : float
            Number of shares to sell.
        order_type : str
            "fok" (fill-or-kill) or "fak" (fill-and-kill).

        Returns
        -------
        dict with keys: bucket, outcome, sold_qty, avg_price, proceeds, fee.
        """
        info = self._buckets.get(bucket, {})
        sub_slug = info.get("slug", "")
        if not sub_slug:
            raise ValueError(f"No sub-market slug for bucket {bucket!r}")

        result = self._engine.sell(sub_slug, outcome, shares, order_type=order_type)
        t = result.trade
        return {
            "bucket": bucket,
            "outcome": outcome,
            "sold_qty": t.shares,
            "avg_price": t.avg_price,
            "proceeds": t.amount_usd,
            "fee": t.fee,
        }

    def close_all_positions(self, portfolio_id: str = None) -> list:
        """Sell ALL open positions in the paper-trader portfolio.

        Returns list of sell result dicts.
        """
        portfolio = self._engine.get_portfolio()
        sells = []
        for pos in portfolio:
            slug_filter = portfolio_id
            try:
                result = self._engine.sell(
                    pos["market_slug"], pos["outcome"], pos["shares"]
                )
                sells.append({
                    "market_slug": pos["market_slug"],
                    "outcome": pos["outcome"],
                    "sold_qty": result.trade.shares,
                    "proceeds": result.trade.amount_usd,
                    "fee": result.trade.fee,
                })
            except Exception as e:
                logger.warning("Failed to sell %s/%s: %s", pos["market_slug"], pos["outcome"], e)
        return sells

    # ── Execution ──────────────────────────────────────────────

    def execute_target_positions(
        self,
        target_positions: dict,
        portfolio_id: str,
        slug: str,
        strategy_key: str,
        prices_dict: dict,
        strategy_context: dict = None,
    ) -> list:
        """Execute target positions through paper-trader's real CLOB.

        1. Resolve sub-market slugs/condition_ids if not cached
        2. Load current positions from paper-trader SQLite
        3. For each bucket, compute delta and execute via engine.buy/sell
        4. Sync current_positions.json for backward compat
        5. Write audit log

        Returns list of fill dicts.
        """
        if slug != self._slug:
            self.resolve_event_buckets(slug)
            self._slug = slug

        fills = []
        ctx = strategy_context or {}

        for bucket, target in target_positions.items():
            target_side = target.get("side")
            target_qty = target.get("quantity", 0)
            info = self._buckets.get(bucket, {})
            cond_id = info.get("condition_id", "")
            if not cond_id:
                logger.warning("No condition_id for bucket '%s', skipping", bucket)
                continue

            market_price = prices_dict.get(bucket, 0.5)
            outcome = "yes" if target_side == "YES" else "no"

            # ── EXIT: sell existing position ──
            if target_qty == 0:
                sub_slug = info.get("slug", "")
                pos_shares = 0.0
                try:
                    for p in self._engine.get_portfolio():
                        if p["market_slug"] == sub_slug and p["outcome"] == outcome:
                            pos_shares = p["shares"]
                            break
                except Exception:
                    pass
                if pos_shares < PM_MIN_QTY:
                    logger.info("Bucket '%s': position %.2fsh < min, skipping sell", bucket, pos_shares)
                    continue
                try:
                    result = self._engine.sell(sub_slug, outcome, pos_shares, order_type="fok")
                    t = result.trade
                    fills.append({
                        "bucket": bucket,
                        "side": target_side,
                        "condition_id": cond_id,
                        "filled_qty": -t.shares,
                        "avg_price": t.avg_price,
                        "total_cost": -t.amount_usd,
                        "fee": t.fee,
                        "action": "SELL",
                    })
                    logger.info("Sold %s %s: %.2fsh @ %.4f, fee=%.4f",
                                bucket, outcome, t.shares, t.avg_price, t.fee)
                except Exception as e:
                    logger.warning("Exit failed for %s: %s", bucket, e)
                continue

            if target_qty < PM_MIN_QTY:
                continue

            amount_usd = target_qty * market_price
            if target_side == "NO":
                yes_price = market_price
                no_price = 1.0 - yes_price
                amount_usd = target_qty * no_price

            if amount_usd < 1.0:
                logger.info("Bucket '%s': amount $%.2f < $1, skipping", bucket, amount_usd)
                continue

            try:
                result = self._engine.buy(cond_id, outcome, amount_usd, order_type="fok")
                t = result.trade
                fill = {
                    "bucket": bucket,
                    "side": target_side,
                    "condition_id": cond_id,
                    "filled_qty": t.shares,
                    "avg_price": t.avg_price,
                    "total_cost": t.amount_usd,
                    "fee": t.fee,
                    "slippage_bps": t.slippage,
                    "levels_filled": t.levels_filled,
                    "is_partial": t.is_partial,
                    "action": "BUY",
                }
                fills.append(fill)
                logger.info(
                    "Bought %s %s: %.2fsh @ %.4f, fee=%.4f, slippage=%.1fbps",
                    bucket, outcome, t.shares, t.avg_price, t.fee, t.slippage
                )

            except InsufficientBalanceError:
                logger.warning("Insufficient balance for %s", bucket)
                break
            except (OrderRejectedError, MarketClosedError) as e:
                logger.warning("Order rejected for %s: %s", bucket, e)
                continue
            except Exception as e:
                logger.error("Failed to execute %s: %s", bucket, e)
                continue

        # ── Build corrected target_positions with real fill prices ──
        # NOTE: reconciler normalizes NO prices (1.0 - price), so pass raw avg_price
        corrected = dict(target_positions)
        for f in fills:
            corrected[f["bucket"]] = dict(corrected.get(f["bucket"], {}))
            corrected[f["bucket"]]["target_price"] = f["avg_price"]
            corrected[f["bucket"]]["quantity"] = f["filled_qty"]

        self._sync_legacy_positions(corrected, portfolio_id, slug, strategy_key, prices_dict, ctx, fills)

        return fills

    def _sync_legacy_positions(
        self, target_positions, portfolio_id, slug, strategy_key,
        prices_dict, ctx, fills
    ):
        """Update current_positions.json and audit log for backward compat."""
        if not target_positions:
            return

        all_pos = load_positions()
        result = reconcile_positions(
            all_pos, target_positions, portfolio_id, slug, strategy_key, ctx
        )
        save_positions(result.positions_updated)

        audit_events = build_audit_events(result, prices_dict)
        write_audit_log(audit_events)

    # ── PnL & Status ───────────────────────────────────────────

    def get_portfolio_pnl(self, portfolio_id: str = None, strategy_keys: list = None) -> dict:
        """Read positions from paper-trader SQLite, format same as
        portfolio_manager.get_portfolio_pnl() for dashboard compatibility."""
        portfolio = self._engine.get_portfolio()
        bal = self._engine.get_balance()

        total_cost = 0.0
        total_market = 0.0
        details = []

        for p in portfolio:
            cost = p["total_cost"]
            market_val = p["current_value"]
            total_cost += cost
            total_market += market_val
            details.append({
                "slug": p["market_slug"],
                "strategy": "",
                "bucket": p["outcome"],
                "side": "YES",  # paper-trader tracks position per outcome
                "quantity": p["shares"],
                "entry_price": p["avg_entry_price"],
                "current_price": p["live_price"],
                "cost_basis": cost,
                "market_value": market_val,
            })

        return {
            "portfolio_id": portfolio_id or "",
            "cost_basis": total_cost,
            "market_value": total_market,
            "unrealized_pnl": total_market - total_cost,
            "details": details,
        }

    def get_position_fees(self) -> dict:
        """Aggregate fees from trade history per (sub_slug, outcome).

        Returns {(market_slug, outcome): total_fee}
        """
        trades = self._engine.get_history(limit=500)
        fees = {}
        for t in trades:
            key = (t.market_slug, t.outcome)
            fees[key] = fees.get(key, 0.0) + t.fee
        return fees

    def get_total_fees(self) -> float:
        trades = self._engine.get_history(limit=500)
        return sum(t.fee for t in trades)

    def get_balance(self) -> dict:
        bal = self._engine.get_balance()
        return {
            "cash": bal["cash"],
            "positions_value": bal["positions_value"],
            "total_value": bal["total_value"],
            "pnl": bal["pnl"],
        }

    def get_trade_history(self, limit: int = 50) -> list:
        trades = self._engine.get_history(limit=limit)
        return [{
            "market_slug": t.market_slug,
            "market_question": t.market_question,
            "outcome": t.outcome,
            "side": t.side,
            "avg_price": t.avg_price,
            "shares": t.shares,
            "amount_usd": t.amount_usd,
            "fee": t.fee,
            "created_at": t.created_at,
        } for t in trades]

    def reset(self):
        self._engine.reset()
        self._engine.init_account(balance=1_000_000.0)
        self._slug = None
        self._buckets = {}
        logger.info("PaperAdapter reset")

    def close(self):
        self._engine.close()
