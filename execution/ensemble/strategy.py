import re
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .params import EnsembleParams

_FORCE_MID_ONLY = False  # set True to disable CLOB execution; FIXME: make this configurable

try:
    from app.services.market_depth_service import walk_book
    _HAVE_WALK_BOOK = True
except ImportError:
    walk_book = None
    _HAVE_WALK_BOOK = False

logger = logging.getLogger(__name__)


def parse_bucket_bounds(bucket: str):
    """Parse bucket label to (lower, upper).

    ``25-26`` → (25.0, 26.0)
    ``>=34``  → (34.0, inf)
    ``<24``   → (-inf, 24.0)
    """
    b = bucket.strip()
    if b.startswith(">="):
        return float(b[2:]), float("inf")
    if b.startswith(">"):
        return float(b[1:]), float("inf")
    if b.startswith("<") or b.startswith("\u2264"):
        return float("-inf"), float(b.lstrip("<\u2264"))
    m = re.match(r"([\d.]+)-([\d.]+)", b)
    if m:
        return float(m.group(1)), float(m.group(2))
    return float("-inf"), float("inf")


class EnsembleStrategy:
    def __init__(self, params: EnsembleParams | None = None):
        self.params = params or EnsembleParams()

    # ── Layer 1: helpers ─────────────────────────────────────────

    def compute_ensemble_probs(self, model_probs: dict) -> dict:
        buckets = set()
        for mk in self.params.model_weights:
            if mk in model_probs:
                buckets.update(model_probs[mk].keys())

        ensemble = {}
        for b in buckets:
            total_w = 0.0
            weighted = 0.0
            for mk, w in self.params.model_weights.items():
                prob = model_probs.get(mk, {}).get(b)
                if prob is not None:
                    weighted += prob * w
                    total_w += w
            if total_w > 0:
                ensemble[b] = weighted / total_w

        total = sum(ensemble.values())
        if total > 0:
            ensemble = {k: v / total for k, v in ensemble.items()}
        return ensemble

    def get_time_mode(self, dt: datetime) -> str:
        t = dt.hour + dt.minute / 60.0
        ms = self.params.morning_start
        rr = self.params.risk_reduction_start
        hf = self.params.hard_flat_start
        if t < ms:
            return "NO_TRADE"
        if t < rr:
            return "RISK_SEEKING"
        if t < hf:
            return "RISK_REDUCTION"
        return "HARD_FLAT_TARGET"

    @staticmethod
    def deterministic_yes_price(bucket: str, max_so_far: float) -> Optional[float]:
        lo, hi = parse_bucket_bounds(bucket)
        if hi != float("inf"):
            if max_so_far >= hi:
                return 0.0
        elif lo != float("-inf"):
            if max_so_far >= lo:
                return 1.0
        return None

    def _price_in_band(self, price: float) -> bool:
        return self.params.min_price <= price <= self.params.max_price

    def _compute_fee(self, shares: float, price: float) -> float:
        return shares * self.params.fee_constant * price * (1.0 - price)

    def _clob_exit_price(self, bucket: str, side: str, qty: float,
                         eff_price: float, clob_depth_yes: dict | None,
                         clob_depth_no: dict | None) -> float | None:
        """CLOB price to close a YES or NO position. Returns None if no bids.

        Walks the bid side selling exactly *qty* shares (trading-sim
        compatible).  If the book has insufficient depth the walk stops
        early and the avg price reflects only the shares actually sold.
        """
        if side == "YES":
            depth = (clob_depth_yes or {}).get(bucket, {})
        else:
            depth = (clob_depth_no or {}).get(bucket, {})

        bids = depth.get("top_bids", [])
        if not bids:
            return None

        # Walk bids, selling exactly qty shares
        remaining = qty
        total_cost = 0.0
        sold = 0.0
        for b in sorted(bids, key=lambda x: x["price"], reverse=True):
            take = min(b["size"], remaining)
            total_cost += take * b["price"]
            sold += take
            remaining -= take
            if remaining <= 0:
                break

        if sold <= 0:
            return None
        return total_cost / sold

    def _clob_price(self, side: str, order_value: float,
                    depth: dict) -> float | None:
        """Walk CLOB book, return avg fill price or None."""
        walked = self._clob_walk(side, order_value, depth)
        return walked["avg_price"] if walked else None

    def _clob_walk(self, side: str, order_value: float,
                   depth: dict) -> dict | None:
        """Walk CLOB book, return full result dict or None."""
        if not _HAVE_WALK_BOOK or not depth or _FORCE_MID_ONLY:
            return None
        asks = sorted(depth.get("top_asks", []), key=lambda x: x["price"])
        bids = sorted(depth.get("top_bids", []), key=lambda x: x["price"], reverse=True)

        if side == "BUY":
            if not asks:
                return None
            result = walk_book("BUY", order_value, asks, bids)
        else:
            if not bids:
                return None
            result = walk_book("SELL", order_value, asks, bids)

        if result["shares_filled"] <= 0 or result["avg_price"] <= 0:
            return None
        return result

    # ── Layer 2: Kelly target optimisation ─────────────────────

    def compute_targets(self, ensemble_probs: dict,
                        market_prices: dict,
                        capital: float,
                        clob_depth_yes: dict | None = None,
                        clob_depth_no: dict | None = None) -> dict:
        """Multi-outcome Kelly with the plan's constraints.

        When ``clob_depth_yes`` / ``clob_depth_no`` are provided, execution
        prices are derived from ``walk_book`` on the per-token CLOB order book
        (multi-level).  Falls back to ``market_price + slippage_fixed``
        when CLOB is unavailable.

        Returns  {bucket: {action, fraction, amount, execution_price, edge, raw_fraction}}
        or empty dict if no candidate meets the edge / price band thresholds.
        """
        candidates = []
        buckets = sorted(ensemble_probs.keys())
        est_amount = self.params.max_per_bucket_side * capital  # worst-case allocation

        for b in buckets:
            p = ensemble_probs[b]
            mkt = market_prices.get(b, 0.5)

            # CLOB price for BUY YES (walk YES ask side, capped by depth)
            depth_yes = (clob_depth_yes or {}).get(b, {})
            total_ask_yes = sum(l["price"] * l["size"] for l in depth_yes.get("top_asks", []))
            capped_yes = min(est_amount, total_ask_yes) if total_ask_yes > 0 else est_amount
            yes_clob = self._clob_price("BUY", capped_yes, depth_yes)
            if yes_clob is None:
                continue
            yes_exec = yes_clob
            yes_edge = p - yes_exec
            if yes_edge > self.params.edge_threshold and self._price_in_band(yes_exec):
                candidates.append(("YES", b, p, yes_exec, yes_edge))

            # CLOB price for BUY NO (walk NO ask side directly, capped by depth)
            depth_no = (clob_depth_no or {}).get(b, {})
            total_ask_no = sum(l["price"] * l["size"] for l in depth_no.get("top_asks", []))
            capped_no = min(est_amount, total_ask_no) if total_ask_no > 0 else est_amount
            no_clob = self._clob_price("BUY", capped_no, depth_no)
            if no_clob is None:
                continue
            no_exec = no_clob
            no_edge = (1.0 - p) - no_exec
            if no_edge > self.params.edge_threshold and self._price_in_band(no_exec):
                candidates.append(("NO", b, 1.0 - p, no_exec, no_edge))

        n = len(candidates)
        if n == 0:
            return {}

        outcomes = buckets
        outcome_probs = np.array([ensemble_probs.get(x, 0.0) for x in outcomes], dtype=float)

        def neg_log_wealth(f):
            total_f = np.sum(f)
            expected = 0.0
            for i, x in enumerate(outcomes):
                p_x = outcome_probs[i]
                if p_x <= 1e-9:
                    continue
                payout = 0.0
                for k, c in enumerate(candidates):
                    if (c[0] == "YES" and c[1] == x) or (c[0] == "NO" and c[1] != x):
                        payout += f[k] / c[3]
                w = 1.0 - total_f + payout
                w = max(w, 1e-9)
                expected += p_x * np.log(w)
            return -expected

        bounds = [(0.0, self.params.max_per_bucket_side) for _ in range(n)]
        cons = [{"type": "ineq", "fun": lambda f: self.params.total_exposure_cap - np.sum(f)}]
        f0 = np.full(n, 0.01)

        try:
            res = minimize(neg_log_wealth, f0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-9})
            opt_f = res.x
        except Exception as exc:
            logger.warning("Kelly optimisation failed: %s", exc)
            opt_f = np.zeros(n)

        targets = {}
        for k, c in enumerate(candidates):
            f_raw = float(opt_f[k])
            if f_raw <= 1e-4:
                continue
            f_final = f_raw * self.params.kelly_fraction
            amount = f_final * capital
            shares = amount / c[3]
            if shares < self.params.min_shares:
                continue

            # Re-walk book with actual Kelly amount for precise price + depth check
            bucket = c[1]
            if self.params.clob_depth_check and c[3] != 0:
                depth = ((clob_depth_yes or {}).get(bucket, {})
                         if c[0] == "YES" else (clob_depth_no or {}).get(bucket, {}))
                walked = self._clob_walk("BUY", amount, depth)
                if walked:
                    refined_price = walked["avg_price"]
                    # Re-check edge with refined price
                    p_ref = c[2] if c[0] == "YES" else (1.0 - c[2])
                    refined_edge = (c[2] - refined_price) if c[0] == "YES" else ((1.0 - c[2]) - refined_price)
                    if refined_edge >= self.params.edge_threshold:
                        c = (c[0], c[1], p_ref, refined_price, refined_edge)
                    # If book can't fully fill, skip
                    if walked["fill_frac"] < 0.99:
                        continue
                    shares = amount / refined_price
                    if shares < self.params.min_shares:
                        continue

            targets[bucket] = {
                "action": "BUY_YES" if c[0] == "YES" else "BUY_NO",
                "fraction": round(f_final, 6),
                "amount": round(amount, 2),
                "execution_price": c[3],
                "edge": round(c[4], 4),
                "raw_fraction": round(f_raw, 6),
            }
        return targets

    # ── Layer 3: full cycle ────────────────────────────────────

    def run_cycle(self, *,
                  timestamp: datetime,
                  ensemble_probs: dict,
                  market_prices: dict,
                  max_so_far: float,
                  current_positions: dict,
                  current_cash: float = 0.0,
                  last_trade_times: dict | None = None,
                  clob_depth: dict | None = None,
                  clob_depth_no: dict | None = None,
                  ) -> dict:
        """Execute one strategy cycle.

        Parameters
        ----------
        last_trade_times : dict, optional
            ``{bucket: datetime}``  —  when each bucket was last rebalanced.
            Used to enforce ``min_rebalance_interval_minutes``.
            Updated copy is returned in the result dict under the same key.

        Returns a dict with keys:
          time_mode, deterministic_events, target_positions, decisions, fills,
          total_fees, total_slippage, cash_delta, last_trade_times.
        """
        if last_trade_times is None:
            last_trade_times = {}

        mode = self.get_time_mode(timestamp)
        cooldown = timedelta(minutes=self.params.min_rebalance_interval_minutes)
        det_events = []
        decisions = []
        fills = []
        total_fees = 0.0
        total_slippage = 0.0
        cash_delta = 0.0

        # Copy trade times so caller can see what we changed
        ltt = dict(last_trade_times)

        # Determine portfolio value for Kelly sizing
        pos_mkt_value = 0.0
        for bucket, pos in current_positions.items():
            price = market_prices.get(bucket, 0.5)
            if pos.get("side") == "NO":
                price = 1.0 - price
            pos_mkt_value += pos.get("quantity", 0) * price
        portfolio_value = max(current_cash + pos_mkt_value, 1.0)

        # ── Deterministic override ────────────────────────────
        det_prices = {}
        for bucket in set(ensemble_probs.keys()) | set(current_positions.keys()):
            dp = self.deterministic_yes_price(bucket, max_so_far)
            if dp is not None:
                det_prices[bucket] = dp

        eff_prices = dict(market_prices)
        for b, dp in det_prices.items():
            eff_prices[b] = dp

        # ── Kelly targets (RISK_SEEKING only) ─────────────────
        if mode == "RISK_SEEKING":
            kelly_targets = self.compute_targets(
                ensemble_probs, eff_prices, portfolio_value,
                clob_depth_yes=clob_depth, clob_depth_no=clob_depth_no,
            )
        else:
            kelly_targets = {}

        all_buckets = set(ensemble_probs.keys()) | set(current_positions.keys())
        target_positions = {}

        for bucket in sorted(all_buckets):
            current = current_positions.get(bucket)
            has_pos = current is not None and current.get("quantity", 0) >= self.params.min_shares

            # ── Deterministic close ─────────────────────────
            if bucket in det_prices:
                dp = det_prices[bucket]
                if has_pos:
                    price = dp if current["side"] == "YES" else (1.0 - dp)
                    qty = current["quantity"]
                    fee = self._compute_fee(qty, price)
                    total_fees += fee
                    cash_delta += qty * price - fee
                    tag = "BREAKOUT_ZERO" if dp == 0.0 else "BREAKOUT_ONE"
                    fills.append({
                        "bucket": bucket, "side": current["side"],
                        "action": tag,
                        "shares_delta": -qty, "execution_price": price,
                        "market_yes_price": market_prices.get(bucket),
                        "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                        "fee": round(fee, 4), "slippage": 0.0,
                        "position_before": qty, "position_after": 0,
                        "reason": f"deterministic_{'zero' if dp==0.0 else 'one'}",
                    })
                    decisions.append({
                        "bucket": bucket, "ensemble_prob": ensemble_probs.get(bucket),
                        "execution_price": price, "edge": 0,
                        "target_shares": 0, "reason": tag,
                    })
                target_positions[bucket] = {"side": "NONE", "quantity": 0}
                continue

            # ── RISK_SEEKING ────────────────────────────────
            if mode == "RISK_SEEKING":
                kt = kelly_targets.get(bucket)

                if kt is None:
                    if has_pos:
                        # no edge → close (handled in post-loop)
                        pass
                    continue

                # Cooldown check
                last_trade = ltt.get(bucket)
                if last_trade is not None and (timestamp - last_trade) < cooldown:
                    decisions.append({
                        "bucket": bucket,
                        "reason": "SKIP_COOLDOWN",
                        "detail": f"last trade {(timestamp-last_trade).total_seconds()/60:.1f}m ago",
                    })
                    if has_pos:
                        target_positions[bucket] = current
                    continue

                target_side = kt["action"].replace("BUY_", "")
                exec_price = kt["execution_price"]
                target_shares = int(kt["amount"] / exec_price)
                current_side = current.get("side") if has_pos else None
                current_qty = current.get("quantity", 0.0) if has_pos else 0.0
                current_entry = current.get("entry_price", 0.0) if has_pos else 0.0

                if target_shares < self.params.min_shares:
                    decisions.append({
                        "bucket": bucket, "ensemble_prob": ensemble_probs.get(bucket),
                        "execution_price": exec_price, "edge": kt["edge"],
                        "raw_kelly_fraction": kt["raw_fraction"],
                        "final_kelly_fraction": kt["fraction"],
                        "target_shares": target_shares,
                        "reason": "SKIP_MIN_SHARES",
                    })
                    if has_pos:
                        target_positions[bucket] = current
                    continue

                if not self._price_in_band(exec_price):
                    decisions.append({
                        "bucket": bucket, "ensemble_prob": ensemble_probs.get(bucket),
                        "execution_price": exec_price, "edge": kt["edge"],
                        "target_shares": target_shares,
                        "reason": "REJECT_PRICE_OUT_OF_RANGE",
                    })
                    if has_pos:
                        target_positions[bucket] = current
                    continue

                # ── Compute delta ──────────────────────────
                if not has_pos:
                    delta = target_shares
                    action = "OPEN"
                    reason = "ENTRY_SIGNAL"
                elif current_side == target_side:
                    delta = target_shares - current_qty
                    if -self.params.min_shares < delta < self.params.min_shares:
                        decisions.append({
                            "bucket": bucket, "ensemble_prob": ensemble_probs.get(bucket),
                            "execution_price": exec_price, "edge": kt["edge"],
                            "raw_kelly_fraction": kt["raw_fraction"],
                            "final_kelly_fraction": kt["fraction"],
                            "target_shares": target_shares,
                            "reason": "SKIP_MIN_SHARES",
                        })
                        target_positions[bucket] = current
                        continue
                    action = "INCREASE" if delta > 0 else "REDUCE"
                    reason = "REBALANCE"
                else:
                    close_price = self._clob_exit_price(
                        bucket, current_side, current_qty,
                        eff_prices.get(bucket, 0.5), clob_depth, clob_depth_no)
                    if close_price is None:
                        target_positions[bucket] = current
                        decisions.append({
                            "bucket": bucket,
                            "reason": "SKIP_NO_PRICE",
                            "detail": "flip_no_bids",
                        })
                        continue
                    delta = target_shares
                    action = "FLIP"
                    reason = "SIDE_CHANGE"
                    cf = self._compute_fee(current_qty, close_price)
                    total_fees += cf
                    cash_delta += current_qty * close_price - cf
                    fills.append({
                        "bucket": bucket, "side": current_side,
                        "action": "CLOSE", "shares_delta": -current_qty,
                        "execution_price": close_price,
                        "market_yes_price": market_prices.get(bucket),
                        "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                        "fee": round(cf, 4), "slippage": self.params.slippage_fixed,
                        "position_before": current_qty, "position_after": 0,
                        "reason": "SIDE_CHANGE",
                    })
                    current_qty = 0.0

                # ── Execute trade ───────────────────────────
                fee = self._compute_fee(abs(delta), exec_price)
                total_fees += fee
                total_slippage += abs(delta) * self.params.slippage_fixed
                cost = delta * exec_price
                if delta > 0:
                    cash_delta -= cost + fee
                else:
                    cash_delta += abs(cost) - fee

                new_qty = current_qty + delta
                # Weighted average entry price
                new_entry = (
                    (current_qty * current_entry + delta * exec_price) / new_qty
                    if new_qty > 0 and current_qty > 0 else exec_price
                )

                fills.append({
                    "bucket": bucket, "side": target_side,
                    "action": action,
                    "shares_delta": delta,
                    "execution_price": round(exec_price, 6),
                    "market_yes_price": market_prices.get(bucket),
                    "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                    "fee": round(fee, 4),
                    "slippage": round(self.params.slippage_fixed, 4),
                    "position_before": current_qty,
                    "position_after": new_qty,
                    "reason": reason,
                })
                decisions.append({
                    "bucket": bucket, "ensemble_prob": ensemble_probs.get(bucket),
                    "market_yes_price": market_prices.get(bucket),
                    "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                    "execution_price": round(exec_price, 6),
                    "edge": kt["edge"],
                    "raw_kelly_fraction": kt["raw_fraction"],
                    "final_kelly_fraction": kt["fraction"],
                    "target_shares": new_qty,
                    "reason": reason,
                })

                if new_qty > 0:
                    target_positions[bucket] = {
                        "side": target_side, "quantity": new_qty,
                        "entry_price": round(new_entry, 6),
                    }
                    ltt[bucket] = timestamp

            # ── NON-RISK_SEEKING modes ─────────────────────
            if self.params.exit_behavior == "settlement_only":
                if mode in ("RISK_REDUCTION", "HARD_FLAT_TARGET", "NO_TRADE"):
                    if has_pos:
                        target_positions[bucket] = current
            elif mode in ("RISK_REDUCTION", "HARD_FLAT_TARGET", "NO_TRADE"):
                if has_pos:
                    price = self._clob_exit_price(
                        bucket, current["side"], current["quantity"],
                        eff_prices.get(bucket, 0.5), clob_depth, clob_depth_no)
                    if price is None or price <= 0:
                        decisions.append({"bucket": bucket, "reason": "SKIP_NO_PRICE"})
                        target_positions[bucket] = current
                        continue
                    qty = current["quantity"]
                    fee = self._compute_fee(qty, price)
                    total_fees += fee
                    total_slippage += qty * self.params.slippage_fixed
                    cash_delta += qty * price - fee
                    fills.append({
                        "bucket": bucket, "side": current["side"],
                        "action": "CLOSE", "shares_delta": -qty,
                        "execution_price": price,
                        "market_yes_price": market_prices.get(bucket),
                        "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                        "fee": round(fee, 4),
                        "slippage": round(self.params.slippage_fixed, 4),
                        "position_before": qty, "position_after": 0,
                        "reason": f"{mode}_EXIT",
                    })
                    decisions.append({
                        "bucket": bucket, "execution_price": price,
                        "reason": f"{mode}_EXIT",
                    })
                    ltt[bucket] = timestamp

        # ── RISK_SEEKING: close positions with no Kelly target ──
        if mode == "RISK_SEEKING" and self.params.hold_behavior != "never_close":
            for bucket, pos in list(current_positions.items()):
                if bucket not in target_positions and pos.get("quantity", 0) >= self.params.min_shares:
                    price = self._clob_exit_price(
                        bucket, pos["side"], pos["quantity"],
                        eff_prices.get(bucket, 0.5), clob_depth, clob_depth_no)
                    if price is None or price <= 0:
                        target_positions[bucket] = pos
                        decisions.append({
                            "bucket": bucket,
                            "reason": "SKIP_NO_PRICE",
                            "detail": "edge_disappeared_no_bids",
                        })
                        continue
                    qty = pos["quantity"]
                    fee = self._compute_fee(qty, price)
                    total_fees += fee
                    total_slippage += qty * self.params.slippage_fixed
                    cash_delta += qty * price - fee
                    fills.append({
                        "bucket": bucket, "side": pos["side"],
                        "action": "CLOSE", "shares_delta": -qty,
                        "execution_price": price,
                        "market_yes_price": market_prices.get(bucket),
                        "fee": round(fee, 4),
                        "slippage": round(self.params.slippage_fixed, 4),
                        "position_before": qty, "position_after": 0,
                        "reason": "EDGE_DISAPPEARED",
                    })
                    decisions.append({
                        "bucket": bucket, "execution_price": price,
                        "reason": "EDGE_DISAPPEARED",
                    })
                    ltt[bucket] = timestamp

        return {
            "time_mode": mode,
            "deterministic_events": det_events,
            "target_positions": target_positions,
            "decisions": decisions,
            "fills": fills,
            "total_fees": round(total_fees, 4),
            "total_slippage": round(total_slippage, 4),
            "cash_delta": round(cash_delta, 4),
            "last_trade_times": ltt,
        }

    # ── Layer 4: settlement ────────────────────────────────────

    def settle_day(self, *,
                   day_max_temp: float,
                   final_positions: dict,
                   bucket_bounds: dict,
                   last_prices: dict) -> dict:
        """Compute final settlement PnL.

        The winning bucket is the one whose bounds contain day_max_temp.
        """
        winner = None
        for bucket, (lo, hi) in bucket_bounds.items():
            if lo <= day_max_temp < hi:
                winner = bucket
                break
        if winner is None and bucket_bounds:
            # pick the bucket with matching bounds from the actual market data
            for bucket, (lo, hi) in bucket_bounds.items():
                if lo <= day_max_temp < hi:
                    winner = bucket
                    break

        total_pnl = 0.0
        details = []
        for bucket, pos in final_positions.items():
            side = pos["side"]
            qty = pos["quantity"]
            entry = pos.get("entry_price", 0)
            settle = 1.0 if (bucket == winner and side == "YES") or (bucket != winner and side == "NO") else 0.0
            gross_pnl = qty * (settle - entry)
            total_pnl += gross_pnl
            details.append({
                "bucket": bucket, "side": side, "quantity": qty,
                "entry_price": entry, "settle_price": settle,
                "gross_pnl": round(gross_pnl, 4),
            })

        return {
            "day_max_temp": day_max_temp,
            "winning_bucket": winner,
            "total_gross_pnl": round(total_pnl, 4),
            "positions": details,
        }
