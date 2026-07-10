from __future__ import annotations
import logging
from datetime import datetime, timedelta

from .params import EnsembleParams
from .strategy import EnsembleStrategy

logger = logging.getLogger(__name__)


class CandidateEnsembleStrategy(EnsembleStrategy):
    """Strategy variant for candidate backtest.

    Changes vs baseline :meth:`run_cycle`:
      1. Do NOT close positions when Kelly produces no target
         (remove ``EDGE_DISAPPEARED`` close).  Hold until:
         - opposite-side signal (FLIP in main loop)
         - risk-reduction window (mode change)
         - deterministic breakout
         - exposure cap exceeded (new logic below)
      2. After all planned trades, if total held fraction > total_exposure_cap,
         reduce the largest positions proportionally.
    """

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
        ltt = dict(last_trade_times)

        # Portfolio value
        pos_mkt_value = 0.0
        for bucket, pos in current_positions.items():
            price = market_prices.get(bucket, 0.5)
            if pos.get("side") == "NO":
                price = 1.0 - price
            pos_mkt_value += pos.get("quantity", 0) * price
        portfolio_value = max(current_cash + pos_mkt_value, 1.0)

        # Deterministic override
        det_prices = {}
        for bucket in set(ensemble_probs.keys()) | set(current_positions.keys()):
            dp = self.deterministic_yes_price(bucket, max_so_far)
            if dp is not None:
                det_prices[bucket] = dp

        eff_prices = dict(market_prices)
        for b, dp in det_prices.items():
            eff_prices[b] = dp

        # Kelly targets (RISK_SEEKING only)
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

            # Deterministic close
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

            # RISK_SEEKING
            if mode == "RISK_SEEKING":
                kt = kelly_targets.get(bucket)

                if kt is None:
                    # ── (C) HOLD position instead of closing ──
                    if has_pos:
                        target_positions[bucket] = current
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

                # Compute delta
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
                    delta = target_shares
                    action = "FLIP"
                    reason = "SIDE_CHANGE"
                    close_price = self._clob_exit_price(
                        bucket, current_side, current_qty,
                        eff_prices.get(bucket, 0.5), clob_depth, clob_depth_no)
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

                # Execute trade
                fee = self._compute_fee(abs(delta), exec_price)
                total_fees += fee
                total_slippage += abs(delta) * self.params.slippage_fixed
                cost = delta * exec_price
                if delta > 0:
                    cash_delta -= cost + fee
                else:
                    cash_delta += abs(cost) - fee

                new_qty = current_qty + delta
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

            # NON-RISK_SEEKING modes
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

        # ── (C) No EDGE_DISAPPEARED close ────────────────────
        # Positions without Kelly targets are held (handled above).
        # Instead, check total exposure and reduce if cap exceeded.

        held_fraction = pos_mkt_value / portfolio_value if portfolio_value > 0 else 0.0
        cap = self.params.total_exposure_cap

        if held_fraction > cap and mode == "RISK_SEEKING":
            # Reduce positions proportionally to bring exposure <= cap
            scale = cap / held_fraction
            for bucket, pos in list(current_positions.items()):
                qty = pos["quantity"]
                new_qty = int(qty * scale)
                if new_qty < self.params.min_shares:
                    new_qty = 0
                delta = new_qty - qty  # negative
                if delta >= 0:
                    continue
                price = self._clob_exit_price(
                    bucket, pos["side"], qty,
                    eff_prices.get(bucket, 0.5), clob_depth, clob_depth_no)
                fee = self._compute_fee(abs(delta), price)
                total_fees += fee
                total_slippage += abs(delta) * self.params.slippage_fixed
                cash_delta += abs(delta) * price - fee
                fills.append({
                    "bucket": bucket, "side": pos["side"],
                    "action": "REDUCE", "shares_delta": delta,
                    "execution_price": price,
                    "market_yes_price": market_prices.get(bucket),
                    "market_no_price": 1.0 - market_prices.get(bucket, 0.5),
                    "fee": round(fee, 4),
                    "slippage": round(self.params.slippage_fixed, 4),
                    "position_before": qty, "position_after": new_qty,
                    "reason": "EXPOSURE_CAP_REDUCTION",
                })
                decisions.append({
                    "bucket": bucket, "execution_price": price,
                    "target_shares": new_qty,
                    "reason": "EXPOSURE_CAP_REDUCTION",
                })
                if new_qty > 0:
                    target_positions[bucket] = {
                        "side": pos["side"],
                        "quantity": new_qty,
                        "entry_price": pos.get("entry_price", 0),
                    }
                else:
                    target_positions.pop(bucket, None)

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
