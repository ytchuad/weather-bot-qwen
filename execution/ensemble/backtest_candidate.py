from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.params import EnsembleParams
from execution.ensemble.strategy import parse_bucket_bounds
from execution.ensemble.strategy_candidate import CandidateEnsembleStrategy
from execution.ensemble.reporting import BacktestReport

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/export")
OUTPUT_DIR = Path("output/backtest_candidate")


class CandidateBacktestRunner:
    """Backtest runner with candidate settings:

    (A) 5-minute dedup (instead of per-second)
    (B) min_rebalance_interval_minutes = 0.0
    (C) hold positions when no Kelly target (handled by CandidateEnsembleStrategy)
    (D) synthetic exit cycles at 14:00 and 15:00
    """

    def __init__(self, strategy: CandidateEnsembleStrategy, output_dir: str | Path = OUTPUT_DIR):
        self.strategy = strategy
        self.output_dir = Path(output_dir)

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _synthetic_snap(source: dict, ts: datetime) -> dict:
        return {
            "timestamp": ts,
            "max_so_far": source.get("max_so_far", 0.0),
            "actual_temp": source.get("actual_temp"),
            "market_prices": dict(source.get("market_prices", {})),
            "market_depth": source.get("market_depth", {}),
            "model_probs": dict(source.get("model_probs", {})),
        }

    @staticmethod
    def _dedup_5min(snaps: list[dict]) -> list[dict]:
        uniq, seen = [], set()
        for snap in snaps:
            ts = snap["timestamp"]
            k = ts.strftime("%Y%m%d%H") + f"{(ts.minute // 5) * 5:02d}"
            if k not in seen:
                seen.add(k)
                uniq.append(snap)
        return uniq

    # ── data loading (5-min dedup) ─────────────────────────────────

    def load_snapshots(self, dates: list[str] | None = None, exclude: list[str] | None = None):
        if exclude is None:
            exclude = []
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        if dates:
            csv_files = [f for f in csv_files if f.stem in dates]
        csv_files = [f for f in csv_files if f.stem not in exclude]
        if not csv_files:
            logger.warning("No CSV files found in %s", DATA_DIR)
            return {}

        by_date = {}
        for fpath in csv_files:
            date_str = fpath.stem
            with open(fpath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_str = row.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue
                    ctx_raw = row.get("context_json", "{}")
                    try:
                        ctx = json.loads(ctx_raw)
                    except (json.JSONDecodeError, TypeError):
                        ctx = {}
                    mp = ctx.get("market_prices", {})
                    if not mp:
                        continue
                    by_date.setdefault(date_str, []).append({
                        "timestamp": ts,
                        "max_so_far": _safe_float(row.get("max_so_far")),
                        "actual_temp": _safe_float(row.get("actual_temp")),
                        "market_prices": mp,
                        "market_depth": ctx.get("market_depth", {}),
                        "model_probs": ctx.get("model_probs", {}),
                    })

        for d in by_date:
            by_date[d].sort(key=lambda x: x["timestamp"])
            by_date[d] = self._dedup_5min(by_date[d])
        return by_date

    # ── main run ───────────────────────────────────────────────────

    def run(self, dates: list[str] | None = None, exclude: list[str] | None = None):
        by_date = self.load_snapshots(dates, exclude)
        if not by_date:
            logger.error("No snapshot data loaded. Aborting.")
            return
        report = BacktestReport()
        params = self.strategy.params
        cash = params.capital
        prev_cash = params.capital
        cumulative_fees = 0.0
        cumulative_slippage = 0.0
        daily_returns = {}
        day_pnl = {}
        all_fills = []
        all_decisions = []
        all_pos_snapshots = []
        all_allocations = []
        equity_curve = []
        total_trades = 0
        skip_counts = {}

        def _process(snap, positions, ltt, dcash, dfees, dslipp, dmax, dactual_max,
                     bucket_bounds, date_str):
            """Process a single snapshot; returns updated float/ints."""
            nonlocal total_trades
            ts = snap["timestamp"]
            max_so_far = snap["max_so_far"] or 0.0
            dmax = max(dmax, max_so_far)
            actual = snap.get("actual_temp")
            if actual is not None:
                dactual_max = max(dactual_max, actual)

            ensemble = self.strategy.compute_ensemble_probs(snap["model_probs"])
            if not ensemble:
                return dcash, dfees, dslipp, dmax, dactual_max, ltt

            result = self.strategy.run_cycle(
                timestamp=ts,
                ensemble_probs=ensemble,
                market_prices=snap["market_prices"],
                max_so_far=max_so_far,
                current_positions=positions,
                current_cash=dcash,
                last_trade_times=ltt,
                clob_depth=snap.get("market_depth"),
            )

            ltt = result["last_trade_times"]

            for fill in result["fills"]:
                bucket = fill["bucket"]
                if fill["position_after"] > 0:
                    positions[bucket] = {
                        "side": fill["side"],
                        "quantity": fill["position_after"],
                        "entry_price": fill["execution_price"],
                    }
                elif bucket in positions:
                    del positions[bucket]

            dcash += result["cash_delta"]
            dfees += result["total_fees"]
            dslipp += result["total_slippage"]

            _ds = ts.strftime("%Y-%m-%d")
            _ts = ts.strftime("%H:%M:%S")
            _ts_iso = ts.isoformat()
            for fill in result["fills"]:
                fill["timestamp"] = _ts_iso
                fill["date"] = _ds
                fill["time"] = _ts
                fill["mode"] = result["time_mode"]
            for dec in result["decisions"]:
                dec["timestamp"] = _ts_iso
                dec["date"] = _ds
                dec["time"] = _ts
                dec["mode"] = result["time_mode"]

            all_fills.extend(result["fills"])
            all_decisions.extend(result["decisions"])

            for dec in result["decisions"]:
                r = dec.get("reason", "")
                if r.startswith("SKIP_") or r.startswith("REJECT_"):
                    skip_counts[r] = skip_counts.get(r, 0) + 1

            for b, p in sorted(positions.items()):
                price = snap["market_prices"].get(b, 0.5)
                if p["side"] == "NO":
                    price = 1.0 - price
                uv = p["quantity"] * (price - p["entry_price"])
                all_pos_snapshots.append({
                    "timestamp": _ts_iso, "date": _ds, "time": _ts,
                    "bucket": b, "side": p["side"], "shares": p["quantity"],
                    "market_price": round(price, 6),
                    "mark_value": round(p["quantity"] * price, 4),
                    "deterministic_status": "",
                    "unrealized_pnl": round(uv, 4),
                })

            for dec in result["decisions"]:
                tsh = dec.get("target_shares", 0)
                if tsh and tsh > 0:
                    all_allocations.append({
                        "timestamp": _ts_iso, "date": _ds, "time": _ts,
                        "bucket": dec.get("bucket"),
                        "side": dec.get("side", ""),
                        "ensemble_prob": dec.get("ensemble_prob"),
                        "market_yes_price": dec.get("market_yes_price"),
                        "market_no_price": dec.get("market_no_price"),
                        "execution_price": dec.get("execution_price"),
                        "edge": dec.get("edge"),
                        "raw_kelly_fraction": dec.get("raw_kelly_fraction"),
                        "final_kelly_fraction": dec.get("final_kelly_fraction"),
                        "target_position": dec.get("target_position", ""),
                        "target_notional": dec.get("target_notional", ""),
                        "target_shares": tsh,
                        "reason": dec.get("reason", ""),
                    })

            pos_mkt = sum(
                p["quantity"] * (
                    snap["market_prices"].get(b, 0.5)
                    if p["side"] == "YES" else 1.0 - snap["market_prices"].get(b, 0.5)
                )
                for b, p in positions.items()
            )
            total_equity = dcash + pos_mkt
            equity_curve.append({
                "timestamp": _ts_iso, "date": _ds, "time": _ts,
                "cash": round(dcash, 4),
                "position_value": round(pos_mkt, 4),
                "total_equity": round(total_equity, 4),
                "unrealized_pnl": round(pos_mkt - sum(
                    p["quantity"] * p["entry_price"] for p in positions.values()
                ), 4),
                "realized_pnl": "",
            })

            total_trades += len(result["fills"])
            return dcash, dfees, dslipp, dmax, dactual_max, ltt

        # ── per-date loop ──────────────────────────────────────
        for date in sorted(by_date.keys()):
            snaps = by_date[date]
            logger.info("Running candidate backtest for %s — %d snapshots", date, len(snaps))

            positions = {}
            last_trade_times = {}
            day_cash = cash
            day_fees = 0.0
            day_slippage = 0.0
            day_max = 0.0
            day_actual_max = -float("inf")

            buckets_set = set()
            for snap in snaps:
                buckets_set.update(snap["market_prices"].keys())
            bucket_bounds = {b: parse_bucket_bounds(b) for b in sorted(buckets_set)}

            prev_ts = None
            for snap in snaps:
                ts = snap["timestamp"]

                # (D) synthetic 14:00 cycle
                if prev_ts is not None and prev_ts.hour < 14 <= ts.hour:
                    syn = self._synthetic_snap(snap, ts.replace(hour=14, minute=0, second=0))
                    day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                        _process(syn, positions, last_trade_times,
                                 day_cash, day_fees, day_slippage,
                                 day_max, day_actual_max, bucket_bounds, date)

                # (D) synthetic 15:00 cycle
                if prev_ts is not None and prev_ts.hour < 15 <= ts.hour:
                    syn = self._synthetic_snap(snap, ts.replace(hour=15, minute=0, second=0))
                    day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                        _process(syn, positions, last_trade_times,
                                 day_cash, day_fees, day_slippage,
                                 day_max, day_actual_max, bucket_bounds, date)

                # normal cycle
                day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                    _process(snap, positions, last_trade_times,
                             day_cash, day_fees, day_slippage,
                             day_max, day_actual_max, bucket_bounds, date)

                prev_ts = ts

            # After last real snapshot, force 15:00 cycle if needed
            if snaps:
                last_real_ts = snaps[-1]["timestamp"]
                if last_real_ts.hour < 15:
                    syn = self._synthetic_snap(
                        snaps[-1], last_real_ts.replace(hour=15, minute=0, second=0))
                    day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                        _process(syn, positions, last_trade_times,
                                 day_cash, day_fees, day_slippage,
                                 day_max, day_actual_max, bucket_bounds, date)

            # ── Settlement ─────────────────────────────────────
            last_prices = snaps[-1]["market_prices"]
            settle_max = day_actual_max if day_actual_max > -float("inf") else day_max
            settlement = self.strategy.settle_day(
                day_max_temp=settle_max,
                final_positions=positions,
                bucket_bounds=bucket_bounds,
                last_prices=last_prices,
            )
            settle_pnl = settlement["total_gross_pnl"]
            day_cash += settle_pnl
            cash = day_cash
            cumulative_fees += day_fees
            cumulative_slippage += day_slippage
            day_pnl_amount = round(cash - prev_cash, 2)
            day_pnl[date] = day_pnl_amount
            daily_returns[date] = round(day_pnl_amount / prev_cash, 6) if prev_cash > 0 else 0.0
            prev_cash = cash

            eq = round(cash, 4)
            equity_curve.append({
                "timestamp": f"{date}T23:59:59", "date": date, "time": "23:59:59",
                "cash": eq, "position_value": 0.0,
                "total_equity": eq, "unrealized_pnl": 0.0,
                "realized_pnl": round(settle_pnl, 4),
            })

            logger.info("  %s \u2192 cap=%.2f  pnl=%+.2f  max=%s  winner=%s",
                        date, cash, day_pnl_amount, settlement["day_max_temp"],
                        settlement["winning_bucket"])

        # ── Stats ──────────────────────────────────────────────
        final_capital = cash
        total_return = (final_capital - params.capital) / params.capital

        if len(daily_returns) >= 2:
            ra = np.array(list(daily_returns.values()))
            sharpe = float(np.mean(ra) / np.std(ra, ddof=1) * np.sqrt(365)) if np.std(ra, ddof=1) > 0 else 0.0
        else:
            sharpe = 0.0

        peak = 0.0
        max_dd = 0.0
        for pt in equity_curve:
            eq = pt["total_equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        yes_trades = sum(1 for f in all_fills if f.get("side") == "YES" and f["shares_delta"] > 0)
        no_trades = sum(1 for f in all_fills if f.get("side") == "NO" and f["shares_delta"] > 0)

        stats = {
            "strategy_params": params,
            "dates": sorted(by_date.keys()),
            "capital_start": params.capital,
            "capital_end": round(final_capital, 2),
            "total_return": round(total_return, 6),
            "total_return_pct": f"{total_return*100:.2f}%",
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": f"{max_dd*100:.2f}%",
            "total_trades": total_trades,
            "total_fees": round(cumulative_fees, 4),
            "total_slippage": round(cumulative_slippage, 4),
            "yes_trades": yes_trades,
            "no_trades": no_trades,
            "daily_returns": daily_returns,
            "day_pnl": day_pnl,
            "skip_counts": skip_counts,
            "deterministic_events": sum(1 for f in all_fills if f.get("action", "").startswith("BREAKOUT_")),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report.write_trade_log(all_fills, self.output_dir / "trade_log.csv")
        report.write_position_snapshot(all_pos_snapshots, self.output_dir / "position_snapshot.csv")
        report.write_allocation_log(all_allocations, self.output_dir / "allocation_log.csv")
        report.write_equity_curve(equity_curve, self.output_dir / "equity_curve.csv")
        report.write_summary_md(stats, self.output_dir / "simulation_summary.md")

        logger.info("=" * 60)
        logger.info("Candidate backtest: %.2f \u2192 %.2f  (%s)",
                    params.capital, final_capital, stats["total_return_pct"])
        logger.info("Sharpe=%.3f  MaxDD=%s  Trades=%d  Fees=%.2f",
                    stats["sharpe"], stats["max_drawdown_pct"],
                    stats["total_trades"], stats["total_fees"])
        return stats


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Candidate backtest (A/B/C/D variants)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Dates to exclude, e.g. --exclude 2026-07-04")
    args = parser.parse_args()

    params = EnsembleParams(capital=1000.0, min_rebalance_interval_minutes=0.0)  # (B)
    strategy = CandidateEnsembleStrategy(params)
    runner = CandidateBacktestRunner(strategy)
    runner.run(exclude=args.exclude or None)
