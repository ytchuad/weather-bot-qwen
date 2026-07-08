"""Run baseline + A/B/C/D variants and print comparison."""

from __future__ import annotations

import csv
import json
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.params import EnsembleParams
from execution.ensemble.strategy import EnsembleStrategy, parse_bucket_bounds
from execution.ensemble.strategy_candidate import CandidateEnsembleStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

DATA_DIR = Path("data/export")


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ── Snapshot loading ──────────────────────────────────────────────

def load_snaps(dates=None, exclude=None) -> dict[str, list[dict]]:
    by_date = {}
    if exclude is None:
        exclude = []
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if dates:
        csv_files = [f for f in csv_files if f.stem in dates]
    csv_files = [f for f in csv_files if f.stem not in exclude]
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
    return by_date


def dedup_1sec(snaps: list[dict]) -> list[dict]:
    uniq, seen = [], set()
    for snap in snaps:
        k = snap["timestamp"].strftime("%Y%m%d%H%M%S")
        if k not in seen:
            seen.add(k)
            uniq.append(snap)
    return uniq


def dedup_5min(snaps: list[dict]) -> list[dict]:
    uniq, seen = [], set()
    for snap in snaps:
        ts = snap["timestamp"]
        k = ts.strftime("%Y%m%d%H") + f"{(ts.minute // 5) * 5:02d}"
        if k not in seen:
            seen.add(k)
            uniq.append(snap)
    return uniq


def synthetic_snap(source: dict, ts: datetime) -> dict:
    return {
        "timestamp": ts,
        "max_so_far": source.get("max_so_far", 0.0),
        "actual_temp": source.get("actual_temp"),
        "market_prices": dict(source.get("market_prices", {})),
        "market_depth": source.get("market_depth", {}),
        "model_probs": dict(source.get("model_probs", {})),
    }


# ── Single-variant runner ─────────────────────────────────────────

Result = dict  # stats dict

def run_variant(label: str, exclude: list[str],
                strategy_cls=EnsembleStrategy,
                params: EnsembleParams | None = None,
                dedup_fn=dedup_1sec,
                use_synthetic_exit: bool = False,
                output_dir: str | Path | None = None,
                dates: list[str] | None = None) -> Result:
    """Run one variant and return stats.  Optionally write CSV reports."""
    if params is None:
        params = EnsembleParams(capital=1000.0)

    strategy = strategy_cls(params)
    by_date = load_snaps(exclude=exclude, dates=dates)
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

    for date in sorted(by_date.keys()):
        snaps = dedup_fn(by_date[date])

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

        def _process(snap, ltt, dcash, dfees, dslipp, dmax, dactual_max):
            nonlocal total_trades
            ts = snap["timestamp"]
            msf = snap["max_so_far"] or 0.0
            dmax = max(dmax, msf)
            actual = snap.get("actual_temp")
            if actual is not None:
                dactual_max = max(dactual_max, actual)

            ensemble = strategy.compute_ensemble_probs(snap["model_probs"])
            if not ensemble:
                return dcash, dfees, dslipp, dmax, dactual_max, ltt

            result = strategy.run_cycle(
                timestamp=ts,
                ensemble_probs=ensemble,
                market_prices=snap["market_prices"],
                max_so_far=msf,
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

        prev_ts = None
        for snap in snaps:
            ts = snap["timestamp"]

            # (D) synthetic exit cycles
            if use_synthetic_exit and prev_ts is not None:
                if prev_ts.hour < 14 <= ts.hour:
                    syn = synthetic_snap(snap, ts.replace(hour=14, minute=0, second=0))
                    day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                        _process(syn, last_trade_times, day_cash, day_fees, day_slippage,
                                 day_max, day_actual_max)

                if prev_ts.hour < 15 <= ts.hour:
                    syn = synthetic_snap(snap, ts.replace(hour=15, minute=0, second=0))
                    day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                        _process(syn, last_trade_times, day_cash, day_fees, day_slippage,
                                 day_max, day_actual_max)

            day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                _process(snap, last_trade_times, day_cash, day_fees, day_slippage,
                         day_max, day_actual_max)

            prev_ts = ts

        # Force end-of-day 15:00 synthetic exit if needed
        if use_synthetic_exit and snaps:
            last_real_ts = snaps[-1]["timestamp"]
            if last_real_ts.hour < 15:
                syn = synthetic_snap(snaps[-1], last_real_ts.replace(hour=15, minute=0, second=0))
                day_cash, day_fees, day_slippage, day_max, day_actual_max, last_trade_times = \
                    _process(syn, last_trade_times, day_cash, day_fees, day_slippage,
                             day_max, day_actual_max)

        # Settlement
        last_prices = snaps[-1]["market_prices"]
        settle_max = day_actual_max if day_actual_max > -float("inf") else day_max
        settlement = strategy.settle_day(
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

    # Stats
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
        "label": label,
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
        "snapshots": sum(len(dedup_fn(by_date[d])) for d in by_date),
        "daily_returns": daily_returns,
        "day_pnl": day_pnl,
        "skip_counts": skip_counts,
        "strategy_params": params,
        "deterministic_events": sum(1 for f in all_fills if f.get("action", "").startswith("BREAKOUT_")),
    }

    # Write CSV output if requested
    if output_dir:
        from execution.ensemble.reporting import BacktestReport
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report = BacktestReport()
        report.write_trade_log(all_fills, out / "trade_log.csv")
        report.write_position_snapshot(all_pos_snapshots, out / "position_snapshot.csv")
        report.write_allocation_log(all_allocations, out / "allocation_log.csv")
        report.write_equity_curve(equity_curve, out / "equity_curve.csv")
        report.write_summary_md(stats, out / "simulation_summary.md")

    return stats


# ── Comparison ────────────────────────────────────────────────────

def print_comparison(results: list[Result]):
    header = f"{'Variant':<18} {'End Cap':>10} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Fees':>8} {'Snaps':>6} {'DetEvts':>7}"
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"{'  ENSEMBLE BACKTEST COMPARISON (excl 07-04)':^{len(header)}}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for r in results:
        print(f"{r['label']:<18} ${r['capital_end']:>8.2f} {r['total_return_pct']:>10} {r['sharpe']:>8.3f} {r['max_drawdown_pct']:>8} {r['total_trades']:>7} ${r['total_fees']:>7.2f} {r['snapshots']:>6} {r['deterministic_events']:>7}")
    print(sep)
    print()

    # Daily breakdown
    print(f"{'Daily PnL':-^60}")
    dates = sorted(results[0]["day_pnl"].keys())
    h = f"{'Date':<12}"
    for r in results:
        h += f"  {r['label']:<16}"
    print(h)
    print("-" * len(h))
    for d in dates:
        line = f"{d:<12}"
        for r in results:
            v = r["day_pnl"].get(d, 0)
            line += f"  ${v:>+8.2f}     "
        print(line)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare backtest variants")
    parser.add_argument("--exclude", nargs="*", default=["2026-07-04"])
    args = parser.parse_args()

    results = []

    # 1. Baseline
    results.append(run_variant(
        "Baseline", exclude=args.exclude,
        output_dir="output/compare_baseline",
    ))

    # 2. A-only: 5-minute dedup
    results.append(run_variant(
        "A (5min dedup)", exclude=args.exclude,
        dedup_fn=dedup_5min,
        output_dir="output/compare_a",
    ))

    # 3. B-only: cooldown = 0
    results.append(run_variant(
        "B (cooldown=0)", exclude=args.exclude,
        params=EnsembleParams(capital=1000.0, min_rebalance_interval_minutes=0.0),
        output_dir="output/compare_b",
    ))

    # 4. C-only: hold positions (CandidateEnsembleStrategy)
    results.append(run_variant(
        "C (hold pos)", exclude=args.exclude,
        strategy_cls=CandidateEnsembleStrategy,
        output_dir="output/compare_c",
    ))

    # 5. D-only: synthetic exit cycles
    results.append(run_variant(
        "D (synth exit)", exclude=args.exclude,
        use_synthetic_exit=True,
        output_dir="output/compare_d",
    ))

    # 6. A+B: 5-min dedup + cooldown=0
    results.append(run_variant(
        "A+B", exclude=args.exclude,
        dedup_fn=dedup_5min,
        params=EnsembleParams(capital=1000.0, min_rebalance_interval_minutes=0.0),
        output_dir="output/compare_ab",
    ))

    print_comparison(results)
