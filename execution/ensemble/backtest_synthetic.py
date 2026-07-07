"""Backtest comparison using synthetic market_prices derived from pm_weighted_temp.

For snapshots where real market_prices are missing (fallback rows),
synthetic prices are generated from a Normal(pm_weighted_temp, std) distribution.
This allows 07-06 fallback data to be included in the backtest.
"""

from __future__ import annotations

import csv
import json
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.params import EnsembleParams
from execution.ensemble.strategy import EnsembleStrategy, parse_bucket_bounds
from execution.ensemble.strategy_candidate import CandidateEnsembleStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

DATA_DIR = Path("data/export")

BUCKETS = ["<24", "25-26", "26-27", "27-28", "28-29", "29-30",
           "30-31", "31-32", "32-33", "33-34", ">=34"]
BUCKET_BOUNDS = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

DEFAULT_STD = 1.27


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _normal_probs(mean: float, std: float) -> dict[str, float]:
    """Bucket probabilities from N(mean, std)."""
    dist = NormalDist(mean, std)
    probs = {}
    prev = -float("inf")
    for i, upper in enumerate(BUCKET_BOUNDS):
        if i == 0:
            p = dist.cdf(upper)
        else:
            p = dist.cdf(upper) - dist.cdf(prev)
        probs[BUCKETS[i]] = max(p, 1e-10)
        prev = upper
    probs[">=34"] = max(1.0 - dist.cdf(34), 1e-10)
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


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

                ap_raw = row.get("all_model_predictions", "{}")
                try:
                    all_preds = json.loads(ap_raw)
                except (json.JSONDecodeError, TypeError):
                    all_preds = {}

                mp = ctx.get("market_prices", {})
                model_probs_from_ctx = ctx.get("model_probs", {})
                pm_temp = _safe_float(row.get("pm_weighted_temp"))
                max_so_far = _safe_float(row.get("max_so_far"))

                # If real market_prices exist, use them as-is
                if mp:
                    by_date.setdefault(date_str, []).append({
                        "timestamp": ts,
                        "max_so_far": max_so_far,
                        "actual_temp": _safe_float(row.get("actual_temp")),
                        "market_prices": mp,
                        "market_depth": ctx.get("market_depth", {}),
                        "model_probs": model_probs_from_ctx,
                    })
                    continue

                # Synthesize from pm_weighted_temp
                if pm_temp is None or pm_temp <= 0:
                    continue
                std = DEFAULT_STD
                synth_prices = _normal_probs(pm_temp, std)

                # Build model_probs from all_model_predictions
                mod_probs = {}
                for mk in ("model_a", "model_b", "model_c"):
                    mean = all_preds.get(mk)
                    if mean is not None:
                        mod_probs[mk] = _normal_probs(mean, std)
                if not mod_probs:
                    continue

                by_date.setdefault(date_str, []).append({
                    "timestamp": ts,
                    "max_so_far": max_so_far,
                    "actual_temp": _safe_float(row.get("actual_temp")),
                    "market_prices": synth_prices,
                    "market_depth": {},
                    "model_probs": mod_probs,
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


def run_variant(label: str, exclude=None, *,
                strategy_cls=EnsembleStrategy,
                params=None,
                dedup_fn=None,
                use_synthetic_exit=False,
                output_dir=None) -> dict:
    from execution.ensemble.backtest import BacktestRunner
    from execution.ensemble.reporting import BacktestReport

    snaps_by_date = load_snaps(exclude=exclude)
    if not snaps_by_date:
        return {"label": label, "error": "no data"}

    strategy = strategy_cls(params or EnsembleParams(capital=1000.0))
    runner = BacktestRunner(strategy, output_dir=output_dir or f"output/synth_{label.lower().replace(' ','_')}")

    initial_capital = strategy.params.capital
    cash = initial_capital
    positions = {}
    last_trade_times = {}
    total_trades = 0
    total_fees = 0.0
    total_det_events = 0
    total_snaps = 0
    daily_pnl: dict[str, float] = {}

    for date_str in sorted(snaps_by_date.keys()):
        snaps = snaps_by_date[date_str]
        if dedup_fn:
            snaps = dedup_fn(snaps)
        total_snaps += len(snaps)

        # Track max_so_far within the day
        day_max = 0.0
        for snap in snaps:
            msf = snap.get("max_so_far") or snap.get("actual_temp") or 0.0
            if msf > day_max:
                day_max = msf

            ts = snap["timestamp"]
            mp = snap["market_prices"]
            md = snap.get("market_depth", {})
            mod_probs = snap.get("model_probs", {})

            # Compute ensemble probs
            try:
                ensemble_probs = strategy.compute_ensemble_probs(mod_probs)
            except Exception:
                continue
            if not ensemble_probs:
                continue

            result = strategy.run_cycle(
                timestamp=ts,
                ensemble_probs=ensemble_probs,
                market_prices=mp,
                max_so_far=day_max,
                current_positions=positions,
                current_cash=cash,
                last_trade_times=last_trade_times,
                clob_depth=md if md else None,
            )

            # Apply fills
            for fill in result.get("fills", []):
                total_trades += 1
                bucket = fill["bucket"]
                qty = fill.get("quantity", 0)
                px = fill.get("price", 0)
                side = fill.get("side", "YES")
                if side == "YES":
                    cost = qty * px
                    if bucket not in positions:
                        positions[bucket] = {"side": "YES", "quantity": 0}
                    positions[bucket]["quantity"] += qty
                    cash -= cost
                elif side == "NO":
                    cost = qty * (1.0 - px)
                    if bucket not in positions:
                        positions[bucket] = {"side": "NO", "quantity": 0}
                    positions[bucket]["quantity"] += qty
                    cash -= cost
                elif side == "SELL":
                    revenue = qty * px
                    if bucket in positions:
                        q = positions[bucket].get("quantity", 0)
                        positions[bucket]["quantity"] = max(q - qty, 0)
                        if positions[bucket]["quantity"] <= 0:
                            positions[bucket] = {"side": "NONE", "quantity": 0}
                    cash += revenue

            total_fees += result.get("total_fees", 0.0)
            total_det_events += len(result.get("deterministic_events", []))
            last_trade_times = result.get("last_trade_times", last_trade_times)

            # Synthetic exit at 14:00 / 15:00
            if use_synthetic_exit:
                from execution.ensemble.backtest_compare import synthetic_snap
                t = ts
                t14 = t.replace(hour=14, minute=0, second=0, microsecond=0)
                t15 = t.replace(hour=15, minute=0, second=0, microsecond=0)
                for exit_ts in (t14, t15):
                    if t <= exit_ts <= t + timedelta(minutes=6):
                        snap2 = synthetic_snap(snap, exit_ts)
                        snap2["max_so_far"] = day_max
                        snap2["market_prices"] = mp
                        snap2["model_probs"] = mod_probs
                        r2 = strategy.run_cycle(
                            timestamp=exit_ts,
                            ensemble_probs=ensemble_probs,
                            market_prices=mp,
                            max_so_far=day_max,
                            current_positions=positions,
                            current_cash=cash,
                            last_trade_times=last_trade_times,
                        )
                        for f2 in r2.get("fills", []):
                            total_trades += 1
                        total_fees += r2.get("total_fees", 0.0)
                        total_det_events += len(r2.get("deterministic_events", []))
                        last_trade_times = r2.get("last_trade_times", last_trade_times)

        # Settle day
        last_prices = snaps[-1]["market_prices"] if snaps else {}
        bucket_bounds = {}
        for b in last_prices:
            lo, hi = parse_bucket_bounds(b)
            bucket_bounds[b] = (lo, hi)
        settlement = strategy.settle_day(
            day_max_temp=day_max,
            final_positions=positions,
            bucket_bounds=bucket_bounds,
            last_prices=last_prices,
        )
        settle_pnl = settlement["total_gross_pnl"]
        cash += settle_pnl
        positions = {k: {"side": "NONE", "quantity": 0} for k in positions}
        daily_pnl[date_str] = settle_pnl

    end_cap = cash + sum(
        p.get("quantity", 0) * 0.5 for p in positions.values()
    )
    total_return = (end_cap / initial_capital - 1.0) * 100.0

    # Simple Sharpe
    returns = [v / initial_capital for v in daily_pnl.values()]
    avg_r = np.mean(returns) if returns else 0.0
    std_r = np.std(returns, ddof=0) if len(returns) > 1 else 1e-6
    sharpe = avg_r / std_r * np.sqrt(252) if std_r > 0 else 0.0

    # Max drawdown
    cum = initial_capital
    peak = cum
    max_dd = 0.0
    for pnl in daily_pnl.values():
        cum += pnl
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "label": label,
        "end_cap": round(end_cap, 2),
        "return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "trades": total_trades,
        "fees": round(total_fees, 2),
        "snaps": total_snaps,
        "det_events": total_det_events,
        "daily_pnl": daily_pnl,
    }


def print_comparison(results: list[dict]):
    print()
    print("=" * 91)
    print("         ENSEMBLE BACKTEST COMPARISON (synthetic 07-06, excl 07-04)        ".center(91))
    print("=" * 91)
    print(f"{'Variant':<20} {'End Cap':>10} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Fees':>8} {'Snaps':>6} {'DetEvts':>7}")
    print("-" * 91)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<20} {'ERROR':>10} {r['error']}")
        else:
            print(f"{r['label']:<20} ${r['end_cap']:>8.2f} {r['return_pct']:>7.2f}% {r['sharpe']:>8.3f} {r['max_dd']:>7.2f}% {r['trades']:>8} ${r['fees']:>6.2f} {r['snaps']:>6} {r['det_events']:>7}")

    # Daily PnL table
    all_dates = sorted({d for r in results for d in r.get("daily_pnl", {})})
    print(f"\n{'-' * 91}")
    print(f"{'Date':<14}", end="")
    for r in results:
        print(f"{r['label'][:16]:<18}", end="")
    print()
    print(f"{'─' * 91}")
    for d in all_dates:
        print(f"{d:<14}", end="")
        for r in results:
            pnl = r.get("daily_pnl", {}).get(d, 0.0)
            print(f" ${pnl:>+8.2f}   ", end="")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic backtest comparison")
    parser.add_argument("--exclude", nargs="*", default=["2026-07-04"])
    args = parser.parse_args()

    results = []

    # 1. Baseline
    results.append(run_variant("Baseline", exclude=args.exclude))

    # 2. A-only
    results.append(run_variant("A (5min dedup)", exclude=args.exclude,
        dedup_fn=dedup_5min))

    # 3. B-only
    results.append(run_variant("B (cooldown=0)", exclude=args.exclude,
        params=EnsembleParams(capital=1000.0, min_rebalance_interval_minutes=0.0)))

    # 4. C-only: hold positions
    results.append(run_variant("C (hold pos)", exclude=args.exclude,
        strategy_cls=CandidateEnsembleStrategy))

    # 5. D-only: synthetic exit cycles
    results.append(run_variant("D (synth exit)", exclude=args.exclude,
        use_synthetic_exit=True))

    # 6. A+B
    results.append(run_variant("A+B", exclude=args.exclude,
        dedup_fn=dedup_5min,
        params=EnsembleParams(capital=1000.0, min_rebalance_interval_minutes=0.0)))

    print_comparison(results)
