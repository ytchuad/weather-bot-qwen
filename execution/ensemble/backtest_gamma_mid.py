"""Per-model hourly PnL using the Gamma (market) mid price.

This mirrors ``backtest_compare_models`` but sets ``gamma_mid=True`` on the
``EnsembleParams`` so every fill/exit is executed at the Gamma mid
(``market_prices``) instead of walking the (pre-fix / unreliable) CLOB book.
The cost model is therefore fees + ``slippage_fixed`` only — a fair,
CLOB-independent view of each model's strategy edge.

Use this to evaluate model performance on the past-week snapshots until the
CLOB market-depth fix is reflected in fresh export data.

Usage:
    python execution/ensemble/backtest_gamma_mid.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from execution.ensemble.backtest_compare import run_variant
from execution.ensemble.params import EnsembleParams

MODELS = [
    "9d", "aws", "baseline", "model_2a", "model_2a1", "model_2a_v2",
    "model_a", "model_b", "model_c", "model_g", "rain_nowcast",
]

# Past week (inclusive). 2026-07-12 is the in-progress day — runs with
# whatever snapshots exist, settlement uses the last observed max_so_far.
TARGET_DATES = [
    "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
    "2026-07-10", "2026-07-11", "2026-07-12",
]

OUTPUT_DIR = Path("output/gamma_mid")


def compute_hourly_mtm(equity_curve: list[dict]) -> dict[str, dict[int, float]]:
    """Hourly unrealized (mark-to-market) PnL per (date, hour).

    For each (date, hour) we take the LAST intraday equity-curve snapshot
    (i.e. the position state at the end of that hour) and read its
    ``unrealized_pnl``. This is the PnL the strategy is *actually sitting
    on* during the hour — unlike cash-flow, it does NOT count the entry
    buy as a loss. Settlement (23:59) rows are excluded.
    """
    last: dict[tuple[str, int], float] = {}
    for e in equity_curve:
        if "T23" in e.get("timestamp", ""):
            continue
        dt = e.get("date", "")
        t = e.get("time", "00:00:00")
        try:
            h = int(t[:2])
        except (ValueError, TypeError):
            continue
        last[(dt, h)] = float(e.get("unrealized_pnl", 0.0))
    result: dict[str, dict[int, float]] = {}
    for (dt, h), v in last.items():
        result.setdefault(dt, {})[h] = round(v, 2)
    return result


def compute_hourly_pnl(fills: list[dict]) -> dict[str, dict[int, float]]:
    """Group fills by (date, hour), return {date: {hour: pnl}}."""
    hourly: dict[tuple[str, int], float] = defaultdict(float)
    for f in fills:
        ts = f["timestamp"]
        date = ts[:10]
        hour = int(ts[11:13])
        d = f["shares_delta"]
        price = f["execution_price"]
        fee = f.get("fee", 0.0)
        if d > 0:
            cash_impact = -(d * price + fee)
        else:
            cash_impact = abs(d) * price - fee
        hourly[(date, hour)] += cash_impact
    result: dict[str, dict[int, float]] = {}
    for (dt, h), pnl in sorted(hourly.items()):
        result.setdefault(dt, {})[h] = round(pnl, 2)
    return result


def run_all() -> list[dict]:
    results = []
    for mk in MODELS:
        params = EnsembleParams(
            capital=1000.0,
            model_weights={mk: 1.0},
            gamma_mid=True,
            # edge threshold is the strategy's pre-defined gate (5%);
            # time slicing (morning/afternoon/evening) comes from params.
        )
        r = run_variant(
            mk, exclude=[], dates=TARGET_DATES,
            params=params,
            output_dir=OUTPUT_DIR / mk,
        )
        hourly = compute_hourly_pnl(r["fills"])
        # MTM: read the per-model equity_curve.csv written by run_variant
        mtm = {}
        eq_path = OUTPUT_DIR / mk / "equity_curve.csv"
        if eq_path.exists():
            import csv as _csv
            with open(eq_path, encoding="utf-8") as _f:
                mtm = compute_hourly_mtm(list(_csv.DictReader(_f)))
        results.append({
            "model": mk,
            "capital_end": r["capital_end"],
            "total_return": r["total_return_pct"],
            "day_pnl": r["day_pnl"],
            "hourly_pnl": hourly,
            "hourly_mtm": mtm,
            "trades": r["total_trades"],
            "skip_counts": r.get("skip_counts", {}),
        })
    return results


def _day_block(results: list[dict], metric_key: str, title: str):
    """Print one per-date hourly table for the given metric (hourly_pnl / hourly_mtm)."""
    HOURS = list(range(8, 20))
    print(f"\n{'-' * 140}")
    print(f"  {title}  (H8-H19 = unrealized/MTM at end of hour; settlement is realized at day end)")
    print(f"{'-' * 140}")
    for date in TARGET_DATES:
        if not any(date in res[metric_key] or date in res["day_pnl"] for res in results):
            continue
        print(f"\n  {date}")
        header = f"{'Model':<18}"
        for h in HOURS:
            header += f"  H{h:02d}     "
        header += f"  Settle    Total"
        print(header)
        print("-" * len(header))
        for res in results:
            hp = res[metric_key].get(date, {})
            trading_sum = sum(hp.get(h, 0.0) for h in HOURS)
            day_total = res["day_pnl"].get(date, 0.0)
            settle = round(day_total - trading_sum, 2)
            line = f"{res['model']:<18}"
            for h in HOURS:
                v = hp.get(h, 0.0)
                line += f" ${v:>+7.2f}"
            line += f"  ${settle:>+7.2f}  ${day_total:>+7.2f}"
            print(line)


def print_results(results: list[dict]):
    HOURS = list(range(8, 20))

    print(f"\n{'=' * 140}")
    print(f"{'PER-MODEL HOURLY PnL  (Gamma mid execution, past week)':^140}")
    print(f"{'=' * 140}\n")

    # PRIMARY: MTM (unrealized) — does not penalize entry buys
    _day_block(results, "hourly_mtm", "TABLE 1 (primary): Hourly UNREALIZED (MTM) PnL  -- what the position is worth during the hour")

    # SECONDARY: cash-flow — for reference only
    _day_block(results, "hourly_pnl", "TABLE 2 (reference): Hourly CASH-FLOW PnL  -- entry buys show as negative, ignored for conclusions")

    # Summary
    print(f"\n{'-' * 140}")
    print(f"  SUMMARY (all days)")
    print(f"{'-' * 140}")
    print(f"{'Model':<18} {'End Cap':>10} {'Return':>10} {'Trades':>7}")
    print("-" * 48)
    for res in sorted(results, key=lambda x: x["capital_end"], reverse=True):
        print(f"{res['model']:<18} ${res['capital_end']:>8.2f} {res['total_return']:>10} {res['trades']:>7}")
    print(f"\nDetailed output in {OUTPUT_DIR}/ (per-model trade_log / equity_curve / summary)\n")


if __name__ == "__main__":
    results = run_all()
    print_results(results)
