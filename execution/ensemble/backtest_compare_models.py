"""Per-model hourly PnL comparison for 07-10 + 07-11 with depth-aware CLOB."""

from __future__ import annotations

from collections import defaultdict

from execution.ensemble.backtest_compare import run_variant
from execution.ensemble.params import EnsembleParams

MODELS = [
    "9d", "aws", "baseline", "model_2a", "model_2a1", "model_2a_v2",
    "model_a", "model_b", "model_c", "model_g", "rain_nowcast",
]

TARGET_DATES = ["2026-07-10", "2026-07-11"]


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
    for (d, h), pnl in sorted(hourly.items()):
        result.setdefault(d, {})[h] = round(pnl, 2)
    return result


def run_all() -> list[dict]:
    results = []
    for mk in MODELS:
        params = EnsembleParams(
            capital=1000.0,
            model_weights={mk: 1.0},
            clob_depth_check=True,
        )
        r = run_variant(
            mk, exclude=[], dates=TARGET_DATES,
            params=params,
        )
        hourly = compute_hourly_pnl(r["fills"])
        results.append({
            "model": mk,
            "capital_end": r["capital_end"],
            "total_return": r["total_return_pct"],
            "day_pnl": r["day_pnl"],
            "hourly_pnl": hourly,
            "trades": r["total_trades"],
        })
    return results


def print_results(results: list[dict]):
    HOURS = list(range(9, 19))  # H9 through H18

    # Header
    print(f"\n{'=' * 120}")
    print(f"{'PER-MODEL HOURLY PnL (depth-aware CLOB, 07-10 + 07-11)':^120}")
    print(f"{'=' * 120}\n")

    for date in TARGET_DATES:
        print(f"\n{'-' * 120}")
        print(f"  {date}")
        print(f"{'-' * 120}")
        header = f"{'Model':<18}"
        for h in HOURS:
            header += f"  H{h:02d}     "
        header += f"  Settle    Total"
        print(header)
        print("-" * len(header))

        for res in results:
            hp = res["hourly_pnl"].get(date, {})
            trading_sum = sum(hp.get(h, 0.0) for h in HOURS)
            day_total = res["day_pnl"].get(date, 0.0)
            settle = round(day_total - trading_sum, 2)

            line = f"{res['model']:<18}"
            for h in HOURS:
                v = hp.get(h, 0.0)
                line += f" ${v:>+7.2f}"
            line += f"  ${settle:>+7.2f}  ${day_total:>+7.2f}"
            print(line)

    # Summary
    print(f"\n{'-' * 120}")
    print(f"  SUMMARY (both days)")
    print(f"{'-' * 120}")
    print(f"{'Model':<18} {'End Cap':>10} {'Return':>10} {'Trades':>7} {'PnL 07-10':>10} {'PnL 07-11':>10}")
    print("-" * 70)
    for res in sorted(results, key=lambda x: x["capital_end"], reverse=True):
        p10 = res["day_pnl"].get("2026-07-10", 0.0)
        p11 = res["day_pnl"].get("2026-07-11", 0.0)
        print(f"{res['model']:<18} ${res['capital_end']:>8.2f} {res['total_return']:>10} {res['trades']:>7} ${p10:>+8.2f} ${p11:>+8.2f}")
    print()


if __name__ == "__main__":
    results = run_all()
    print_results(results)
