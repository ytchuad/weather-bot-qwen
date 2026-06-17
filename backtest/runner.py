"""Backtest runner — replays historical snapshots through paper-trader.

Usage:
    python -m backtest.runner --slug highest-temperature-in-hong-kong --days 7

Output:
    - Paper-trader SQLite with all historical trades
    - Summary DataFrame with PnL over time
    - Optionally, a CSV report
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import argparse
import pandas as pd
from tqdm import tqdm

from execution.paper_adapter import PaperAdapter
from execution.strategy_engine import compute_enhanced_orders
from execution.strategy_runner import run_single_strategy_cycle
from backtest.snapshot_builder import build_snapshots, count_snapshots

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _patch_engine_api(engine, slug: str, adapter):
    """Replace engine's live API calls with offline mock versions."""
    from pm_trader.models import Market
    from backtest.snapshot_builder import BUCKET_DEFS
    import uuid

    # Populate mock buckets so adapter doesn't call Gamma API
    adapter._slug = slug
    adapter._buckets = {}
    for bname, _, _ in BUCKET_DEFS:
        adapter._buckets[bname] = {
            "condition_id": str(uuid.uuid4()),
            "slug": f"{slug}-{bname}",
            "yes_token_id": f"yes_{bname}",
            "no_token_id": f"no_{bname}",
        }

    def mock_get_market(slug_or_id):
        # Find which bucket this slug/condition_id belongs to
        bucket = None
        cond_id = slug_or_id
        for bname, info in adapter._buckets.items():
            if info["slug"] == slug_or_id or info["condition_id"] == slug_or_id:
                bucket = bname
                cond_id = info["condition_id"]
                break
        if bucket is None:
            bucket = "mock"
        return Market(
            condition_id=cond_id,
            slug=adapter._buckets.get(bucket, {}).get("slug", f"{slug}-{bucket}"),
            question=f"Backtest {bucket}",
            description="",
            outcomes=["Yes", "No"],
            outcome_prices=[0.5, 0.5],
            tokens=[{"outcome": "Yes", "token_id": f"yes_{bucket}"},
                    {"outcome": "No", "token_id": f"no_{bucket}"}],
            active=True,
            closed=False,
            volume=0.0,
            liquidity=0.0,
            end_date="2099-12-31T23:59:59Z",
            fee_rate_bps=0,
            tick_size=0.01,
        )

    def mock_get_order_book(token_id):
        from pm_trader.models import OrderBook, OrderBookLevel
        return OrderBook(
            asks=[OrderBookLevel(price=0.90, size=10000)],
            bids=[OrderBookLevel(price=0.10, size=10000)],
        )

    def mock_get_fee_rate(token_id):
        return 0

    def mock_get_midpoint(token_id):
        return 0.5

    engine.api.get_market = mock_get_market
    engine.api.get_order_book = mock_get_order_book
    engine.api.get_fee_rate = mock_get_fee_rate
    engine.api.get_midpoint = mock_get_midpoint


def run_backtest(
    slug: str = "highest-temperature-in-hong-kong",
    strategy_key: str = "enhanced_v1_paper",
    capital: float = 10_000.0,
    date_from: str = None,
    date_to: str = None,
    sample_every_n: int = 6,
    data_dir: str = "data",
    use_mock: bool = True,
    report_path: str = None,
) -> pd.DataFrame:
    """Run full backtest and return results DataFrame.

    Parameters
    ----------
    slug : str
        Polymarket event slug.
    strategy_key : str
        Strategy to use.
    capital : float
        Starting capital for paper-trader account.
    date_from, date_to : str, optional
        Date range (YYYY-MM-DD).
    sample_every_n : int
        Use every Nth row.  6 ≈ hourly for 10-min data.
    data_dir : str
        Path to data directory.
    use_mock : bool
        Use synthetic predictions if True.
    report_path : str, optional
        Save results CSV to this path.

    Returns
    -------
    pd.DataFrame with columns: snapshot_time, cash, cost_basis,
    market_value, unrealized_pnl, num_positions.
    """
    # Fresh paper-trader account
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bt_dir = Path(f"data/backtest_{run_id}")
    adapter = PaperAdapter(bt_dir)

    # Monkey-patch engine API for offline backtesting
    _patch_engine_api(adapter._engine, slug, adapter)
    adapter._engine.init_account(balance=capital)

    snapshots = list(build_snapshots(
        strategy_key=strategy_key,
        slug=slug,
        date_from=date_from,
        date_to=date_to,
        data_dir=data_dir,
        use_mock=use_mock,
        sample_every_n=sample_every_n,
    ))
    total = len(snapshots)
    results = []

    for snap in tqdm(snapshots, total=total, desc="Backtesting"):
        try:
            # Use current available capital
            bal = adapter.get_balance()
            avail_capital = bal.get("cash", capital)

            target_positions, decisions = compute_enhanced_orders(
                target_probs=snap["target_probs"],
                prices_dict=snap["prices_dict"],
                token_ids_dict=snap["token_ids_dict"],
                capital=avail_capital * 0.95,
                mock_slippage=True,
                dt_now=snap["context"]["dt_now"],
                current_positions={},
                model_key=strategy_key,
                slug=slug,
                temp_now=snap["context"].get("temp_now"),
                max_so_far=snap["context"].get("max_so_far"),
                rain_regime=snap["context"].get("rain_regime"),
                model_std=snap["context"].get("model_std", 1.0),
                recent_price_volatility=snap["context"].get("recent_price_volatility", 0.0),
            )

            if target_positions:
                ctx = {
                    "strategy_key": strategy_key,
                    "scheduler_source": "backtest",
                    "selected_model": strategy_key,
                }
                adapter.execute_target_positions(
                    target_positions, "backtest_pf", slug, strategy_key,
                    snap["prices_dict"], ctx,
                )

            pnl = adapter.get_portfolio_pnl(portfolio_id="backtest_pf")
            bal2 = adapter.get_balance()
            results.append({
                "snapshot_time": snap["snapshot_time"],
                "cash": bal2.get("cash", 0),
                "cost_basis": pnl["cost_basis"],
                "market_value": pnl["market_value"],
                "unrealized_pnl": pnl["unrealized_pnl"],
                "num_positions": len(pnl["details"]),
            })

        except Exception as e:
            logger.warning("Snapshot %s failed: %s", snap["snapshot_time"], e)
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
        df = df.sort_values("snapshot_time").reset_index(drop=True)
        df["net_pnl"] = df["market_value"] - df["cost_basis"]

    if report_path and not df.empty:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Report saved to %s", out)

    adapter.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest weather strategy")
    parser.add_argument("--slug", default="highest-temperature-in-hong-kong")
    parser.add_argument("--strategy", default="enhanced_v1_paper")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--from", dest="date_from", help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="YYYY-MM-DD")
    parser.add_argument("--sample", type=int, default=6,
                        help="Every Nth row (default 6 = hourly)")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--report", default="data/backtest_results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count snapshots and exit")
    args = parser.parse_args()

    if args.dry_run:
        n = count_snapshots(args.date_from, args.date_to, args.sample)
        print(f"Snapshots: {n}")
        return

    df = run_backtest(
        slug=args.slug,
        strategy_key=args.strategy,
        capital=args.capital,
        date_from=args.date_from,
        date_to=args.date_to,
        sample_every_n=args.sample,
        use_mock=args.mock,
        report_path=args.report,
    )

    if not df.empty:
        print(f"\nBacktest complete: {len(df)} snapshots")
        print(f"  Start: {df['snapshot_time'].min()}")
        print(f"  End:   {df['snapshot_time'].max()}")
        print(f"  Final unrealized PnL: ${df['net_pnl'].iloc[-1]:,.2f}")
        print(f"  Worst PnL: ${df['net_pnl'].min():,.2f}")
        print(f"  Report: {args.report}")
    else:
        print("No results — no snapshots processed.")


if __name__ == "__main__":
    main()
