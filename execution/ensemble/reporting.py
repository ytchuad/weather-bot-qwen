import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FIELDS_TRADE_LOG = [
    "timestamp", "date", "time", "mode", "bucket", "side",
    "action", "shares_delta", "execution_price",
    "market_yes_price", "market_no_price",
    "fee", "slippage", "position_before", "position_after", "reason",
]

REQUIRED_FIELDS_POSITION = [
    "timestamp", "date", "time", "bucket", "side", "shares",
    "market_price", "mark_value", "deterministic_status", "unrealized_pnl",
]

REQUIRED_FIELDS_ALLOCATION = [
    "timestamp", "date", "time", "bucket", "side",
    "ensemble_prob", "market_yes_price", "market_no_price",
    "execution_price", "edge", "raw_kelly_fraction", "final_kelly_fraction",
    "target_position", "target_notional", "target_shares", "reason",
]

REQUIRED_FIELDS_EQUITY = [
    "timestamp", "date", "time", "cash", "position_value",
    "total_equity", "unrealized_pnl", "realized_pnl",
]


class BacktestReport:
    @staticmethod
    def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info("Wrote %d rows → %s", len(rows), path)

    def write_trade_log(self, events: list[dict], path: Path):
        self._write_csv(path, events, REQUIRED_FIELDS_TRADE_LOG)

    def write_position_snapshot(self, snapshots: list[dict], path: Path):
        self._write_csv(path, snapshots, REQUIRED_FIELDS_POSITION)

    def write_allocation_log(self, allocations: list[dict], path: Path):
        self._write_csv(path, allocations, REQUIRED_FIELDS_ALLOCATION)

    def write_equity_curve(self, curve: list[dict], path: Path):
        self._write_csv(path, curve, REQUIRED_FIELDS_EQUITY)

    def write_summary_md(self, stats: dict, path: Path):
        lines = []
        _a = lines.append

        _a("# Simulation Summary")
        _a("")
        _a(f"- **Backtest date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        _a(f"- **Dates analysed**: {', '.join(stats['dates'])}")
        _a(f"- **Strategy**: Ensemble A/B/C (equal-weight)")
        _a("")

        _a("## Capital & Return")
        _a("")
        _a(f"- **Capital start**: ${stats['capital_start']:.2f}")
        _a(f"- **Capital end**: ${stats['capital_end']:.2f}")
        _a(f"- **Total return**: {stats['total_return_pct']}")
        _a(f"- **Sharpe ratio** (daily, annualised): {stats['sharpe']}")
        _a(f"- **Max drawdown**: {stats['max_drawdown_pct']}")
        _a("")

        _a("## Trades")
        _a("")
        _a(f"- **Total trades**: {stats['total_trades']}")
        _a(f"- **Total fees**: ${stats['total_fees']:.2f}")
        _a(f"- **Total slippage**: ${stats['total_slippage']:.2f}")
        _a(f"- **YES buy trades**: {stats['yes_trades']}")
        _a(f"- **NO buy trades**: {stats['no_trades']}")
        _a("")

        _a("## Daily Performance")
        _a("")
        _a("| Date | PnL | Return |")
        _a("|------|-----|--------|")
        day_pnl = stats.get("day_pnl", {})
        daily_returns = stats.get("daily_returns", {})
        for d in stats["dates"]:
            pnl = day_pnl.get(d, 0)
            ret = daily_returns.get(d, 0)
            _a(f"| {d} | ${pnl:+.2f} | {ret*100:+.2f}% |")
        _a("")

        _a("## Risk-Reduction Mode Summary")
        _a("")
        n_risk_reduction = sum(
            1 for f in stats.get("_all_modes", {}).get("RISK_REDUCTION", [])
        )
        _a(f"- Cycles in risk-reduction mode (14:00-15:00): {n_risk_reduction}")
        _a("")

        _a("## Trading Window Summary")
        _a("")
        _a("- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing")
        _a("- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk")
        _a("- **After 15:00**: HARD_FLAT_TARGET — exit all positions")
        _a("")

        _a("## Breakout Handling Summary")
        _a("")
        det_events = stats.get("deterministic_events", 0)
        _a(f"- Deterministic events triggered: {det_events}")
        _a("")

        _a("## Skipped / Rejected Trades by Reason")
        _a("")
        skip_counts = stats.get("skip_counts", {})
        if skip_counts:
            _a("| Reason | Count |")
            _a("|--------|-------|")
            for reason, cnt in sorted(skip_counts.items()):
                _a(f"| {reason} | {cnt} |")
        else:
            _a("(none)")
        _a("")

        _a("## Parameters")
        _a("")
        p = stats.get("strategy_params", {})
        _a(f"- Capital: ${p.capital}")
        _a(f"- Model weights: {p.model_weights}")
        _a(f"- Edge threshold: {p.edge_threshold}")
        _a(f"- Kelly fraction: {p.kelly_fraction}")
        _a(f"- Max per bucket side: {p.max_per_bucket_side}")
        _a(f"- Total exposure cap: {p.total_exposure_cap}")
        _a(f"- Price band: [{p.min_price}, {p.max_price}]")
        _a(f"- Min shares: {p.min_shares}")
        _a(f"- Slippage fixed: {p.slippage_fixed}")
        _a(f"- Fee constant: {p.fee_constant}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote summary → %s", path)
