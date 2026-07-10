# execution/paper_trade_harness.py
"""Snapshot-isolated paper trade harness for backtesting strategies.

Usage as module:
    python -m execution.paper_trade_harness --strategy enhanced_v2_paper

Usage as library:
    from execution.paper_trade_harness import PaperTradeHarness, Scenario, Backtest
    harness = PaperTradeHarness("enhanced_v2_paper")
    bt = harness.run([scenario1, scenario2, ...])

Architecture
────────────
Scenarios are fed to the harness in time order.  Each scenario represents
the market state at a snapshot ::

    { buckets: {bucket_name: {prob, market_price}},
      weather: {temp_now, max_so_far, rain_regime, ...},
      time: datetime,
      hours_to_settlement: float,
      capital: float,
      drawdown_pct: float,
      context: {} }             # extra per-scenario context

The harness builds GateInput per bucket and passes it through the strategy's
four pipelines (entry / exit / sizing / rebalance).  Target positions are
computed from Kelly sizing × sizing multipliers, then a simulated fill is
applied using mock_slippage.  A trade log records every cycle.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from execution.gates import GateInput, product_of_multipliers
from execution.gates.pipeline import GatePipeline
from execution.strategy_config import Strategy
from execution.strategy_factory import build_strategy, get_factory

logger = logging.getLogger(__name__)

# ─── Data classes ──────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """Market state at one point in time.

    Attributes:
        buckets: {bucket_name: {"prob": float, "market_price": float}}.
            prob is the model's probability for this bucket.
        weather: rain regime, temperatures, etc.
        time: datetime of this snapshot.
        hours_to_settlement: hours until question settles.
        capital: available capital in the account.
        drawdown_pct: current drawdown from peak (negative = down).
        context: extra fields passed through to GateInput.
    """
    buckets: dict
    weather: dict = field(default_factory=dict)
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hours_to_settlement: float = 24.0
    capital: float = 10000.0
    drawdown_pct: float = 0.0
    context: dict = field(default_factory=dict)

    # Derived convenience
    @property
    def rain_regime(self) -> str:
        return self.weather.get("rain_regime", "no_rain")

    @property
    def temp_now(self) -> Optional[float]:
        return self.weather.get("temp_now")

    @property
    def max_so_far(self) -> Optional[float]:
        return self.weather.get("max_so_far")

    @property
    def model_key(self) -> str:
        return self.context.get("model_key", "baseline")

    @property
    def post_mean(self) -> Optional[float]:
        """Temperature forecast (posterior mean) for boundary proximity checks."""
        return self.weather.get("post_mean") or self.weather.get("temp_now")


@dataclass
class Position:
    """A live position in one bucket. Simulated paper trade."""
    bucket: str
    quantity: float          # positive = YES (bought), negative = NO
    entry_price: float       # price at which we entered
    entry_time: datetime
    model_prob_at_entry: float


@dataclass
class Trade:
    """One completed trade (exit of a position)."""
    bucket: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    side: str          # "YES" or "NO"
    entry_price: float
    exit_price: float
    pnl: float         # signed: positive = profit, negative = loss
    reason: str        # exit reason code
    strategy: str


@dataclass
class CycleResult:
    """Output of one strategy cycle (one scenario evaluation)."""
    time: datetime
    bucket: str
    model_prob: float
    market_price: float
    entry_ok: bool
    exit_ok: bool            # True = hold, False = exit
    sizing_factor: float
    target_qty: float
    current_qty: float
    kelly_size: float
    edge: float
    gate_results: dict       # {pipeline_name: list of gate outputs}
    rebalance_triggered: bool
    rebalance_reason: str
    action: str              # "NONE" | "ENTER" | "REDUCE" | "EXIT"
    action_qty: float


@dataclass
class BacktestResult:
    """Cumulative result of a full backtest across all scenarios."""
    strategy: str
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    num_trades: int
    num_cycles: int
    cycle_results: list[CycleResult]
    trade_log: list[Trade]
    capital_history: list[tuple[datetime, float]]

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3) if np.isfinite(self.sharpe_ratio) else 0.0,
            "num_trades": self.num_trades,
            "num_cycles": self.num_cycles,
        }


# ─── Kelly sizing ───────────────────────────────────────────────────────────

def kelly_fraction(win_prob: float, price: float) -> float:
    """Classic Kelly: f* = (p*b - q) / b, where price = 1/b."""
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price         # odds received
    q = 1.0 - win_prob
    kelly = (win_prob * b - q) / b
    return max(0.0, kelly)


# ─── Position tracker ──────────────────────────────────────────────────────

class PositionTracker:
    """Keeps track of live positions and computes PnL."""

    def __init__(self):
        self.positions: dict[str, Position] = {}

    def get(self, bucket: str) -> Position | None:
        return self.positions.get(bucket)

    def apply_trade(self, bucket: str, qty: float, price: float,
                    model_prob: float, dt: datetime) -> None:
        """Execute a trade: positive qty = BUY YES, negative = SELL/FLAT."""
        existing = self.positions.get(bucket)
        if existing is None:
            if abs(qty) > 0.01:
                side = "YES" if qty > 0 else "NO"
                self.positions[bucket] = Position(
                    bucket=bucket,
                    quantity=abs(qty),
                    entry_price=price,
                    entry_time=dt,
                    model_prob_at_entry=model_prob,
                )
        else:
            net = existing.quantity - abs(qty)
            if abs(net) < 0.01:
                del self.positions[bucket]
            else:
                existing.quantity = abs(net)

    def exit_all(self, bucket: str, exit_price: float, exit_time: datetime,
                 reason: str, strategy: str) -> Trade | None:
        """Close a position and return the resulting Trade."""
        pos = self.positions.pop(bucket, None)
        if pos is None:
            return None

        side = "YES" if pos.quantity > 0 else "NO"
        entry_val = pos.quantity * pos.entry_price
        exit_val = pos.quantity * exit_price
        # PnL for YES position: we paid entry_price, collect exit_price
        pnl = exit_val - entry_val if side == "YES" else entry_val - exit_val

        return Trade(
            bucket=bucket,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            quantity=pos.quantity,
            side=side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            reason=reason,
            strategy=strategy,
        )

    @property
    def total_exposure(self) -> float:
        return sum(abs(p.quantity * p.entry_price) for p in self.positions.values())

    def snapshot(self, buckets: dict) -> dict[str, dict]:
        """Return current positions as dict matching downstream expectation."""
        return {
            bucket: {
                "quantity": (p.quantity if p.quantity > 0 else -p.quantity),
                "side": "YES" if p.quantity >= 0 else "NO",
                "entry_price": p.entry_price,
            }
            for bucket, p in self.positions.items()
            if bucket in buckets
        }


# ─── Slippagesim ────────────────────────────────────────────────────────

class SlippageSim:
    """Mock slippage: larger orders get worse fills.

    In production the real order-adapter computes this from Polymarket
    order-book depth.  Here we approximate: order size / price → slippage %
    so we can detect when a strategy tries to over-size.
    """

    def __init__(self, base_slippage: float = 0.005, price_impact: float = 0.01):
        self.base = base_slippage
        self.price_impact = price_impact

    def fill(self, quantity: float, price: float, capital: float) -> dict:
        """Return a mock filled bet result."""
        notional = abs(quantity) * price
        if notional < 0.01:
            return {"filled": False, "adjusted_quantity": 0,
                    "slippage_pct": 0.0}

        # Simulate slippage: bigger orders → more slippage
        size_fraction = min(1.0, (notional / capital))
        slip = self.base + self.price_impact * size_fraction
        slippage_cost = notional * slip
        adjusted_notional = max(0.0, notional - slippage_cost)
        adjusted_qty = adjusted_notional / price if price > 0 else 0

        return {
            "_sim": True,
            "filled": adjusted_qty > 0.01,
            "adjusted_quantity": adjusted_qty,
            "original_quantity": quantity,
            "slippage_pct": round(slip * 100, 2),
            "adjusted_bet_notional": round(adjusted_notional, 2),
        }


# ─── Main harness ───────────────────────────────────────────────────────────

class PaperTradeHarness:
    """Isolated paper-trade runner for a Strategy.

    Parameters
    ----------
    strategy_or_key : Strategy | str
        Either a Strategy object or a strategy key string.
    capital : float, default 10 000
        Starting capital.
    kelly_fraction : float, default 0.25
        Fraction of full Kelly to bet.
    seed : int, optional
        Random seed for reproducible slippage simulation.
    """

    def __init__(
        self,
        strategy_or_key: Strategy | str = "enhanced_v2_paper",
        capital: float = 10_000.0,
        kelly_fraction: float = 0.25,
        seed: int | None = 42,
        _override_pipelines: tuple | None = None,
    ):
        if _override_pipelines is not None:
            # Used by Strategy Builder to inject custom pipelines from the
            # form UI without needing a full Strategy object.
            ep, ex, sz, rb = _override_pipelines
            self.strategy = object.__new__(Strategy)
            self.strategy.key = strategy_or_key if isinstance(strategy_or_key, str) else "custom"
            self.strategy.entry_pipeline = ep
            self.strategy.exit_pipeline = ex
            self.strategy.sizing_pipeline = sz
            self.strategy.rebalance_pipeline = rb
            self.strategy.kelly_fraction = kelly_fraction
            self.strategy.label = strategy_or_key if isinstance(strategy_or_key, str) else "custom"
            self.strategy_key = self.strategy.key
        elif isinstance(strategy_or_key, str):
            self.strategy: Strategy = build_strategy(strategy_or_key)
            self.strategy_key = strategy_or_key
        else:
            self.strategy = strategy_or_key
            self.strategy_key = strategy_or_key.key

        self.capital = capital
        self.kelly_frac = kelly_fraction
        self.slippage_sim = SlippageSim()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tracker = PositionTracker()
        self.trade_log: list[Trade] = []
        self.capital_history: list[tuple[datetime, float]] = []
        self.cycle_results: list[CycleResult] = []

        # Track prior probabilities for rebalance trigger
        self._last_probs: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self, scenarios: list[Scenario]) -> BacktestResult:
        """Run the strategy across a list of scenarios in time order.

        Returns a BacktestResult with all cycles, trades, and summary stats.
        """
        self.tracker = PositionTracker()
        self.trade_log = []
        self.capital_history = []
        self.cycle_results = []
        self._last_probs = {}
        self._last_prices = {}

        initial_capital = self.capital
        running_capital = initial_capital
        running_pnl: float = 0.0

        for scenario in scenarios:
            probs = {b: d["prob"] for b, d in scenario.buckets.items()}
            prices = {b: d["market_price"] for b, d in scenario.buckets.items()}

            cycle_capital = running_capital + running_pnl

            for bucket, prob in probs.items():
                market_price = prices.get(bucket, 0.5)
                edge = prob - market_price

                # --- Determine current position ---
                pos = self.tracker.get(bucket)
                current_qty = pos.quantity if pos else 0.0

                # --- Build GateInput ---
                context = {
                    **scenario.context,
                    "post_mean": scenario.post_mean or 31.0,
                    "drawdown_pct": scenario.drawdown_pct,
                    "hours_to_settlement": scenario.hours_to_settlement,
                    "rain_regime": scenario.rain_regime,
                    "max_so_far": scenario.max_so_far,
                    "temp_now": scenario.temp_now,
                    "nowcast_stale": scenario.context.get("nowcast_stale", False),
                    "data_missing": scenario.context.get("data_missing", False),
                    "model_key": scenario.model_key,
                    "capital": cycle_capital,
                    "current_positions": self.tracker.snapshot(scenario.buckets),
                    "probs_old": self._last_probs,
                    "probs_new": probs,
                    "target_quantity": 0.0,   # set below after sizing
                }

                inp = GateInput(
                    bucket=bucket,
                    model_prob=prob,
                    market_price=market_price,
                    model_std=scenario.context.get("model_std", 1.5),
                    dt_now=scenario.time,
                    position={"side": "YES", "quantity": current_qty} if current_qty > 0 else None,
                    context=context,
                )

                # --- Slipped Kelly sizing ---
                kelly_size = kelly_fraction(prob, market_price) * self.kelly_frac * cycle_capital
                if kelly_size < 0:
                    kelly_size = 0.0

                # Run entry pipeline (short-circuits on fail)
                entry_ok, entry_results = self.strategy.evaluate_entry(inp)
                entry_gate_names = [r.metadata.get("_gate_name", "?") for r in entry_results]

                # Run exit pipeline (all gates, multiplier product ≤ 0 → exit)
                exit_ok, exit_results = self.strategy.evaluate_exit(inp)
                exit_gate_names = [r.metadata.get("_gate_name", "?") for r in exit_results]
                exit_mult = product_of_multipliers(exit_results)

                # Run sizing pipeline (all gates, multiplier product = size factor)
                sizing_ok, sizing_results = self.strategy.evaluate_sizing(inp)
                sizing_factor = product_of_multipliers(sizing_results)

                # Run rebalance pipeline (short-circuits on first trigger)
                rebal_ok, rebal_results = self.strategy.evaluate_rebalance(inp)
                rebal_gate_names = [r.metadata.get("_gate_name", "?") for r in rebal_results]

                # Target quantity after sizing
                target_raw = kelly_size * sizing_factor
                target_qty = max(0.0, min(target_raw, cycle_capital * 0.2))  # cap at 20% capital per bucket

                # Update context with computed target
                inp.context["target_quantity"] = target_qty

                # ── Decision ───────────────────────────────────────────────
                action = "NONE"
                action_qty = 0.0
                rebalance_triggered = rebal_ok
                rebalance_reason = rebal_results[0].reason_code if rebal_results else ""

                if current_qty > 0:
                    if exit_mult <= 0:
                        # Exit triggered
                        trade = self.tracker.exit_all(bucket, market_price, scenario.time,
                                                       "GATE_EXIT", self.strategy_key)
                        if trade:
                            self.trade_log.append(trade)
                            running_pnl += trade.pnl
                        action = "EXIT"
                        action_qty = current_qty
                else:
                    if entry_ok:
                        # Entry triggered — all entry gates passed
                        slip = self.slippage_sim.fill(target_qty, market_price, cycle_capital)
                        filled_qty = slip["adjusted_quantity"]
                        if slip["filled"] and filled_qty > 0.01:
                            self.tracker.apply_trade(bucket, filled_qty, market_price,
                                                     prob, scenario.time)
                            action = "ENTER"
                            action_qty = filled_qty

                # Record cycle
                self.cycle_results.append(CycleResult(
                    time=scenario.time,
                    bucket=bucket,
                    model_prob=prob,
                    market_price=market_price,
                    entry_ok=entry_ok,
                    exit_ok=exit_ok,
                    sizing_factor=round(sizing_factor, 4),
                    target_qty=round(action_qty, 2),
                    current_qty=round(current_qty, 2),
                    kelly_size=round(kelly_size, 2),
                    edge=round(edge, 4),
                    gate_results={
                        "entry": entry_gate_names,
                        "exit": exit_gate_names,
                        "sizing": [r.metadata.get("_gate_name", "?") for r in sizing_results],
                        "rebalance": rebal_gate_names,
                    },
                    rebalance_triggered=rebalance_triggered,
                    rebalance_reason=rebalance_reason,
                    action=action,
                    action_qty=round(action_qty, 2),
                ))

            # Update capital
            running_capital = initial_capital + running_pnl
            # Add exposure cost (simplified: reduce capital by unrealised PnL estimate)
            # For a proper mark-to-market we'd update entry prices, but here we
            # just record the running capital snapshot.
            self.capital_history.append((scenario.time, running_capital))

        # Close all open positions at final prices
        final_prices = {}
        if scenarios:
            last = scenarios[-1]
            final_prices = {b: d["market_price"] for b, d in last.buckets.items()}
        for bucket in list(self.tracker.positions.keys()):
            price = final_prices.get(bucket, 0.5)
            trade = self.tracker.exit_all(bucket, price,
                                           scenarios[-1].time if scenarios else datetime.now(),
                                           "END_OF_BACKTEST", self.strategy_key)
            if trade:
                self.trade_log.append(trade)
                running_pnl += trade.pnl

        final_capital = initial_capital + running_pnl

        return BacktestResult(
            strategy=self.strategy_key,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_pnl=running_pnl,
            total_return_pct=running_pnl / initial_capital if initial_capital else 0,
            max_drawdown=self._compute_max_drawdown(),
            sharpe_ratio=self._compute_sharpe(),
            num_trades=len(self.trade_log),
            num_cycles=len(self.cycle_results),
            cycle_results=self.cycle_results,
            trade_log=self.trade_log,
            capital_history=self.capital_history,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_max_drawdown(self) -> float:
        """Peak-to-trough drawdown from capital history."""
        if not self.capital_history:
            return 0.0
        peak = self.capital_history[0][1]
        max_dd = 0.0
        for _, cap in self.capital_history:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _compute_sharpe(self, risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio (simplified, uses daily returns)."""
        if len(self.capital_history) < 2:
            return 0.0
        pcts = [(b - a) / a for (ta, a), (tb, b) in zip(self.capital_history, self.capital_history[1:])]
        if not pcts:
            return 0.0
        mean_ret = np.mean(pcts)
        std_ret = np.std(pcts, ddof=1) if len(pcts) > 1 else 1.0
        if std_ret == 0:
            return 0.0
        # Annualise assuming 365 data points
        return (mean_ret - risk_free) / std_ret * np.sqrt(len(pcts))


# ─── Scenario builders (helpers for backtesting) ───────────────────────────

def _maybe_import_inference():
    """Lazy import of predict_bucket_probabilities — may not be available in all envs."""
    try:
        from models.inference import predict_bucket_probabilities
        return predict_bucket_probabilities
    except ImportError:
        return None


def _probs_from_temperature(mean: float, std: float,
                              bucket_names: list[str]) -> dict[str, float]:
    """Compute bucket probabilities from a normal temperature forecast.

    Uses scipy if available, otherwise a manual normal-CDF approximation.
    Returns {bucket_name: probability}.
    """
    try:
        from scipy.stats import norm
        cdf = norm.cdf
    except ImportError:
        # Manual normal CDF approximation (Abramowitz & Stegun)
        def cdf(x):
            z = abs(x) / 1.4142135623730951
            t = 1.0 / (1.0 + 0.3275911 * z)
            p = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
            return 0.5 * (1.0 + p) if x >= 0 else 0.5 * (1.0 - p)

    # Build market-style dicts with lower/upper bounds
    from execution.gates.entry import _parse_bucket_bounds
    markets = []
    for bk in bucket_names:
        lo, hi = _parse_bucket_bounds(bk)
        markets.append({"name": bk, "lower": lo, "upper": hi})

    probs = {}
    for m in markets:
        lo, hi = m["lower"], m["upper"]
        if lo == float("-inf"):
            lo_cdf = 0.0
        else:
            lo_cdf = cdf((lo - mean) / std) if std > 0 else (1.0 if mean > lo else 0.0)
        if hi == float("inf"):
            hi_cdf = 1.0
        else:
            hi_cdf = cdf((hi - mean) / std) if std > 0 else (1.0 if mean > hi else 0.0)
        probs[m["name"]] = max(0.0, hi_cdf - lo_cdf)

    # Normalise to sum to 1
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def generate_synthetic_scenarios(
    buckets: dict,
    n_cycles: int = 50,
    start_time: datetime | None = None,
    hours_per_cycle: float = 2.0,
    capital: float = 10_000.0,
    base_rain_regime: str = "no_rain",
    add_noise: bool = True,
    seed: int | None = None,
) -> list[Scenario]:
    """Generate N synthetic scenarios with physically-consistent probability distributions.

    Bucket probabilities are derived from a drifting temperature forecast
    (post_mean ± model_std), so boundary proximity checks work correctly.
    Market prices drift with a lag behind true probabilities to create
    tradeable edges.
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bucket_names = list(buckets.keys())
    if not bucket_names:
        return []

    # Initialise: temperature forecast starts near 30°C, std ~1.5°C
    n = len(bucket_names)
    post_mean = 30.0
    model_std = 1.5
    # Market prices start from a slightly off-centre distribution
    init_probs = _probs_from_temperature(post_mean, model_std, bucket_names)
    prices = np.array([buckets[b].get("market_price", init_probs.get(b, 0.1)) for b in bucket_names])
    # Normalise initial prices
    prices = prices / prices.sum() if prices.sum() > 0 else np.ones(n) / n

    scenarios = []
    h2s = 24.0

    for i in range(n_cycles):
        t = start_time + timedelta(hours=i * hours_per_cycle)
        h2s = max(0.0, h2s - hours_per_cycle)

        if add_noise:
            # Temperature forecast drifts
            post_mean += np.random.randn() * 0.3
            post_mean = np.clip(post_mean, 26.0, 36.0)
            model_std = np.clip(1.0 + np.random.randn() * 0.3, 0.5, 3.5)

        # Derive bucket probabilities from temperature forecast
        probs = _probs_from_temperature(post_mean, model_std, bucket_names)

        # Market prices lag behind true probabilities to create edges
        prob_arr = np.array([probs.get(b, 0.0) for b in bucket_names])
        if add_noise:
            prices = np.clip(
                prices + 0.4 * (prob_arr - prices) + np.random.randn(n) * 0.003,
                0.005, 0.95,
            )
            # Re-normalise prices
            prices = prices / prices.sum() if prices.sum() > 0 else np.ones(n) / n

        bucket_dict = {}
        for j, bk in enumerate(bucket_names):
            bucket_dict[bk] = {
                "prob": round(float(prob_arr[j]), 4),
                "market_price": round(float(prices[j]), 4),
            }

        # Simulate weather
        rain_regime = base_rain_regime
        if t.hour >= 12 and t.hour <= 15 and random.random() < 0.3:
            rain_regime = random.choice(["weak_rain", "moderate_or_heavy_rain"])

        # Temperature observed values consistent with the forecast
        scenario = Scenario(
            buckets=bucket_dict,
            weather={
                "rain_regime": rain_regime,
                "temp_now": float(np.clip(post_mean + np.random.randn() * 0.3, 24.0, 38.0)),
                "max_so_far": float(np.clip(post_mean + abs(np.random.randn()) * 0.5, 24.0, 40.0)),
                "post_mean": float(post_mean),
            },
            time=t,
            hours_to_settlement=h2s,
            capital=capital,
            drawdown_pct=random.uniform(-0.05, 0.02) if add_noise else 0.0,
            context={
                "model_std": model_std,
                "model_key": random.choice(["baseline", "model_a", "model_c"]),
            },
        )
        scenarios.append(scenario)

    return scenarios


# ─── CLI ───────────────────────────────────────────────────────────────────

def _cli():
    """Run a backtest from the command line."""
    import argparse, sys

    parser = argparse.ArgumentParser(description="Paper-trade strategy harness")
    parser.add_argument("--strategy", "-s", default="enhanced_v2_paper",
                        help="Strategy key to test")
    parser.add_argument("--cycles", "-n", type=int, default=50,
                        help="Number of synthetic scenarios")
    parser.add_argument("--capital", "-c", type=float, default=10_000.0,
                        help="Starting capital")
    parser.add_argument("--kelly-fraction", "-k", type=float, default=0.25,
                        help="Kelly fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (set for reproducible slippage)")
    parser.add_argument("--output", "-o", type=str,
                        help="Write result JSON to file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    if args.strategy not in get_factory():
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available: {list(get_factory().keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Paper Trade Backtest")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Capital    : ${args.capital:,.2f}")
    print(f"  Kelly frac : {args.kelly_fraction:.0%}")
    print(f"  Cycles     : {args.cycles}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}\n")

    # Build synthetic markets (temperature buckets)
    temp_buckets = {
        "25-26": {"market_price": 0.05},
        "26-27": {"market_price": 0.08},
        "27-28": {"market_price": 0.12},
        "28-29": {"market_price": 0.18},
        "29-30": {"market_price": 0.20},
        "30-31": {"market_price": 0.15},
        "31-32": {"market_price": 0.10},
        "32-33": {"market_price": 0.07},
        "33-34": {"market_price": 0.04},
        "34-35": {"market_price": 0.02},
        "35-36": {"market_price": 0.01},
        ">=36":  {"market_price": 0.005},
    }

    scenarios = generate_synthetic_scenarios(
        buckets=temp_buckets,
        n_cycles=args.cycles,
        capital=args.capital,
        seed=args.seed,
    )

    harness = PaperTradeHarness(
        strategy_or_key=args.strategy,
        capital=args.capital,
        kelly_fraction=args.kelly_fraction,
        seed=args.seed,
    )

    result = harness.run(scenarios)
    summary = result.summary()

    print(f"Results")
    print(f"  Final capital : ${result.final_capital:,.2f}")
    print(f"  Total PnL     : ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:.1f}%)")
    print(f"  Max drawdown  : {summary['max_drawdown']:.1f}%")
    print(f"  Sharpe ratio  : {summary['sharpe_ratio']}")
    print(f"  Trades        : {summary['num_trades']}")
    print(f"  Cycles        : {summary['num_cycles']}")
    print()

    # Trade log
    if result.trade_log:
        print("Trade log (last 10)")
        print(f"  {'Bucket':<12} {'In':>6} {'Out':>6} {'Qty':>6} {'PnL':>8}  Reason")
        for t in result.trade_log[-10:]:
            print(f"  {t.bucket:<12} {t.entry_price:>6.3f} {t.exit_price:>6.3f} "
                  f"{t.quantity:>6.2f} {t.pnl:>+8.2f}  {t.reason}")

    if args.output:
        out = {
            "summary": summary,
            "capital_history": [(str(t), round(c, 2)) for t, c in result.capital_history],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResult written to {args.output}")


if __name__ == "__main__":
    _cli()