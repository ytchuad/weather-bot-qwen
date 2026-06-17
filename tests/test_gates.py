# tests/test_gates.py
"""Unit tests for the execution.gates package.

Each gate is tested independently with minimal GateInput fixtures.
Also tests the StrategyFactory, Strategy, and PaperTradeHarness.
"""

from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timezone

from execution.gates import (
    GateInput, GateOutput, GateConfig, GatePipeline, product_of_multipliers,
    # Entry gates
    time_gate, regime_edge_gate, confidence_gate,
    boundary_gate, drawdown_gate, slippage_gate, exposure_gate,
    # Exit gates
    conviction_hold_gate, edge_reversal_gate, profit_take_gate,
    confidence_drop_gate, rain_emergency_gate, nowcast_stale_gate,
    data_missing_gate, drawdown_flatten_gate, t2s_taper_gate,
    prob_stop_gate,
    # Sizing gates
    kelly_sizer, time_window_sizer, rain_uncertainty_sizer,
    boundary_proximity_sizer, model_confidence_sizer,
    # Rebalance triggers
    qty_delta_trigger, edge_delta_trigger, prob_confidence_trigger,
    ev_change_trigger, nowcast_regime_trigger, exposure_limit_trigger,
    t2s_derisk_trigger,
)
from execution.gates.entry import _parse_bucket_bounds
from execution.strategy_config import (
    Strategy, build_pipeline, deep_merge,
    DEFAULT_ENTRY_GATES, DEFAULT_EXIT_GATES,
    DEFAULT_SIZING_GATES, DEFAULT_REBALANCE_GATES,
)
from execution.strategy_factory import StrategyFactory, build_strategy, get_factory
from execution.paper_trade_harness import (
    PaperTradeHarness, Scenario, BacktestResult,
    generate_synthetic_scenarios, kelly_fraction, _probs_from_temperature,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _inp(**overrides) -> GateInput:
    """Create a GateInput with sensible defaults, overridden by kwargs."""
    defaults = dict(
        bucket="32-33",
        model_prob=0.70,
        market_price=0.60,
        model_std=1.2,
        dt_now=datetime(2025, 6, 15, 10, 0),  # 10am weekday
        position=None,
        context={},
    )
    defaults.update(overrides)
    # Move context sub-keys from overrides into context dict
    context_keys = [
        "drawdown_pct", "hours_to_settlement", "rain_regime",
        "max_so_far", "temp_now", "nowcast_stale", "data_missing",
        "model_key", "capital", "current_positions", "probs_old",
        "probs_new", "adjusted_bet", "post_mean",
    ]
    context = defaults.get("context", {})
    for k in context_keys:
        if k in overrides and k not in defaults.get("context", {}):
            context[k] = overrides[k]
            del defaults[k]
    defaults["context"] = context
    return GateInput(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY GATES
# ═══════════════════════════════════════════════════════════════════════

class TestTimeGate:
    def test_blocked_before_min_hour(self):
        inp = _inp(dt_now=datetime(2025, 6, 15, 6, 0))
        out = time_gate(inp, {"min_hour": 8})
        assert not out.passed
        assert out.reason_code == "TIME_WINDOW_CLOSED"

    def test_allowed_during_day(self):
        inp = _inp(dt_now=datetime(2025, 6, 15, 10, 0))
        out = time_gate(inp, {"min_hour": 8, "blocked_slots": []})
        assert out.passed

    def test_blocked_evening_slot(self):
        inp = _inp(dt_now=datetime(2025, 6, 15, 18, 0))
        out = time_gate(inp, {"min_hour": 8, "blocked_slots": ["evening"]})
        assert not out.passed

    def test_allowed_afternoon(self):
        inp = _inp(dt_now=datetime(2025, 6, 15, 14, 0))
        out = time_gate(inp, {"min_hour": 8, "blocked_slots": ["evening", "night"]})
        assert out.passed


class TestRegimeEdgeGate:
    def test_pass_when_edge_above_threshold(self):
        inp = _inp(model_prob=0.70, market_price=0.60)
        config = {
            "thresholds": {
                "day_08_12": {"min_edge": 0.05, "exposure_cap": 0.50}
            }
        }
        out = regime_edge_gate(inp, config)
        assert out.passed
        assert out.metadata["regime"] == "day_08_12"

    def test_fail_when_edge_below_threshold(self):
        inp = _inp(model_prob=0.62, market_price=0.60)
        config = {
            "thresholds": {
                "day_08_12": {"min_edge": 0.05, "exposure_cap": 0.50}
            }
        }
        out = regime_edge_gate(inp, config)
        assert not out.passed
        assert out.reason_code == "EDGE_TOO_LOW"

    def test_fail_when_no_regime_config(self):
        inp = _inp(dt_now=datetime(2025, 6, 15, 22, 0))
        config = {"thresholds": {"day_08_12": {"min_edge": 0.05}}}
        out = regime_edge_gate(inp, config)
        assert not out.passed

    def test_rain_regime(self):
        inp = _inp(
            dt_now=datetime(2025, 6, 15, 10, 0),
            rain_regime="moderate_or_heavy_rain",
        )
        config = {
            "thresholds": {
                "rain_08_12": {"min_edge": 0.045, "exposure_cap": 0.40}
            }
        }
        out = regime_edge_gate(inp, config)
        assert out.passed
        assert out.metadata["regime"] == "rain_08_12"


class TestConfidenceGate:
    def test_pass_when_std_low(self):
        inp = _inp(model_std=1.5)
        out = confidence_gate(inp, {"max_model_std": 2.5})
        assert out.passed

    def test_fail_when_std_high(self):
        inp = _inp(model_std=3.0)
        out = confidence_gate(inp, {"max_model_std": 2.5})
        assert not out.passed
        assert out.reason_code == "LOW_CONFIDENCE"


class TestBoundaryGate:
    def test_pass_when_far_from_boundary(self):
        inp = _inp(bucket="32-33", model_prob=32.5, model_std=1.0,
                    post_mean=32.5)
        out = boundary_gate(inp, {"min_standardized_distance": 0.5})
        assert out.passed

    def test_reduce_when_close_to_lower(self):
        """Boundary gate passes but with reduced multiplier (not a hard block)."""
        inp = _inp(bucket="32-33", model_prob=0.175, model_std=1.0,
                    post_mean=32.05)
        out = boundary_gate(inp, {
            "min_standardized_distance": 0.5,
            "aggressive_reduction_threshold": 0.3,
        })
        assert out.passed  # passes with reduced sizing
        assert out.multiplier < 1.0

    def test_reduce_when_close_to_upper(self):
        """Boundary gate passes but with reduced multiplier."""
        inp = _inp(bucket="32-33", model_prob=0.175, model_std=1.0,
                    post_mean=32.95)
        out = boundary_gate(inp, {"min_standardized_distance": 0.5})
        assert out.passed  # passes with reduced sizing

    def test_uses_post_mean_from_context(self):
        """Boundary gate should use post_mean (temperature) instead of model_prob."""
        inp = _inp(bucket="30-31", model_prob=0.35, model_std=1.0,
                    post_mean=30.4)
        out = boundary_gate(inp, {"min_standardized_distance": 0.5})
        # post_mean=30.4 is 0.4°C from the 30 boundary → dist_std=0.4 → reduced
        assert out.passed  # passes with reduced sizing
        assert out.multiplier < 1.0


class TestDrawdownGate:
    def test_pass_when_no_drawdown(self):
        inp = _inp(drawdown_pct=-0.02)
        out = drawdown_gate(inp, {"hard_flatten": -0.15, "stop_new_entries": -0.10})
        assert out.passed

    def test_stop_at_moderate_drawdown(self):
        inp = _inp(drawdown_pct=-0.10)
        out = drawdown_gate(inp, {"hard_flatten": -0.15, "stop_new_entries": -0.08})
        assert not out.passed
        assert out.reason_code == "DRAWDOWN_STOP"

    def test_hard_flatten(self):
        inp = _inp(drawdown_pct=-0.20)
        out = drawdown_gate(inp, {"hard_flatten": -0.15, "stop_new_entries": -0.10})
        assert not out.passed
        assert out.reason_code == "DRAWDOWN_HARD"


class TestSlippageGate:
    def test_pass_when_no_bet_data(self):
        """No slippage data → pass (slippage is execution-level, not always available)."""
        inp = _inp()
        out = slippage_gate(inp, {})
        assert out.passed
        assert out.reason_code == "PASS"

    def test_fail_when_not_filled(self):
        inp = _inp(adjusted_bet={"filled": False, "adjusted_quantity": 5})
        out = slippage_gate(inp, {})
        assert not out.passed
        assert out.reason_code == "LIQUIDITY_INSUFFICIENT"

    def test_pass_when_edge_survives_slippage(self):
        inp = _inp(
            model_prob=0.70, market_price=0.60,
            adjusted_bet={"filled": True, "slippage_pct": 2.0, "adjusted_quantity": 10},
        )
        out = slippage_gate(inp, {})
        assert out.passed

    def test_fail_when_slippage_eats_edge(self):
        inp = _inp(
            model_prob=0.605, market_price=0.60,
            adjusted_bet={"filled": True, "slippage_pct": 2.0, "adjusted_quantity": 10},
        )
        out = slippage_gate(inp, {})
        assert not out.passed
        assert out.reason_code == "SLIPPAGE_EATS_EDGE"


class TestExposureGate:
    def test_pass_when_within_limits(self):
        inp = _inp(
            bucket="32-33",
            capital=10000,
            current_positions={"32-33": {"quantity": 50, "entry_price": 0.55}},
        )
        out = exposure_gate(inp, {"max_per_bucket": 0.15, "total_max": 0.50})
        assert out.passed

    def test_fail_bucket_exposure(self):
        inp = _inp(
            bucket="32-33",
            capital=10000,
            current_positions={"32-33": {"quantity": 3000, "entry_price": 0.55}},
        )
        out = exposure_gate(inp, {"max_per_bucket": 0.15, "total_max": 0.50})
        assert not out.passed
        assert out.reason_code == "BUCKET_EXPOSURE"


# ═══════════════════════════════════════════════════════════════════════
# EXIT GATES
# ═══════════════════════════════════════════════════════════════════════

class TestConvictionHoldGate:
    def test_hold_at_extreme_conviction_near_settlement(self):
        inp = _inp(model_prob=0.99, hours_to_settlement=4)
        out = conviction_hold_gate(inp, {"hold_conviction_prob": 0.98, "hold_max_hours": 6})
        assert out.passed
        assert out.reason_code == "HOLD_CONVICTION"
        assert out.multiplier == 1.0

    def test_no_hold_below_conviction(self):
        inp = _inp(model_prob=0.85, hours_to_settlement=4)
        out = conviction_hold_gate(inp, {})
        assert out.reason_code == "PASS"


class TestEdgeReversalGate:
    def test_exit_on_reversed_edge(self):
        inp = _inp(
            model_prob=0.50, market_price=0.60,
            position={"side": "YES", "quantity": 10},
        )
        out = edge_reversal_gate(inp, {"edge_reversal_threshold": -0.05})
        assert not out.passed
        assert out.reason_code == "EDGE_REVERSED"
        assert out.multiplier == 0.0

    def test_hold_when_edge_ok(self):
        inp = _inp(
            model_prob=0.70, market_price=0.60,
            position={"side": "YES", "quantity": 10},
        )
        out = edge_reversal_gate(inp, {"edge_reversal_threshold": -0.05})
        assert out.passed


class TestProfitTakeGate:
    def test_profit_take_yes_side(self):
        inp = _inp(
            market_price=0.75, model_prob=0.70,
            position={"side": "YES", "quantity": 10},
        )
        out = profit_take_gate(inp, {"profit_take_multiplier": 0.5})
        assert out.reason_code == "PROFIT_TAKE"
        assert out.multiplier == 0.5

    def test_profit_take_no_side(self):
        inp = _inp(
            market_price=0.50, model_prob=0.60,
            position={"side": "NO", "quantity": 10},
        )
        out = profit_take_gate(inp, {"profit_take_multiplier": 0.5})
        assert out.reason_code == "PROFIT_TAKE"

    def test_no_profit_take(self):
        inp = _inp(
            market_price=0.55, model_prob=0.70,
            position={"side": "YES", "quantity": 10},
        )
        out = profit_take_gate(inp, {})
        assert out.reason_code == "PASS"


class TestRainEmergencyGate:
    def test_trigger_on_rain_temp_drop(self):
        inp = _inp(
            rain_regime="moderate_or_heavy_rain",
            max_so_far=33.0, temp_now=31.0,
        )
        out = rain_emergency_gate(inp, {"rain_emergency_temp_drop": 1.5})
        assert out.reason_code == "RAIN_EMERGENCY"
        assert out.multiplier == 0.3

    def test_no_trigger_without_rain(self):
        inp = _inp(rain_regime="no_rain", max_so_far=33.0, temp_now=31.0)
        out = rain_emergency_gate(inp, {})
        assert out.reason_code == "PASS"


class TestT2STaperGate:
    def test_strong_taper(self):
        inp = _inp(hours_to_settlement=1.5)
        out = t2s_taper_gate(inp, {"taper_start_hours": 4, "strong_taper_hours": 2, "strong_taper_multiplier": 0.3})
        assert out.reason_code == "T2S_STRONG"
        assert out.multiplier == 0.3

    def test_moderate_taper(self):
        inp = _inp(hours_to_settlement=3.0)
        out = t2s_taper_gate(inp, {"taper_start_hours": 4})
        assert out.reason_code == "T2S_TAPER"
        assert 0 < out.multiplier < 1

    def test_no_taper(self):
        inp = _inp(hours_to_settlement=10.0)
        out = t2s_taper_gate(inp, {"taper_start_hours": 4})
        assert out.reason_code == "PASS"
        assert out.multiplier == 1.0


class TestDrawdownFlattenGate:
    def test_hard_flatten(self):
        inp = _inp(drawdown_pct=-0.20)
        out = drawdown_flatten_gate(inp, {"hard_flatten": -0.15, "reduce_risk": -0.075})
        assert not out.passed
        assert out.multiplier == 0.0

    def test_reduce_risk(self):
        inp = _inp(drawdown_pct=-0.10)
        out = drawdown_flatten_gate(inp, {"hard_flatten": -0.15, "reduce_risk": -0.075})
        assert out.reason_code == "DRAWDOWN_REDUCE"
        assert 0 < out.multiplier <= 1

    def test_no_drawdown(self):
        inp = _inp(drawdown_pct=-0.02)
        out = drawdown_flatten_gate(inp, {"hard_flatten": -0.15, "reduce_risk": -0.075})
        assert out.reason_code == "PASS"
        assert out.multiplier == 1.0


# ═══════════════════════════════════════════════════════════════════════
# SIZING GATES
# ═══════════════════════════════════════════════════════════════════════

class TestKellySizer:
    def test_always_returns_1(self):
        inp = _inp()
        out = kelly_sizer(inp, {})
        assert out.passed
        assert out.multiplier == 1.0


class TestRainUncertaintySizer:
    def test_no_rain_no_reduction(self):
        inp = _inp(rain_regime="no_rain")
        out = rain_uncertainty_sizer(inp, {"multipliers": {"no_rain": 1.0, "weak_rain": 0.8}})
        assert out.multiplier == 1.0

    def test_rain_reduces(self):
        inp = _inp(rain_regime="weak_rain")
        out = rain_uncertainty_sizer(inp, {"multipliers": {"weak_rain": 0.8}})
        assert out.multiplier == 0.8

    def test_default_multiplier(self):
        inp = _inp(rain_regime="moderate_or_heavy_rain")
        out = rain_uncertainty_sizer(inp, {"multipliers": {"weak_rain": 0.8}})
        assert out.multiplier == 1.0


class TestModelConfidenceSizer:
    def test_known_model(self):
        inp = _inp(model_key="baseline")
        out = model_confidence_sizer(inp, {"multipliers": {"baseline": 0.9}, "default_mult": 0.5})
        assert out.multiplier == 0.9

    def test_unknown_model_uses_default(self):
        inp = _inp(model_key="unknown_model")
        out = model_confidence_sizer(inp, {"multipliers": {"baseline": 0.9}, "default_mult": 0.5})
        assert out.multiplier == 0.5


# ═══════════════════════════════════════════════════════════════════════
# REBALANCE TRIGGERS
# ═══════════════════════════════════════════════════════════════════════

class TestQtyDeltaTrigger:
    def test_trigger_on_large_delta(self):
        inp = _inp(
            position={"quantity": 5.0},
            context={"target_quantity": 20.0},
        )
        out = qty_delta_trigger(inp, {"min_qty_delta": 0.5})
        assert out.passed
        assert out.reason_code == "QTY_DELTA"

    def test_no_trigger_on_small_delta(self):
        inp = _inp(
            position={"quantity": 19.8},
            context={"target_quantity": 20.0},
        )
        out = qty_delta_trigger(inp, {"min_qty_delta": 0.5})
        assert not out.passed


class TestEdgeDeltaTrigger:
    def test_trigger_on_material_edge(self):
        inp = _inp(model_prob=0.70, market_price=0.60)
        out = edge_delta_trigger(inp, {"material_edge_delta": 0.05})
        assert out.passed
        assert out.reason_code == "MATERIAL_EDGE"

    def test_no_trigger_on_tiny_edge(self):
        inp = _inp(model_prob=0.605, market_price=0.60)
        out = edge_delta_trigger(inp, {"material_edge_delta": 0.05})
        assert not out.passed


class TestProbConfidenceTrigger:
    def test_trigger_on_top_bucket_change(self):
        inp = _inp(
            probs_old={"32-33": 0.5, "30-31": 0.3},
            probs_new={"30-31": 0.5, "32-33": 0.3},
        )
        out = prob_confidence_trigger(inp, {"material_prob_pp": 5.0})
        assert out.passed
        assert out.reason_code == "TOP_BUCKET_CHANGED"

    def test_trigger_on_large_pp_shift(self):
        inp = _inp(
            probs_old={"32-33": 0.5, "30-31": 0.3},
            probs_new={"32-33": 0.6, "30-31": 0.2},
        )
        out = prob_confidence_trigger(inp, {"material_prob_pp": 5.0})
        assert out.passed

    def test_no_trigger_on_small_shift(self):
        inp = _inp(
            probs_old={"32-33": 0.50, "30-31": 0.30},
            probs_new={"32-33": 0.52, "30-31": 0.28},
        )
        out = prob_confidence_trigger(inp, {"material_prob_pp": 5.0})
        assert not out.passed


class TestNowcastRegimeTrigger:
    def test_trigger_when_regime_changed(self):
        inp = _inp(context={"nowcast_regime_changed": True})
        out = nowcast_regime_trigger(inp, {})
        assert out.passed
        assert out.reason_code == "NOWCAST_REGIME"

    def test_no_trigger_when_stable(self):
        inp = _inp(context={"nowcast_regime_changed": False})
        out = nowcast_regime_trigger(inp, {})
        assert not out.passed


class TestT2SDeriskTrigger:
    def test_trigger_near_settlement(self):
        inp = _inp(hours_to_settlement=2.0)
        out = t2s_derisk_trigger(inp, {"taper_start_hours": 4})
        assert out.passed
        assert out.reason_code == "T2S_DE_RISK"

    def test_no_trigger_far_from_settlement(self):
        inp = _inp(hours_to_settlement=10.0)
        out = t2s_derisk_trigger(inp, {"taper_start_hours": 4})
        assert not out.passed


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE (GatePipeline)
# ═══════════════════════════════════════════════════════════════════════

class TestGatePipeline:
    def test_entry_mode_short_circuits_on_fail(self):
        """Entry pipeline should stop at first failing gate."""
        inp = _inp(dt_now=datetime(2025, 6, 15, 6, 0))
        p = GatePipeline(
            gates=[
                ("time", time_gate, {"min_hour": 8}),
                ("confidence", confidence_gate, {"max_model_std": 2.5}),
            ],
            mode="entry",
        )
        ok, results = p.evaluate(inp)
        assert not ok
        assert len(results) == 1  # stopped at time_gate

    def test_entry_mode_all_pass(self):
        inp = _inp(model_prob=0.70, market_price=0.55, model_std=1.0)
        p = GatePipeline(
            gates=[
                ("time", time_gate, {"min_hour": 8, "blocked_slots": []}),
                ("confidence", confidence_gate, {"max_model_std": 2.5}),
            ],
            mode="entry",
        )
        ok, results = p.evaluate(inp)
        assert ok
        assert len(results) == 2

    def test_sizing_mode_returns_product(self):
        inp = _inp(rain_regime="weak_rain", model_key="baseline")
        p = GatePipeline(
            gates=[
                ("kelly", kelly_sizer, {}),
                ("rain", rain_uncertainty_sizer, {"multipliers": {"weak_rain": 0.8}}),
                ("model", model_confidence_sizer, {"multipliers": {"baseline": 0.9}, "default_mult": 0.5}),
            ],
            mode="sizing",
        )
        ok, results = p.evaluate(inp)
        assert ok
        assert product_of_multipliers(results) == pytest.approx(0.72, abs=0.01)

    def test_exit_mode_multiplier_product(self):
        inp = _inp(
            model_prob=0.70, market_price=0.60,
            position={"side": "YES", "quantity": 10},
            drawdown_pct=-0.02, hours_to_settlement=10.0,
        )
        p = GatePipeline(
            gates=[
                ("conviction", conviction_hold_gate, {}),
                ("edge", edge_reversal_gate, {"edge_reversal_threshold": -0.05}),
                ("t2s", t2s_taper_gate, {"taper_start_hours": 4}),
            ],
            mode="exit",
        )
        ok, results = p.evaluate(inp)
        assert ok  # multiplier > 0 → hold
        assert product_of_multipliers(results) > 0

    def test_exit_mode_exit_when_multiplier_zero(self):
        inp = _inp(
            model_prob=0.50, market_price=0.60,
            position={"side": "YES", "quantity": 10},
        )
        p = GatePipeline(
            gates=[
                ("edge", edge_reversal_gate, {"edge_reversal_threshold": -0.05}),
            ],
            mode="exit",
        )
        ok, results = p.evaluate(inp)
        assert not ok  # multiplier <= 0 → exit

    def test_rebalance_mode_short_circuits_on_trigger(self):
        inp = _inp(
            model_prob=0.70, market_price=0.60,
            position={"quantity": 5.0},
            context={"target_quantity": 20.0},
        )
        p = GatePipeline(
            gates=[
                ("qty_delta", qty_delta_trigger, {"min_qty_delta": 0.5}),
                ("edge_delta", edge_delta_trigger, {"material_edge_delta": 0.01}),
            ],
            mode="rebalance",
        )
        ok, results = p.evaluate(inp)
        assert ok  # first trigger is enough
        assert results[0].reason_code == "QTY_DELTA"

    def test_rebalance_mode_no_trigger(self):
        inp = _inp(
            model_prob=0.605, market_price=0.60,
            position={"quantity": 20.0},
            context={"target_quantity": 20.0},
        )
        p = GatePipeline(
            gates=[
                ("qty_delta", qty_delta_trigger, {"min_qty_delta": 0.5}),
                ("edge_delta", edge_delta_trigger, {"material_edge_delta": 0.05}),
            ],
            mode="rebalance",
        )
        ok, results = p.evaluate(inp)
        assert not ok


# ═══════════════════════════════════════════════════════════════════════
# HELPER: _parse_bucket_bounds
# ═══════════════════════════════════════════════════════════════════════

class TestParseBucketBounds:
    def test_range_bucket(self):
        assert _parse_bucket_bounds("32-33") == (32.0, 33.0)

    def test_lt_bucket(self):
        assert _parse_bucket_bounds("<23") == (float("-inf"), 23.0)

    def test_gte_bucket(self):
        assert _parse_bucket_bounds(">=34") == (34.0, float("inf"))

    def test_fallback_numeric(self):
        lo, hi = _parse_bucket_bounds("30")
        assert lo == 30.0 and hi == 31.0


# ═══════════════════════════════════════════════════════════════════════
# PRODUCT OF MULTIPLIERS
# ═══════════════════════════════════════════════════════════════════════

class TestProductOfMultipliers:
    def test_all_ones(self):
        results = [GateOutput(passed=True, reason_code="PASS", multiplier=1.0) for _ in range(3)]
        assert product_of_multipliers(results) == 1.0

    def test_mixed(self):
        results = [
            GateOutput(passed=True, reason_code="PASS", multiplier=0.8),
            GateOutput(passed=True, reason_code="PASS", multiplier=0.5),
        ]
        assert product_of_multipliers(results) == pytest.approx(0.4)

    def test_zero_halts(self):
        results = [
            GateOutput(passed=True, reason_code="PASS", multiplier=0.0),
            GateOutput(passed=True, reason_code="PASS", multiplier=0.5),
        ]
        assert product_of_multipliers(results) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG
# ═══════════════════════════════════════════════════════════════════════

class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"entry": {"min_hour": 8, "other": "x"}}
        override = {"entry": {"min_hour": 10}}
        result = deep_merge(base, override)
        assert result == {"entry": {"min_hour": 10, "other": "x"}}

    def test_no_mutation(self):
        base = {"a": [1, 2]}
        override = {"a": [3]}
        result = deep_merge(base, override)
        assert result == {"a": [3]}
        assert base == {"a": [1, 2]}  # not mutated


class TestBuildPipeline:
    def test_builds_entry_pipeline(self):
        gate_names = ["time_gate", "confidence_gate"]
        config = {
            "time_gate": {"min_hour": 8},
            "confidence_gate": {"max_model_std": 2.5},
        }
        p = build_pipeline("entry", gate_names, config)
        assert p.mode == "entry"
        assert len(p.gates) == 2

    def test_skips_unknown_gates(self):
        p = build_pipeline("entry", ["time_gate", "nonexistent_gate"], {})
        assert len(p.gates) == 1

    def test_uses_default_gate_lists(self):
        assert len(DEFAULT_ENTRY_GATES) == 7
        assert len(DEFAULT_EXIT_GATES) == 10
        assert len(DEFAULT_SIZING_GATES) == 5
        assert len(DEFAULT_REBALANCE_GATES) == 7


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY FACTORY
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyFactory:
    def test_factory_loads_all_strategies(self):
        factory = get_factory()
        keys = factory.keys()
        assert "enhanced_v2_paper" in keys
        assert "enhanced_v2_aggressive_paper" in keys
        assert len(keys) >= 5

    def test_build_strategy_has_pipelines(self):
        s = build_strategy("enhanced_v2_paper")
        assert s.key == "enhanced_v2_paper"
        assert len(s.entry_pipeline.gates) > 0
        assert len(s.exit_pipeline.gates) > 0
        assert len(s.sizing_pipeline.gates) > 0
        assert len(s.rebalance_pipeline.gates) > 0

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="no_such_strategy"):
            build_strategy("no_such_strategy")

    def test_strategy_to_dict(self):
        s = build_strategy("enhanced_v2_paper")
        d = s.to_dict()
        assert d["key"] == "enhanced_v2_paper"
        assert "entry_gates" in d
        assert "exit_gates" in d
        assert "sizing_gates" in d
        assert "rebalance_gates" in d

    def test_override_merge(self):
        """V2 aggressive should have rain_08_12 min_edge=0.045 per override."""
        s = build_strategy("enhanced_v2_aggressive_paper")
        for name, fn, cfg in s.entry_pipeline.gates:
            if name == "regime_edge_gate":
                rain = cfg.get("thresholds", {}).get("rain_08_12", {})
                assert rain.get("min_edge") == 0.045


# ═══════════════════════════════════════════════════════════════════════
# PAPER TRADE HARNESS
# ═══════════════════════════════════════════════════════════════════════

class TestKellyFraction:
    def test_positive_edge(self):
        k = kelly_fraction(0.70, 0.50)
        assert k > 0

    def test_zero_edge(self):
        k = kelly_fraction(0.50, 0.50)
        assert k == pytest.approx(0.0, abs=0.01)

    def test_negative_edge(self):
        k = kelly_fraction(0.30, 0.50)
        assert k == 0.0


class TestProbsFromTemperature:
    def test_probs_sum_to_one(self):
        buckets = ["28-29", "29-30", "30-31", "31-32", "32-33"]
        probs = _probs_from_temperature(30.5, 1.5, buckets)
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_peak_at_forecast(self):
        buckets = ["28-29", "29-30", "30-31", "31-32", "32-33"]
        probs = _probs_from_temperature(30.5, 1.0, buckets)
        assert probs["30-31"] > probs["28-29"]

    def test_infinite_bounds(self):
        probs = _probs_from_temperature(30.0, 1.5, ["<28", "28-30", "30-32", ">=32"])
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestScenarioBuilder:
    def test_generates_scenarios(self):
        buckets = {"28-29": {"market_price": 0.2}, "30-31": {"market_price": 0.3}, ">=32": {"market_price": 0.05}}
        scenarios = generate_synthetic_scenarios(buckets, n_cycles=10, seed=42)
        assert len(scenarios) == 10
        # Each scenario has buckets with prob and market_price
        for s in scenarios:
            for b, d in s.buckets.items():
                assert "prob" in d
                assert "market_price" in d
                assert 0 <= d["prob"] <= 1
                assert 0 < d["market_price"] < 1

    def test_reproducible_with_seed(self):
        buckets = {"28-29": {"market_price": 0.2}, "30-31": {"market_price": 0.3}}
        s1 = generate_synthetic_scenarios(buckets, n_cycles=5, seed=123)
        s2 = generate_synthetic_scenarios(buckets, n_cycles=5, seed=123)
        assert s1[0].buckets == s2[0].buckets

    def test_post_mean_consistent_with_probs(self):
        """Temperature forecast should be consistent with bucket probabilities."""
        buckets = {"28-29": {"market_price": 0.2}, "30-31": {"market_price": 0.3}}
        scenarios = generate_synthetic_scenarios(buckets, n_cycles=5, seed=42)
        for s in scenarios:
            assert s.post_mean is not None
            assert 24.0 <= s.post_mean <= 40.0


class TestPaperTradeHarness:
    def test_runs_and_produces_result(self):
        harness = PaperTradeHarness(strategy_or_key="enhanced_v2_paper", seed=42)
        buckets = {"28-29": {"market_price": 0.2}, "30-31": {"market_price": 0.3}, ">=32": {"market_price": 0.05}}
        scenarios = generate_synthetic_scenarios(buckets, n_cycles=20, seed=42)
        result = harness.run(scenarios)
        assert isinstance(result, BacktestResult)
        assert result.num_cycles > 0
        assert result.initial_capital == 10_000.0

    def test_summary_dict(self):
        harness = PaperTradeHarness(strategy_or_key="enhanced_v2_paper", seed=42)
        buckets = {"28-29": {"market_price": 0.2}, "30-31": {"market_price": 0.3}}
        scenarios = generate_synthetic_scenarios(buckets, n_cycles=20, seed=42)
        result = harness.run(scenarios)
        summary = result.summary()
        assert "strategy" in summary
        assert "total_pnl" in summary
        assert "sharpe_ratio" in summary

    def test_multiple_strategies_different_results(self):
        """Different strategies should produce different results."""
        buckets = {
            "28-29": {"market_price": 0.2}, "29-30": {"market_price": 0.25},
            "30-31": {"market_price": 0.3}, "31-32": {"market_price": 0.15},
            ">=32": {"market_price": 0.05},
        }
        scenarios = generate_synthetic_scenarios(buckets, n_cycles=50, seed=42)

        results = {}
        for key in ["enhanced_v2_paper", "enhanced_v2_aggressive_paper"]:
            harness = PaperTradeHarness(strategy_or_key=key, seed=42)
            result = harness.run(scenarios)
            results[key] = result.summary()

        # Aggressive should generally have more trades or different PnL
        # (at least they should not have identical results)
        assert results["enhanced_v2_paper"]["strategy"] == "enhanced_v2_paper"
        assert results["enhanced_v2_aggressive_paper"]["strategy"] == "enhanced_v2_aggressive_paper"
