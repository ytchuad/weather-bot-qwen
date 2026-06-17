# execution/gates/__init__.py
"""Pluggable gate framework for strategy entry/exit/sizing/rebalance decisions.

Each gate is a standalone function with the standard signature:

    (GateInput, GateConfig) -> GateOutput

Gates are composed into pipelines via GatePipeline, which runs them
sequentially and short-circuits on the first blocking gate.
"""

from execution.gates.types import GateInput, GateOutput, GateConfig, GateFn
from execution.gates.pipeline import GatePipeline, product_of_multipliers

# Entry gates
from execution.gates.entry import (
    time_gate,
    regime_edge_gate,
    confidence_gate,
    boundary_gate,
    drawdown_gate,
    slippage_gate,
    exposure_gate,
)

# Exit gates
from execution.gates.exit import (
    conviction_hold_gate,
    edge_reversal_gate,
    profit_take_gate,
    confidence_drop_gate,
    rain_emergency_gate,
    nowcast_stale_gate,
    data_missing_gate,
    drawdown_flatten_gate,
    t2s_taper_gate,
    prob_stop_gate,
)

# Sizing gates
from execution.gates.sizing import (
    kelly_sizer,
    time_window_sizer,
    rain_uncertainty_sizer,
    boundary_proximity_sizer,
    model_confidence_sizer,
)

# Rebalance triggers
from execution.gates.rebalance import (
    qty_delta_trigger,
    edge_delta_trigger,
    prob_confidence_trigger,
    ev_change_trigger,
    nowcast_regime_trigger,
    exposure_limit_trigger,
    t2s_derisk_trigger,
)

__all__ = [
    # Core types
    "GateInput", "GateOutput", "GateConfig", "GateFn",
    "GatePipeline", "product_of_multipliers",
    # Entry gates
    "time_gate", "regime_edge_gate", "confidence_gate",
    "boundary_gate", "drawdown_gate", "slippage_gate", "exposure_gate",
    # Exit gates
    "conviction_hold_gate", "edge_reversal_gate", "profit_take_gate",
    "confidence_drop_gate", "rain_emergency_gate", "nowcast_stale_gate",
    "data_missing_gate", "drawdown_flatten_gate", "t2s_taper_gate",
    "prob_stop_gate",
    # Sizing gates
    "kelly_sizer", "time_window_sizer", "rain_uncertainty_sizer",
    "boundary_proximity_sizer", "model_confidence_sizer",
    # Rebalance triggers
    "qty_delta_trigger", "edge_delta_trigger", "prob_confidence_trigger",
    "ev_change_trigger", "nowcast_regime_trigger", "exposure_limit_trigger",
    "t2s_derisk_trigger",
]
