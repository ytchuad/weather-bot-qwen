# execution/strategy_config.py
"""Strategy configuration dataclasses and pipeline composition.

A Strategy is a composition of four gate pipelines (entry, exit, sizing,
rebalance) plus metadata like label, description, and kelly_fraction.

The factory (strategy_factory.py) builds Strategy objects from JSON config,
resolving `extends` chains and mapping gate names to function references.
"""

from __future__ import annotations

import json
import logging
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from execution.gates import GatePipeline, GateFn, GateConfig

logger = logging.getLogger(__name__)


# ── Gate Registry ─────────────────────────────────────────────────────

# Maps gate name (string) → gate function reference.
# Populated by strategy_factory at import time.
GATE_REGISTRY: dict[str, GateFn] = {}


def register_gate(name: str, fn: GateFn) -> None:
    """Register a gate function by name."""
    GATE_REGISTRY[name] = fn


def _bootstrap_registry() -> None:
    """Populate GATE_REGISTRY with all built-in gates."""
    if GATE_REGISTRY:
        return  # already bootstrapped

    from execution.gates import (
        # Entry
        time_gate, regime_edge_gate, confidence_gate,
        boundary_gate, drawdown_gate, slippage_gate, exposure_gate,
        # Exit
        conviction_hold_gate, edge_reversal_gate, profit_take_gate,
        confidence_drop_gate, rain_emergency_gate, nowcast_stale_gate,
        data_missing_gate, drawdown_flatten_gate, t2s_taper_gate,
        prob_stop_gate,
        # Sizing
        kelly_sizer, time_window_sizer, rain_uncertainty_sizer,
        boundary_proximity_sizer, model_confidence_sizer,
        # Rebalance
        qty_delta_trigger, edge_delta_trigger, prob_confidence_trigger,
        ev_change_trigger, nowcast_regime_trigger, exposure_limit_trigger,
        t2s_derisk_trigger,
    )

    mapping = {
        "time_gate": time_gate,
        "regime_edge_gate": regime_edge_gate,
        "confidence_gate": confidence_gate,
        "boundary_gate": boundary_gate,
        "drawdown_gate": drawdown_gate,
        "slippage_gate": slippage_gate,
        "exposure_gate": exposure_gate,
        "conviction_hold_gate": conviction_hold_gate,
        "edge_reversal_gate": edge_reversal_gate,
        "profit_take_gate": profit_take_gate,
        "confidence_drop_gate": confidence_drop_gate,
        "rain_emergency_gate": rain_emergency_gate,
        "nowcast_stale_gate": nowcast_stale_gate,
        "data_missing_gate": data_missing_gate,
        "drawdown_flatten_gate": drawdown_flatten_gate,
        "t2s_taper_gate": t2s_taper_gate,
        "prob_stop_gate": prob_stop_gate,
        "kelly_sizer": kelly_sizer,
        "time_window_sizer": time_window_sizer,
        "rain_uncertainty_sizer": rain_uncertainty_sizer,
        "boundary_proximity_sizer": boundary_proximity_sizer,
        "model_confidence_sizer": model_confidence_sizer,
        "qty_delta_trigger": qty_delta_trigger,
        "edge_delta_trigger": edge_delta_trigger,
        "prob_confidence_trigger": prob_confidence_trigger,
        "ev_change_trigger": ev_change_trigger,
        "nowcast_regime_trigger": nowcast_regime_trigger,
        "exposure_limit_trigger": exposure_limit_trigger,
        "t2s_derisk_trigger": t2s_derisk_trigger,
    }
    GATE_REGISTRY.update(mapping)


_bootstrap_registry()


# ── Strategy Dataclass ────────────────────────────────────────────────

@dataclass
class Strategy:
    """A composable strategy built from gate pipelines.

    Attributes:
        key: Unique strategy identifier (e.g. 'enhanced_v2_paper').
        label: Human-readable label.
        description: One-line description.
        paper_only: If True, only runs in paper mode.
        entry_pipeline: GatePipeline for entry decisions.
        exit_pipeline: GatePipeline for exit/reduce decisions.
        sizing_pipeline: GatePipeline for position sizing.
        rebalance_pipeline: GatePipeline for rebalance triggers.
        kelly_fraction: Fraction of Kelly criterion to use.
        entry_rules: Additional entry rules (min_hour, only_on_event_date).
        exposure_limits: Per-slot exposure caps (legacy, for V1 compat).
    """

    key: str
    label: str = ""
    description: str = ""
    paper_only: bool = True
    entry_pipeline: GatePipeline = field(default_factory=lambda: GatePipeline(gates=[], mode="entry"))
    exit_pipeline: GatePipeline = field(default_factory=lambda: GatePipeline(gates=[], mode="exit"))
    sizing_pipeline: GatePipeline = field(default_factory=lambda: GatePipeline(gates=[], mode="sizing"))
    rebalance_pipeline: GatePipeline = field(default_factory=lambda: GatePipeline(gates=[], mode="rebalance"))
    kelly_fraction: float = 0.25
    entry_rules: dict = field(default_factory=dict)
    exposure_limits: dict | None = None

    def evaluate_entry(self, inp) -> tuple[bool, list]:
        """Run entry pipeline. Returns (passed, gate_outputs)."""
        return self.entry_pipeline.evaluate(inp)

    def evaluate_exit(self, inp) -> tuple[bool, list]:
        """Run exit pipeline. Returns (hold, gate_outputs)."""
        return self.exit_pipeline.evaluate(inp)

    def evaluate_sizing(self, inp) -> tuple[bool, list]:
        """Run sizing pipeline. Returns (True, gate_outputs). Multiplier product = final size factor."""
        return self.sizing_pipeline.evaluate(inp)

    def evaluate_rebalance(self, inp) -> tuple[bool, list]:
        """Run rebalance pipeline. Returns (triggered, gate_outputs)."""
        return self.rebalance_pipeline.evaluate(inp)

    def size_factor(self, inp) -> float:
        """Convenience: run sizing pipeline and return the product of multipliers."""
        _, results = self.sizing_pipeline.evaluate(inp)
        from execution.gates.pipeline import product_of_multipliers
        return product_of_multipliers(results)

    def to_dict(self) -> dict:
        """Serialize strategy metadata (excluding pipeline callbacks)."""
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "paper_only": self.paper_only,
            "kelly_fraction": self.kelly_fraction,
            "entry_rules": self.entry_rules,
            "exposure_limits": self.exposure_limits,
            "entry_gates": [name for name, _, _ in self.entry_pipeline.gates],
            "exit_gates": [name for name, _, _ in self.exit_pipeline.gates],
            "sizing_gates": [name for name, _, _ in self.sizing_pipeline.gates],
            "rebalance_gates": [name for name, _, _ in self.rebalance_pipeline.gates],
        }


# ── Pipeline Builder ──────────────────────────────────────────────────

# Default gate lists for each pipeline mode. Used when a strategy
# doesn't specify its own gate list (i.e. inherits from defaults).

DEFAULT_ENTRY_GATES = [
    "time_gate",
    "regime_edge_gate",
    "confidence_gate",
    "boundary_gate",
    "drawdown_gate",
    "slippage_gate",
    "exposure_gate",
]

DEFAULT_EXIT_GATES = [
    "conviction_hold_gate",
    "edge_reversal_gate",
    "profit_take_gate",
    "confidence_drop_gate",
    "rain_emergency_gate",
    "nowcast_stale_gate",
    "data_missing_gate",
    "drawdown_flatten_gate",
    "t2s_taper_gate",
    "prob_stop_gate",
]

DEFAULT_SIZING_GATES = [
    "kelly_sizer",
    "time_window_sizer",
    "rain_uncertainty_sizer",
    "boundary_proximity_sizer",
    "model_confidence_sizer",
]

DEFAULT_REBALANCE_GATES = [
    "qty_delta_trigger",
    "edge_delta_trigger",
    "prob_confidence_trigger",
    "ev_change_trigger",
    "nowcast_regime_trigger",
    "exposure_limit_trigger",
    "t2s_derisk_trigger",
]


def build_pipeline(
    mode: str,
    gate_names: list[str],
    config: dict,
) -> GatePipeline:
    """Build a GatePipeline from a list of gate name strings and a config dict.

    Config is a nested dict like:
        {
            "time_gate": {"min_hour": 8, "blocked_slots": ["evening", "night"]},
            "regime_edge_gate": {"thresholds": {...}},
            ...
        }

    Each gate gets its own sub-dict from the config, keyed by gate name.
    """
    _bootstrap_registry()
    gates: list[tuple[str, GateFn, GateConfig]] = []
    for name in gate_names:
        fn = GATE_REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown gate '%s' — skipping", name)
            continue
        gate_config = config.get(name, {})
        gates.append((name, fn, gate_config))
    return GatePipeline(gates=gates, mode=mode)


# ── Config Deep Merge ─────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result
