# execution/strategy_factory.py
"""StrategyFactory — build Strategy objects from V2 config (or new V3 JSON).

Reads ``config/paper_strategies.json`` (or a V3 successor), merges the shared
``defaults`` with per-strategy ``override`` blocks, and instantiates the four
GatePipelines (entry / exit / sizing / rebalance) that a Strategy needs.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from execution.gates.pipeline import GatePipeline
from execution.strategy_config import (
    Strategy,
    build_pipeline,
    deep_merge,
    DEFAULT_ENTRY_GATES,
    DEFAULT_EXIT_GATES,
    DEFAULT_SIZING_GATES,
    DEFAULT_REBALANCE_GATES,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/paper_strategies.json")


# ── Config-shape mappers ─────────────────────────────────────────────

def _map_entry_config(raw: dict) -> dict:
    """Convert old V2 ``entry`` block into per-gate config dict."""
    cfg: dict[str, Any] = {}

    cfg["time_gate"] = {
        "min_hour": raw.get("min_hour", 8),
        "blocked_slots": ["evening", "night"],
    }

    thresholds = raw.get("regime_thresholds", {})
    if thresholds:
        cfg["regime_edge_gate"] = {"thresholds": thresholds}

    prob_conf = raw.get("probability_confidence", {})
    if prob_conf:
        cfg["confidence_gate"] = prob_conf
    elif "max_model_std" in raw:
        cfg["confidence_gate"] = {"max_model_std": raw["max_model_std"]}

    bound = raw.get("boundary_proximity", {})
    if bound:
        cfg["boundary_gate"] = bound
        cfg["boundary_proximity_sizer"] = bound

    dd = raw.get("drawdown", {})
    if dd:
        cfg["drawdown_gate"] = dd
        cfg["drawdown_flatten_gate"] = dd

    return cfg


def _map_exit_config(raw: dict) -> dict:
    """Convert old V2 ``exit`` block into per-gate config dict."""
    cfg: dict[str, Any] = {}

    if "hold_conviction_prob" in raw:
        cfg["conviction_hold_gate"] = {
            "hold_conviction_prob": raw["hold_conviction_prob"],
            "hold_max_hours": raw.get("hold_max_hours", 6),
        }
    if "edge_reversal_threshold" in raw:
        cfg["edge_reversal_gate"] = {"edge_reversal_threshold": raw["edge_reversal_threshold"]}
    if "profit_take_multiplier" in raw:
        cfg["profit_take_gate"] = {"profit_take_multiplier": raw["profit_take_multiplier"]}
    if "stop_prob" in raw:
        cfg["prob_stop_gate"] = {"stop_prob": raw["stop_prob"]}

    tts = raw.get("time_to_settlement", {})
    if tts:
        cfg["t2s_taper_gate"] = tts
        cfg["t2s_derisk_trigger"] = tts

    dd = raw.get("drawdown", {})
    if dd:
        cfg["drawdown_flatten_gate"] = dd
        cfg["drawdown_gate"] = dd

    rain = raw.get("rain_emergency_temp_drop")
    if rain is not None:
        cfg["rain_emergency_gate"] = {"rain_emergency_temp_drop": rain}

    return cfg


def _map_sizing_config(raw: dict) -> dict:
    """Convert old V2 ``position_sizing`` block into per-gate config dict."""
    cfg: dict[str, Any] = {}

    tw = raw.get("time_window_multiplier", [])
    if tw:
        cfg["time_window_sizer"] = {"multipliers": tw}

    rw = raw.get("rain_uncertainty_multiplier", {})
    if rw:
        cfg["rain_uncertainty_sizer"] = {"multipliers": rw}

    bp = raw.get("boundary_proximity", {})
    if bp:
        cfg["boundary_proximity_sizer"] = bp

    max_b = raw.get("max_per_bucket")
    total_m = raw.get("total_max")
    if max_b is not None or total_m is not None:
        cfg["exposure_gate"] = {
            "max_per_bucket": max_b if max_b is not None else 0.15,
            "total_max": total_m if total_m is not None else 0.50,
        }

    return cfg


def _map_rebalance_config(raw: dict) -> dict:
    """Convert old V2 ``rebalance`` block into per-gate config dict."""
    cfg: dict[str, Any] = {}

    mqd = raw.get("min_qty_delta")
    if mqd is not None:
        cfg["qty_delta_trigger"] = {"min_qty_delta": mqd}

    med = raw.get("material_edge_delta")
    if med is not None:
        cfg["edge_delta_trigger"] = {"material_edge_delta": med}

    mpp = raw.get("material_prob_pp")
    if mpp is not None:
        cfg["prob_confidence_trigger"] = {"material_prob_pp": mpp}

    mev = raw.get("material_ev_pp")
    if mev is not None:
        cfg["ev_change_trigger"] = {"material_ev_pp": mev}

    return cfg


def build_gate_config(merged: dict) -> dict:
    """Flat per-gate config dict from a fully-merged strategy definition."""
    cfg: dict[str, Any] = {}
    entry = merged.get("entry", {})
    sizing = merged.get("position_sizing", {})
    rebalance = merged.get("rebalance", {})
    exit_block = merged.get("exit", {})

    cfg.update(_map_entry_config(entry))
    cfg.update(_map_sizing_config(sizing))
    cfg.update(_map_rebalance_config(rebalance))
    cfg.update(_map_exit_config(exit_block))

    if "model_selection" in entry:
        ms = entry["model_selection"]
        cfg.setdefault("model_confidence_sizer", {}).update(ms)

    return cfg


# ── StrategyFactory ────────────────────────────────────────────────────

class StrategyFactory:
    """Creates :class:`Strategy` instances from JSON config."""

    def __init__(self, source: dict | None = None):
        self._source = source or self._load_defaults()
        self._defaults: dict = self._source.get("defaults", {})
        self._raw_strategies: dict = self._source.get("strategies", {})

    def keys(self) -> list[str]:
        return list(self._raw_strategies.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._raw_strategies

    def get(self, key: str) -> Strategy:
        """Build a composable Strategy from config, merging defaults + overrides."""
        if key not in self._raw_strategies:
            raise KeyError(f"No strategy named '{key}'")

        raw = self._raw_strategies[key]
        merged = self._merge(base=copy.deepcopy(self._defaults),
                             override=raw.get("override", {}))
        gate_cfg = build_gate_config(merged)

        entry_names = _gate_list(raw, "entry_gates", DEFAULT_ENTRY_GATES)
        exit_names = _gate_list(raw, "exit_gates", DEFAULT_EXIT_GATES)
        sizing_names = _gate_list(raw, "sizing_gates", DEFAULT_SIZING_GATES)
        rebalance_names = _gate_list(raw, "rebalance_gates", DEFAULT_REBALANCE_GATES)

        entry_pipeline = build_pipeline("entry", entry_names, gate_cfg)
        exit_pipeline = build_pipeline("exit", exit_names, gate_cfg)
        sizing_pipeline = build_pipeline("sizing", sizing_names, gate_cfg)
        rebalance_pipeline = build_pipeline("rebalance", rebalance_names, gate_cfg)

        entry_rules = {}
        if isinstance(merged.get("entry"), dict):
            entry_rules = merged["entry"].get("entry_rules", {})

        return Strategy(
            key=key,
            label=raw.get("label", key),
            description=raw.get("description", ""),
            paper_only=raw.get("paper_only", True),
            entry_pipeline=entry_pipeline,
            exit_pipeline=exit_pipeline,
            sizing_pipeline=sizing_pipeline,
            rebalance_pipeline=rebalance_pipeline,
            entry_rules=entry_rules,
            exposure_limits=raw.get("exposure_limits"),
        )

    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        return deep_merge(base, override)

    @staticmethod
    def _load_defaults() -> dict:
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("%s not found; using empty defaults.", DEFAULT_CONFIG_PATH)
        return {}


def _gate_list(raw: dict, key: str, default: list[str]) -> list[str]:
    gates = raw.get("gates", {})
    if isinstance(gates, dict) and key in gates:
        return list(gates[key])
    return list(default)


# ── Module-level convenience ────────────────────────────────────────

_FACTORY: StrategyFactory | None = None


def get_factory() -> StrategyFactory:
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = StrategyFactory()
    return _FACTORY


def build_strategy(key: str) -> Strategy:
    """Convenience: build a single Strategy by name."""
    return get_factory().get(key)


def list_strategies() -> list[str]:
    """All registered strategy keys."""
    return get_factory().keys()