# execution/gates/rebalance.py
"""Rebalance trigger gates — each checks ONE condition for rebalancing.

In the rebalance pipeline (mode="rebalance"), gates short-circuit on
the first trigger (passed=True). Any single trigger is sufficient to
trigger a rebalance; the pipeline stops immediately.

Convention: passed=True means "rebalance IS triggered". The metadata
includes the trigger name and relevant values for audit.
"""

from __future__ import annotations

import logging

from execution.gates.types import GateInput, GateOutput, GateConfig

logger = logging.getLogger(__name__)


# ── Qty Delta Trigger ─────────────────────────────────────────────────

def qty_delta_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when position quantity differs from target.

    Config key: min_qty_delta (float, default 0.5).
    Reads `target_quantity` from inp.context.
    """
    min_delta = config.get("min_qty_delta", 0.5)
    target_qty = inp.context.get("target_quantity", 0.0)

    if inp.position is None:
        current_qty = 0.0
    else:
        current_qty = inp.position.get("quantity", 0.0)

    delta = abs(target_qty - current_qty)
    if delta >= min_delta:
        return GateOutput(
            passed=True,
            reason_code="QTY_DELTA",
            detail=f"delta={delta:.1f} >= {min_delta}",
            metadata={"trigger": "QTY_DELTA", "delta": delta},
        )

    return GateOutput(passed=False, reason_code="PASS", detail=f"delta={delta:.1f} < {min_delta}")


# ── Edge Delta Trigger ────────────────────────────────────────────────

def edge_delta_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when edge crosses a material threshold.

    Config key: material_edge_delta (float, default 0.01).
    """
    min_edge = config.get("material_edge_delta", 0.01)
    edge = abs(inp.model_prob - inp.market_price)

    if edge >= min_edge:
        return GateOutput(
            passed=True,
            reason_code="MATERIAL_EDGE",
            detail=f"edge={edge:.4f} >= {min_edge}",
            metadata={"trigger": "MATERIAL_EDGE", "edge": edge},
        )

    return GateOutput(passed=False, reason_code="PASS", detail=f"edge={edge:.4f} < {min_edge}")


# ── Prob Confidence Trigger ───────────────────────────────────────────

def prob_confidence_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when probability has shifted materially.

    Compares probs_old vs probs_new. Triggered if the top bucket
    changed OR if any bucket shifted by >= material_prob_pp.

    Config key: material_prob_pp (float, default 5.0).
    """
    min_pp = config.get("material_prob_pp", 5.0)
    old = inp.probs_old
    new = inp.probs_new

    if not old or not new:
        return GateOutput(passed=False, reason_code="PASS", detail="no old/new probs to compare")

    top_old = max(old, key=old.get) if old else ""
    top_new = max(new, key=new.get) if new else ""

    if top_old != top_new:
        return GateOutput(
            passed=True,
            reason_code="TOP_BUCKET_CHANGED",
            detail=f"top: {top_old} -> {top_new}",
            metadata={"trigger": "TOP_BUCKET_CHANGED", "top_old": top_old, "top_new": top_new},
        )

    # Check max pp change
    all_keys = set(list(old.keys()) + list(new.keys()))
    max_pp = 0.0
    for k in all_keys:
        pp = abs(new.get(k, 0) - old.get(k, 0)) * 100
        max_pp = max(max_pp, pp)

    if max_pp >= min_pp:
        return GateOutput(
            passed=True,
            reason_code="PROB_CONFIDENCE_CHANGE",
            detail=f"max_pp_change={max_pp:.1f}pp >= {min_pp}",
            metadata={"trigger": "PROB_CONFIDENCE_CHANGE", "max_pp": max_pp},
        )

    return GateOutput(passed=False, reason_code="PASS", detail=f"max_pp={max_pp:.1f}pp")


# ── EV Change Trigger ─────────────────────────────────────────────────

def ev_change_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when expected value shifted materially.

    Config key: material_ev_pp (float, default 2.0).
    """
    min_ev = config.get("material_ev_pp", 2.0)
    old = inp.probs_old
    new = inp.probs_new

    if not old or not new:
        return GateOutput(passed=False, reason_code="PASS", detail="no old/new probs")

    all_keys = set(list(old.keys()) + list(new.keys()))
    for k in all_keys:
        ev_old = old.get(k, 0) * inp.market_price
        ev_new = new.get(k, 0) * inp.market_price
        if abs(ev_new - ev_old) * 100 >= min_ev:
            return GateOutput(
                passed=True,
                reason_code="MATERIAL_EV_CHANGE",
                detail=f"EV change on {k}: {ev_old:.4f} -> {ev_new:.4f}",
                metadata={"trigger": "MATERIAL_EV_CHANGE", "bucket": k},
            )

    return GateOutput(passed=False, reason_code="PASS", detail="no material EV change")


# ── Nowcast Regime Trigger ─────────────────────────────────────────────

def nowcast_regime_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when nowcast regime has changed.

    Reads `nowcast_regime_changed` from inp.context.
    """
    changed = inp.context.get("nowcast_regime_changed", False)

    if changed:
        return GateOutput(
            passed=True,
            reason_code="NOWCAST_REGIME",
            detail="nowcast regime changed",
            metadata={"trigger": "NOWCAST_REGIME"},
        )

    return GateOutput(passed=False, reason_code="PASS", detail="nowcast regime stable")


# ── Exposure Limit Trigger ─────────────────────────────────────────────

def exposure_limit_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when total exposure exceeds limits.

    Config key: total_max (float, default 0.50).
    """
    total_max = config.get("total_max", 0.50)
    capital = inp.capital

    total_exposure = 0.0
    for pos in inp.current_positions.values():
        if isinstance(pos, dict):
            total_exposure += pos.get("quantity", 0) * pos.get("entry_price", 0)

    exposure_pct = total_exposure / capital if capital > 0 else 0

    if exposure_pct >= total_max:
        return GateOutput(
            passed=True,
            reason_code="EXPOSURE_LIMIT",
            detail=f"exposure={exposure_pct:.1%} >= {total_max:.1%}",
            metadata={"trigger": "EXPOSURE_LIMIT", "exposure_pct": exposure_pct},
        )

    return GateOutput(passed=False, reason_code="PASS", detail=f"exposure={exposure_pct:.1%}")


# ── T2S De-Risk Trigger ───────────────────────────────────────────────

def t2s_derisk_trigger(inp: GateInput, config: GateConfig) -> GateOutput:
    """Trigger rebalance when time-to-settlement is within taper window.

    Config key: taper_start_hours (float, default 4).
    """
    taper_start = config.get("taper_start_hours", 4)

    if inp.hours_to_settlement < taper_start:
        return GateOutput(
            passed=True,
            reason_code="T2S_DE_RISK",
            detail=f"t2s={inp.hours_to_settlement:.1f}h < {taper_start}h",
            metadata={"trigger": "T2S_DE_RISK", "hours": inp.hours_to_settlement},
        )

    return GateOutput(passed=False, reason_code="PASS", detail=f"t2s={inp.hours_to_settlement:.1f}h")
