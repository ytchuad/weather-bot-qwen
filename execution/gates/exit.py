# execution/gates/exit.py
"""Exit gates — each evaluates a single condition for exiting a position.

In the exit pipeline (mode="exit"), all gates run and their multipliers
are multiplied together. A combined multiplier <= 0 means exit; between
0 and ~0.8 means reduce; above means hold.
"""

from __future__ import annotations

import logging

from execution.gates.types import GateInput, GateOutput, GateConfig

logger = logging.getLogger(__name__)


# ── Conviction Hold Gate ──────────────────────────────────────────────

def conviction_hold_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Hold if extreme conviction and close to settlement.

    Config keys:
        hold_conviction_prob (float, default 0.98)
        hold_max_hours (float, default 6)
    """
    conv_prob = config.get("hold_conviction_prob", 0.98)
    max_hours = config.get("hold_max_hours", 6)

    if inp.model_prob >= conv_prob or inp.model_prob <= (1 - conv_prob):
        if inp.hours_to_settlement <= max_hours:
            return GateOutput(
                passed=True,
                reason_code="HOLD_CONVICTION",
                detail=f"extreme conviction prob={inp.model_prob:.3f} within {max_hours}h",
                multiplier=1.0,
            )

    return GateOutput(passed=True, reason_code="PASS", detail="no conviction hold")


# ── Edge Reversal Gate ────────────────────────────────────────────────

def edge_reversal_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Exit if edge has reversed beyond threshold.

    Config key: edge_reversal_threshold (float, default -0.05).
    """
    threshold = config.get("edge_reversal_threshold", -0.05)

    if inp.position is None:
        return GateOutput(passed=True, reason_code="PASS", detail="no position")

    side = inp.position.get("side", "YES")
    edge = inp.model_prob - inp.market_price if side == "YES" else inp.market_price - inp.model_prob

    if edge < threshold:
        return GateOutput(
            passed=False,
            reason_code="EDGE_REVERSED",
            detail=f"edge={edge:.4f} < {threshold}",
            multiplier=0.0,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"edge={edge:.4f}")


# ── Profit Take Gate ──────────────────────────────────────────────────

def profit_take_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position when market price has moved against the model.

    Config key: profit_take_multiplier (float, default 0.5 — reduces to 50%).
    """
    pt_mult = config.get("profit_take_multiplier", 0.5)

    if inp.position is None:
        return GateOutput(passed=True, reason_code="PASS", detail="no position")

    side = inp.position.get("side", "YES")

    # Take profit when market > model for YES, or market < model for NO
    if side == "YES" and inp.market_price > inp.model_prob:
        return GateOutput(
            passed=True,
            reason_code="PROFIT_TAKE",
            detail=f"market={inp.market_price:.3f} > model={inp.model_prob:.3f}",
            multiplier=pt_mult,
        )
    if side == "NO" and inp.market_price < inp.model_prob:
        return GateOutput(
            passed=True,
            reason_code="PROFIT_TAKE",
            detail=f"market={inp.market_price:.3f} < model={inp.model_prob:.3f}",
            multiplier=pt_mult,
        )

    return GateOutput(passed=True, reason_code="PASS", detail="no profit take")


# ── Confidence Drop Gate ──────────────────────────────────────────────

def confidence_drop_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position when model confidence has deteriorated.

    Config key: max_model_std (float, default 2.5). Reduces by 50% if
    model_std > 1.5x the configured max.
    """
    max_std = config.get("max_model_std", 2.5)

    if inp.model_std > max_std * 1.5:
        return GateOutput(
            passed=True,
            reason_code="CONFIDENCE_DROP",
            detail=f"std={inp.model_std:.2f} > 1.5x max={max_std * 1.5:.2f}",
            multiplier=0.5,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"std={inp.model_std:.2f}")


# ── Rain Emergency Gate ───────────────────────────────────────────────

def rain_emergency_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position during rain-induced temperature drops.

    Config key: rain_emergency_temp_drop (float, default 1.5°C).
    """
    temp_drop_threshold = config.get("rain_emergency_temp_drop", 1.5)

    msf = inp.max_so_far
    tn = inp.temp_now
    rain = inp.rain_regime

    if rain not in ("weak_rain", "moderate_or_heavy_rain"):
        return GateOutput(passed=True, reason_code="PASS", detail="no rain regime")

    if msf is not None and tn is not None:
        temp_drop = msf - tn
        if temp_drop > temp_drop_threshold:
            return GateOutput(
                passed=True,
                reason_code="RAIN_EMERGENCY",
                detail=f"temp_drop={temp_drop:.1f} > {temp_drop_threshold}",
                multiplier=0.3,
            )

    return GateOutput(passed=True, reason_code="PASS", detail="no rain emergency")


# ── Nowcast Stale Gate ────────────────────────────────────────────────

def nowcast_stale_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position when nowcast data is stale.

    Config keys:
        model_c_stale_mult (float, default 0.7)
        rain_nowcast_stale_mult (float, default 0.5)
    """
    mc_mult = config.get("model_c_stale_mult", 0.7)
    rn_mult = config.get("rain_nowcast_stale_mult", 0.5)

    if not inp.nowcast_stale:
        return GateOutput(passed=True, reason_code="PASS", detail="nowcast fresh")

    mk = inp.model_key
    mult = rn_mult if mk in ("rain_nowcast",) else mc_mult

    return GateOutput(
        passed=True,
        reason_code="NOWCAST_STALE",
        detail=f"nowcast stale, model={mk}",
        multiplier=mult,
    )


# ── Data Missing Gate ─────────────────────────────────────────────────

def data_missing_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position when data feed is missing.

    Config key: data_missing_mult (float, default 0.5).
    """
    mult = config.get("data_missing_mult", 0.5)

    if not inp.data_missing:
        return GateOutput(passed=True, reason_code="PASS", detail="data OK")

    return GateOutput(
        passed=True,
        reason_code="DATA_MISSING",
        detail="data feed missing",
        multiplier=mult,
    )


# ── Drawdown Flatten Gate ─────────────────────────────────────────────

def drawdown_flatten_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Apply drawdown-based position reduction.

    Config keys:
        hard_flatten (float, default -0.15) — multiplier=0
        reduce_risk (float, default -0.075) — multiplier=ratio
        stop_new_entries (float, default -0.10) — no new entries but hold
    """
    dd = inp.drawdown_pct
    hard = config.get("hard_flatten", -0.15)
    reduce = config.get("reduce_risk", -0.075)

    if dd <= hard:
        return GateOutput(
            passed=False,
            reason_code="DRAWDOWN_HARD",
            detail=f"drawdown={dd:.1%} <= hard={hard:.1%}",
            multiplier=0.0,
        )

    if dd <= reduce:
        # Normalise between hard=0.0 and reduce=1.0
        if reduce != hard:
            ratio = max(0.0, min(1.0, (dd - hard) / (reduce - hard)))
        else:
            ratio = 0.0
        return GateOutput(
            passed=True,
            reason_code="DRAWDOWN_REDUCE",
            detail=f"drawdown={dd:.1%} <= reduce={reduce:.1%}, mult={ratio:.2f}",
            multiplier=ratio,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"drawdown={dd:.1%}")


# ── T2S Taper Gate ────────────────────────────────────────────────────

def t2s_taper_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """De-risk positions as settlement time approaches.

    Config keys:
        taper_start_hours (float, default 4)
        strong_taper_hours (float, default 2)
        strong_taper_multiplier (float, default 0.3)
    """
    h = inp.hours_to_settlement
    taper_start = config.get("taper_start_hours", 4)
    strong_hours = config.get("strong_taper_hours", 2)
    strong_mult = config.get("strong_taper_multiplier", 0.3)

    if h < strong_hours:
        return GateOutput(
            passed=True,
            reason_code="T2S_STRONG",
            detail=f"h={h:.1f} < {strong_hours}",
            multiplier=strong_mult,
        )

    if h < taper_start:
        mult = h / taper_start
        return GateOutput(
            passed=True,
            reason_code="T2S_TAPER",
            detail=f"h={h:.1f} < {taper_start}, mult={mult:.2f}",
            multiplier=mult,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"t2s={h:.1f}h")


# ── Prob Below Stop Gate ──────────────────────────────────────────────

def prob_stop_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position when top-bucket probability is below stop level.

    Config key: stop_prob (float, default 0.05).
    """
    stop_prob = config.get("stop_prob", 0.05)
    top_prob = inp.context.get("prob_top_bucket", 0.0)

    if top_prob < stop_prob:
        return GateOutput(
            passed=True,
            reason_code="PROB_BELOW_STOP",
            detail=f"top_prob={top_prob:.3f} < stop={stop_prob}",
            multiplier=0.3,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"top_prob={top_prob:.3f}")
