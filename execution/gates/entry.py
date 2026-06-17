# execution/gates/entry.py
"""Entry gates — each evaluates a single condition for entering a position.

All gates follow the standard (GateInput, GateConfig) -> GateOutput signature.
They are composed via GatePipeline(mode="entry") which short-circuits on
the first gate that returns passed=False.
"""

from __future__ import annotations

import logging

from execution.gates.types import GateInput, GateOutput, GateConfig

logger = logging.getLogger(__name__)


# ── Time Gate ─────────────────────────────────────────────────────────

def time_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Block entries before min_hour or in blocked time slots."""
    min_hour = config.get("min_hour", 8)
    blocked_slots = config.get("blocked_slots", ["evening", "night"])

    hour = inp.dt_now.hour
    if hour < min_hour:
        return GateOutput(
            passed=False,
            reason_code="TIME_WINDOW_CLOSED",
            detail=f"hour={hour} < min_hour={min_hour}",
            multiplier=0.0,
        )

    slot = _time_slot(hour)
    if slot in blocked_slots:
        return GateOutput(
            passed=False,
            reason_code="TIME_WINDOW_CLOSED",
            detail=f"{slot}: entries blocked",
            multiplier=0.0,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"slot={slot}")


def _time_slot(hour: int) -> str:
    if 8 <= hour < 12:
        return "morning"
    if 12 <= hour < 16:
        return "afternoon"
    if 16 <= hour < 20:
        return "evening"
    return "night"


# ── Regime Edge Gate ──────────────────────────────────────────────────

def regime_edge_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Check if edge meets the regime-specific threshold.

    Config keys:
        thresholds: dict mapping regime_key -> {min_edge, exposure_cap}
    The regime is derived from hour + rain_regime.
    """
    thresholds = config.get("thresholds", {})
    regime = _get_entry_regime(inp.dt_now.hour, inp.rain_regime)
    regime_cfg = thresholds.get(regime, {})

    min_edge = regime_cfg.get("min_edge")
    if min_edge is None:
        return GateOutput(
            passed=False,
            reason_code="EDGE_TOO_LOW",
            detail=f"no entry in regime={regime}",
            multiplier=0.0,
        )

    edge = inp.model_prob - inp.market_price
    exposure_cap = regime_cfg.get("exposure_cap", 0.50)

    if edge >= min_edge:
        return GateOutput(
            passed=True,
            reason_code="PASS",
            detail=f"edge={edge:.4f} >= {min_edge} (regime={regime})",
            metadata={"regime": regime, "exposure_cap": exposure_cap},
        )

    return GateOutput(
        passed=False,
        reason_code="EDGE_TOO_LOW",
        detail=f"edge={edge:.4f} < {min_edge} (regime={regime})",
        multiplier=0.0,
    )


def _get_entry_regime(hour: int, rain_regime: str) -> str:
    if 8 <= hour < 12:
        if rain_regime in ("moderate_or_heavy_rain", "weak_rain"):
            return "rain_08_12"
        return "day_08_12"
    if 12 <= hour < 16:
        return "slot_12_16"
    return "slot_16_24"


# ── Confidence Gate ───────────────────────────────────────────────────

def confidence_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Block entry if model uncertainty is too high.

    Config key: max_model_std (float, default 2.5).
    """
    max_std = config.get("max_model_std", 2.5)
    if inp.model_std > max_std:
        return GateOutput(
            passed=False,
            reason_code="LOW_CONFIDENCE",
            detail=f"std={inp.model_std:.2f} > max={max_std}",
            multiplier=0.0,
        )
    return GateOutput(passed=True, reason_code="PASS", detail=f"std={inp.model_std:.2f} <= {max_std}")


# ── Boundary Proximity Gate ───────────────────────────────────────────

def boundary_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce or block entry when prediction is too close to a bucket boundary.

    Config keys:
        min_standardized_distance (float, default 0.5)
        aggressive_reduction_threshold (float, default 0.3)
        aggressive_reduction_multiplier (float, default 0.5)

    Uses the temperature forecast from inp.context["post_mean"] (or
    inp.model_prob as fallback) against the bucket's temperature bounds.
    """
    min_dist = config.get("min_standardized_distance", 0.5)
    agg_thresh = config.get("aggressive_reduction_threshold", 0.3)
    agg_mult = config.get("aggressive_reduction_multiplier", 0.5)

    # Use the temperature forecast for boundary checks, not bucket probability.
    # post_mean (temperature) lives in context; model_prob is the bucket probability.
    temperature = inp.context.get("post_mean", inp.model_prob)

    # Parse bucket bounds from bucket label.
    bucket_lo, bucket_hi = _parse_bucket_bounds(inp.bucket)

    # Compute standardized distance from nearest boundary.
    dist_raw = 999.0
    if bucket_lo != float("-inf"):
        dist_raw = min(dist_raw, temperature - bucket_lo)
    if bucket_hi != float("inf"):
        dist_raw = min(dist_raw, bucket_hi - temperature)

    dist_std = dist_raw / inp.model_std if inp.model_std > 0 else 999.0

    if dist_std < agg_thresh:
        return GateOutput(
            passed=True,
            reason_code="BOUNDARY_TOO_CLOSE",
            detail=f"dist_std={dist_std:.2f} < agg_thresh={agg_thresh}",
            multiplier=agg_mult,
            metadata={"distance_raw": dist_raw, "distance_std": dist_std},
        )

    if dist_std < min_dist:
        ratio = dist_std / min_dist
        return GateOutput(
            passed=True,
            reason_code="BOUNDARY_CLOSE",
            detail=f"dist_std={dist_std:.2f} < min_dist={min_dist}, ratio={ratio:.2f}",
            multiplier=max(agg_mult, ratio),
            metadata={"distance_raw": dist_raw, "distance_std": dist_std},
        )

    return GateOutput(
        passed=True,
        reason_code="PASS",
        detail=f"dist_std={dist_std:.2f} >= {min_dist}",
        metadata={"distance_raw": dist_raw, "distance_std": dist_std},
    )


import re as _re


def _parse_bucket_bounds(bucket_name: str):
    """Extract (lower, upper) from bucket label like '32-33', '<23', '>=34'."""
    s = bucket_name.strip()
    if s.startswith("<"):
        return float("-inf"), float(s[1:])
    if s.startswith(">="):
        return float(s[2:]), float("inf")
    if "-" in s:
        parts = s.split("-")
        return float(parts[0]), float(parts[1])
    # Fallback: try numeric
    m = _re.search(r"([\d.]+)", s)
    if m:
        val = float(m.group(1))
        if any(kw in s.lower() for kw in ["below", "lower", "under", "<"]):
            return float("-inf"), val + 1.0
        if any(kw in s.lower() for kw in ["higher", "above", "over", ">"]):
            return val, float("inf")
        return val, val + 1.0
    return float("-inf"), float("inf")


# ── Drawdown Gate ─────────────────────────────────────────────────────

def drawdown_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Block new entries when drawdown exceeds thresholds.

    Config keys:
        stop_new_entries (float, default -0.10)
        hard_flatten (float, default -0.15)

    If drawdown_pct <= hard_flatten: multiplier=0, block.
    If drawdown_pct <= stop_new_entries: multiplier=0, block.
    Otherwise: pass.
    """
    dd = inp.drawdown_pct
    hard = config.get("hard_flatten", -0.15)
    stop = config.get("stop_new_entries", -0.10)

    if dd <= hard:
        return GateOutput(
            passed=False,
            reason_code="DRAWDOWN_HARD",
            detail=f"drawdown={dd:.1%} <= hard_flatten={hard:.1%}",
            multiplier=0.0,
        )

    if dd <= stop:
        return GateOutput(
            passed=False,
            reason_code="DRAWDOWN_STOP",
            detail=f"drawdown={dd:.1%} <= stop_entries={stop:.1%}",
            multiplier=0.0,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"drawdown={dd:.1%}")


# ── Slippage Gate ─────────────────────────────────────────────────────

def slippage_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Block entry if slippage-adjusted edge is not positive.

    Reads adjusted_bet from inp.context to check slippage.
    If no adjusted_bet is available the gate passes (slippage check
    is execution-level, not always present at strategy time).
    """
    bet = inp.adjusted_bet
    if not bet:
        return GateOutput(
            passed=True,
            reason_code="PASS",
            detail="no slippage data — skipping slippage check",
        )

    if not bet.get("filled", False):
        return GateOutput(
            passed=False,
            reason_code="LIQUIDITY_INSUFFICIENT",
            detail=f"qty={bet.get('adjusted_quantity', 0)}",
            multiplier=0.0,
        )

    slippage_pct = bet.get("slippage_pct", 0)
    edge = inp.model_prob - inp.market_price
    edge_after = edge - (slippage_pct / 100)

    if edge_after <= 0:
        return GateOutput(
            passed=False,
            reason_code="SLIPPAGE_EATS_EDGE",
            detail=f"edge_after={edge_after:.4f}",
            multiplier=0.0,
        )

    return GateOutput(passed=True, reason_code="PASS", detail=f"edge_after={edge_after:.4f}")


# ── Exposure Gate ─────────────────────────────────────────────────────

def exposure_gate(inp: GateInput, config: GateConfig) -> GateOutput:
    """Block entry if per-bucket or total exposure exceeds limits.

    Config keys:
        max_per_bucket (float, default 0.15) — fraction of capital
        total_max (float, default 0.50) — fraction of capital
    """
    max_per_bucket = config.get("max_per_bucket", 0.15)
    total_max = config.get("total_max", 0.50)
    capital = inp.capital

    # Check existing bucket exposure
    existing = inp.current_positions.get(inp.bucket, {})
    existing_qty = existing.get("quantity", 0) if isinstance(existing, dict) else 0
    existing_entry = existing.get("entry_price", 0.5) if isinstance(existing, dict) else 0.5
    bucket_exposure = (existing_qty * existing_entry) / capital if capital > 0 else 0

    if bucket_exposure >= max_per_bucket:
        return GateOutput(
            passed=False,
            reason_code="BUCKET_EXPOSURE",
            detail=f"bucket_exposure={bucket_exposure:.1%} >= {max_per_bucket:.1%}",
            multiplier=0.0,
        )

    # Check total exposure
    total_exposure = 0.0
    for pos in inp.current_positions.values():
        if isinstance(pos, dict):
            total_exposure += pos.get("quantity", 0) * pos.get("entry_price", 0)

    if total_exposure / capital >= total_max if capital > 0 else False:
        return GateOutput(
            passed=False,
            reason_code="TOTAL_EXPOSURE",
            detail=f"total_exposure={total_exposure/capital:.1%} >= {total_max:.1%}",
            multiplier=0.0,
        )

    return GateOutput(passed=True, reason_code="PASS", detail="exposure OK")
