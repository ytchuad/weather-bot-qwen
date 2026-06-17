# execution/gates/sizing.py
"""Sizing gates — each computes a multiplicative factor for position size.

In the sizing pipeline (mode="sizing"), ALL gates run and their
multipliers are multiplied together to produce the final size factor.
The base Kelly bet is then multiplied by this factor.

Convention: sizing gates always return passed=True; they communicate
via the `multiplier` field only.
"""

from __future__ import annotations

import logging

from execution.gates.types import GateInput, GateOutput, GateConfig

logger = logging.getLogger(__name__)


# ── Kelly Sizer ───────────────────────────────────────────────────────

def kelly_sizer(inp: GateInput, config: GateConfig) -> GateOutput:
    """Base Kelly criterion sizing (no modification).

    This gate is a no-op (multiplier=1.0) that serves as the pipeline
    entry point. The actual Kelly computation happens in the strategy
    layer before the sizing pipeline is called.
    """
    return GateOutput(passed=True, reason_code="PASS", detail="kelly base", multiplier=1.0)


# ── Time Window Sizer ─────────────────────────────────────────────────

def time_window_sizer(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position size based on time-of-day window.

    Config key: multipliers — list of {hours: [lo, hi], value: float}.
    """
    multipliers = config.get("multipliers", [])
    hour = inp.dt_now.hour

    for entry in multipliers:
        lo, hi = entry.get("hours", [0, 24])
        if lo <= hour < hi:
            mult = entry.get("value", 1.0)
            return GateOutput(
                passed=True,
                reason_code="TIME_WINDOW",
                detail=f"hour={hour}, window={lo}-{hi}, mult={mult}",
                multiplier=mult,
            )

    # Default: reduce outside known windows
    return GateOutput(
        passed=True,
        reason_code="TIME_WINDOW",
        detail=f"hour={hour}, no window match, default 0.3",
        multiplier=0.3,
    )


# ── Rain Uncertainty Sizer ────────────────────────────────────────────

def rain_uncertainty_sizer(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position size when rain introduces uncertainty.

    Config key: multipliers — dict mapping rain_regime -> float.
    """
    multipliers = config.get("multipliers", {})
    mult = multipliers.get(inp.rain_regime, 1.0)

    return GateOutput(
        passed=True,
        reason_code="RAIN_UNCERTAINTY",
        detail=f"regime={inp.rain_regime}, mult={mult}",
        multiplier=mult,
    )


# ── Boundary Proximity Sizer ──────────────────────────────────────────

def boundary_proximity_sizer(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position size when prediction is near a bucket boundary.

    Config keys:
        min_standardized_distance (float, default 0.5)
        aggressive_reduction_threshold (float, default 0.3)
        aggressive_reduction_multiplier (float, default 0.5)

    Uses the temperature forecast from inp.context["post_mean"] (or
    inp.model_prob as fallback) against the bucket's temperature bounds.
    """
    from execution.gates.entry import _parse_bucket_bounds

    min_dist = config.get("min_standardized_distance", 0.5)
    agg_thresh = config.get("aggressive_reduction_threshold", 0.3)
    agg_mult = config.get("aggressive_reduction_multiplier", 0.5)

    temperature = inp.context.get("post_mean", inp.model_prob)
    bucket_lo, bucket_hi = _parse_bucket_bounds(inp.bucket)

    dist_raw = 999.0
    if bucket_lo != float("-inf"):
        dist_raw = min(dist_raw, temperature - bucket_lo)
    if bucket_hi != float("inf"):
        dist_raw = min(dist_raw, bucket_hi - temperature)

    dist_std = dist_raw / inp.model_std if inp.model_std > 0 else 999.0

    if dist_std < agg_thresh:
        mult = agg_mult
    elif dist_std < min_dist:
        mult = max(agg_mult, dist_std / min_dist)
    else:
        mult = 1.0

    return GateOutput(
        passed=True,
        reason_code="BOUNDARY_PROXIMITY",
        detail=f"dist_std={dist_std:.2f}, mult={mult:.2f}",
        multiplier=mult,
        metadata={"distance_std": dist_std, "distance_raw": dist_raw},
    )


# ── Model Confidence Sizer ─────────────────────────────────────────────

def model_confidence_sizer(inp: GateInput, config: GateConfig) -> GateOutput:
    """Reduce position size when using a less reliable model.

    Config key: multipliers — dict mapping model_key -> float.
    """
    multipliers = config.get("multipliers", {})
    default_mult = config.get("default_mult", 0.5)
    mult = multipliers.get(inp.model_key, default_mult)

    return GateOutput(
        passed=True,
        reason_code="MODEL_CONFIDENCE",
        detail=f"model={inp.model_key}, mult={mult}",
        multiplier=mult,
    )
