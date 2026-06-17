# execution/gates/types.py
"""Standard gate interface — shared data types and function signature."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class GateInput:
    """Standard input passed to every gate.

    Callers populate this; gates read only what they need.
    The `context` dict carries cross-cutting data (drawdown, T2S, rain, etc.)
    that individual gates can inspect.
    """

    bucket: str
    model_prob: float
    market_price: float
    model_std: float
    dt_now: datetime
    position: Optional[dict] = None
    context: dict = field(default_factory=dict)

    # Convenience aliases into context
    @property
    def drawdown_pct(self) -> float:
        return self.context.get("drawdown_pct", 0.0)

    @property
    def hours_to_settlement(self) -> float:
        return self.context.get("hours_to_settlement", 24.0)

    @property
    def rain_regime(self) -> str:
        return self.context.get("rain_regime", "no_rain")

    @property
    def max_so_far(self) -> Optional[float]:
        return self.context.get("max_so_far")

    @property
    def temp_now(self) -> Optional[float]:
        return self.context.get("temp_now")

    @property
    def nowcast_stale(self) -> bool:
        return self.context.get("nowcast_stale", True)

    @property
    def data_missing(self) -> bool:
        return self.context.get("data_missing", False)

    @property
    def model_key(self) -> str:
        return self.context.get("model_key", "baseline")

    @property
    def capital(self) -> float:
        return self.context.get("capital", 10000.0)

    @property
    def current_positions(self) -> dict:
        return self.context.get("current_positions", {})

    @property
    def probs_old(self) -> dict:
        return self.context.get("probs_old", {})

    @property
    def probs_new(self) -> dict:
        return self.context.get("probs_new", {})

    @property
    def adjusted_bet(self) -> Optional[dict]:
        return self.context.get("adjusted_bet")


@dataclass
class GateOutput:
    """Standard output returned by every gate.

    Attributes:
        passed: True if the gate allows the action to proceed.
        reason_code: Machine-readable code (e.g. "EDGE_TOO_LOW", "PASS").
        detail: Human-readable explanation for audit logs.
        multiplier: Sizing multiplier applied by this gate.
            1.0 = no effect, 0.0 = full block, 0.5 = halve position.
        metadata: Extra data that downstream gates or the pipeline can read.
    """

    passed: bool
    reason_code: str
    detail: str = ""
    multiplier: float = 1.0
    metadata: dict = field(default_factory=dict)


# GateConfig is just a dict — keeps things flexible without over-engineering.
# Each gate knows which keys it reads from its config dict.
GateConfig = dict

# The standard gate function signature.
GateFn = Callable[[GateInput, GateConfig], GateOutput]
