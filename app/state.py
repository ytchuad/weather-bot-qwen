# app/state.py
"""Centralised session-state manager.

All st.session_state keys are namespaced under ``app.`` to avoid collisions
with Streamlit internals or other libraries.

Usage:
    from .state import AppState
    state = AppState()

    state.target_date = some_date          # setter
    preds = state.pred_intra               # getter

Do NOT access raw ``st.session_state`` keys from pages/components; always go
through AppState.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import streamlit as st

from .config import DEFAULT_REFRESH_INTERVAL_MS

# Single prefix for all keys
_NS = "app"

# ── helpers ──────────────────────────────────────────────────────────
def _key(name: str) -> str:
    return f"{_NS}.{name}"

def _get(name: str, default: Any = None) -> Any:
    return st.session_state.get(_key(name), default)

def _set(name: str, value: Any) -> None:
    st.session_state[_key(name)] = value

def _pop(name: str, default: Any = None) -> Any:
    return st.session_state.pop(_key(name), default)


# ── AppState ──────────────────────────────────────────────────────────
@dataclass
class AppState:
    """Typed accessor over st.session_state for the dashboard."""

    # -- event / market -------------------------------------------------
    @property
    def selected_event(self) -> dict | None:
        return _get("selected_event")

    @selected_event.setter
    def selected_event(self, value: dict | None):
        _set("selected_event", value)

    @property
    def target_date(self) -> date | None:
        return _get("target_date")

    @target_date.setter
    def target_date(self, value: date | None):
        _set("target_date", value)

    @property
    def is_min_temp(self) -> bool:
        return _get("is_min_temp", False)

    @is_min_temp.setter
    def is_min_temp(self, value: bool):
        _set("is_min_temp", value)

    @property
    def markets(self) -> list[dict]:
        return _get("markets", [])

    @markets.setter
    def markets(self, value: list[dict]):
        _set("markets", value)

    @property
    def pm_events(self) -> list[dict]:
        return _get("pm_events", [])

    @pm_events.setter
    def pm_events(self, value: list[dict]):
        _set("pm_events", value)

    # -- model predictions ----------------------------------------------
    @property
    def pred_9d(self) -> dict | None:
        return _get("pred_9d")

    @pred_9d.setter
    def pred_9d(self, value: dict | None):
        _set("pred_9d", value)

    @property
    def pred_aws(self) -> dict | None:
        return _get("pred_aws")

    @pred_aws.setter
    def pred_aws(self, value: dict | None):
        _set("pred_aws", value)

    @property
    def pred_intra(self) -> dict[str, dict]:
        return _get("pred_intra", {})

    @pred_intra.setter
    def pred_intra(self, value: dict[str, dict]):
        _set("pred_intra", value)

    @property
    def selected_model(self) -> str:
        return _get("selected_model", "9d")

    @selected_model.setter
    def selected_model(self, value: str):
        _set("selected_model", value)

    # -- weather state --------------------------------------------------
    @property
    def intraday_state(self) -> dict | None:
        return _get("intraday_state")

    @intraday_state.setter
    def intraday_state(self, value: dict | None):
        _set("intraday_state", value)

    @property
    def live_temp(self) -> tuple | None:
        return _get("live_temp")

    @live_temp.setter
    def live_temp(self, value: tuple | None):
        _set("live_temp", value)

    @property
    def rain_kwargs(self) -> dict:
        return _get("rain_kwargs", {})

    @rain_kwargs.setter
    def rain_kwargs(self, value: dict):
        _set("rain_kwargs", value)

    # -- strategy / portfolio -------------------------------------------
    @property
    def portfolio_id(self) -> str:
        return _get("portfolio_id", "weather_main")

    @portfolio_id.setter
    def portfolio_id(self, value: str):
        _set("portfolio_id", value)

    @property
    def capital(self) -> float:
        return _get("capital", 10000.0)

    @capital.setter
    def capital(self, value: float):
        _set("capital", value)

    @property
    def kelly_fraction(self) -> float:
        return _get("kelly_fraction", 0.5)

    @kelly_fraction.setter
    def kelly_fraction(self, value: float):
        _set("kelly_fraction", value)

    @property
    def bias(self) -> float:
        return _get("bias", 0.0)

    @bias.setter
    def bias(self, value: float):
        _set("bias", value)

    @property
    def std_mult(self) -> float:
        return _get("std_mult", 1.0)

    @std_mult.setter
    def std_mult(self, value: float):
        _set("std_mult", value)

    @property
    def scheduler_on(self) -> bool:
        return _get("scheduler_on", False)

    @scheduler_on.setter
    def scheduler_on(self, value: bool):
        _set("scheduler_on", value)

    @property
    def strategy_last_run(self) -> datetime | None:
        return _get("strategy_last_run")

    @strategy_last_run.setter
    def strategy_last_run(self, value: datetime | None):
        _set("strategy_last_run", value)

    @property
    def refresh_interval_ms(self) -> int:
        return _get("refresh_interval_ms", DEFAULT_REFRESH_INTERVAL_MS)

    @refresh_interval_ms.setter
    def refresh_interval_ms(self, value: int):
        _set("refresh_interval_ms", value)

    @property
    def last_decisions(self) -> list[dict]:
        return _get("last_decisions", [])

    @last_decisions.setter
    def last_decisions(self, value: list[dict]):
        _set("last_decisions", value)

    # -- convenience methods --------------------------------------------
    def init_defaults(self) -> None:
        """Seed session-state with defaults on first run (idempotent)."""
        defaults = {
            "selected_event": None,
            "target_date": None,
            "is_min_temp": False,
            "markets": [],
            "pm_events": [],
            "pred_9d": None,
            "pred_aws": None,
            "pred_intra": {},
            "selected_model": "9d",
            "intraday_state": None,
            "live_temp": None,
            "rain_kwargs": {},
            "portfolio_id": "weather_main",
            "capital": 10000.0,
            "kelly_fraction": 0.5,
            "bias": 0.0,
            "std_mult": 1.0,
            "scheduler_on": False,
            "strategy_last_run": None,
            "refresh_interval_ms": DEFAULT_REFRESH_INTERVAL_MS,
            "last_decisions": [],
            "model_pin": None,
        }
        for name, default in defaults.items():
            if _key(name) not in st.session_state:
                _set(name, default)

    def clear_cache(self) -> None:
        """Clear all cached data and model predictions."""
        _set("pred_9d", None)
        _set("pred_aws", None)
        _set("pred_intra", {})
        _set("intraday_state", None)
        _set("rain_kwargs", {})
        _set("markets", [])
        _set("live_temp", None)

    def get_probs_for(self, model_key: str) -> dict[str, float]:
        """Return bucket→probability dict for a given model key."""
        if model_key == "9d":
            pred = self.pred_9d
            return pred.get("probs", {}) if pred else {}
        if model_key == "aws":
            pred = self.pred_aws
            return pred.get("probs", {}) if pred else {}
        intra = self.pred_intra.get(model_key, {})
        return intra.get("probs", {})

    def get_market_prices(self) -> dict[str, float]:
        """Return {bucket_name: yes_price} from current markets."""
        return {m.get("bucket", m.get("name", "")): m.get("yes_price", 0.5) for m in self.markets}

    # -- model data migration from old dashboard.py ---------------------
    @property
    def model_preferences(self) -> list[str] | None:
        """Which models to show in the grid (None = show all)."""
        return _get("model_preferences", None)

    @model_preferences.setter
    def model_preferences(self, value: list[str] | None):
        _set("model_preferences", value)

    @property
    def model_pin(self) -> list[str]:
        """Pinned model keys for the hub strip (None = compute default)."""
        return _get("model_pin", None)

    @model_pin.setter
    def model_pin(self, value: list[str]):
        _set("model_pin", value)
