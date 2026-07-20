"""Strategy-context adapter over the shared canonical sampling cycle."""

from __future__ import annotations

from typing import Any

from execution.strategy_account import StrategyAccount


def build_strategy_context(acct: StrategyAccount) -> dict[str, Any]:
    """Return an account execution context backed by one shared cycle.

    Weather, all model outputs, market metadata and normalized YES/NO books
    are built by :mod:`app.services.canonical_cycle`.  This adapter only adds
    account-local derived fields required by the existing paper strategy API.
    """
    from app.services.canonical_cycle import build_strategy_context as _build

    return _build(acct)
