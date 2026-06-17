# app/services/strategy_service.py
"""Strategy and portfolio management service.

Wraps the existing execution/ modules (strategy_runner, portfolio_manager,
rebalancer, paper_adapter) behind a clean interface so pages don't need
to know about import paths or error-handling patterns.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from ..config import (
    STRATEGY_CONFIG_PATH,
    PORTFOLIO_CONFIG_PATH,
    PORTFOLIO_STATE_PATH,
    STRATEGY_MODEL_ALIASES,
)

logger = logging.getLogger(__name__)

# ── lazy imports with graceful degradation ──────────────────────────────

_rebalancer = None
_strategy_runner = None
_portfolio_manager = None
_paper_adapter = None
_import_errors: list[str] = []


def _init_modules() -> None:
    """Attempt imports once; store results module-level."""
    global _rebalancer, _strategy_runner, _portfolio_manager, _paper_adapter, _import_errors
    if _rebalancer is not None:
        return  # already attempted

    try:
        from execution import rebalancer as _r
        _rebalancer = _r
    except Exception as e:
        _import_errors.append(f"rebalancer: {e}")

    try:
        from execution import strategy_runner as _sr
        _strategy_runner = _sr
    except Exception as e:
        _import_errors.append(f"strategy_runner: {e}")

    try:
        from execution import portfolio_manager as _pm
        _portfolio_manager = _pm
    except Exception as e:
        _import_errors.append(f"portfolio_manager: {e}")

    try:
        from execution import paper_adapter as _pa
        _paper_adapter = _pa
    except Exception as e:
        _import_errors.append(f"paper_adapter: {e}")


def has_rebalancer() -> bool:
    _init_modules()
    return _rebalancer is not None


def has_strategy_runner() -> bool:
    _init_modules()
    return _strategy_runner is not None


def has_portfolio_manager() -> bool:
    _init_modules()
    return _portfolio_manager is not None


# ── strategy registry ────────────────────────────────────────────────

def load_strategy_registry() -> dict:
    _init_modules()
    if _strategy_runner is not None:
        try:
            return _strategy_runner.load_strategy_registry()
        except Exception:
            pass
    return {"version": 1, "strategies": {}}


def save_strategy_registry(registry: dict) -> None:
    """Save strategy registry back to config."""
    _init_modules()
    if _strategy_runner is not None:
        try:
            _strategy_runner.save_strategy_registry(registry)
        except Exception as e:
            logger.warning("save_strategy_registry failed: %s", e)


def load_strategy_state() -> dict:
    _init_modules()
    if _strategy_runner is not None:
        try:
            return _strategy_runner.load_strategy_state()
        except Exception:
            pass
    return {"version": 1, "accounts": {}}


# ── portfolio CRUD ───────────────────────────────────────────────────

def load_portfolio_config() -> dict:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.load_portfolio_config()
        except Exception:
            pass
    return {"version": 1, "portfolios": {}}


def save_portfolio_config(cfg: dict) -> None:
    _init_modules()
    if _portfolio_manager is not None:
        _portfolio_manager.save_portfolio_config(cfg)


def load_portfolio_state() -> dict:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.load_portfolio_state()
        except Exception:
            pass
    return {"version": 1, "portfolios": {}}


def save_portfolio_state(state: dict) -> None:
    _init_modules()
    if _portfolio_manager is not None:
        _portfolio_manager.save_portfolio_state(state)


def list_portfolios() -> dict[str, dict]:
    cfg = load_portfolio_config()
    return cfg.get("portfolios", {})


def get_portfolio(pid: str) -> dict | None:
    return load_portfolio_config().get("portfolios", {}).get(pid)


def create_portfolio(pid: str, label: str, capital: float, strategies: list[str], strategy_models: dict | None = None) -> None:
    cfg = load_portfolio_config()
    if strategy_models is None:
        strategy_models = {}
    cfg.setdefault("portfolios", {})[pid] = {
        "label": label,
        "capital": capital,
        "strategies": strategies,
        "strategy_models": strategy_models,
        "watched_slugs": [],
    }
    save_portfolio_config(cfg)


def delete_portfolio(pid: str) -> None:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            _portfolio_manager.delete_portfolio(pid)
        except Exception:
            cfg = load_portfolio_config()
            cfg.get("portfolios", {}).pop(pid, None)
            save_portfolio_config(cfg)


def reset_portfolio(pid: str) -> None:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            _portfolio_manager.reset_portfolio(pid)
        except Exception:
            pass


# ── PnL ──────────────────────────────────────────────────────────────

def get_pnl(pid: str, current_prices: dict[str, float]) -> dict:
    """Return {cost_basis, unrealized_pnl, market_value, total_fees, details[]}."""
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.get_portfolio_pnl(pid, current_prices=current_prices)
        except Exception:
            pass
    return {"unrealized_pnl": 0, "cost_basis": 0, "market_value": 0, "total_fees": 0, "details": []}


def get_pnl_history(pid: str) -> pd.DataFrame:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.load_pnl_history(pid)
        except Exception:
            pass
    return pd.DataFrame()


def get_portfolio_exposure(pid: str) -> float:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.get_portfolio_exposure(pid)
        except Exception:
            pass
    return 0.0


# ── account balance (paper-trader) ───────────────────────────────────

def get_account_balance() -> dict | None:
    """Return {cash, positions_value, total_value} or None."""
    _init_modules()
    if _paper_adapter is not None:
        try:
            return _paper_adapter._get_adapter().get_balance()
        except Exception:
            pass
    return None


def get_trade_history(limit: int = 100) -> list[dict]:
    _init_modules()
    if _paper_adapter is not None:
        try:
            return _paper_adapter._get_adapter().get_trade_history(limit=limit)
        except Exception:
            pass
    return []


# ── strategy execution ───────────────────────────────────────────────

def run_portfolio(
    pid: str,
    strategy_contexts: dict[str, dict] | None = None,
    prices_dict: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """Execute all strategies in a portfolio. Returns list of result dicts."""
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.run_portfolio(
                pid,
                strategy_contexts=strategy_contexts,
                force=force,
                prices_dict=prices_dict or {},
            )
        except Exception as e:
            logger.warning("run_portfolio failed: %s", e)
    return []


def resolve_expired_markets() -> list:
    """Settle any markets whose resolution date has passed."""
    _init_modules()
    if _paper_adapter is not None:
        try:
            return _paper_adapter._get_adapter().resolve_expired_markets()
        except Exception as e:
            logger.warning("resolve_expired_markets failed: %s", e)
    return []


def sell_position(bucket: str, outcome: str, quantity: float) -> dict | None:
    """Sell a paper position. Returns adapter result or None."""
    _init_modules()
    if _paper_adapter is not None:
        try:
            adapter = _paper_adapter._get_adapter()
            return adapter.sell_position(bucket, outcome, quantity)
        except Exception as e:
            logger.warning("sell_position failed: %s", e)
    return None


def discover_events_for_portfolio(pid: str) -> dict:
    _init_modules()
    if _portfolio_manager is not None:
        try:
            return _portfolio_manager.discover_events_for_portfolio(pid)
        except Exception:
            pass
    return {}


# ── model→strategy key mapping ───────────────────────────────────────

def get_probs_for_strategy(
    model_key: str,
    probs_9d: dict,
    probs_aws: dict,
    all_intra_probs: dict,
    probs_intra: dict,
) -> dict:
    """Look up the right probability dict for a strategy's assigned model key."""
    if model_key == "9d":
        return probs_9d
    if model_key == "aws":
        return probs_aws
    intra_key = STRATEGY_MODEL_ALIASES.get(model_key, model_key)
    if all_intra_probs and intra_key:
        p = all_intra_probs.get(intra_key)
        if p:
            return p
    return probs_intra if probs_intra else probs_9d


def build_strategy_contexts(
    pf: dict,
    base_ctx: dict,
    probs_9d: dict,
    probs_aws: dict,
    all_intra_probs: dict,
    probs_intra: dict,
    scheduler_source: str = "manual",
) -> dict[str, dict]:
    """Build per-strategy context dicts with appropriate model probabilities."""
    strategy_models = pf.get("strategy_models", {})
    contexts: dict[str, dict] = {}
    for sk in pf.get("strategies", []):
        mk = strategy_models.get(sk, sk)
        ctx = dict(base_ctx)
        try:
            ctx["target_probs"] = get_probs_for_strategy(
                mk, probs_9d, probs_aws, all_intra_probs, probs_intra
            )
        except NameError:
            ctx["target_probs"] = {}
        ctx["model_key"] = mk
        ctx["scheduler_source"] = scheduler_source
        contexts[sk] = ctx
    return contexts


# ── enhanced version / time slot ─────────────────────────────────────

def get_enhanced_version() -> str | None:
    _init_modules()
    if _strategy_runner is not None:
        try:
            from execution.strategy_engine import ENHANCED_VERSION
            return ENHANCED_VERSION
        except Exception:
            pass
    return None


def get_time_slot(dt_now: datetime) -> str | None:
    _init_modules()
    if _strategy_runner is not None:
        try:
            from execution.strategy_engine import get_time_slot as _gts
            return _gts(dt_now)
        except Exception:
            pass
    return None


def get_effective_exposure_limit(dt_now: datetime) -> float:
    _init_modules()
    if _strategy_runner is not None:
        try:
            from execution.strategy_engine import get_effective_exposure_limit as _gel
            return _gel(dt_now)
        except Exception:
            pass
    return 0.0
