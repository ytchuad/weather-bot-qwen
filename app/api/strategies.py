from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.cache import prediction_cache
from app.api.schemas import Suggestion, SuggestRequest, SuggestResponse
from app.services.model_service import calculate_kelly, run_all_models
from execution.strategy_account import StrategyAccount, StrategyAccountStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["Strategies"])


# ── Strategy Schemas ───────────────────────────────────────────────────────────

class StrategyCreate(BaseModel):
    id: str
    label: str
    model: str = "baseline"
    capital: float = 10000.0
    initial_capital: float | None = None
    market_template: str = "hk-tmax"
    from_strategy_key: str | None = None


class StrategyUpdate(BaseModel):
    status: str | None = None
    scheduler_on: bool | None = None
    capital: float | None = None
    params: dict[str, Any] | None = None


class StrategyTrade(BaseModel):
    bucket: str
    entry_time: str
    exit_time: str | None
    side: str
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None
    reason: str | None


# ── Portfolio Endpoint ───────────────────────────────────────────────────────

@router.get("/portfolio")
def get_portfolio():
    """Get aggregate portfolio statistics."""
    store = StrategyAccountStore()
    accounts = store.list()
    
    total_capital = sum(a.capital for a in accounts)
    total_initial = sum(a.initial_capital for a in accounts)
    total_pnl = total_capital - total_initial
    total_return_pct = (total_pnl / total_initial * 100) if total_initial > 0 else 0
    active_count = len([a for a in accounts if a.status == "running" and a.scheduler_on])
    
    return {
        "total_capital": total_capital,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "active_strategies": active_count,
        "count": len(accounts),
    }


# ── Strategy Account Endpoints ────────────────────────────────────────────────

@router.get("")
def list_strategies():
    """List all strategy accounts."""
    store = StrategyAccountStore()
    return {"strategies": [a.to_dict() for a in store.list()]}


@router.post("")
def create_strategy(req: StrategyCreate):
    """Create a new strategy account."""
    store = StrategyAccountStore()
    
    # Check if already exists
    if store.load(req.id):
        raise HTTPException(status_code=400, detail="Strategy already exists")
    
    initial = req.initial_capital if req.initial_capital is not None else req.capital
    acct = StrategyAccount(
        id=req.id,
        label=req.label,
        model=req.model,
        capital=req.capital,
        initial_capital=initial,
        market_template=req.market_template,
        from_strategy_key=req.from_strategy_key,
    )
    store.save(acct)
    return {"status": "created", "strategy": acct.to_dict()}


@router.patch("/{sid}")
def update_strategy(sid: str, req: StrategyUpdate):
    """Update strategy account status or parameters."""
    store = StrategyAccountStore()
    acct = store.load(sid)
    if not acct:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    if req.status:
        acct.status = req.status
        acct.scheduler_on = req.status == "running"
    if req.capital:
        acct.capital = req.capital
    if req.params:
        acct.params.update(req.params)
    
    store.save(acct)
    return {"status": "updated", "strategy": acct.to_dict()}


@router.get("/{sid}/trades")
def get_strategy_trades(sid: str, limit: int = 50):
    """Get recent trades for a strategy."""
    # This would integrate with paper_trade_audit.parquet in real implementation
    return {"trades": []}


@router.post("/{sid}/reset")
def reset_strategy(sid: str):
    """Reset strategy capital to initial value, clear trade history."""
    store = StrategyAccountStore()
    acct = store.load(sid)
    if not acct:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    acct.capital = acct.initial_capital
    acct.last_run = None
    store.save(acct)
    return {"status": "reset", "strategy": acct.to_dict()}


@router.delete("/{sid}")
def delete_strategy(sid: str):
    """Delete a strategy account."""
    store = StrategyAccountStore()
    if not store.load(sid):
        raise HTTPException(status_code=404, detail="Strategy not found")
    store.delete(sid)
    return {"status": "deleted", "id": sid}


@router.post("/suggest", response_model=SuggestResponse)
@prediction_cache
def suggest_strategy(req: SuggestRequest):
    from app.services.market_service import fetch_event_markets
    from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now

    target_date = _parse_date(req.date)
    target_date_str = req.date

    hkt = hkt_now()
    is_today = target_date == hkt.date()

    hko = fetch_hko_data(target_date_str)
    _sd = target_date_str.replace("-", "")
    state = get_intraday_state(_sd)
    from app.services.weather_service import compute_rain_kwargs

    rain_kwargs = compute_rain_kwargs(_sd, hkt_now())
    markets = fetch_event_markets(target_date_str, is_min_temp=req.is_min_temp)

    forecast_key = "forecast_min" if req.is_min_temp else "forecast_max"
    forecast_aws = hko.get(forecast_key) if hko else None

    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=req.is_min_temp,
        bias=0.0,
        std_mult=1.0,
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=forecast_aws,
        is_today=is_today,
    )

    suggestions = []
    for mk, pred in results.items():
        if mk == "_intraday_error" or not pred.get("probs"):
            continue
        for bucket_name, model_prob in pred["probs"].items():
            market_price = _find_market_price(markets, bucket_name)
            if market_price is None:
                continue
            edge = model_prob - market_price
            if edge > 0.01:
                kelly = calculate_kelly(model_prob, market_price, req.kelly_fraction)
                action = "buy_yes" if model_prob > market_price else "buy_no"
            else:
                kelly = 0.0
                action = "pass"

            suggestions.append(
                Suggestion(
                    bucket=bucket_name,
                    market_price=market_price,
                    model_prob=model_prob,
                    edge=edge,
                    kelly_fraction=kelly,
                    action=action,
                )
            )

    suggestions.sort(key=lambda s: s.edge, reverse=True)
    return SuggestResponse(date=req.date, suggestions=suggestions)


def _parse_date(d: str) -> date_type:
    try:
        return date_type.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date: {d}. Use YYYY-MM-DD.")


def _find_market_price(markets: list[dict], bucket_name: str) -> float | None:
    for m in markets:
        if m.get("bucket") == bucket_name or m.get("name") == bucket_name:
            return m.get("yes_price")
    return None
