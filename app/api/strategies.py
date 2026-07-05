from __future__ import annotations

import logging
import threading
import time
from datetime import date as date_type, datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.cache import prediction_cache
from app.api.schemas import Suggestion, SuggestRequest, SuggestResponse
from app.services.model_service import calculate_kelly, run_all_models
from app.services.market_depth_service import get_global_depth_cache, compute_execution_estimate
from execution.strategy_account import StrategyAccount, StrategyAccountStore
from app.services.market_service import fetch_today_event, fetch_event_markets
from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now
from features.strategy_snapshot_logger import (
    write_snapshot,
    read_snapshots,
    calc_pm_weighted_temp,
    calc_model_predicted_temp,
)

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
    initial_capital: float | None = None
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
    if req.capital is not None:  # 建议改为 is not None，防止资金设为0时被跳过
        acct.capital = req.capital
    if req.initial_capital is not None:  # 新增：处理初始资金更新
        acct.initial_capital = req.initial_capital
    if req.params:
        acct.params.update(req.params)
    
    store.save(acct)
    return {"status": "updated", "strategy": acct.to_dict()}


@router.get("/{sid}/trades")
def get_strategy_trades(sid: str, limit: int = 50):
    """Get recent trades and decisions for a strategy."""
    import pandas as pd
    from pathlib import Path
    
    audit_path = Path("data/paper_trade_audit.parquet")
    if not audit_path.exists():
        return {"trades": []}
    
    try:
        df = pd.read_parquet(audit_path)
        if "strategy_key" in df.columns:
            df = df[df["strategy_key"] == sid]
        elif "sid" in df.columns:
            df = df[df["sid"] == sid]
        
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp", ascending=False)
        
        records = df.head(limit).to_dict(orient="records")
        
        # Convert NaN to None for JSON serialization
        for r in records:
            for k, v in list(r.items()):
                if pd.isna(v):
                    r[k] = None
        
        return {"trades": records}
    except Exception as e:
        logger.error(f"Error reading trades for {sid}: {e}")
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
    from app.services.market_service import fetch_today_event, fetch_event_markets
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
    
    # 關鍵修正：先將日期轉換為 Event Slug，再抓取市場數據
    today_event = fetch_today_event(target_date_str)
    slug = today_event.get("slug") if today_event else None
    markets = fetch_event_markets(slug, is_min_temp=req.is_min_temp) if slug else []
    
    if not markets:
        logger.warning(f"No Polymarket event found for {target_date_str}. Suggestions will lack market prices.")

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
            # 如果找不到市場價格，使用 0.5 作為預設值，並標記為 pass
            market_price = _find_market_price(markets, bucket_name)
            if market_price is None:
                market_price = 0.5

            # 計算 YES 和 NO 的 Edge
            edge_yes = model_prob - market_price
            edge_no = market_price - model_prob

            action = "pass"
            kelly = 0.0
            edge = 0.0

            # 只有在市場價格不是預設值時才計算 Edge
            if markets:
                if edge_yes > 0.01:
                    action = "buy_yes"
                    edge = edge_yes
                    kelly = calculate_kelly(model_prob, market_price, req.kelly_fraction)
                elif edge_no > 0.01:
                    action = "buy_no"
                    edge = edge_no
                    kelly = calculate_kelly(1 - model_prob, 1 - market_price, req.kelly_fraction)

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


# ── Run All Strategies Endpoint ──────────────────────────────────────────────

@router.get("/{sid}/chart")
def get_strategy_chart(
    sid: str,
    date: str | None = None,
    slug: str | None = None,
    model_key: str | None = None,
):
    """Return time-series data for the strategy chart.

    Reads from the snapshot SQLite database — no models or APIs are touched.
    Returns raw arrays for the frontend ECharts component to render.
    """
    from app.services.weather_service import hkt_now as _hkt_now
    target_date = date or _hkt_now().strftime("%Y-%m-%d")

    rows = read_snapshots(
        strategy_key=sid,
        slug=slug,
        date=target_date,
        model_key=model_key,
    )

    if not rows:
        return {
            "strategy_key": sid,
            "slug": slug or "",
            "date": target_date,
            "timestamps": [],
            "market_temps": [],
            "model_temps": [],
            "actual_temps": [],
            "trades": [],
        }

    # Load trades from audit log for the same strategy/date
    trades = _load_chart_trades(sid, target_date, slug)

    return {
        "strategy_key": sid,
        "slug": rows[0].get("slug", slug or ""),
        "date": target_date,
        "timestamps": [r["timestamp"] for r in rows],
        "market_temps": [r.get("pm_weighted_temp") for r in rows],
        "model_temps": [r.get("model_predicted_temp") for r in rows],
        "actual_temps": [r.get("actual_temp") for r in rows],
        "trades": trades,
    }


def _load_chart_trades(
    sid: str,
    date: str,
    slug: str | None = None,
) -> list[dict]:
    """Load trade events from the audit log for chart markers."""
    import pandas as pd
    from pathlib import Path

    audit_path = Path("data/paper_trade_audit.parquet")
    if not audit_path.exists():
        return []

    try:
        df = pd.read_parquet(audit_path)
        df = df[df["strategy_key"] == sid] if "strategy_key" in df.columns else df
        if slug and "slug" in df.columns:
            df = df[df["slug"] == slug]
        if "timestamp" in df.columns:
            df = df[df["timestamp"].str.startswith(date)]

        records = []
        for _, r in df.iterrows():
            action = r.get("action", "")
            if action in ("EXIT_ZERO", "EXIT_DUST", "FLIP", "EXIT", "NEW", "INCREASE", "DECREASE"):
                records.append({
                    "time": r.get("timestamp", ""),
                    "bucket": r.get("bucket", ""),
                    "action": action,
                    "qty": float(r.get("qty_after", 0)) - float(r.get("qty_before", 0)),
                    "side": r.get("side_after") or r.get("side_before", ""),
                })
        return records[-200:]
    except Exception as e:
        logger.warning("Failed to load trades for chart: %s", e)
        return []


@router.post("/run-all")
def run_all_strategies():
    """Run all enabled strategies once. Used by frontend polling and manual triggers."""
    from execution.strategy_runner import run_enabled_strategies_once
    results = run_enabled_strategies_once()
    return {"results": results, "total": len(results)}


# ── Background Scheduler ──────────────────────────────────────────────────────

_scheduler_thread: threading.Thread | None = None
_scheduler_alive = False

def _build_strategy_context(acct: StrategyAccount) -> dict:
    """Build context for strategy execution from account data."""
    from execution.market_templates import resolve_slug
    from app.services.weather_service import compute_rain_kwargs
    from app.services.market_service import fetch_event_markets
    
    target_date = hkt_now().date()
    target_date_str = target_date.strftime("%Y-%m-%d")
    _sd = target_date_str.replace("-", "")
    
    is_min_temp = acct.market_template == "hk-tmin"
    today_event = fetch_today_event(target_date_str)
    slug = today_event.get("slug") if today_event else None
    markets = fetch_event_markets(slug, is_min_temp=is_min_temp) if slug else []
    
    if not markets:
        logger.warning("No markets found for %s", slug)
        return {}
    
    hko = fetch_hko_data(target_date_str)
    state = get_intraday_state(_sd)
    rain_kwargs = compute_rain_kwargs(_sd, hkt_now())
    forecast_key = "forecast_min" if is_min_temp else "forecast_max"
    forecast_aws = hko.get(forecast_key) if hko else None
    params = acct.params or {}
    
    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=is_min_temp,
        bias=params.get("bias", 0.0),
        std_mult=params.get("std_mult", 1.0),
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=forecast_aws,
        forecast_max=hko.get("forecast_max") if hko else None,
        forecast_min=hko.get("forecast_min") if hko else None,
        is_today=True,
    )
    
    model = acct.model
    target_probs = results.get(model, {}).get("probs", {})
    if not target_probs:
        logger.error("Model %s produced no probs for strategy %s", model, acct.id)
        return {}
    
    prices_dict = {m["bucket"]: m.get("yes_price", 0.5) for m in markets}
    token_ids_dict = {m["bucket"]: m.get("token_id", "") for m in markets}

    # Read CLOB order-book depth from the background cache (refreshed every 10 s)
    depth_cache = get_global_depth_cache()
    depth_cache.update_token_ids(
        {b: t for b, t in token_ids_dict.items() if t}
    )
    market_depth = depth_cache.get()

    post_mean = results.get(model, {}).get("mean") if results else None

    model_stds = {}
    if results:
        for mk, pred in results.items():
            if mk != "_intraday_error" and pred.get("std") is not None:
                model_stds[mk] = pred["std"]

    context_json = {}
    if state:
        for k in ("temp_30m_ago", "temp_60m_ago", "temp_120m_ago",
                  "min_so_far", "rh_now", "temp_change_30m", "temp_change_60m",
                  "time_since_max", "time_since_min",
                  "temp_volatility_60m", "temp_acceleration_60m",
                  "rh_change_60m", "dew_point_change_60m",
                  "dew_point_spread_change_60m"):
            v = state.get(k)
            if v is not None:
                context_json[k] = v
    if hko:
        for k in ("max_since_midnight", "min_since_midnight", "forecast_max", "forecast_min"):
            v = hko.get(k)
            if v is not None:
                context_json[k] = v
    for k in ("rain_60m", "rain_120m", "rain_data_ok",
              "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
              "rainfall_30m_missing_flag", "rainfall_data_age_minutes",
              "rain_data_gap_flag", "rain_regime"):
        v = rain_kwargs.get(k)
        if v is not None:
            context_json[k] = v
    # buffer debug info for Model 2A stability monitoring
    if state and state.get("df_today") is not None:
        _df = state["df_today"]
        context_json["buffer_len"] = len(_df)
        if len(_df) >= 30:
            context_json["temp_at_idx30"] = float(_df["temp"].iloc[-30])
        if len(_df) >= 60:
            context_json["temp_at_idx60"] = float(_df["temp"].iloc[-60])
            if "rh" in _df.columns and _df["rh"].iloc[-60] is not None:
                context_json["rh_at_idx60"] = float(_df["rh"].iloc[-60])
    if model_stds:
        context_json["model_stds"] = model_stds
    # per-bucket probabilities for each model
    if results:
        _probs = {}
        for mk, pred in results.items():
            if mk != "_intraday_error" and pred.get("probs"):
                _probs[mk] = pred["probs"]
        if _probs:
            context_json["model_probs"] = _probs
    # Polymarket prices per bucket
    if prices_dict:
        context_json["market_prices"] = prices_dict
    # CLOB order-book depth per bucket
    if market_depth:
        context_json["market_depth"] = market_depth
    # Gamma market metadata (best bid/ask, spread, last trade, volume)
    gamma_market_info = {}
    for m in markets:
        bucket = m.get("bucket")
        if not bucket:
            continue
        info = {}
        for k in ("token_id", "conditionId", "bestBid", "bestAsk",
                  "spread", "lastTradePrice", "liquidityClob", "volume24hrClob"):
            v = m.get(k)
            if v is not None:
                info[k] = v
        if info:
            gamma_market_info[bucket] = info
    if gamma_market_info:
        context_json["gamma_market_info"] = gamma_market_info
    # Log all Model 2A features for stability diagnostics
    if results and "model_2a" in results:
        _m2a_raw = results["model_2a"].get("raw", {})
        _m2a_f = _m2a_raw.get("_features")
        if _m2a_f:
            context_json["model_2a_features"] = _m2a_f

    return dict(
        capital=acct.capital,
        model_key=model,
        mock_slippage=True,
        bias=params.get("bias", 0.0),
        std_mult=params.get("std_mult", 1.0),
        kelly_fraction=params.get("kelly_fraction", 0.25),
        slug=slug,
        target_probs=target_probs,
        prices_dict=prices_dict,
        token_ids_dict=token_ids_dict,
        temp_now=state.get("temp_now") if state else None,
        max_so_far=state.get("max_so_far") if state else None,
        rain_regime=rain_kwargs.get("rain_regime", "no_rain"),
        model_std=1.5,
        recent_price_volatility=0.0,
        hours_to_settlement=24.0,
        # snapshot extras
        markets=markets,
        post_mean=post_mean,
        is_min_temp=is_min_temp,
        target_date_str=target_date_str,
        all_results=results,
        context_json=context_json,
    )


def _write_cycle_snapshot(context: dict, acct) -> None:
    """Write a snapshot record from strategy context regardless of cycle status."""
    markets = context.get("markets", [])
    prices_dict = context.get("prices_dict", {})
    pm_temp = calc_pm_weighted_temp(markets, prices_dict)
    actual_temp = context.get("temp_now")
    max_so_far = context.get("max_so_far")
    post_mean = context.get("post_mean")
    model_predicted = calc_model_predicted_temp(max_so_far, post_mean)

    all_results = context.get("all_results", {})
    all_model_preds = {}
    for mk, pred in all_results.items():
        if mk != "_intraday_error" and pred.get("mean") is not None:
            all_model_preds[mk] = pred["mean"]

    write_snapshot({
        "timestamp": hkt_now().isoformat(),
        "snapshot_date": context.get("target_date_str", hkt_now().strftime("%Y-%m-%d")),
        "slug": context.get("slug", ""),
        "strategy_key": acct.id,
        "model_key": acct.model,
        "pm_weighted_temp": pm_temp,
        "model_predicted_temp": model_predicted,
        "actual_temp": actual_temp,
        "max_so_far": max_so_far,
        "predicted_upside": post_mean,
        "model_std": context.get("model_std", 1.5),
        "all_model_predictions": all_model_preds,
        "context_json": context.get("context_json"),
    })


def _scheduler_loop():
    """Background thread that runs enabled strategies on cooldown interval."""
    global _scheduler_alive
    logger.info("Strategy scheduler started")
    
    while _scheduler_alive:
        try:
            from execution.strategy_runner import run_single_strategy_cycle, load_strategy_registry
            from execution.strategy_account import get_store
            
            store = get_store()
            accounts = store.get_running()
            
            if not accounts:
                time.sleep(30)
                continue
            
            for acct in accounts:
                if acct.last_run:
                    try:
                        last = datetime.fromisoformat(acct.last_run)
                        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                        if elapsed < 300:
                            continue
                    except (ValueError, TypeError):
                        pass
                
                registry = load_strategy_registry()
                sdef = registry.get("strategies", {}).get(acct.id)
                if not sdef:
                    continue
                
                context = _build_strategy_context(acct)
                if context:
                    # ── write snapshot (always, before strategy cycle) ───
                    try:
                        _write_cycle_snapshot(context, acct)
                    except Exception as snap_err:
                        logger.warning("Failed to write snapshot for %s: %s", acct.id, snap_err)

                    result = run_single_strategy_cycle(
                        strategy_key=acct.id,
                        strategy_config=sdef,
                        portfolio_id=acct.id,
                        event_slug=context.get("slug"),
                        **context,
                    )
                    logger.info("Strategy %s executed: %s", acct.id, result.get("status"))
                    store.set_last_run(acct.id)
                
        except Exception as exc:
            logger.exception("Scheduler error: %s", exc)
        
        time.sleep(30)


def start_scheduler():
    """Start background scheduler thread (call from FastAPI lifespan)."""
    global _scheduler_thread, _scheduler_alive
    if _scheduler_thread and _scheduler_thread.is_alive():
        return

    _seed_default_accounts_if_empty()

    # Start the CLOB depth background cache (refreshes every 10 s)
    try:
        from app.services.market_depth_service import get_global_depth_cache
        get_global_depth_cache().start()
    except Exception as e:
        logger.warning("Failed to start depth cache: %s", e)

    _scheduler_alive = True
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="strategy-scheduler")
    _scheduler_thread.start()
    logger.info("Background strategy scheduler started")


def _seed_default_accounts_if_empty() -> None:
    """Seed default strategy accounts if none exist and a default file is present."""
    from pathlib import Path
    store = StrategyAccountStore()
    if store.list():
        return
    default_path = Path("config/default_strategy_accounts.json")
    if not default_path.exists():
        logger.warning("No default accounts file at %s — skip seeding", default_path)
        return
    import json
    raw = json.loads(default_path.read_text(encoding="utf-8"))
    entries = raw.get("strategies", {})
    for sid, data in entries.items():
        store.save(StrategyAccount.from_dict(data))
    logger.info("Seeded %d default strategy accounts from %s", len(entries), default_path)


def stop_scheduler():
    """Stop background scheduler thread."""
    global _scheduler_alive
    _scheduler_alive = False