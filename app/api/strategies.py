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
from execution.strategy_account import StrategyAccount, StrategyAccountStore
from app.services.weather_service import hkt_now
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
    from app.services.market_service import fetch_event_markets, resolve_event_slug_for_kind
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
    try:
        slug = resolve_event_slug_for_kind(target_date, is_min_temp=req.is_min_temp)
    except Exception:
        slug = None
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
        input_status={
            "forecast_input_status": hko.get("forecast_input_status") if hko else None,
            "forecast_source": hko.get("forecast_source") if hko else None,
            "forecast_issue_time": hko.get("forecast_issue_time") if hko else None,
            "forecast_target_date": hko.get("forecast_target_date") if hko else None,
        },
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
    """Adapt the shared canonical cycle for this account.

    This function intentionally contains no network, model or CLOB fetch.
    The canonical builder owns those operations and is keyed by sampling
    cycle rather than account ID.
    """
    from app.services.canonical_cycle import (
        build_strategy_context_from_cycle,
        get_canonical_cycle,
    )

    try:
        cycle = get_canonical_cycle(is_min_temp=acct.market_template == "hk-tmin")
    except Exception as exc:
        logger.warning("Canonical cycle unavailable for %s: %s", acct.id, exc)
        return {}
    return build_strategy_context_from_cycle(cycle, acct)


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
        time.sleep(10)  # wait for first refresh cycle to complete
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
