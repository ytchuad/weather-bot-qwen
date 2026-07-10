# execution/auto_runner.py
"""Headless auto-runner — run enabled strategies outside of Streamlit.

Intended as a cron job entry point (GitHub Actions, Windows Task Scheduler).
Reads ``data/strategy_accounts.json``, runs every enabled strategy whose
cooldown has elapsed, and commits updated ``data/*.json`` back.

Usage
-----
    python -m execution.auto_runner                 # run all due strategies
    python -m execution.auto_runner --force          # skip cooldown check
    python -m execution.auto_runner --list           # list enabled strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_runner")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_accounts() -> dict:
    """Load ``data/strategy_accounts.json``."""
    p = Path("data/strategy_accounts.json")
    if not p.exists():
        logger.warning("strategy_accounts.json not found at %s", p.resolve())
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("strategies", {})


def is_due(acct: dict, force: bool = False) -> bool:
    """Check cooldown (default 5 min)."""
    if force:
        return True
    if not acct.get("scheduler_on"):
        return False
    if acct.get("status") != "running":
        return False
    last = acct.get("last_run")
    if last is None:
        return True
    try:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
        return elapsed >= 300
    except (TypeError, ValueError):
        return True


def run_strategy(sid: str, acct: dict, force: bool = False) -> dict:
    """Run one strategy cycle and return a result dict."""
    from execution.strategy_runner import run_single_strategy_cycle
    from execution.strategy_account import StrategyAccountStore
    from execution.market_templates import resolve_slug
    from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now, compute_rain_kwargs
    from app.services.market_service import fetch_today_event, fetch_event_markets
    from app.services.market_depth_service import get_global_depth_cache
    from app.services.model_service import run_all_models
    from features.strategy_snapshot_logger import (
        write_snapshot,
        calc_pm_weighted_temp,
        calc_model_predicted_temp,
    )

    store = StrategyAccountStore()

    # Load registry
    registry_path = Path("config/paper_strategies.json")
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        logger.error("paper_strategies.json not found")
        return {"status": "error", "error": "config not found"}

    sdef = registry.get("strategies", {}).get(sid)
    if sdef is None:
        logger.error("Strategy '%s' not in paper_strategies.json", sid)
        return {"status": "error", "error": f"strategy {sid} not in registry"}

    if not is_due(acct, force):
        return {"status": "skipped", "reason": "cooldown"}

    model = acct.get("model", "baseline")
    capital = acct.get("capital", 10_000.0)
    params = acct.get("params", {})
    template = acct.get("market_template", "hk-tmax")
    is_min_temp = template == "hk-tmin"

    try:
        # --- 資料抓取 ---
        target_date = hkt_now().date()
        target_date_str = target_date.strftime("%Y-%m-%d")
        _sd = target_date_str.replace("-", "")
        
        today_event = fetch_today_event(target_date_str)
        slug = today_event.get("slug") if today_event else resolve_slug(template)
        markets = fetch_event_markets(slug, is_min_temp=is_min_temp) if slug else []
        
        if not markets:
            logger.error("No markets found for %s", slug)
            return {"status": "error", "error": "No markets found"}
            
        hko = fetch_hko_data(target_date_str)
        state = get_intraday_state(_sd)
        rain_kwargs = compute_rain_kwargs(_sd, hkt_now())
        forecast_key = "forecast_min" if is_min_temp else "forecast_max"
        forecast_aws = hko.get(forecast_key) if hko else None
        
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
            is_today=True
        )
        
        target_probs = results.get(model, {}).get("probs", {})
        if not target_probs:
            logger.error("Model %s produced no probs", model)
            return {"status": "error", "error": "No model predictions"}
            
        prices_dict = {m["bucket"]: m.get("yes_price", 0.5) for m in markets}
        token_ids_dict = {m["bucket"]: m.get("token_id", "") for m in markets}
        no_token_ids_dict = {m["bucket"]: m.get("no_token_id", "") for m in markets}

        # Read CLOB depth from the background cache (refreshed every 10 s)
        depth_cache = get_global_depth_cache()
        depth_cache.update_token_ids(
            {b: t for b, t in token_ids_dict.items() if t},
            {b: t for b, t in no_token_ids_dict.items() if t},
        )
        market_depth = depth_cache.get()
        market_depth_no = depth_cache.get_no()

        # --- 構建 Context ---
        context = dict(
            capital=capital,
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
            nowcast_stale=False,
            data_missing=False,
            drawdown_pct=0.0,
            post_mean=results.get(model, {}).get("mean", 30.0)
        )

        # --- 執行策略 ---
        result = run_single_strategy_cycle(
            strategy_key=sid,
            strategy_config=sdef,
            portfolio_id=sid,
            event_slug=slug,
            **context,
        )

        store.set_last_run(sid)
        logger.info(
            "Cycle %s: status=%s  decisions=%s",
            sid, result.get("status"),
            len(result.get("decisions", [])),
        )

        # ── write snapshot for chart ─────────────────────────
        if result.get("status") == "completed":
            try:
                pm_temp = calc_pm_weighted_temp(markets, prices_dict)
                actual_temp = state.get("temp_now") if state else None
                max_so_far = state.get("max_so_far") if state else None
                post_mean = results.get(model, {}).get("mean") if results else None
                model_predicted = calc_model_predicted_temp(max_so_far, post_mean)

                all_model_preds = {}
                if results:
                    for mk, pred in results.items():
                        if mk != "_intraday_error" and pred.get("mean") is not None:
                            all_model_preds[mk] = pred["mean"]

                # ── assemble context_json for debugging ──
                _ctx = {}
                if state:
                    for k in ("temp_30m_ago", "temp_60m_ago", "temp_120m_ago",
                              "min_so_far", "rh_now", "temp_change_30m", "temp_change_60m",
                              "time_since_max", "time_since_min"):
                        v = state.get(k)
                        if v is not None:
                            _ctx[k] = v
                if hko:
                    for k in ("max_since_midnight", "min_since_midnight", "forecast_max", "forecast_min"):
                        v = hko.get(k)
                        if v is not None:
                            _ctx[k] = v
                for k in ("rain_60m", "rain_120m", "rain_data_ok",
                          "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
                          "rainfall_30m_missing_flag", "rainfall_data_age_minutes",
                          "rain_data_gap_flag", "rain_regime"):
                    v = rain_kwargs.get(k)
                    if v is not None:
                        _ctx[k] = v
                if results:
                    _stds = {}
                    _probs = {}
                    for mk, pred in results.items():
                        if mk != "_intraday_error":
                            if pred.get("std") is not None:
                                _stds[mk] = pred["std"]
                            if pred.get("probs"):
                                _probs[mk] = pred["probs"]
                    if _stds:
                        _ctx["model_stds"] = _stds
                    if _probs:
                        _ctx["model_probs"] = _probs
                # Polymarket prices per bucket
                if prices_dict:
                    _ctx["market_prices"] = prices_dict
                # CLOB order-book depth per bucket
                if market_depth:
                    _ctx["market_depth"] = market_depth
                if market_depth_no:
                    _ctx["market_depth_no"] = market_depth_no
                # Gamma market metadata
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
                    _ctx["gamma_market_info"] = gamma_market_info

                write_snapshot({
                    "timestamp": hkt_now().isoformat(),
                    "snapshot_date": target_date_str,
                    "slug": slug,
                    "strategy_key": sid,
                    "model_key": model,
                    "pm_weighted_temp": pm_temp,
                    "model_predicted_temp": model_predicted,
                    "actual_temp": actual_temp,
                    "max_so_far": max_so_far,
                    "predicted_upside": post_mean,
                    "model_std": context.get("model_std", 1.5),
                    "all_model_predictions": all_model_preds,
                    "context_json": _ctx,
                })
            except Exception as snap_err:
                logger.warning("Failed to write snapshot for %s: %s", sid, snap_err)

        return result

    except Exception as exc:
        logger.exception("Strategy %s failed: %s", sid, exc)
        return {"status": "error", "error": str(exc)}


def list_enabled() -> list[dict]:
    """Display enabled strategies."""
    accounts = load_accounts()
    if not accounts:
        print("No strategy accounts found.")
        return []

    rows = []
    for sid, acct in sorted(accounts.items()):
        on = acct.get("scheduler_on", False)
        status = acct.get("status", "unknown")
        model = acct.get("model", "?")
        capital = acct.get("capital", 0)
        last = acct.get("last_run", "never")[:19]
        rows.append({
            "id": sid,
            "active": "✅" if on and status == "running" else "⏸",
            "model": model,
            "capital": capital,
            "last_run": last,
            "market": acct.get("market_template", "hk-tmax"),
        })
        print(f"  {rows[-1]['active']} {sid:35s} model={model:15s} "
              f"capital={capital:>8.0f}  last={last}")
    return rows


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Headless strategy auto-runner for GitHub Actions cron",
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip cooldown check")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List enabled strategies and exit")
    args = parser.parse_args()

    if args.list:
        list_enabled()
        return

    accounts = load_accounts()
    if not accounts:
        logger.warning("No strategy accounts found. Run migration first:\n"
                       "  python -m execution.strategy_account migrate")
        return

    # Start CLOB depth cache
    try:
        from app.services.market_depth_service import get_global_depth_cache
        get_global_depth_cache().start()
    except Exception as e:
        logger.warning("Failed to start depth cache: %s", e)

    enabled = [(sid, acct) for sid, acct in accounts.items()
               if acct.get("scheduler_on") and acct.get("status") == "running"]

    if not enabled:
        logger.info("No running strategies (enable via Streamlit or edit "
                    "data/strategy_accounts.json directly).")
        return

    results = []
    for sid, acct in enabled:
        result = run_strategy(sid, acct, force=args.force)
        results.append({"strategy": sid, "result": result})
        status = result.get("status", "error")
        if status == "completed":
            logger.info("  ✅ %s completed", sid)
        elif status == "skipped":
            logger.info("  ⏸ %s skipped (cooldown)", sid)
        else:
            logger.warning("  ❌ %s: %s", sid, result.get("error", "unknown"))

    # Write run summary for auditing
    summary_path = Path("data/auto_runner_log.json")
    try:
        existing = []
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                existing = json.load(f)
        existing.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs": results,
        })
        # Keep last 1000 entries
        existing = existing[-1000:]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write run log: %s", exc)


if __name__ == "__main__":
    main()