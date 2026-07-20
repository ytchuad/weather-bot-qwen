# execution/auto_runner.py
"""Headless auto-runner backed by the shared canonical sampling cycle.

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
from collections.abc import Mapping
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
    path = Path("data/strategy_accounts.json")
    if not path.exists():
        logger.warning("strategy_accounts.json not found at %s", path.resolve())
        return {}
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
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
    """Run one account from one shared canonical cycle."""
    from app.services.canonical_cycle import (
        build_strategy_context_from_cycle,
        get_canonical_cycle,
    )
    from execution.strategy_account import StrategyAccount, StrategyAccountStore
    from execution.strategy_runner import run_single_strategy_cycle

    registry_path = Path("config/paper_strategies.json")
    if not registry_path.exists():
        logger.error("paper_strategies.json not found")
        return {"status": "error", "error": "config not found"}
    with registry_path.open(encoding="utf-8") as handle:
        registry = json.load(handle)

    strategy_config = registry.get("strategies", {}).get(sid)
    if strategy_config is None:
        logger.error("Strategy '%s' not in paper_strategies.json", sid)
        return {"status": "error", "error": f"strategy {sid} not in registry"}
    if not is_due(acct, force):
        return {"status": "skipped", "reason": "cooldown"}

    try:
        account_data = dict(acct)
        account_data.setdefault("id", sid)
        account = StrategyAccount.from_dict(account_data)
        cycle = get_canonical_cycle(is_min_temp=account.market_template == "hk-tmin")
        context = build_strategy_context_from_cycle(cycle, account)
        if not context:
            logger.error("No strategy context for %s", sid)
            return {"status": "error", "error": "No strategy context"}

        result = run_single_strategy_cycle(
            strategy_key=sid,
            strategy_config=strategy_config,
            portfolio_id=sid,
            event_slug=context.get("slug"),
            **context,
        )
        StrategyAccountStore().set_last_run(sid)
        logger.info(
            "Cycle %s: status=%s decisions=%s",
            sid,
            result.get("status"),
            len(result.get("decisions", [])),
        )

        # Preserve the existing chart snapshot side effect.  This is a
        # strategy/presentation log, not Layer A persistence.
        if result.get("status") == "completed":
            try:
                from app.services.weather_service import hkt_now
                from features.strategy_snapshot_logger import (
                    calc_model_predicted_temp,
                    calc_pm_weighted_temp,
                    write_snapshot,
                )

                markets = context.get("markets", [])
                prices = context.get("prices_dict", {})
                all_results = context.get("all_results", {})
                all_model_predictions = {
                    str(model_key): prediction.get("mean")
                    for model_key, prediction in all_results.items()
                    if not str(model_key).startswith("_")
                    and isinstance(prediction, Mapping)
                    and prediction.get("mean") is not None
                }
                post_mean = context.get("post_mean")
                max_so_far = context.get("max_so_far")
                write_snapshot(
                    {
                        "timestamp": hkt_now().isoformat(),
                        "snapshot_date": context.get("target_date_str", hkt_now().strftime("%Y-%m-%d")),
                        "slug": context.get("slug", ""),
                        "strategy_key": sid,
                        "model_key": account.model,
                        "pm_weighted_temp": calc_pm_weighted_temp(markets, prices),
                        "model_predicted_temp": calc_model_predicted_temp(max_so_far, post_mean),
                        "actual_temp": context.get("temp_now"),
                        "max_so_far": max_so_far,
                        "predicted_upside": post_mean,
                        "model_std": context.get("model_std", 1.5),
                        "all_model_predictions": all_model_predictions,
                        "context_json": context.get("context_json"),
                    }
                )
            except Exception as snapshot_error:
                logger.warning("Failed to write snapshot for %s: %s", sid, snapshot_error)

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
        print(
            f"  {rows[-1]['active']} {sid:35s} model={model:15s} "
            f"capital={capital:>8.0f}  last={last}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless strategy auto-runner for GitHub Actions cron",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Skip cooldown check")
    parser.add_argument("--list", "-l", action="store_true", help="List enabled strategies and exit")
    args = parser.parse_args()

    if args.list:
        list_enabled()
        return

    accounts = load_accounts()
    if not accounts:
        logger.warning(
            "No strategy accounts found. Run migration first:\n"
            "  python -m execution.strategy_account migrate"
        )
        return

    # Start one shared background CLOB cache; run_strategy only reads its
    # canonical bundle and never starts a per-account refresh.
    try:
        from app.services.market_depth_service import get_global_depth_cache

        get_global_depth_cache().start()
    except Exception as exc:
        logger.warning("Failed to start depth cache: %s", exc)

    enabled = [
        (sid, acct)
        for sid, acct in accounts.items()
        if acct.get("scheduler_on") and acct.get("status") == "running"
    ]
    if not enabled:
        logger.info(
            "No running strategies (enable via Streamlit or edit "
            "data/strategy_accounts.json directly)."
        )
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

    # Write run summary for auditing.
    summary_path = Path("data/auto_runner_log.json")
    try:
        existing = []
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
        existing.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs": results,
        })
        existing = existing[-1000:]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write run log: %s", exc)


if __name__ == "__main__":
    main()
