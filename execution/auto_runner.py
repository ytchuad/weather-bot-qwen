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

    try:
        event_slug = resolve_slug(template)

        # Build context similar to what the Streamlit UI passes
        context = dict(
            capital=capital,
            model_key=model,
            mock_slippage=True,
            bias=params.get("bias", 0.0),
            std_mult=params.get("std_mult", 1.0),
            kelly_fraction=params.get("kelly_fraction", 0.25),
            portfolio_id=sid,
            slug=event_slug,
        )

        result = run_single_strategy_cycle(
            strategy_key=sid,
            strategy_config=sdef,
            portfolio_id=sid,
            event_slug=event_slug,
            **context,
        )

        store.set_last_run(sid)
        logger.info(
            "Cycle %s: status=%s  decisions=%s",
            sid, result.get("status"),
            len(result.get("decisions", [])),
        )
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