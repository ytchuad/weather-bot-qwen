# app/main.py
"""Weather Quant Dashboard — modular Streamlit multi-page entry point."""

import logging
import threading
import time
from datetime import datetime, timezone

import streamlit as st

from .config import APP_TITLE, APP_FAVICON

logger = logging.getLogger(__name__)

# ---- page config (must be first st call) ----
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- import pages ----
from .pages.page_hub import run as run_hub
from .pages.page_intraday import run as run_intraday
from .pages.page_strategies import run as run_strategies
from .pages.page_analytics import run as run_analytics
from .pages.page_health import run as run_health


# ---- page definitions ----
hub_page = st.Page(
    run_hub,
    title="Hub",
    icon="📊",
    url_path="hub",
    default=True,
)

intraday_page = st.Page(
    run_intraday,
    title="Intraday",
    icon="📈",
    url_path="intraday",
)

strategies_page = st.Page(
    run_strategies,
    title="Strategies",
    icon="⚡",
    url_path="strategies",
)

analytics_page = st.Page(
    run_analytics,
    title="Analytics",
    icon="📉",
    url_path="analytics",
)

health_page = st.Page(
    run_health,
    title="Health",
    icon="🏥",
    url_path="health",
)


# ---- navigation ----
pg = st.navigation(
    {
        "📊 Dashboard": [hub_page, intraday_page],
        "💼 Trading": [strategies_page],
        "📉 Analysis": [analytics_page, health_page],
    }
)

# ---- sidebar footer (after nav so it appears at bottom) ----
st.sidebar.markdown("---")
st.sidebar.caption("Weather Quant Bot · HKO × Polymarket")

# ── Auto-runner daemon ────────────────────────────────────────────────
# Starts once per session, loops until the page closes.  It polls
# strategy_accounts.json and runs any strategy with scheduler_on=True
# that is due.  Writes results back to JSON so the Live tab picks them
# up on next re-render.

_scheduler_interval_sec = 30


def _scheduler_loop():
    """Session-level background thread."""
    import logging as _log
    _log.basicConfig(level=_log.WARNING)
    _log.getLogger("execution.scheduler").setLevel(_log.INFO)
    _log_sched = _log.getLogger("execution.scheduler")

    while st.session_state.get("_scheduler_alive", False):
        try:
            from execution.strategy_account import StrategyAccountStore, get_store

            store = get_store()
            running = store.get_running()

            for acct in running:
                try:
                    if acct.last_run:
                        last = datetime.fromisoformat(acct.last_run)
                        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                        if elapsed < 300:  # 5-min cooldown
                            continue

                    # Build minimal context and run via existing service
                    from execution.market_templates import resolve_slug
                    from execution.strategy_runner import run_single_strategy_cycle
                    from .services.strategy_service import load_strategy_registry

                    registry = load_strategy_registry()
                    sdef = registry.get("strategies", {}).get(acct.id)
                    if sdef is None:
                        continue

                    event_slug = resolve_slug(acct.market_template)
                    params = acct.params or {}

                    context = dict(
                        capital=acct.capital,
                        model_key=acct.model,
                        mock_slippage=True,
                        bias=params.get("bias", 0.0),
                        std_mult=params.get("std_mult", 1.0),
                        kelly_fraction=params.get("kelly_fraction", 0.25),
                        portfolio_id=acct.id,
                        slug=event_slug,
                    )

                    result = run_single_strategy_cycle(
                        strategy_key=acct.id,
                        strategy_config=sdef,
                        portfolio_id=acct.id,
                        event_slug=event_slug,
                        **context,
                    )

                    store.set_last_run(acct.id)
                    _log_sched.info("Ran %s: %s", acct.id, result.get("status"))

                except Exception as exc:
                    _log_sched.warning("Scheduler cycle for %s failed: %s", acct.id, exc)

        except Exception as exc:
            _log_sched.warning("Scheduler iteration failed: %s", exc)

        time.sleep(_scheduler_interval_sec)


if "_scheduler_started" not in st.session_state:
    st.session_state["_scheduler_alive"] = True
    st.session_state["_scheduler_started"] = True
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler")
    t.start()

pg.run()
