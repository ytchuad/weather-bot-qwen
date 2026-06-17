# streamlit_app.py
"""Streamlit Cloud entry point.

When Streamlit Cloud runs this file, the repo root is on sys.path,
so absolute imports like ``from app.config import X`` work correctly.
The ``app/`` sub-modules use relative imports which also work because
they are imported as part of the ``app`` package.
"""
import sys
from pathlib import Path

# Guarantee the repo root is on sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Main dashboard logic ─────────────────────────────────────────────
import logging
import threading
import time
from datetime import datetime, timezone

import streamlit as st
from app.config import APP_TITLE, APP_FAVICON

logger = logging.getLogger(__name__)

# ---- page config (must be first st call) ----
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_FAVICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- page runners (direct import, no st.navigation) ----
from collections import OrderedDict
import json

from app.pages.page_hub import run as run_hub
from app.pages.page_intraday import run as run_intraday
from app.pages.page_strategies import run as run_strategies
from app.pages.page_analytics import run as run_analytics
from app.pages.page_health import run as run_health

_PAGE_RUNNERS = OrderedDict([
    ("hub", run_hub),
    ("intraday", run_intraday),
    ("strategies", run_strategies),
    ("analytics", run_analytics),
    ("health", run_health),
])


# ── Global UI: glass theme + top nav (rendered on EVERY page) ────────
from app.components.glass_theme import inject_glass_css
from app.components.top_nav import render_top_nav

inject_glass_css()

# Session-state-based page routing (no st.navigation / URL pathname change)
if "page" not in st.session_state:
    st.session_state.page = "hub"

_current_page_key = st.session_state.page
render_top_nav(current=_current_page_key)

# Hidden nav trigger zone — JS bridge clicks these buttons to trigger
# st.session_state.page changes + st.rerun() (no browser reload).
_PAGE_KEYS = list(_PAGE_RUNNERS.keys())

st.markdown('<div id="_nav-trigger-markers" class="top-nav-hidden" '
            f'data-keys=\'{json.dumps(_PAGE_KEYS)}\'></div>',
            unsafe_allow_html=True)

_nt_cols = st.columns(len(_PAGE_KEYS))
for _i, _key in enumerate(_PAGE_KEYS):
    with _nt_cols[_i]:
        if st.button("", key=f"_nt_{_key}"):
            st.session_state.page = _key
            st.rerun()

# SPA JS bridge — clicks hidden trigger button instead of window.location
_NAV_JS_BRIDGE = """<script>
(function(){
  var marker = document.getElementById('_nav-trigger-markers');
  if (!marker) { setTimeout(arguments.callee, 300); return; }
  if (document.body.hasAttribute('data-spa-wired')) return;
  document.body.setAttribute('data-spa-wired', '1');

  var keys = JSON.parse(marker.getAttribute('data-keys'));

  document.body.addEventListener('click', function(e){
    var link = e.target.closest('.top-nav-link[data-nav-key]');
    if (!link) return;
    e.preventDefault();
    e.stopPropagation();
    var key = link.getAttribute('data-nav-key');
    if (!key) return;

    /* Fade-out */
    var main = document.querySelector('section[data-testid="stMain"]');
    if (main) { main.style.opacity = '0'; main.style.transition = 'opacity 0.12s ease'; }

    /* Find hidden trigger button and click it */
    var idx = keys.indexOf(key);
    if (idx >= 0) {
      var container = marker.closest('[data-testid="stVerticalBlock"]');
      if (!container) return;
      var hblock = container.querySelector('div[data-testid="stHorizontalBlock"]');
      if (!hblock) return;
      var cols = hblock.querySelectorAll('div[data-testid="column"]');
      if (cols[idx]) {
        var btn = cols[idx].querySelector('button');
        if (btn) { setTimeout(function(){ btn.click(); }, 120); }
      }
    }
  });
})();
</script>"""
st.html(_NAV_JS_BRIDGE, unsafe_allow_javascript=True)


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
                    from app.services.strategy_service import load_strategy_registry

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

_PAGE_RUNNERS[st.session_state.page]()
