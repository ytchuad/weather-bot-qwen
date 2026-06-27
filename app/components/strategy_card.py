# app/components/strategy_card.py
"""Strategy card — a self-contained card for one strategy account.

Rendered in the Live tab of the Strategy Dashboard.  Each card shows:
  - status toggle, model, capital, market
  - PnL, Sharpe, max DD, trade count
  - Expandable detail: positions, trade history, gate results
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..state import AppState
from execution.strategy_account import StrategyAccount, StrategyAccountStore
from execution.market_templates import resolve_slug


def strategy_card(
    acct: StrategyAccount,
    state: AppState,
    store: StrategyAccountStore,
) -> None:
    """Render one strategy card (collapsed by default)."""
    sid = acct.id
    capital = acct.capital
    params = acct.params or {}

    # Resolve today's event slug from the market template
    event_slug = resolve_slug(acct.market_template)

    with st.container(border=True):
        # ── Header row ──────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1.5])
        with c1:
            status_icon = "⚡" if acct.status == "running" else "⏸"
            st.markdown(f"**{status_icon} {acct.label}**  `{sid}`")

        with c2:
            new_capital = st.number_input(
                "Capital ($)",
                value=int(capital),
                step=1000,
                key=f"capital_{sid}",
                label_visibility="collapsed",
            )

        with c3:
            new_status = st.toggle(
                "Active",
                value=acct.scheduler_on,
                key=f"toggle_{sid}",
                help="Enable live paper trading",
            )

        with c4:
            if st.button("🔄 Run Now", key=f"run_{sid}", use_container_width=True):
                _run_single_strategy(sid, acct, state)

        # Update capital / status if changed
        if new_capital != int(capital):
            acct.capital = float(new_capital)
            store.save(acct)
            st.rerun()

        if new_status != acct.scheduler_on:
            acct.scheduler_on = new_status
            acct.status = "running" if new_status else "paused"
            store.save(acct)
            st.rerun()

        # ── Info row ────────────────────────────────────────────────
        model_label = {
            "baseline": "Baseline", "rain_nowcast": "Rain Nowcast",
            "rain_observed": "Rain Observed", "model_a": "Model A",
            "model_b": "Model B", "model_c": "Model C", "model_d": "Model D",
            "model_e": "Model E", "model_g": "Model G (Gap+Max)", "model_2a": "Model 2A (Core+Wind)",
            "9d": "9-Day", "aws": "AWS HF",
        }.get(acct.model, acct.model)

        st.markdown(
            f"📊 **Model:** {model_label} · "
            f"📅 **Market:** {event_slug} · "
            f"🔄 **Kelly:** {params.get('kelly_fraction', 0.25):.0%} · "
            f"📏 **Bias:** {params.get('bias', 0.0):+.1f}°C · "
            f"📐 **Std Mult:** {params.get('std_mult', 1.0):.1f}x"
        )

        # ── PnL summary row (read from data files) ──────────────────
        _render_pnl_summary(sid)

        # ── Expandable detail ───────────────────────────────────────
        with st.expander("📊 Details", expanded=False):
            _render_strategy_detail(sid, acct, event_slug, state)


def _render_pnl_summary(sid: str) -> None:
    """Read PnL from data files and show metrics row."""
    try:
        from execution.portfolio_manager import get_portfolio_pnl
        pnl = get_portfolio_pnl(sid, current_prices={})
    except Exception:
        pnl = None

    if pnl and pnl.get("details"):
        upnl = pnl.get("unrealized_pnl", 0)
        mv = pnl.get("market_value", 0)
        cost = pnl.get("cost_basis", 0)
        ret = (upnl / cost * 100) if cost > 0 else 0.0

        cols = st.columns(4)
        cols[0].metric("Unrealized PnL", f"${upnl:+,.2f}", delta=f"{ret:+.1f}%")
        cols[1].metric("Market Value", f"${mv:,.2f}")
        cols[2].metric("Cost Basis", f"${cost:,.2f}")
        cols[3].metric("Positions", len(pnl.get("details", [])))
    else:
        st.caption("No PnL data yet — run the strategy to see results.")


def _render_strategy_detail(
    sid: str, acct: StrategyAccount, event_slug: str, state: AppState,
) -> None:
    """Expanded view: positions, trade history, and per-strategy params."""
    tab_pos, tab_trades, tab_params = st.tabs(
        ["📦 Positions", "📋 Trade History", "⚙️ Params"]
    )

    # ── Positions tab ──────────────────────────────────────────────
    with tab_pos:
        _render_positions(sid, event_slug)

    # ── Trade history tab ──────────────────────────────────────────
    with tab_trades:
        _render_trade_history(sid)

    # ── Per-strategy params tab ────────────────────────────────────
    with tab_params:
        _render_strategy_params(sid, acct)


def _render_positions(sid: str, event_slug: str) -> None:
    """Show positions for this strategy from current_positions.json."""
    try:
        from execution.portfolio_manager import get_portfolio_positions
        positions = get_portfolio_positions(sid)
    except Exception:
        positions = {}

    if not positions:
        st.info("No open positions.")
        return

    rows = []
    for slug, strat_data in positions.items():
        for sk, buckets in (strat_data if isinstance(strat_data, dict) else {}).items():
            for bucket, pos in buckets.items():
                rows.append({
                    "Event": slug,
                    "Bucket": bucket,
                    "Side": pos.get("side", ""),
                    "Qty": round(pos.get("quantity", 0), 2),
                    "Entry $": round(pos.get("entry_price", 0), 4),
                })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")


def _render_trade_history(sid: str) -> None:
    """Show trade history filtered to this strategy."""
    try:
        from execution.paper_adapter import _get_adapter
        trades = _get_adapter().get_trade_history(limit=100)
    except Exception:
        trades = []

    if not trades:
        st.info("No trades yet.")
        return

    df = pd.DataFrame(trades)
    display_cols = [c for c in ["timestamp", "event_slug", "bucket", "side",
                                 "quantity", "price", "pnl", "reason"]
                    if c in df.columns]
    if display_cols:
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_strategy_params(sid: str, acct: StrategyAccount) -> None:
    """Editable per-strategy parameters."""
    from ..services.strategy_service import load_strategy_registry, save_strategy_registry

    params = dict(acct.params or {})

    cola, colb = st.columns(2)
    with cola:
        new_bias = st.slider(
            "Bias offset (°C)", -2.0, 2.0, params.get("bias", 0.0), 0.1,
            key=f"bias_{sid}",
        )
        new_std = st.slider(
            "Std multiplier", 0.5, 2.0, params.get("std_mult", 1.0), 0.1,
            key=f"std_{sid}",
        )
    with colb:
        new_kelly = st.slider(
            "Kelly fraction", 0.05, 1.0, params.get("kelly_fraction", 0.25), 0.05,
            key=f"kelly_{sid}",
        )

    if st.button("💾 Save Params", key=f"save_params_{sid}"):
        acct.params = {
            "bias": new_bias,
            "std_mult": new_std,
            "kelly_fraction": new_kelly,
        }
        store = StrategyAccountStore()
        store.save(acct)
        st.success("Params saved!")
        st.rerun()

    # ── Gate config (read-only for now) ─────────────────────────────
    st.markdown("**Gate config** (read-only — edit in Builder tab)")
    if acct.from_strategy_key:
        registry = load_strategy_registry()
        strategy_def = registry.get("strategies", {}).get(acct.from_strategy_key, {})
        if strategy_def:
            st.code(
                f"Strategy: {acct.from_strategy_key}\n"
                f"Entry: {strategy_def.get('entry_point', 'N/A')}\n"
                f"Module: {strategy_def.get('module', 'N/A')}",
                language="text",
            )


def _run_single_strategy(sid: str, acct: StrategyAccount, state: AppState) -> None:
    """Execute one cycle of this strategy."""
    try:
        from execution.strategy_runner import run_single_strategy_cycle
        from ..services.strategy_service import load_strategy_registry

        registry = load_strategy_registry()
        sdef = registry.get("strategies", {}).get(sid)
        if sdef is None:
            st.error(f"Strategy '{sid}' not found in registry")
            return

        event_slug = resolve_slug(acct.market_template)

        # Build context from strategy params
        params = acct.params or {}
        context = dict(
            capital=acct.capital,
            model_key=acct.model,
            mock_slippage=True,
            slug=event_slug,
            bias=params.get("bias", 0.0),
            std_mult=params.get("std_mult", 1.0),
            kelly_fraction=params.get("kelly_fraction", 0.25),
            portfolio_id=sid,
        )

        result = run_single_strategy_cycle(
            strategy_key=sid,
            strategy_config=sdef,
            portfolio_id=sid,
            event_slug=event_slug,
            **context,
        )

        # Update last_run
        store = StrategyAccountStore()
        store.set_last_run(sid)

        if result.get("status") == "completed":
            st.success(f"Cycle completed — {result.get('error', '')}")
        elif result.get("status") == "error":
            st.error(f"Cycle error: {result.get('error')}")
        else:
            st.info(f"Cycle {result.get('status')}: {result.get('error', 'no detail')}")

    except Exception as e:
        st.error(f"Failed to run strategy: {e}")
