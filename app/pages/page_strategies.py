# app/pages/page_strategies.py
"""Strategy Dashboard — unified Live / Builder / Lab page.

Replaces the old Portfolio (page_portfolio.py) and Execute (page_execute.py)
pages with a single strategy-centric page.
"""

from __future__ import annotations

import streamlit as st

from ..state import AppState
from ..components.sidebar import render_sidebar
from ..components.strategy_card import strategy_card
from ..components.strategy_builder import strategy_builder_form
from execution.strategy_account import StrategyAccountStore


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ---- sidebar ----
    render_sidebar(state)

    st.header("📊 Strategy Dashboard")

    tabs = st.tabs(["📋 Live", "🔧 Builder", "🧪 Lab"])

    # ========================================================================
    # Tab 1: Live — strategy cards
    # ========================================================================
    with tabs[0]:
        st.subheader("Live Paper-Trading Strategies")
        st.caption("Toggle strategies ON to start live trading. Expand for detail.")

        store = StrategyAccountStore()
        accounts = store.list()

        if not accounts:
            st.warning(
                "No strategy accounts found. Use the **Builder** tab to "
                "create one, or run the migration: "
                "`python -m execution.strategy_account migrate`"
            )
        else:
            for acct in accounts:
                strategy_card(acct, state, store)

    # ========================================================================
    # Tab 2: Builder — create / edit strategy config
    # ========================================================================
    with tabs[1]:
        strategy_builder_form()

    # ========================================================================
    # Tab 3: Lab — synthetic backtest (same as old Execute Tab 5)
    # ========================================================================
    with tabs[2]:
        _render_lab()


# ── Lab tab (preserved from old page_execute.py Tab 5) ───────────────

def _render_lab() -> None:
    """Strategy Lab — paper-trade backtest with synthetic data."""
    st.subheader("🧪 Strategy Lab — Paper Trade Backtest")
    st.caption(
        "Run a synthetic backtest to compare strategy variants "
        "before deploying live."
    )

    try:
        from execution.strategy_factory import get_factory
        from execution.paper_trade_harness import (
            PaperTradeHarness,
            generate_synthetic_scenarios,
        )
        _lab_available = True
    except ImportError:
        _lab_available = False

    if not _lab_available:
        st.error("Strategy Lab modules not available. Check execution/ imports.")
        return

    factory = get_factory()
    strategy_keys = factory.keys()

    if not strategy_keys:
        st.warning("No strategies registered. Use the Builder tab to create one.")
        return

    # --- Configuration ---
    c1, c2 = st.columns(2)
    with c1:
        selected_strat = st.selectbox(
            "Strategy",
            strategy_keys,
            key="lab_strategy",
            help="Select which strategy variant to backtest",
        )
        n_cycles = st.number_input(
            "Number of cycles",
            min_value=10, max_value=500, value=100, step=10,
            help="Each cycle = one time snapshot across all buckets",
        )
        seed = st.number_input(
            "Random seed",
            min_value=0, max_value=9999, value=42, step=1,
            help="Fixed seed for reproducible results",
        )
    with c2:
        capital = st.number_input(
            "Starting capital ($)",
            min_value=100, value=10000, step=1000,
        )
        kelly_frac = st.slider(
            "Kelly fraction",
            min_value=0.05, max_value=1.0, value=0.25, step=0.05,
            help="Fraction of full Kelly to use. Lower = more conservative.",
        )
        hours_per_cycle = st.slider(
            "Hours per cycle",
            min_value=0.5, max_value=4.0, value=2.0, step=0.5,
        )

    temp_buckets = {
        "25-26": {"market_price": 0.05},
        "26-27": {"market_price": 0.08},
        "27-28": {"market_price": 0.12},
        "28-29": {"market_price": 0.18},
        "29-30": {"market_price": 0.20},
        "30-31": {"market_price": 0.15},
        "31-32": {"market_price": 0.10},
        "32-33": {"market_price": 0.07},
        "33-34": {"market_price": 0.04},
        "34-35": {"market_price": 0.02},
        "35-36": {"market_price": 0.01},
        ">=36": {"market_price": 0.005},
    }

    # --- Run backtest ---
    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_cycles}-cycle backtest for `{selected_strat}`..."):
            scenarios = generate_synthetic_scenarios(
                buckets=temp_buckets, n_cycles=n_cycles,
                capital=capital, seed=seed,
                hours_per_cycle=hours_per_cycle,
            )
            harness = PaperTradeHarness(
                strategy_or_key=selected_strat,
                capital=capital, kelly_fraction=kelly_frac, seed=seed,
            )
            result = harness.run(scenarios)

        st.session_state["lab_last_result"] = result
        _display_lab_result(result, n_cycles, selected_strat)

    # --- Compare all strategies ---
    st.markdown("---")
    st.subheader("⚖️ Compare All Strategies")
    if st.button("🔄 Run Comparison", use_container_width=True):
        comparison_rows = []
        progress = st.progress(0)
        for idx, sk in enumerate(strategy_keys):
            scenarios = generate_synthetic_scenarios(
                buckets=temp_buckets, n_cycles=n_cycles,
                capital=capital, seed=seed,
            )
            harness = PaperTradeHarness(
                strategy_or_key=sk, capital=capital,
                kelly_fraction=kelly_frac, seed=seed,
            )
            result = harness.run(scenarios)
            s = result.summary()
            comparison_rows.append({
                "Strategy": sk,
                "PnL": s["total_pnl"],
                "Return %": s["total_return_pct"],
                "Max DD %": s["max_drawdown"],
                "Sharpe": s["sharpe_ratio"],
                "Trades": s["num_trades"],
            })
            progress.progress((idx + 1) / len(strategy_keys))

        if comparison_rows:
            import pandas as pd
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            best = comp_df.loc[comp_df["Return %"].idxmax()]
            st.success(
                f"🏆 Best strategy: **{best['Strategy']}** "
                f"(Return: {best['Return %']:+.2f}%, "
                f"Sharpe: {best['Sharpe']:.3f})"
            )


def _display_lab_result(result, n_cycles, selected_strat):
    """Display a single backtest result."""
    import pandas as pd
    summary = result.summary()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final Capital", f"${result.final_capital:,.2f}")
    m2.metric("Total PnL", f"${summary['total_pnl']:+,.2f}",
              delta=f"{summary['total_return_pct']:+.2f}%")
    m3.metric("Max Drawdown", f"{summary['max_drawdown']:.2f}%")
    m4.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.3f}")
    m5.metric("Trades", f"{summary['num_trades']}")

    if result.trade_log:
        st.subheader("📋 Trade Log")
        trade_df = pd.DataFrame([{
            "Bucket": t.bucket,
            "Entry Time": t.entry_time.strftime("%H:%M"),
            "Exit Time": t.exit_time.strftime("%H:%M"),
            "Qty": round(t.quantity, 2),
            "Side": t.side,
            "Entry": round(t.entry_price, 3),
            "Exit": round(t.exit_price, 3),
            "PnL": round(t.pnl, 2),
            "Reason": t.reason,
        } for t in result.trade_log[-50:]])
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

    if result.capital_history and len(result.capital_history) > 1:
        st.subheader("📈 Capital History")
        cap_df = pd.DataFrame(
            [(t.strftime("%H:%M"), c) for t, c in result.capital_history],
            columns=["Time", "Capital"],
        )
        st.line_chart(cap_df, x="Time", y="Capital")

    if result.cycle_results:
        st.subheader("🔬 Recent Cycle Details")
        recent = result.cycle_results[-20:]
        cycle_df = pd.DataFrame([{
            "Time": cr.time.strftime("%H:%M"),
            "Bucket": cr.bucket,
            "Edge": cr.edge,
            "Entry": "✅" if cr.entry_ok else "❌",
            "Sizing": f"{cr.sizing_factor:.2f}x",
            "Action": cr.action,
            "Qty": round(cr.action_qty, 2),
            "Rebalance": "⚡" if cr.rebalance_triggered else "—",
        } for cr in recent if cr.action != "NONE"])
        if not cycle_df.empty:
            st.dataframe(cycle_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active cycles in the last 20 snapshots.")
