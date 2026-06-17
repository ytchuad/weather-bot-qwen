# app/pages/page_execute.py
"""DEPRECATED — merged into page_strategies.py.

This page has been replaced by the unified Strategy Dashboard
(``app/pages/page_strategies.py``) which combines portfolio management,
strategy execution, and the strategy lab into a single page.

Use the **Strategies** page (tab: Live / Builder / Lab) instead.
"""

import streamlit as st


def run() -> None:
    st.error(
        "This page has been removed. 🗑️\n\n"
        "Strategy execution is now part of the **Strategy Dashboard** — "
        "use the **Strategies** page in the sidebar instead.\n\n"
        "• **Live** tab — toggle strategies on/off, view PnL, expand for details\n"
        "• **Builder** tab — create/edit strategy config, select model, tune gates\n"
        "• **Lab** tab — synthetic backtest before deploying live"
    )
    if st.button("📊 Go to Strategy Dashboard"):
        st.switch_page("app/pages/page_strategies.py")
