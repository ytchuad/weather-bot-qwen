# app/pages/page_portfolio.py
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
        "Portfolio management is now part of the **Strategy Dashboard** — "
        "use the **Strategies** page in the sidebar instead.\n\n"
        "Each strategy is now a self-contained account with its own capital, "
        "model, and market template."
    )
    if st.button("📊 Go to Strategy Dashboard"):
        st.switch_page("app/pages/page_strategies.py")
