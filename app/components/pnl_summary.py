# app/components/pnl_summary.py
"""PnL summary metrics row."""

import streamlit as st


def pnl_summary(pnl_data: dict) -> None:
    """Render 5-column PnL summary: cost_basis, unrealized_pnl, total_fees, market_value, net_pnl.

    Args:
        pnl_data: dict from StrategyService.get_pnl() with keys like
            cost_basis, unrealized_pnl, total_fees, market_value.
    """
    total_fees = pnl_data.get("total_fees", 0.0)
    unrealized = pnl_data.get("unrealized_pnl", 0.0)
    net_pnl = unrealized - total_fees

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cost Basis", f"${pnl_data.get('cost_basis', 0):,.2f}")
    c2.metric(
        "Unrealized PnL",
        f"${unrealized:+,.2f}",
        delta_color="normal" if unrealized >= 0 else "inverse",
    )
    c3.metric("Total Fees", f"${total_fees:,.2f}")
    c4.metric("Market Value", f"${pnl_data.get('market_value', 0):,.2f}")
    c5.metric(
        "Net PnL (after fees)",
        f"${net_pnl:+,.2f}",
        delta_color="normal" if net_pnl >= 0 else "inverse",
    )
