# app/components/trade_history.py
"""Trade history expander — strategy decisions + paper-trader logs."""

import pandas as pd
import streamlit as st

from ..services.strategy_service import get_trade_history


def trade_history(
    last_decisions: list[dict] | None = None,
    limit: int = 100,
) -> None:
    """Render expander with strategy decisions and paper-trader trade logs.

    Args:
        last_decisions: list of strategy decision dicts from last run.
        limit: max number of paper-trader trades to show.
    """
    has_decisions = bool(last_decisions)
    trades = get_trade_history(limit=limit)
    has_trades = bool(trades)

    if not has_decisions and not has_trades:
        st.info("No trade history yet.")
        return

    with st.expander("📜 Trade History & Decisions", expanded=False):
        if has_decisions:
            st.markdown("**🧠 Latest Strategy Decisions**")
            df_dec = pd.DataFrame(last_decisions)
            show_cols = [c for c in ["bucket", "action", "reason", "detail"] if c in df_dec.columns]
            if show_cols:
                st.dataframe(df_dec[show_cols], use_container_width=True, hide_index=True)

        if has_trades:
            st.markdown("**📋 Paper Trader History**")
            rows = []
            for t in trades:
                rows.append({
                    "Time": str(t.get("created_at", "")),
                    "Market": t.get("market_question", t.get("market_slug", "")),
                    "Side": t.get("side", ""),
                    "Outcome": t.get("outcome", ""),
                    "Qty": t.get("shares", 0),
                    "Price": f"{t.get('avg_price', 0) * 100:.1f}¢",
                    "Amount": f"${t.get('amount_usd', 0):,.2f}",
                    "Fee": f"${t.get('fee', 0):,.4f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
