# app/components/recommendation_table.py
"""Recommendation table — per-bucket edge, action, and Kelly sizing."""

import pandas as pd
import streamlit as st

from ..services.model_service import calculate_kelly


def recommendation_table(
    model_probs: dict[str, float],
    market_prices: dict[str, float],
    bucket_labels: list[str] | None = None,
    capital: float = 10000.0,
    kelly_frac: float = 0.5,
    kelly_label: str = "Half Kelly",
) -> None:
    """Render table of bucket-level model vs market with action recommendations.

    Args:
        model_probs: {bucket: model_probability}.
        market_prices: {bucket: yes_price}.
        bucket_labels: ordered bucket names (defaults to model_probs keys).
        capital: total capital for display context.
        kelly_frac: kelly fraction (0.25, 0.5, etc.).
        kelly_label: display label for kelly column header.
    """
    if not model_probs:
        st.info("No model predictions to show.")
        return

    labels = bucket_labels or list(model_probs.keys())
    rows = []
    for bucket in labels:
        p_mkt = market_prices.get(bucket, 0.5)
        p_mod = model_probs.get(bucket, 0.0)
        edge_yes = p_mod - p_mkt
        edge_no = (1.0 - p_mod) - (1.0 - p_mkt)
        kelly_yes = calculate_kelly(p_mod, p_mkt, kelly_frac)
        kelly_no = calculate_kelly(1.0 - p_mod, 1.0 - p_mkt, kelly_frac)

        if edge_yes > 0.02:
            action, kelly_pct = "🟢 BUY YES", kelly_yes
        elif edge_no > 0.02:
            action, kelly_pct = "🔴 BUY NO", kelly_no
        else:
            action, kelly_pct = "⚪ HOLD", 0.0

        rows.append({
            "Bucket": bucket,
            "Model": f"{p_mod:.1%}",
            "Market": f"{p_mkt:.1%}",
            "Edge (Yes)": f"{edge_yes:+.1%}",
            "Edge (No)": f"{edge_no:+.1%}",
            "Action": action,
            f"Kelly % ({kelly_label.split()[0]})": f"{kelly_pct:.1%}" if kelly_pct > 0 else "—",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
