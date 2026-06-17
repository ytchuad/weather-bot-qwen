# app/components/bucket_bars.py
"""Bucket probability comparison chart — Model(s) vs Market."""

import streamlit as st
import plotly.graph_objects as go

from ..config import COLORS


def bucket_bars(
    probs_dict: dict[str, dict[str, float]],
    market_prices: dict[str, float],
    bucket_labels: list[str],
    title: str = "Probability Comparison",
) -> None:
    """Render grouped bar chart comparing model probabilities to market prices.

    Args:
        probs_dict: {model_name: {bucket: probability}}.
        market_prices: {bucket: yes_price}.
        bucket_labels: ordered list of bucket names for x-axis.
        title: chart title.
    """
    if not market_prices and not probs_dict:
        st.info("No market or model data to display.")
        return

    labels = bucket_labels or list(market_prices.keys())
    mk_probs = [market_prices.get(b, 0) for b in labels]

    fig = go.Figure()

    # Market bars (outline style)
    fig.add_trace(go.Bar(
        name="Market",
        x=labels,
        y=mk_probs,
        marker_color="rgba(255,255,255,0)",
        marker_line=dict(color=COLORS["market"], width=2.5),
        text=[f"{v:.1%}" for v in mk_probs],
        textposition="outside",
        textfont=dict(color=COLORS["market"], size=10),
        hovertemplate="Market: %{y:.1%}<extra></extra>",
    ))

    color_list = [COLORS.get(k, COLORS["neutral"]) for k in probs_dict]
    for i, (model_name, model_dict) in enumerate(probs_dict.items()):
        if not model_dict:
            continue
        m_probs = [model_dict.get(b, 0) for b in labels]
        color = color_list[i] if i < len(color_list) else COLORS["neutral"]
        fig.add_trace(go.Bar(
            name=model_name,
            x=labels,
            y=m_probs,
            marker_color=color,
            marker_line=dict(color=color, width=1),
            opacity=0.7,
            text=[f"{v:.1%}" for v in m_probs],
            textposition="outside",
            textfont=dict(color=color, size=10),
            hovertemplate=f"{model_name}: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        barmode="group",
        height=320,
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(l=40, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
