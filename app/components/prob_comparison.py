# app/components/prob_comparison.py
"""Model vs Market probability comparison — interactive pill selector + horizontal bars."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from ..config import COLORS, MODEL_LABELS, TMAX_BUCKETS, TMIN_BUCKETS


_CSS = """
<style>
.wqb-prob-panel {
  background: #14171F;
  border: 1px solid #1F2330;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 16px;
}
</style>
"""


def render_prob_comparison(
    selected_model: str,
    all_results: dict,
    market_prices: dict,
    bucket_labels: list,
) -> None:
    """Render probability comparison panel."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="wqb-prob-panel">', unsafe_allow_html=True)

    # Ensure selected model is valid
    available = list(all_results.keys())
    if selected_model not in available:
        selected_model = available[0] if available else ""

    # Get probs for the selected model
    selected_pred = all_results.get(selected_model, {})
    selected_probs = selected_pred.get("probs", {})

    # Build figure
    if not market_prices and not selected_probs:
        st.info("No market or model data to display.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    labels = bucket_labels or list(market_prices.keys())
    mk_probs = [market_prices.get(b, 0) for b in labels]
    model_probs = [selected_probs.get(b, 0) for b in labels]

    # Edge annotations
    edges = []
    for b in labels:
        p_mkt = market_prices.get(b, 0.5)
        p_mod = selected_probs.get(b, 0.0)
        edges.append(p_mod - p_mkt)

    fig = go.Figure()

    # Market bars
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

    # Selected model bars
    fig.add_trace(go.Bar(
        name=MODEL_LABELS.get(selected_model, selected_model),
        x=labels,
        y=model_probs,
        marker_color=COLORS["primary"],
        marker_line=dict(color=COLORS["primary"], width=1),
        opacity=0.7,
        text=[f"{v:.1%}" for v in model_probs],
        textposition="outside",
        textfont=dict(color=COLORS["primary"], size=10),
        hovertemplate="%{data.name}: %{y:.1%}<extra></extra>",
    ))

    # Edge annotations as scatter trace (dots + text between bars)
    edge_colors = ["#00D68F" if e >= 0 else "#FF4D6D" for e in edges]
    edge_texts = [f"{e:+.1%}" for e in edges]
    fig.add_trace(go.Scatter(
        x=labels,
        y=[max(mk_probs[i], model_probs[i]) + 0.05 for i in range(len(labels))],
        mode="text",
        text=edge_texts,
        textposition="middle center",
        textfont=dict(size=9, color=edge_colors),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        barmode="group",
        height=320,
        yaxis=dict(tickformat=".0%", range=[0, 1.15]),
        margin=dict(l=40, r=20, t=40, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
