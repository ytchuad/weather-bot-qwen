# app/components/recommendation_cards.py
"""Card-per-bucket recommendation layout with profit/loss projections."""

from __future__ import annotations

import streamlit as st

from ..services.model_service import calculate_kelly
from ..config import COLORS


_CSS = """
<style>
.wqb-rec-wrap {
  display: flex; flex-wrap: wrap; gap: 10px;
  margin-bottom: 16px;
}
.wqb-rec-card {
  flex: 1 1 220px; min-width: 200px;
  background: #14171F;
  border: 1px solid #1F2330;
  border-radius: 10px;
  padding: 12px 14px;
  position: relative;
  overflow: hidden;
}
.wqb-rec-card-edge-green { border-top: 2px solid #00D68F; }
.wqb-rec-card-edge-red   { border-top: 2px solid #FF4D6D; }
.wqb-rec-card-edge-grey  { border-top: 2px solid #6B7280; }
.wqb-rec-bucket {
  font-size: 14px; font-weight: 600; color: #E6E9EF;
  margin-bottom: 8px;
}
.wqb-rec-edge-row {
  display: flex; justify-content: space-between;
  font-size: 12px; color: #6B7280;
  margin-bottom: 4px;
}
.wqb-rec-action {
  display: inline-block;
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 600;
  text-transform: uppercase;
}
.wqb-rec-action-yes { background: rgba(0, 214, 143, 0.15); color: #00D68F; }
.wqb-rec-action-no  { background: rgba(255, 77, 109, 0.15); color: #FF4D6D; }
.wqb-rec-action-hold{ background: rgba(107, 114, 128, 0.15); color: #6B7280; }
/* Total exposure summary card */
.wqb-rec-summary {
  flex: 1 1 100%;
  background: linear-gradient(135deg, #14171F 0%, #1A1F2E 100%);
  border: 1px solid #2A3040;
  border-radius: 10px;
  padding: 14px 18px;
  margin-bottom: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.wqb-rec-summary-label {
  font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;
}
.wqb-rec-summary-value {
  font-size: 22px; font-weight: 700; color: #00E5FF;
}
.wqb-rec-summary-sub {
  font-size: 12px; color: #6B7280; margin-left: 8px;
}
</style>
"""


def render_recommendation_cards(
    model_probs: dict,
    market_prices: dict,
    bucket_labels: list,
    capital: float,
    kelly_frac: float,
) -> None:
    """Render recommendation cards."""
    st.markdown(_CSS, unsafe_allow_html=True)

    if not model_probs:
        st.info("No model predictions to show.")
        return

    # Sort buckets by edge, descending
    labels = bucket_labels or list(market_prices.keys())
    rows = []
    for bucket in labels:
        p_mkt = market_prices.get(bucket, 0.5)
        p_mod = model_probs.get(bucket, 0.0)
        edge_yes = p_mod - p_mkt
        edge_no = (1.0 - p_mod) - (1.0 - p_mkt)
        kelly_yes = calculate_kelly(p_mod, p_mkt, kelly_frac)
        kelly_no = calculate_kelly(1.0 - p_mod, 1.0 - p_mkt, kelly_frac)

        if edge_yes > 0.02:
            action_text, kelly_pct = "BUY YES", kelly_yes
            action_class = "wqb-rec-action-yes"
            edge_sign = "green"
        elif edge_no > 0.02:
            action_text, kelly_pct = "BUY NO", kelly_no
            action_class = "wqb-rec-action-no"
            edge_sign = "red"
        else:
            action_text, kelly_pct = "HOLD", 0.0
            action_class = "wqb-rec-action-hold"
            edge_sign = "grey"

        rows.append({
            "bucket": bucket,
            "model": p_mod,
            "market": p_mkt,
            "edge": edge_yes if action_text == "BUY YES" else (edge_no if action_text == "BUY NO" else 0),
            "action_text": action_text,
            "action_class": action_class,
            "kelly_pct": kelly_pct,
            "edge_sign": edge_sign,
        })

    # Sort by edge descending
    rows.sort(key=lambda r: r["edge"], reverse=True)

    edge_rows = [r for r in rows if r["kelly_pct"] > 0]
    no_edge_rows = [r for r in rows if r["kelly_pct"] <= 0]

    # --- Total exposure summary -------------------------------------------
    total_exposure = sum(r["kelly_pct"] * capital for r in edge_rows)
    n_trades = len(edge_rows)
    st.markdown(
        f'<div class="wqb-rec-summary">'
        f'<div>'
        f'  <div class="wqb-rec-summary-label">Total Recommended Exposure</div>'
        f'  <div class="wqb-rec-summary-value">${total_exposure:,.0f}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'  <div class="wqb-rec-summary-label">Active Trades</div>'
        f'  <div style="font-size:18px; font-weight:600; color:#E6E9EF;">{n_trades}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Edge cards -------------------------------------------------------
    if edge_rows:
        st.markdown('<div class="wqb-rec-wrap">', unsafe_allow_html=True)
        for row in edge_rows:
            size = row["kelly_pct"] * capital
            card_cls = f"wqb-rec-card wqb-rec-card-edge-{row['edge_sign']}"
            html = (
                f'<div class="{card_cls}">'
                f'<div class="wqb-rec-bucket">{row["bucket"]}</div>'
                f'<div class="wqb-rec-edge-row">'
                f'  <span>Model: {row["model"]:.1%}</span>'
                f'  <span>Market: {row["market"]:.1%}</span>'
                f'</div>'
                f'<div style="margin:6px 0;">'
                f'  <span class="wqb-rec-action {row["action_class"]}">{row["action_text"]}</span>'
                f'  <span style="float:right; font-size:13px; font-weight:600; color:#E6E9EF;">${size:,.0f}</span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- No-edge buckets in expander --------------------------------------
    if no_edge_rows:
        with st.expander(f"📦 All buckets ({len(no_edge_rows)} with no edge)"):
            st.markdown('<div class="wqb-rec-wrap">', unsafe_allow_html=True)
            for row in no_edge_rows:
                card_cls = f"wqb-rec-card wqb-rec-card-edge-{row['edge_sign']}"
                html = (
                    f'<div class="{card_cls}">'
                    f'<div class="wqb-rec-bucket">{row["bucket"]}</div>'
                    f'<div class="wqb-rec-edge-row">'
                    f'  <span>Model: {row["model"]:.1%}</span>'
                    f'  <span>Market: {row["market"]:.1%}</span>'
                    f'</div>'
                    f'<div style="margin:6px 0;">'
                    f'  <span class="wqb-rec-action {row["action_class"]}">{row["action_text"]}</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
