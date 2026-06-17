# app/components/kpi_strip.py
"""Five-hero KPI cards for the top of the Hub page."""

from __future__ import annotations

import streamlit as st

from ..services.model_service import calculate_kelly
from ..config import COLORS, MODEL_LABELS


_CSS = """
<style>
.wqb-kpi-wrap {
  display: flex; gap: 12px; flex-wrap: wrap;
  margin-bottom: 16px;
}
.wqb-kpi-card {
  flex: 1 1 18%; min-width: 160px;
  background: #14171F;
  border: 1px solid #1F2330;
  border-radius: 10px;
  padding: 14px 16px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.wqb-kpi-card:hover { border-color: #2A3040; }
.wqb-kpi-card::before {
  content:""; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: #1F2330;
}
.wqb-kpi-label {
  font-size: 11px; color: #6B7280;
  text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.wqb-kpi-value {
  font-size: 22px; font-weight: 700; color: #E6E9EF;
}
.wqb-kpi-sub {
  font-size: 12px; color: #6B7280; margin-top: 2px;
}
.wqb-kpi-accent::before { background: #00E5FF; }
.wqb-kpi-green  .wqb-kpi-value { color: #00D68F; }
.wqb-kpi-red    .wqb-kpi-value { color: #FF4D6D; }
</style>
"""


def _card(label: str, value: str, sub: str = "", css_class: str = "") -> str:
    cls = f"wqb-kpi-card {css_class}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="wqb-kpi-label">{label}</div>'
        f'<div class="wqb-kpi-value">{value}</div>'
        f'<div class="wqb-kpi-sub">{sub}</div>'
        f'</div>'
    )


def render_kpi_strip(
    active_model_key: str,
    all_results: dict,
    market_prices: dict,
    bucket_labels: list,
    current_obs: float | None,
    capital: float,
    kelly_frac: float,
    settle_hours_left: float,
) -> None:
    """Render the 5 hero KPI cards."""
    st.markdown(_CSS, unsafe_allow_html=True)

    active_pred = all_results.get(active_model_key, {})
    active_mean = active_pred.get("mean", 0)
    active_std = active_pred.get("std", 0)
    active_label = MODEL_LABELS.get(active_model_key, active_model_key)
    ac_label = f"Model: {active_label}"
    ac_val = f"{active_mean:.1f}°C ± {active_std:.2f}"
    ac_sub = ""

    # 2. Current Observed
    co_val = f"{current_obs:.1f}°C" if current_obs is not None else "N/A"
    co_sub = ""

    # 3. Best Edge
    best_edge = -1.0
    best_bucket = ""
    for bucket in bucket_labels:
        p_mkt = market_prices.get(bucket, 0.5)
        p_mod = active_pred.get("probs", {}).get(bucket, 0.0)
        edge = abs(p_mod - p_mkt)
        if edge > best_edge:
            best_edge = edge
            best_bucket = bucket

    be_sub = f"Bucket: {best_bucket}" if best_bucket else ""
    be_val = f"{best_edge:+.1%}" if best_bucket else "N/A"

    # 4. Recommended Size
    total_kelly = 0.0
    for bucket in bucket_labels:
        p_mkt = market_prices.get(bucket, 0.5)
        p_mod = active_pred.get("probs", {}).get(bucket, 0.0)
        if p_mod > p_mkt:
            total_kelly += calculate_kelly(p_mod, p_mkt, kelly_frac)
    rs_val = f"${total_kelly * capital:,.0f}"
    rs_sub = f"({total_kelly:.1%} total Kelly)"

    # 5. Settlement Countdown
    h = int(settle_hours_left)
    m = int((settle_hours_left - h) * 60)
    sc_val = f"{h:02d}:{m:02d}"
    sc_sub = "Left until settlement"

    cards_html = '<div class="wqb-kpi-wrap">' + "".join(
        [
            _card("Active Model Temp", ac_val, ac_sub, "wqb-kpi-accent"),
            _card("Current Observed", co_val, co_sub, "wqb-kpi-accent"),
            _card("Best Edge", be_val, be_sub, "wqb-kpi-green" if best_edge > 0 else ""),
            _card("Recommended Size", rs_val, rs_sub, "wqb-kpi-green" if total_kelly > 0 else ""),
            _card("Settlement Countdown", sc_val, sc_sub, "wqb-kpi-red" if settle_hours_left < 2 else ""),
        ]
    ) + "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
