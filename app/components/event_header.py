# app/components/event_header.py
"""Event header card — auto-loaded event title, date picker, Tmax/Tmin toggle."""

from __future__ import annotations

import streamlit as st

from ..services.weather_service import hkt_now
from ..state import AppState


_CSS = """
<style>
.wqb-event-card {
  background: #14171F;
  border: 1px solid #1F2330;
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 8px;
}
.wqb-event-title {
  font-size: 18px; font-weight: 600; color: #E6E9EF;
  margin: 0 0 4px 0; line-height: 1.3;
}
.wqb-event-meta {
  font-size: 12px; color: #6B7280;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
</style>
"""

LABEL_TEMP = "temp_label"
SELECTED_EVENT = "selected_event"
IS_MIN_TEMP = "is_min_temp"
TARGET_DATE = "target_date"


def render_event_header(state: AppState) -> None:
    """Render event header card with title, slug, date input, Tmax/Tmin toggle, and live clock."""
    st.markdown(_CSS, unsafe_allow_html=True)

    sel = state.selected_event or {}
    title = sel.get("title", "(no event loaded)")
    slug = sel.get("slug", "")
    pm_url = f"https://polymarket.com/event/{slug}" if slug else ""

    today = hkt_now().date()
    target_date = state.target_date or today

    st.markdown('<div class="wqb-event-card">', unsafe_allow_html=True)

    left, right = st.columns([3.2, 2.0], gap="medium")

    with left:
        if pm_url:
            link_html = f' &nbsp;·&nbsp; <a href="{pm_url}" target="_blank" style="color:#6B7280; text-decoration:none;">polymarket ↗</a>'
        else:
            link_html = ""
        html = (
            f'<p class="wqb-event-title">{title}</p>'
            f'<div class="wqb-event-meta">'
            f'<span>slug: <code>{slug if slug else "—"}</code></span>'
            f'{link_html}'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    with right:
        date_col, metric_col = st.columns([1.5, 1.0])
        with date_col:
            new_date = st.date_input(
                "Date",
                value=target_date,
                key="event_header_date",
                label_visibility="collapsed",
            )
            if new_date != target_date:
                state.target_date = new_date
        with metric_col:
            metric = st.segmented_control(
                "Metric",
                options=["Tmax", "Tmin"],
                default=("Tmin" if state.is_min_temp else "Tmax"),
                key="event_header_metric",
                label_visibility="collapsed",
            )
            new_is_min = metric == "Tmin"
            if new_is_min != state.is_min_temp:
                state.is_min_temp = new_is_min
                state.selected_event = None  # Force re-resolve

        clock = hkt_now().strftime("%Y-%m-%d %H:%M:%S HKT")
        st.markdown(
            f'<div style="text-align:right; font-size:11px; color:#6B7280; margin-top:4px;">'
            f'<span style="color:#00D68F; font-weight:600;">●</span> {clock}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
