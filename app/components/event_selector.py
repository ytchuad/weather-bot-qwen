# app/components/event_selector.py
"""Polymarket event search & selection UI."""

import streamlit as st

from ..services.market_service import search_events, parse_date_from_event
from ..services.weather_service import hkt_now


def event_selector(state) -> None:
    """Render event search bar, results list, and event details.

    On selection, populates state.selected_event, state.target_date,
    state.is_min_temp, and state.markets.
    """
    st.header("🔍 Polymarket Market Search")

    cq, cb = st.columns([4, 1])
    with cq:
        search_query = st.text_input(
            "Search keyword",
            value="hong-kong-temperature",
            label_visibility="collapsed",
        )
    with cb:
        search_btn = st.button("🔍 Search", use_container_width=True)

    if search_btn or not state.pm_events:
        with st.spinner("Searching Polymarket API..."):
            state.pm_events = search_events(search_query)

    pm_events = state.pm_events
    if not pm_events:
        st.warning("No events found. Try a different keyword.")
        st.stop()

    options = {f"{e['title']}": e for e in pm_events}
    selected_title = st.selectbox("Select target market", list(options.keys()))
    selected_event = options[selected_title]
    slug = selected_event["slug"]

    parsed_date = parse_date_from_event(selected_event["title"], slug)
    default_date = parsed_date if parsed_date else hkt_now().date()
    is_min_temp = "lowest" in selected_event["title"].lower() or "lowest" in slug.lower()

    temp_label = "Tmin" if is_min_temp else "Tmax"

    state.selected_event = selected_event
    state.target_date = default_date
    state.is_min_temp = is_min_temp

    st.caption(f"🎯 **Slug**: `{slug}` | **Metric**: {temp_label} | **Date**: {default_date}")
