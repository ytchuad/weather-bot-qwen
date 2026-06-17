# app/components/sidebar.py
"""Shared sidebar controls — date, model params, refresh actions."""

import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ..config import ROOT_DIR
from ..state import AppState
from ..services.weather_service import hkt_now


def render_sidebar(state: AppState) -> None:
    """Render sidebar with date picker, model params, and action buttons."""
    st.sidebar.header("📅 Weather Data Alignment")

    target_date = st.sidebar.date_input(
        "Target date (HKO data)",
        value=state.target_date or hkt_now().date(),
    )
    target_date_dt = pd.Timestamp(target_date)
    target_date_str = target_date_dt.strftime("%Y%m%d")

    state.target_date = target_date
    st.session_state["app.target_date_str"] = target_date_str

    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Actions")

    if st.sidebar.button("🔄 Force refresh live data"):
        st.cache_data.clear()
        state.clear_cache()
        st.rerun()

    if st.sidebar.button("🧹 Clear __pycache__"):
        for p in ROOT_DIR.rglob("__pycache__"):
            shutil.rmtree(p, ignore_errors=True)
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("🔄 Sync HKO forecast"):
        with st.spinner("Syncing HKO 9-day forecast..."):
            from features.live_feature_builder import update_forecast_database
            update_forecast_database()
        st.rerun()
