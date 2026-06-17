from __future__ import annotations

import streamlit as st

from .pages.page_hub import run as run_hub
from .pages.page_intraday import run as run_intraday
from .pages.page_strategies import run as run_strategies
from .pages.page_analytics import run as run_analytics
from .pages.page_health import run as run_health

hub_page = st.Page(run_hub, title="Hub", url_path="hub", default=True)
intraday_page = st.Page(run_intraday, title="Intraday", url_path="intraday")
strategies_page = st.Page(run_strategies, title="Strategies", url_path="strategies")
analytics_page = st.Page(run_analytics, title="Analytics", url_path="analytics")
health_page = st.Page(run_health, title="Health", url_path="health")

pages = {
    "Hub": hub_page,
    "Intraday": intraday_page,
    "Strategies": strategies_page,
    "Analytics": analytics_page,
    "Health": health_page,
}

ALL_PAGES: list[st.Page] = list(pages.values())
