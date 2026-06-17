from __future__ import annotations

import streamlit as st

_PAGES = [
    ("hub", "Hub"),
    ("intraday", "Intraday"),
    ("strategies", "Strategies"),
    ("analytics", "Analytics"),
    ("health", "Health"),
]


def render_top_nav(current: str = "hub") -> None:
    """Render the fixed top navigation bar.

    Parameters
    ----------
    current : str
        The URL-path key of the currently active page (e.g. "hub", "intraday").
        Used to highlight the active nav link.
    """
    links_html = ""
    for key, title in _PAGES:
        is_active = key == current
        active_cls = "active" if is_active else ""
        # Use <span> instead of <a> — avoids href status-bar text.
        # JS bridge intercepts clicks on data-nav-key and triggers
        # Streamlit-internal page switching via st.switch_page().
        links_html += (
            f'<span class="top-nav-link {active_cls}" data-nav-key="{key}" '
            f'role="button" tabindex="0">{title}</span>'
        )

    st.markdown(
        f'<div class="top-nav">'
        f'<div class="top-nav-inner">'
        f'<div class="top-nav-links">{links_html}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def mark_refreshed() -> None:
    pass
