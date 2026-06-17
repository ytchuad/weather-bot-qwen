# app/components/metric_row.py
"""Row of KPI metric cards."""

import streamlit as st
from typing import Sequence


def metric_row(metrics: Sequence[dict], columns: int = 4) -> None:
    """Render a row of st.metric cards.

    Args:
        metrics: list of dicts with keys:
            - label (str): metric title
            - value (str): display value
            - delta (str, optional): delta indicator
            - delta_color (str, optional): "normal" or "inverse"
        columns: number of columns in the row
    """
    if not metrics:
        return
    cols = st.columns(columns)
    for i, m in enumerate(metrics):
        col = cols[i % columns]
        col.metric(
            m.get("label", ""),
            m.get("value", "—"),
            delta=m.get("delta"),
            delta_color=m.get("delta_color", "normal"),
        )
