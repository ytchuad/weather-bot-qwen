# app/components/intraday_path.py
"""Intraday temperature path chart — enhanced with shaded now band & stats."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import COLORS


def intraday_path(
    df_today: pd.DataFrame,
    market_bounds: dict[str, float] | None = None,
    is_min_temp: bool = False,
    title: str = "Temperature Path",
    bucket_labels: list[str] | None = None,
) -> None:
    """Render intraday temperature path with enhanced visual elements.

    Args:
        df_today: DataFrame with 'datetime' and 'temp' columns.
        market_bounds: optional dict like {"Market Floor": 23, "Market Ceil": 34}.
        is_min_temp: if True, show cumulative min; else cumulative max.
        title: chart title.
        bucket_labels: optional list of bucket labels for hover index lookup.
    """
    if df_today is None or df_today.empty:
        st.info("No intraday data available.")
        return

    fig = go.Figure()

    # Precompute helpers for hover
    cum_series = df_today["temp"].cummin() if is_min_temp else df_today["temp"].cummax()
    temp_series = df_today["temp"]
    delta_from_cum = temp_series - cum_series

    # Observed path
    fig.add_trace(go.Scatter(
        x=df_today["datetime"],
        y=df_today["temp"],
        mode="lines+markers",
        name="Observed",
        line=dict(color=COLORS["temperature"], width=2),
        marker=dict(size=3, color=COLORS["temperature"]),
        customdata=pd.DataFrame({
            "cum": cum_series,
            "delta": delta_from_cum,
        }).values,
        hovertemplate=(
            "%{x|%H:%M}<br>"
            "<b>%{y:.1f}°C</b><br>"
            "Cumulative: %{customdata[0]:.1f}°C<br>"
            "Δ from cum: %{customdata[1]:+.1f}°C"
            "<extra></extra>"
        ),
    ))

    # Cumulative max/min
    if not is_min_temp:
        fig.add_trace(go.Scatter(
            x=df_today["datetime"],
            y=df_today["temp"].cummax(),
            mode="lines",
            name="Cumulative Max",
            line=dict(color="#F59E0B", width=1, dash="dot"),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df_today["datetime"],
            y=df_today["temp"].cummin(),
            mode="lines",
            name="Cumulative Min",
            line=dict(color="#22D3EE", width=1, dash="dot"),
        ))

    # Market bounds as shaded rectangles
    if market_bounds:
        for label, val in market_bounds.items():
            fig.add_hline(
                y=val,
                line=dict(color="#F59E0B", width=1, dash="dash"),
                annotation_text=label,
                annotation_position="right",
                annotation_font=dict(color="#F59E0B", size=10),
            )

    # Shaded "now" band (last 1 hour window)
    if not df_today.empty and len(df_today) >= 2:
        now = df_today["datetime"].iloc[-1]
        one_hour_ago = df_today["datetime"].iloc[max(0, len(df_today) - 7)]
        fig.add_vrect(
            x0=one_hour_ago, x1=now,
            fillcolor="rgba(0, 229, 255, 0.06)",
            line_width=0,
        )

    # "Now" line + annotation
    if not df_today.empty:
        now = df_today["datetime"].iloc[-1]
        fig.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=now, x1=now, y0=0, y1=1,
            line=dict(color=COLORS["temperature"], width=1, dash="dot"),
        )
        fig.add_annotation(
            x=now, y=1.02,
            xref="x", yref="paper",
            text="Now",
            showarrow=False,
            font=dict(color=COLORS["temperature"], size=10),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=340,
        xaxis_title="Time",
        yaxis_title="°C",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Footer stats
    if not df_today.empty:
        temp_series = df_today["temp"]
        n_records = len(temp_series)
        min_val = temp_series.min()
        max_val = temp_series.max()
        # Delta 60m: compare last vs entry ~ hour ago (or ~6 rows at 10min spacing)
        if len(temp_series) >= 7:
            delta_60m = temp_series.iloc[-1] - temp_series.iloc[-7]
            delta_str = f"{'+' if delta_60m >= 0 else ''}{delta_60m:.1f}°C"
        else:
            delta_str = "N/A"

        st.markdown(
            f"<div style='font-size:11px; color:#6B7280; margin-top:4px;'>"
            f"Records: {n_records} · Min: {min_val:.1f}°C · Max: {max_val:.1f}°C · Δ60m: {delta_str}"
            f"</div>",
            unsafe_allow_html=True,
        )
