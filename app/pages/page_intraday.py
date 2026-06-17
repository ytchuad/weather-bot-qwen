# app/pages/page_intraday.py
"""Intraday page — temperature path, rain metrics, model detail expander."""

import pandas as pd
import streamlit as st

from ..state import AppState
from ..config import MODEL_LABELS
from ..components.sidebar import render_sidebar
from ..components.intraday_path import intraday_path
from ..components.metric_row import metric_row
from ..services.weather_service import (
    fetch_hko_data, get_intraday_state, compute_rain_kwargs, hkt_now,
)
from ..services.model_service import run_all_models
from ..services.market_service import fetch_event_markets, parse_date_from_event
from ..services.today_event_resolver import resolve_today_event


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ---- sidebar ----
    render_sidebar(state)

    st.header("📈 Intraday Temperature & Rain Analysis")

    if state.selected_event is None:
        ev = resolve_today_event()
        if ev:
            state.selected_event = ev
            parsed = parse_date_from_event(ev.get("title", ""), ev.get("slug", ""))
            if parsed:
                state.target_date = parsed
            state.is_min_temp = ("lowest" in ev.get("title", "").lower() or "lowest" in ev.get("slug", "").lower())
            pass

    slug = state.selected_event.get("slug", "") if state.selected_event else ""
    if not slug:
        st.markdown('<div style="padding:60px 32px;max-width:1280px;margin:0 auto;"><div class="glass-card"><p style="color:#8F9BB7;">No event found for today.</p></div></div>', unsafe_allow_html=True)
        return
    is_min_temp = state.is_min_temp
    target_date = state.target_date
    target_date_str = pd.Timestamp(target_date).strftime("%Y%m%d") if target_date else hkt_now().strftime("%Y%m%d")
    is_today = pd.Timestamp(target_date).normalize() == pd.Timestamp(hkt_now()).normalize() if target_date else True

    temp_label = "Tmin" if is_min_temp else "Tmax"

    # ---- data loading ----
    with st.spinner("Loading intraday data..."):
        intra_state = get_intraday_state(target_date_str)
        if not intra_state:
            today_str = hkt_now().strftime("%Y%m%d")
            if today_str != target_date_str:
                intra_state = get_intraday_state(today_str)
        state.intraday_state = intra_state

    with st.spinner("Loading rain data..."):
        rain_kwargs = {}
        if intra_state:
            rain_kwargs = compute_rain_kwargs(target_date_str, hkt_now())
        state.rain_kwargs = rain_kwargs

    with st.spinner("Loading HKO data..."):
        hko = fetch_hko_data(target_date_str)

    # ---- rain metrics ----
    if rain_kwargs:
        st.subheader("🌧️ Rain Context")
        rain_metrics = [
            {"label": "Rain (60m)", "value": f"{rain_kwargs.get('rain_60m', 0):.1f} mm"},
            {"label": "Rain (120m)", "value": f"{rain_kwargs.get('rain_120m', 0):.1f} mm"},
            {"label": "Rain Data OK", "value": str(rain_kwargs.get("rain_data_ok", False))},
        ]
        prev_temp = rain_kwargs.get("prev_18_temp")
        if prev_temp is not None:
            rain_metrics.append({"label": "Prev 18h Temp", "value": f"{prev_temp:.1f}°C"})
        metric_row(rain_metrics, columns=min(len(rain_metrics), 4))

    # ---- temperature path chart ----
    st.subheader("🌡️ Temperature Path")

    if intra_state and intra_state.get("df_today") is not None and not intra_state["df_today"].empty:
        markets = fetch_event_markets(slug, is_min_temp)
        market_bounds = None
        if markets:
            try:
                first_b = markets[0]["bucket"]
                last_b = markets[-1]["bucket"]
                lo = float(first_b.split("-")[0]) if "-" in first_b else float(first_b.replace("<", "").replace(">=", ""))
                hi = float(last_b.split("-")[-1]) if "-" in last_b else float(last_b.replace("<", "").replace(">=", ""))
                market_bounds = {"Market Floor": lo, "Market Ceil": hi}
            except (ValueError, IndexError):
                pass

        intraday_path(
            intra_state["df_today"],
            market_bounds=market_bounds,
            is_min_temp=is_min_temp,
            title=f"{temp_label} Intraday Path — {target_date_str}",
        )
    else:
        st.info("No intraday temperature data available for this date.")

    # ---- current state summary ----
    st.subheader("📋 Current State")
    if intra_state:
        cur_metrics = []
        if intra_state.get("temp_now") is not None:
            cur_metrics.append({"label": "Current Temp", "value": f"{intra_state['temp_now']:.1f}°C"})
        if intra_state.get("max_so_far") is not None:
            cur_metrics.append({"label": "Max So Far", "value": f"{intra_state['max_so_far']:.1f}°C"})
        if intra_state.get("min_so_far") is not None:
            cur_metrics.append({"label": "Min So Far", "value": f"{intra_state['min_so_far']:.1f}°C"})
        if intra_state.get("n_records") is not None:
            cur_metrics.append({"label": "Records Today", "value": str(intra_state["n_records"])})
        if cur_metrics:
            metric_row(cur_metrics, columns=min(len(cur_metrics), 4))

    # ---- intraday model details ----
    st.markdown("---")
    st.subheader("🔬 Intraday Model Predictions")

    with st.spinner("Running intraday models..."):
        markets = fetch_event_markets(slug, is_min_temp)
        all_results = run_all_models(
            target_date=target_date,
            target_date_str=target_date_str,
            is_min_temp=is_min_temp,
            bias=state.bias,
            std_mult=state.std_mult,
            state=intra_state,
            rain_kwargs=rain_kwargs,
            markets=markets,
            forecast_aws_val=hko.get("forecast_max" if not is_min_temp else "forecast_min"),
            is_today=is_today,
        )
        intra_preds = {k: v for k, v in all_results.items() if k not in ("9d", "aws")}
        state.pred_intra = intra_preds

    if intra_preds:
        rows = []
        for mk, ip in intra_preds.items():
            raw = ip.get("raw", {})
            p10_key = "pred_morning_min_p10" if is_min_temp else "pred_tmax_p10"
            p50_key = "pred_morning_min_p50" if is_min_temp else "pred_tmax_p50"
            p90_key = "pred_morning_min_p90" if is_min_temp else "pred_tmax_p90"
            p10 = raw.get(p10_key)
            p50 = raw.get(p50_key)
            p90 = raw.get(p90_key)
            label = MODEL_LABELS.get(mk, mk)
            rows.append({
                "Model": label,
                "P10": f"{p10:.1f}" if p10 is not None else "N/A",
                "P50": f"{p50:.1f}" if p50 is not None else "N/A",
                "P90": f"{p90:.1f}" if p90 is not None else "N/A",
                "Post Mean": f"{ip.get('post_mean', 0):.2f}",
                "Post Std": f"{ip.get('post_std', 0):.2f}",
                "Prior Source": ip.get("prior_source", ""),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No intraday model predictions available.")

    # ---- hourly AWS forecast ----
    st.markdown("---")
    st.subheader("📡 AWS Hourly Forecast")
    aws_hourly = hko.get("aws_hourly")
    if aws_hourly:
        st.dataframe(pd.DataFrame(aws_hourly), use_container_width=True)
    else:
        st.info("No AWS hourly forecast data available.")


if __name__ == "__main__":
    run()
