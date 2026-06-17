# app/pages/page_analytics.py
"""Analytics page — forward test, model performance, Brier scores, calibration."""

import pandas as pd
import streamlit as st

from ..state import AppState
from ..config import MODEL_LABELS
from ..components.sidebar import render_sidebar
from ..services.backtest_service import (
    evaluate_forward_test,
    load_forward_test_log,
    load_performance_log,
)


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ---- sidebar ----
    render_sidebar(state)

    st.header("📊 Analytics & Forward Testing")

    tabs = st.tabs(["🧪 Forward Test", "📈 Model Performance", "📋 Raw Logs"])

    # ========================================================================
    # Tab 1: Forward Test
    # ========================================================================
    with tabs[0]:
        st.subheader("Forward Test Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            test_capital = st.number_input(
                "Initial capital",
                min_value=100,
                value=int(state.capital),
                step=1000,
                key="ft_capital",
            )
        with col2:
            test_kelly = st.selectbox(
                "Kelly fraction",
                [0.25, 0.5, 0.75, 1.0],
                index=1,
                key="ft_kelly",
            )

        # Persist results in session_state
        ft_key = "app.forward_test_results"
        if st.button("🔄 Run Forward Test", use_container_width=True):
            with st.spinner("Evaluating forward test performance..."):
                try:
                    results = evaluate_forward_test(test_capital, test_kelly)
                    st.session_state[ft_key] = results
                except Exception as e:
                    st.error(f"Forward test evaluation failed: {e}")
                    st.session_state.pop(ft_key, None)

        results = st.session_state.get(ft_key)

        if results:
            for engine, perf in results.items():
                label = MODEL_LABELS.get(engine, engine)
                with st.expander(f"📊 {label}", expanded=engine == "9d"):
                    # Summary metrics
                    summary = perf.get("summary", {})
                    if summary:
                        cols = st.columns(4)
                        cols[0].metric("Bankroll", f"${summary.get('final_bankroll', 0):,.2f}")
                        cols[1].metric("Total Return", f"{summary.get('total_return_pct', 0):+.1f}%")
                        cols[2].metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
                        cols[3].metric("Max DD", f"{summary.get('max_drawdown_pct', 0):.1f}%")

                    # PnL chart
                    pnl_df = perf.get("pnl_df")
                    if pnl_df is not None and not pnl_df.empty:
                        st.line_chart(pnl_df, x="date", y="bankroll", use_container_width=True)

                    # Brier scores
                    brier_df = perf.get("brier_df")
                    if brier_df is not None and not brier_df.empty:
                        st.caption("Brier Scores by Bucket")
                        st.dataframe(brier_df, use_container_width=True, hide_index=True)

    # ========================================================================
    # Tab 2: Model Performance
    # ========================================================================
    with tabs[1]:
        st.subheader("Model Performance Metrics")

        try:
            perf_log = load_performance_log()
            if perf_log is not None and not perf_log.empty:
                # Summary stats
                st.caption(f"**{len(perf_log)}** performance records")

                if "engine" in perf_log.columns:
                    engines = perf_log["engine"].unique()
                    for eng in engines:
                        eng_data = perf_log[perf_log["engine"] == eng]
                        label = MODEL_LABELS.get(eng, eng)
                        st.markdown(f"**{label}**")
                        cols = st.columns(4)
                        if "mae" in eng_data.columns:
                            cols[0].metric("MAE", f"{eng_data['mae'].mean():.2f}°C")
                        if "rmse" in eng_data.columns:
                            cols[1].metric("RMSE", f"{eng_data['rmse'].mean():.2f}°C")
                        if "bias" in eng_data.columns:
                            cols[2].metric("Bias", f"{eng_data['bias'].mean():+.2f}°C")
                        if "brier" in eng_data.columns:
                            cols[3].metric("Brier", f"{eng_data['brier'].mean():.4f}")

                # Raw table
                st.dataframe(perf_log, use_container_width=True, hide_index=True)
            else:
                st.info("No performance log available. Run models and record results to populate this view.")
        except Exception as e:
            st.warning(f"Could not load performance log: {e}")

        # Forward test log
        st.markdown("---")
        st.subheader("Forward Test Log")
        try:
            ft_log = load_forward_test_log()
            if ft_log is not None and not ft_log.empty:
                st.dataframe(ft_log, use_container_width=True, hide_index=True)
            else:
                st.info("No forward test log available.")
        except Exception as e:
            st.warning(f"Could not load forward test log: {e}")

    # ========================================================================
    # Tab 3: Raw Logs
    # ========================================================================
    with tabs[2]:
        st.subheader("Raw Log Data")

        st.markdown("**Performance Log**")
        try:
            perf_log = load_performance_log()
            if perf_log is not None and not perf_log.empty:
                st.dataframe(perf_log, use_container_width=True)
            else:
                st.info("No performance log.")
        except Exception as e:
            st.error(str(e))

        st.markdown("**Forward Test Log**")
        try:
            ft_log = load_forward_test_log()
            if ft_log is not None and not ft_log.empty:
                st.dataframe(ft_log, use_container_width=True)
            else:
                st.info("No forward test log.")
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    run()
