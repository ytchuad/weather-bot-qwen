# app/services/backtest_service.py
"""Forward-test evaluation and historical model performance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..config import (
    FORWARD_TEST_LOG,
    HISTORICAL_TEMP_PATH,
    PERF_LOG_PATH,
    CACHE_TTL_LONG,
)

logger = logging.getLogger(__name__)


def evaluate_forward_test(
    initial_capital: float = 10000.0,
    kelly_frac: float = 0.5,
) -> dict:
    """Evaluate forward-test log and return performance dict per engine.

    Returns dict like {engine: {summary, pnl_df, brier_df}}.
    Raises ValueError or returns empty dict if no data available.
    """
    if not FORWARD_TEST_LOG.exists():
        return {}

    df_log = pd.read_parquet(FORWARD_TEST_LOG)
    df_log["target_date"] = pd.to_datetime(df_log["target_date"]).dt.normalize()

    if "model_version" not in df_log.columns:
        df_log["model_version"] = "9d"

    def _safe_float(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower()
        if s in ("inf", "infinity"):
            return 999.0
        if s in ("-inf", "-infinity"):
            return -999.0
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    df_log["lower_bound"] = df_log["lower_bound"].apply(_safe_float)
    df_log["upper_bound"] = df_log["upper_bound"].apply(_safe_float)

    actuals_max: dict = {}
    actuals_min: dict = {}
    if HISTORICAL_TEMP_PATH.exists():
        df_obs = pd.read_parquet(HISTORICAL_TEMP_PATH)
        df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.normalize()
        actuals_max = dict(zip(df_obs["date"], df_obs["tmax"]))
        if "tmin" in df_obs.columns:
            actuals_min = dict(zip(df_obs["date"], df_obs["tmin"]))

    results: dict[str, dict] = {}
    for version in ("9d", "aws"):
        df_v = df_log[df_log["model_version"] == version].copy()
        if df_v.empty:
            continue
        brier_scores: list[dict] = []
        daily_pnl: list[dict] = []
        current_bankroll = float(initial_capital)
        for target_date in sorted(df_v["target_date"].unique()):
            group = df_v[df_v["target_date"] == target_date]
            is_min_market = group["market_type"].iloc[0] == "lowest" if "market_type" in group.columns else False
            actual_val = actuals_min.get(target_date) if is_min_market else actuals_max.get(target_date)
            if pd.notna(actual_val):
                group = group.copy()
                group["actual_outcome"] = group.apply(
                    lambda row: 1 if row["lower_bound"] <= actual_val < row["upper_bound"] else 0,
                    axis=1,
                )
                bs = np.mean((group["model_prob"] - group["actual_outcome"]) ** 2)
                brier_scores.append({"date": target_date, "brier_score": float(bs)})
                day_pnl = 0.0
                for _, row in group.iterrows():
                    edge = row["edge"]
                    market_price = min(row["market_price"], 0.99)
                    if edge > 0.02:
                        raw_kelly = edge / (1.0 - market_price)
                        bet_pct = min(raw_kelly * kelly_frac, 0.10)
                        bet_amount = current_bankroll * bet_pct
                        if row["actual_outcome"] == 1:
                            day_pnl += bet_amount * (1.0 / market_price - 1.0)
                        else:
                            day_pnl -= bet_amount
                current_bankroll += day_pnl
                daily_pnl.append({"date": target_date, "pnl": day_pnl, "bankroll": current_bankroll})

        pnl_df = pd.DataFrame(daily_pnl)
        brier_df = pd.DataFrame(brier_scores)
        summary: dict = {
            "final_bankroll": current_bankroll,
            "total_return_pct": (current_bankroll - initial_capital) / initial_capital * 100.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
        }
        if not pnl_df.empty and pnl_df["bankroll"].nunique() > 1:
            daily_returns = pnl_df["bankroll"].pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                summary["sharpe"] = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            peak = pnl_df["bankroll"].cummax()
            drawdown = (pnl_df["bankroll"] - peak) / peak * 100.0
            summary["max_drawdown_pct"] = float(drawdown.min())

        results[version] = {
            "brier_df": brier_df,
            "pnl_df": pnl_df,
            "summary": summary,
        }
    return results


@st.cache_data(ttl=CACHE_TTL_LONG)
def load_performance_log() -> pd.DataFrame | None:
    if PERF_LOG_PATH.exists():
        try:
            return pd.read_parquet(PERF_LOG_PATH).sort_values("date")
        except Exception:
            pass
    return None


@st.cache_data(ttl=CACHE_TTL_LONG)
def load_forward_test_log() -> pd.DataFrame | None:
    if FORWARD_TEST_LOG.exists():
        try:
            return pd.read_parquet(FORWARD_TEST_LOG)
        except Exception:
            pass
    return None
