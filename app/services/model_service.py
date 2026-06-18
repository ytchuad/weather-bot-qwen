# app/services/model_service.py
"""Model inference orchestration.

Coordinates 9-day XGBoost, AWS high-freq, and all intraday LightGBM models
(minute-level A/B/C/D/E). Handles Bayesian fusion and bucket-probability
computation for downstream UI.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ..config import (
    DEFAULT_TMAX_FORECAST_DELTA,
    DEFAULT_TMIN_FORECAST_DELTA,
    MODEL_KEYS,
)

logger = logging.getLogger(__name__)


# ── 9-day XGBoost ────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def predict_9d(
    target_date: date,
    is_min_temp: bool,
    bias: float = 0.0,
    std_mult: float = 1.0,
) -> dict | None:
    """Run 9-day XGBoost inference. Returns {mean, std} or None."""
    try:
        from features.live_feature_builder import build_features_for_date
        from models.inference import predict_distribution

        model_type = "tmin" if is_min_temp else "tmax"
        dt = pd.Timestamp(target_date).to_pydatetime()
        features, meta = build_features_for_date(dt)
        if features is None:
            return None
        mean, std = predict_distribution(
            features, model_type,
            meta.get("hko_spread") if meta else None,
            std_mult,
        )
        if np.isnan(mean) or np.isnan(std):
            return None
        mean += bias
        return {"mean": float(mean), "std": float(std)}
    except Exception as e:
        logger.warning("predict_9d failed: %s", e)
        return None


# ── AWS High-Freq ────────────────────────────────────────────────────

def predict_aws(
    forecast_aws: float | None,
    fallback_mean: float,
    fallback_std: float,
    bias: float = 0.0,
) -> dict:
    """Build AWS prediction from HKO forecast value.

    Falls back to 9d mean/std when AWS data is unavailable.
    """
    if forecast_aws is not None:
        mean = forecast_aws + bias
        std = fallback_std * 0.8  # AWS has narrower uncertainty
        source = "⚡ AWS High-Freq"
    else:
        mean = fallback_mean
        std = fallback_std
        source = "⚠️ AWS 無數據 (沿用 9D)"
    return {"mean": float(mean), "std": float(std), "source": source}


# ── Intraday (all models) ────────────────────────────────────────────

def predict_intraday_all(
    target_date_str: str,
    is_min_temp: bool,
    state: dict | None,
    rain_kwargs: dict | None = None,
    forecast_max: float | None = None,
    forecast_min: float | None = None,
) -> dict[str, dict]:
    """Run all intraday models and fuse with prior.

    Returns:
        {model_key: {post_mean, post_std, probs, raw_pred}} for each
        intraday model that succeeded. Also includes '_error' key on failure.
    """
    if state is None:
        return {}

    from models.intraday_inference import (
        predict_intraday_tmax_all,
        predict_intraday_tmin_all,
        combine_with_prior,
    )

    rain_kw = rain_kwargs or {}
    hour_now = state["time_now"].hour
    minutes_since_midnight = hour_now * 60 + state["time_now"].minute

    common = dict(
        current_datetime=state["time_now"],
        temp_60min_ago=state.get("temp_60m_ago", state["temp_now"]),
        temp_now=state["temp_now"],
        forecast_tmax=forecast_max if forecast_max is not None else (state.get("max_so_far", 30.0) + DEFAULT_TMAX_FORECAST_DELTA),
        forecast_tmin=forecast_min if forecast_min is not None else (state.get("min_so_far", 10.0) + DEFAULT_TMIN_FORECAST_DELTA),
        temp_120m_ago=state.get("temp_120m_ago", state["temp_60m_ago"]),
        max_so_far=state.get("max_so_far"),
        min_so_far=state.get("min_so_far"),
        rainfall_60m_filled=rain_kw.get("rain_60m", 0.0),
        rainfall_120m_filled=rain_kw.get("rain_120m", 0.0),
        rainfall_60m_missing_flag=rain_kw.get("rainfall_60m_missing_flag", 1),
        rainfall_120m_missing_flag=rain_kw.get("rainfall_120m_missing_flag", 1),
        temp_change_30min=state.get("temp_change_30m", 0.0),
        temp_change_60min=state.get("temp_change_60m", 0.0),
        time_since_max_so_far=state.get("time_since_max", 0.0),
        time_since_min_so_far=state.get("time_since_min", 0.0),
        hour=hour_now,
        minutes_since_midnight=minutes_since_midnight,
        rh_current=state.get("rh_now", 50.0),
        temp_buffer=state.get("df_today", pd.DataFrame()).get("temp", pd.Series()).dropna().tolist() if state.get("df_today") is not None else None,
        rh_buffer=state.get("df_today", pd.DataFrame()).get("rh", pd.Series()).dropna().tolist() if state.get("df_today") is not None else None,
    )

    # Merge rain kwargs — only pass keys that predict_intraday_tmax/tmin_all
    # actually accepts via **rain_kwargs.  The function strips non-param
    # keys internally (rh_current, temp_buffer, etc.), but raw keys like
    # rain_60m / rain_120m / rain_data_ok are NOT valid params and cause
    # "unexpected keyword argument" errors.
    _ALLOWED_RAIN_KWARGS = {
        "prev_18_temp", "prev_21_temp", "prev_2359_temp",
        "prev_evening_temp_change", "prev_evening_temp_min",
        "prev_evening_temp_range", "prev_evening_temp_slope",
        "prev_evening_rh_mean", "prev_evening_rh_max",
        "prev_evening_dew_point_mean",
        "prev_evening_rainfall_18_24", "prev_evening_rain_flag",
        "rainfall_30m_filled", "rainfall_30m_missing_flag",
        "rainfall_data_age_minutes", "rain_data_gap_flag",
        # buffer variants for Model D/E
        "temp_buffer_long", "rh_buffer_long",
    }
    for k, v in rain_kw.items():
        if k in _ALLOWED_RAIN_KWARGS and k not in common:
            common[k] = v

    try:
        if is_min_temp:
            raw_preds = predict_intraday_tmin_all(**common)
        else:
            common_tmax = dict(common)
            common_tmax["time_since_max_so_far"] = state.get("time_since_max", 0.0)
            raw_preds = predict_intraday_tmax_all(**common_tmax)
    except Exception as e:
        logger.warning("predict_intraday_all failed: %s", e)
        return {"_error": str(e)}

    if not raw_preds:
        return {}

    results: dict[str, dict] = {}
    for mk, raw_pred in raw_preds.items():
        if raw_pred is None:
            continue
        try:
            pm, ps = combine_with_prior(0.0, 1.0, raw_pred, weight=0.0)
        except Exception:
            pm = raw_pred.get("post_mean", raw_pred.get("pred_morning_min_p50", np.nan))
            ps = raw_pred.get("post_std", 1.0)
        if np.isnan(pm) or np.isnan(ps):
            continue
        results[mk] = {
            "post_mean": float(pm),
            "post_std": float(ps),
            "raw": raw_pred,
        }
    return results


# ── bucket probabilities ─────────────────────────────────────────────

def compute_bucket_probs(
    mean: float,
    std: float,
    markets: list[dict],
    is_today: bool,
    is_min_temp: bool,
    max_sf: float | None = None,
    min_sf: float | None = None,
) -> dict[str, float]:
    """Compute bucket probabilities via predict_bucket_probabilities.

    Returns a dict keyed by **bucket label** (e.g. "32-33") so that
    downstream UI components (bucket_bars, recommendation_table) can
    look up probabilities by the same canonical keys they use for
    market_prices.
    """
    try:
        from models.inference import predict_bucket_probabilities

        raw_probs = predict_bucket_probabilities(
            mean, std, markets,
            max_since_midnight=max_sf,
            min_since_midnight=min_sf,
            is_today=is_today,
            is_min_temp=is_min_temp,
        )

        # predict_bucket_probabilities keys probs by market['name'] (the
        # display title from Polymarket, e.g. "Temperature 32C or higher?").
        # Remap to canonical bucket labels so the UI can cross-reference.
        name_to_bucket = {m["name"]: m["bucket"] for m in markets}
        remapped: dict[str, float] = {}
        for k, v in raw_probs.items():
            bucket_key = name_to_bucket.get(k, k)
            remapped[bucket_key] = v
        return remapped
    except Exception as e:
        logger.warning("compute_bucket_probs failed: %s", e)
        return {}


# ── orchestration ────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def run_all_models(
    target_date: date,
    target_date_str: str,
    is_min_temp: bool,
    bias: float,
    std_mult: float,
    state: dict | None,
    rain_kwargs: dict,
    markets: list[dict],
    forecast_aws_val: float | None,
    is_today: bool,
) -> dict[str, dict]:
    """Run all models and compute bucket probabilities for each.

    Returns:
        {model_key: {mean, std, probs, source}} where probs is
        {bucket_name: probability}. The 'probs' are already bucket-level.
    """
    output: dict[str, dict] = {}

    # 9-day
    pred_9d = predict_9d(target_date, is_min_temp, bias, std_mult)
    if pred_9d:
        pred_9d["probs"] = compute_bucket_probs(
            pred_9d["mean"], pred_9d["std"], markets,
            is_today, is_min_temp,
            max_sf=state.get("max_so_far") if state else None,
            min_sf=state.get("min_so_far") if state else None,
        )
        pred_9d["source"] = "🧠 9-Day XGBoost"
    else:
        # fallback
        fallback_temp = forecast_aws_val if forecast_aws_val else (22.0 if is_min_temp else 32.0)
        pred_9d = {
            "mean": fallback_temp + bias,
            "std": 1.5 * std_mult,
            "source": "⚠️ 9-Day Fallback",
        }
        pred_9d["probs"] = compute_bucket_probs(
            pred_9d["mean"], pred_9d["std"], markets, is_today, is_min_temp,
        )
    output["9d"] = pred_9d

    # AWS
    pred_aws = predict_aws(forecast_aws_val, pred_9d["mean"], pred_9d["std"], bias)
    pred_aws["probs"] = compute_bucket_probs(
        pred_aws["mean"], pred_aws["std"], markets, is_today, is_min_temp,
    )
    output["aws"] = pred_aws

    # Intraday
    if state and markets:
        intra_preds = predict_intraday_all(
            target_date_str, is_min_temp, state, rain_kwargs,
            forecast_max=None, forecast_min=None,
        )
        if intra_preds:
            if "_error" in intra_preds:
                output["_intraday_error"] = intra_preds["_error"]
            else:
                for mk, ip in intra_preds.items():
                    ip["probs"] = compute_bucket_probs(
                        ip["post_mean"], ip["post_std"], markets,
                        is_today, is_min_temp,
                        max_sf=state.get("max_so_far"),
                        min_sf=state.get("min_so_far"),
                    )
                    ip["mean"] = ip["post_mean"]
                    ip["std"] = ip["post_std"]
                    ip["source"] = f"🔥 {mk}"
                    output[mk] = ip

    return output


def calculate_kelly(model_prob: float, market_price: float, kelly_frac: float) -> float:
    """Simple single-outcome Kelly fraction."""
    market_price = min(market_price, 0.99)
    edge = model_prob - market_price
    if edge > 0.01:
        f = edge / (1.0 - market_price)
        return min(f * kelly_frac, 0.10)
    return 0.0
