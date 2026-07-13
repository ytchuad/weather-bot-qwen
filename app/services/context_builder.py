from __future__ import annotations

import logging
from typing import Any

from execution.strategy_account import StrategyAccount

logger = logging.getLogger(__name__)


def build_strategy_context(acct: StrategyAccount) -> dict[str, Any]:
    """Fetch weather data, run models, build full execution context for one account.

    Returns an empty dict if markets or model predictions are unavailable
    (caller should skip that cycle).
    """
    from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now, compute_rain_kwargs
    from app.services.market_service import fetch_today_event, fetch_event_markets
    from app.services.market_depth_service import get_global_depth_cache
    from app.services.model_service import run_all_models
    from execution.market_templates import resolve_slug

    target_date = hkt_now().date()
    target_date_str = target_date.strftime("%Y-%m-%d")
    _sd = target_date_str.replace("-", "")
    is_min_temp = acct.market_template == "hk-tmin"
    params = acct.params or {}

    today_event = fetch_today_event(target_date_str)
    slug = today_event.get("slug") if today_event else resolve_slug(acct.market_template)
    markets = fetch_event_markets(slug, is_min_temp=is_min_temp) if slug else []
    if not markets:
        logger.warning("No markets found for %s (slug=%s)", acct.id, slug)
        return {}

    hko = fetch_hko_data(target_date_str)
    state = get_intraday_state(_sd)
    _msf = state.get("max_so_far")
    _tnow = state.get("temp_now")
    _drop_from_max = (_msf - _tnow) if (_msf is not None and _tnow is not None) else 0.0
    rain_kwargs = compute_rain_kwargs(
        _sd, hkt_now(),
        drop_from_max=_drop_from_max,
        temp_change_60m=state.get("temp_change_60m", 0.0),
    )
    forecast_key = "forecast_min" if is_min_temp else "forecast_max"
    forecast_aws = hko.get(forecast_key) if hko else None

    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=is_min_temp,
        bias=params.get("bias", 0.0),
        std_mult=params.get("std_mult", 1.0),
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=forecast_aws,
        forecast_max=hko.get("forecast_max") if hko else None,
        forecast_min=hko.get("forecast_min") if hko else None,
        is_today=True,
    )

    model = acct.model
    target_probs = results.get(model, {}).get("probs", {})
    if not target_probs:
        logger.error("Model %s produced no probs for %s", model, acct.id)
        return {}

    prices_dict = {m["bucket"]: m.get("yes_price", 0.5) for m in markets}
    token_ids_dict = {m["bucket"]: m.get("token_id", "") for m in markets}
    no_token_ids_dict = {m["bucket"]: m.get("no_token_id", "") for m in markets}

    depth_cache = get_global_depth_cache()
    depth_cache.update_token_ids(
        {b: t for b, t in token_ids_dict.items() if t},
        {b: t for b, t in no_token_ids_dict.items() if t},
    )
    market_depth = depth_cache.get()
    market_depth_no = depth_cache.get_no()

    post_mean = results.get(model, {}).get("mean") if results else None

    context_json: dict[str, Any] = {}
    if state:
        for k in ("temp_30m_ago", "temp_60m_ago", "temp_120m_ago",
                   "min_so_far", "rh_now", "temp_change_30m", "temp_change_60m",
                   "time_since_max", "time_since_min",
                   "temp_volatility_60m", "temp_acceleration_60m",
                   "rh_change_60m", "dew_point_change_60m",
                   "dew_point_spread_change_60m"):
            v = state.get(k)
            if v is not None:
                context_json[k] = v
    if hko:
        for k in ("max_since_midnight", "min_since_midnight", "forecast_max", "forecast_min"):
            v = hko.get(k)
            if v is not None:
                context_json[k] = v
    for k in ("rain_60m", "rain_120m", "rain_data_ok",
               "rainfall_60m_missing_flag", "rainfall_120m_missing_flag",
               "rainfall_30m_missing_flag", "rainfall_data_age_minutes",
               "rain_data_gap_flag", "rain_regime",
               # Model 2B rainfall features
               "rainfall_60m", "rainfall_120m", "has_recent_rainfall_obs",
               "rain_intensity_max_120m", "rain_cooling_60m",
               "rain_after_max_flag", "post_peak_rain_flag"):
        v = rain_kwargs.get(k)
        if v is not None:
            context_json[k] = v

    if state and state.get("df_today") is not None:
        _df = state["df_today"]
        context_json["buffer_len"] = len(_df)
        if len(_df) >= 30:
            context_json["temp_at_idx30"] = float(_df["temp"].iloc[-30])
        if len(_df) >= 60:
            context_json["temp_at_idx60"] = float(_df["temp"].iloc[-60])
            if "rh" in _df.columns and _df["rh"].iloc[-60] is not None:
                context_json["rh_at_idx60"] = float(_df["rh"].iloc[-60])

    model_stds = {}
    if results:
        for mk, pred in results.items():
            if mk != "_intraday_error" and pred.get("std") is not None:
                model_stds[mk] = pred["std"]
    if model_stds:
        context_json["model_stds"] = model_stds

    if results:
        _probs = {}
        for mk, pred in results.items():
            if mk != "_intraday_error" and pred.get("probs"):
                _probs[mk] = pred["probs"]
        if _probs:
            context_json["model_probs"] = _probs

    if prices_dict:
        context_json["market_prices"] = prices_dict
    if market_depth:
        context_json["market_depth"] = market_depth
    if market_depth_no:
        context_json["market_depth_no"] = market_depth_no

    gamma_market_info = {}
    for m in markets:
        bucket = m.get("bucket")
        if not bucket:
            continue
        info = {}
        for k in ("token_id", "conditionId", "bestBid", "bestAsk",
                   "spread", "lastTradePrice", "liquidityClob", "volume24hrClob"):
            v = m.get(k)
            if v is not None:
                info[k] = v
        if info:
            gamma_market_info[bucket] = info
    if gamma_market_info:
        context_json["gamma_market_info"] = gamma_market_info

    if results and "model_2a" in results:
        _m2a_raw = results["model_2a"].get("raw", {})
        _m2a_f = _m2a_raw.get("_features")
        if _m2a_f:
            context_json["model_2a_features"] = _m2a_f

    _meta = results.pop("_feature_metadata", {})
    if _meta:
        context_json["feature_metadata"] = _meta

    return dict(
        capital=acct.capital,
        model_key=model,
        mock_slippage=True,
        bias=params.get("bias", 0.0),
        std_mult=params.get("std_mult", 1.0),
        kelly_fraction=params.get("kelly_fraction", 0.25),
        slug=slug,
        target_probs=target_probs,
        prices_dict=prices_dict,
        token_ids_dict=token_ids_dict,
        temp_now=state.get("temp_now") if state else None,
        max_so_far=state.get("max_so_far") if state else None,
        rain_regime=rain_kwargs.get("rain_regime", "no_rain"),
        model_std=1.5,
        recent_price_volatility=0.0,
        hours_to_settlement=24.0,
        nowcast_stale=False,
        data_missing=False,
        drawdown_pct=0.0,
        markets=markets,
        post_mean=post_mean,
        is_min_temp=is_min_temp,
        target_date_str=target_date_str,
        all_results=results,
        context_json=context_json,
    )
