# app/services/model_service.py
"""Model inference orchestration.

Coordinates 9-day XGBoost, AWS high-freq, and all intraday LightGBM models
(minute-level A/B/C/D/E). Handles Bayesian fusion and bucket-probability
computation for downstream UI.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from cachetools import TTLCache, cached

from ..config import (
    DEFAULT_TMAX_FORECAST_DELTA,
    DEFAULT_TMIN_FORECAST_DELTA,
    CACHE_TTL_MEDIUM,
    HKT_OFFSET,
)
from features.input_status import (
    InputStatus,
    build_forecast_status_from_values,
    build_observation_buffer_status,
    jsonable,
    make_status_bundle,
)

logger = logging.getLogger(__name__)

_medium_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_MEDIUM)


def _model_2a_lineage_metadata() -> dict:
    """Return lineage metadata without reloading trained model artifacts."""
    from inference.model_2a_adapters import Model2AV1Adapter, Model2AV2Adapter

    metadata = {}
    for key, adapter_cls in (("model_2a", Model2AV1Adapter), ("model_2a_v2", Model2AV2Adapter)):
        adapter = adapter_cls()
        metadata[key] = {
            "model_version": adapter.model_version,
            "feature_version": adapter.feature_version,
            "artifact_identity": adapter.artifact_identity,
            "artifact_directory": str(adapter.artifact_directory),
            "feature_spec": str(adapter.feature_spec_path),
            "feature_spec_path": str(adapter.feature_spec_path),
            "feature_list_path": str(adapter.feature_list_path),
            "wind_fields": "wind_highland_only" if adapter.model_version == "v1" else "wind_offshore_highland_only",
        }
    return metadata


def _rebuild_legacy_forecast_status(
    *,
    forecast_max: float | None,
    forecast_min: float | None,
    decision_timestamp,
    input_status: dict,
) -> dict:
    """Recalculate forecast age at the actual inference decision time."""
    existing = input_status.get("forecast_input_status") if isinstance(input_status, dict) else None
    existing_max = existing.get("forecast_max", {}) if isinstance(existing, dict) else {}
    existing_min = existing.get("forecast_min", {}) if isinstance(existing, dict) else {}
    issue_time = (
        input_status.get("forecast_issue_time")
        if isinstance(input_status, dict)
        else None
    ) or existing_max.get("forecast_issue_time")
    source = (
        input_status.get("forecast_source")
        if isinstance(input_status, dict)
        else None
    ) or existing_max.get("forecast_source")
    anomalies = []
    if isinstance(existing, dict):
        anomalies.extend(existing_max.get("continuity_anomaly", []))
        anomalies.extend(existing_min.get("continuity_anomaly", []))
    return build_forecast_status_from_values(
        forecast_max=forecast_max,
        forecast_min=forecast_min,
        decision_timestamp=decision_timestamp,
        forecast_issue_time=issue_time,
        forecast_target_date=(
            input_status.get("forecast_target_date")
            if isinstance(input_status, dict)
            else None
        ) or existing_max.get("forecast_target_date"),
        forecast_source=source,
        previous_forecast_max=existing_max.get("previous_forecast_value"),
        previous_forecast_min=existing_min.get("previous_forecast_value"),
        continuity_anomaly=list(dict.fromkeys(anomalies)),
    )


def _build_nowcast_status(nowcast_features: dict, decision_timestamp) -> dict:
    issue_time = nowcast_features.get("_issue_time") if nowcast_features else None
    if not nowcast_features:
        return {
            "rain_nowcast": InputStatus.fallback(
                None,
                fallback_method="unavailable",
                decision_timestamp=decision_timestamp,
                source_name="hko_rain_nowcast",
                raw_status="unavailable",
            ).to_dict()
        }
    status = {}
    for key in (
        "rain_nc_sum_0_60m",
        "rain_nc_sum_0_120m",
        "rain_nowcast_age_minutes",
        "rain_nowcast_missing_flag",
    ):
        if key not in nowcast_features:
            continue
        status[key] = InputStatus.from_value(
            nowcast_features.get(key),
            source_timestamp=issue_time,
            decision_timestamp=decision_timestamp,
            source_name="hko_rain_nowcast",
            stale_after_minutes=180.0,
            observation_method="nowcast_issue",
        ).to_dict()
    return status


# ── 9-day XGBoost ────────────────────────────────────────────────────

@cached(_medium_cache)
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
    input_status: dict | None = None,
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

    # Check if minute-level buffer has sufficient data
    df_today = state.get("df_today")
    buffer_sufficient = df_today is not None and len(df_today) >= 30
    if not buffer_sufficient:
        logger.warning(
            f"Insufficient intraday buffer for {target_date_str}: "
            f"{len(df_today) if df_today is not None else 0} rows. "
            "Predictions may be degraded."
        )

    rain_kw = rain_kwargs or {}
    nc_features = {}
    # Inject nowcast features for Model C (silently merge, no crash if missing)
    try:
        from features.nowcast_loader import get_nowcast_features
        nc = get_nowcast_features(snapshot_time=state.get("time_now"))
        if nc:
            nc_features = nc
    except Exception:
        pass

    hour_now = state["time_now"].hour
    minutes_since_midnight = hour_now * 60 + state["time_now"].minute

    # Collect live pressure + wind data for Model 2A
    from ..services.weather_service import (
        compute_pressure_kwargs,
        compute_wind_kwargs,
        fetch_hko_ilens_forecast,
        hkt_now,
    )
    # ``state['time_now']`` is the latest observation timestamp, not the
    # inference decision time.  Use the actual orchestration time so ages are
    # truthful when an observation buffer is delayed.
    decision_timestamp = hkt_now()
    pressure_kw = {}
    wind_kw = {}
    try:
        try:
            pressure_kw = compute_pressure_kwargs(
                decision_timestamp=decision_timestamp
            )
        except TypeError:
            # Preserve compatibility with test doubles and older integrations.
            pressure_kw = compute_pressure_kwargs()
        try:
            wind_kw = compute_wind_kwargs(decision_timestamp=decision_timestamp)
        except TypeError:
            wind_kw = compute_wind_kwargs()
    except Exception as _e:
        logger.warning("Failed to collect live pressure/wind data: %s", _e)

    # Compute forecast freshness from HKO XML ModelTime
    forecast_age_minutes = None
    forecast_lead_days = None
    try:
        from ..services.weather_service import HKO_FORECAST_URL_TEMPLATE
        import time as _time_mod
        import requests as _req
        _url = HKO_FORECAST_URL_TEMPLATE.format(ts=int(_time_mod.time() * 1000))
        _r = _req.get(_url, timeout=10)
        _r.raise_for_status()
        _data = _r.json()
        _model_time_str = _data.get("ModelTime", "")
        if _model_time_str:
            _model_dt = pd.to_datetime(str(_model_time_str), format="%Y%m%d%H")
            _model_dt_hkt = _model_dt + HKT_OFFSET
            _now = hkt_now()
            forecast_age_minutes = (_now - _model_dt_hkt).total_seconds() / 60
            _target_dt = pd.to_datetime(target_date_str, format="%Y%m%d")
            forecast_lead_days = (_target_dt.date() - _model_dt_hkt.date()).days
    except Exception as _e:
        logger.debug("Could not compute forecast freshness: %s", _e)
        forecast_age_minutes = None
        forecast_lead_days = None

    # Compute obs_data_age_minutes from state observation timestamp
    obs_data_age_minutes = None
    if state.get("time_now") and state.get("df_today") is not None and not state["df_today"].empty:
        _last_obs = state["df_today"]["datetime"].iloc[-1]
        _now = hkt_now()
        obs_data_age_minutes = (_now - _last_obs).total_seconds() / 60

    common = dict(
        current_datetime=state["time_now"],
        temp_60min_ago=state.get("temp_60m_ago", state["temp_now"]),
        temp_now=state["temp_now"],
        forecast_tmax=forecast_max if forecast_max is not None else (
            state.get("max_so_far", 30.0) if hour_now >= 20 else
            state.get("max_so_far", 30.0) + DEFAULT_TMAX_FORECAST_DELTA
        ),
        forecast_tmin=forecast_min if forecast_min is not None else (
            state.get("min_so_far", 10.0) if hour_now >= 20 else
            state.get("min_so_far", 10.0) + DEFAULT_TMIN_FORECAST_DELTA
        ),
        temp_120m_ago=state.get("temp_120m_ago", state["temp_60m_ago"]),
        max_so_far=state.get("max_so_far"),
        min_so_far=state.get("min_so_far"),
        rainfall_60m_filled=rain_kw.get("rain_60m", 0.0),
        rainfall_120m_filled=rain_kw.get("rain_120m", 0.0),
        rainfall_60m_missing_flag=rain_kw.get("rainfall_60m_missing_flag", 1),
        rainfall_120m_missing_flag=rain_kw.get("rainfall_120m_missing_flag", 1),
        temp_change_30min=state.get("temp_change_30m", 0.0),
        temp_change_60min=state.get("temp_change_60m", 0.0),
        temp_volatility_60m=state.get("temp_volatility_60m", 0.0),
        temp_acceleration_60m=state.get("temp_acceleration_60m", 0.0),
        rh_change_60m=state.get("rh_change_60m", 0.0),
        dew_point_change_60m=state.get("dew_point_change_60m", 0.0),
        dew_point_spread_change_60m=state.get("dew_point_spread_change_60m", 0.0),
        time_since_max_so_far=state.get("time_since_max", 0.0),
        time_since_min_so_far=state.get("time_since_min", 0.0),
        hour=hour_now,
        minutes_since_midnight=minutes_since_midnight,
        rh_current=state.get("rh_now", 50.0),
        temp_buffer=state.get("df_today", pd.DataFrame()).get("temp", pd.Series()).dropna().tolist() if state.get("df_today") is not None else None,
        rh_buffer=state.get("df_today", pd.DataFrame()).get("rh", pd.Series()).dropna().tolist() if state.get("df_today") is not None else None,
        # Model 2A / Model G pressure data
        pressure_current=pressure_kw.get("pressure_current"),
        pressure_30m_ago=pressure_kw.get("pressure_30m_ago"),
        pressure_change_60m=pressure_kw.get("pressure_change_60m", 0.0),
        pressure_change_180m=pressure_kw.get("pressure_change_180m", 0.0),
        wind_ref_mean=wind_kw.get("wind_ref_mean", 0.0),
        wind_ref_max=wind_kw.get("wind_ref_max", 0.0),
        wind_victoria_harbour_mean=wind_kw.get("wind_victoria_harbour_mean", 0.0),
        wind_victoria_harbour_max=wind_kw.get("wind_victoria_harbour_max", 0.0),
        wind_highland_mean=wind_kw.get("wind_highland_mean", 0.0),
        wind_highland_max=wind_kw.get("wind_highland_max", 0.0),
        wind_offshore_highland_mean=wind_kw.get("wind_offshore_highland_mean", 0.0),
        wind_offshore_highland_max=wind_kw.get("wind_offshore_highland_max", 0.0),
        wind_all_change_60m=wind_kw.get("wind_all_change_60m", 0.0),
        wind_kings_park_current=wind_kw.get("wind_kings_park_current", 0.0),
        forecast_age_minutes=forecast_age_minutes,
        forecast_lead_days=forecast_lead_days,
        obs_data_age_minutes=obs_data_age_minutes,
        wind_data_age_minutes=None,
    )

    _ALLOWED_RAIN_KWARGS = {
        "prev_18_temp", "prev_21_temp", "prev_2359_temp",
        "prev_evening_temp_change", "prev_evening_temp_min",
        "prev_evening_temp_range", "prev_evening_temp_slope",
        "prev_evening_rh_mean", "prev_evening_rh_max",
        "prev_evening_dew_point_mean",
        "prev_evening_rainfall_18_24", "prev_evening_rain_flag",
        "rainfall_30m_filled", "rainfall_30m_missing_flag",
        # Model 2B rainfall features (computed in compute_rain_kwargs)
        "rainfall_60m", "rainfall_120m", "has_recent_rainfall_obs",
        "rain_intensity_max_120m", "rain_cooling_60m",
        "rain_after_max_flag", "post_peak_rain_flag",
        "rainfall_data_age_minutes", "rain_data_gap_flag",
        "temp_buffer_long", "rh_buffer_long",
    }
    for k, v in rain_kw.items():
        if k in _ALLOWED_RAIN_KWARGS and k not in common:
            common[k] = v

    # Inject nowcast features directly (bypass ALLOWED_RAIN_KWARGS whitelist)
    common.update({k: v for k, v in nc_features.items() if k != "_issue_time" and k not in common})

    observation_values = {
        key: state.get(key)
        for key in (
            "temp_now", "rh_now", "max_so_far", "min_so_far",
            "temp_30m_ago", "temp_60m_ago", "temp_120m_ago",
            "temp_change_30m", "temp_change_60m", "temp_volatility_60m",
            "temp_acceleration_60m", "rh_change_60m",
            "dew_point_change_60m", "dew_point_spread_change_60m",
            "time_since_max", "time_since_min",
        )
    }
    observation_buffer_status = build_observation_buffer_status(
        state.get("df_today"),
        decision_timestamp=decision_timestamp,
        values=observation_values,
    )
    previous_rh_status = (state.get("weather_input_status") or {}).get(
        "rh_current", {}
    )
    if previous_rh_status.get("fallback_method") == "climatological_default":
        observation_buffer_status["rh_current"] = InputStatus.fallback(
            state.get("rh_now", 50.0),
            fallback_method="climatological_default",
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs",
            raw_status="synthetic_fallback",
            observation_method="fallback",
        ).to_dict()
    weather_input_status = {
        key: observation_buffer_status.get(key)
        for key in (
            "temp_current", "rh_current", "pressure_current",
            "dew_point_current", "max_so_far", "min_so_far",
            "obs_data_age_minutes",
        )
        if key in observation_buffer_status
    }
    forecast_input_status = _rebuild_legacy_forecast_status(
        forecast_max=forecast_max,
        forecast_min=forecast_min,
        decision_timestamp=decision_timestamp,
        input_status=input_status or {},
    )
    status_bundle = make_status_bundle(
        {
            "weather_input_status": weather_input_status,
            "observation_buffer_status": observation_buffer_status,
            "wind_input_status": wind_kw.get("_input_status", {}),
            "pressure_input_status": pressure_kw.get("_input_status", {}),
            "forecast_input_status": forecast_input_status,
            "rain_input_status": rain_kw.get("_input_status", {}),
            "nowcast_input_status": _build_nowcast_status(
                nc_features, decision_timestamp
            ),
        },
        decision_timestamp=decision_timestamp,
    )

    # Fetch i-lens forecast (same source as training) for Model 2A1
    ilens_forecast = None
    ilens_forecast_tmax = None
    ilens_forecast_tmin = None
    ilens_forecast_age_minutes = None
    ilens_forecast_lead_days = None
    try:
        ilens_forecast = fetch_hko_ilens_forecast(target_date_str)
        if ilens_forecast:
            ilens_forecast_tmax = ilens_forecast.get("forecast_tmax")
            ilens_forecast_tmin = ilens_forecast.get("forecast_tmin")
            if ilens_forecast.get("forecast_issue_date") and ilens_forecast.get("forecast_issue_time"):
                _issue = pd.to_datetime(
                    f"{ilens_forecast['forecast_issue_date']} {ilens_forecast['forecast_issue_time']}"
                )
                _issue_hkt = _issue.tz_localize("Asia/Hong_Kong") if hasattr(_issue, 'tz_localize') else _issue
                _now = hkt_now()
                ilens_forecast_age_minutes = (_now - _issue_hkt).total_seconds() / 60
                _target_dt = pd.to_datetime(target_date_str, format="%Y%m%d")
                ilens_forecast_lead_days = (_target_dt.date() - _issue_hkt.date()).days
    except Exception as _e:
        logger.debug("Could not fetch i-lens forecast: %s", _e)

    try:
        if is_min_temp:
            raw_preds = predict_intraday_tmin_all(**common)
        else:
            common_tmax = dict(common)
            common_tmax["input_status"] = status_bundle
            common_tmax["time_since_max_so_far"] = state.get("time_since_max", 0.0)
            common_tmax["ilens_forecast_tmax"] = ilens_forecast_tmax
            common_tmax["ilens_forecast_tmin"] = ilens_forecast_tmin
            common_tmax["ilens_forecast_age_minutes"] = ilens_forecast_age_minutes
            common_tmax["ilens_forecast_lead_days"] = ilens_forecast_lead_days
            # Model 4 forecast rain/humidity features (from i-lens weather description)
            try:
                if ilens_forecast and ilens_forecast.get("forecast_rain_prob"):
                    _fc_df = pd.DataFrame([{
                        "forecast_rain_prob": ilens_forecast["forecast_rain_prob"],
                        "forecast_weather_desc": ilens_forecast.get("forecast_weather_desc", ""),
                        "forecast_min_rh": ilens_forecast.get("forecast_min_rh"),
                        "forecast_max_rh": ilens_forecast.get("forecast_max_rh"),
                    }])
                    from features.model_4_feature_builder import build_forecast_features_m4
                    _m4_fc = build_forecast_features_m4(_fc_df)
                    common_tmax.update(_m4_fc)
            except Exception as _m4e:
                logger.debug("Model 4 forecast features failed: %s", _m4e)
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
        # Apply degradation handling for insufficient buffer data
        if not buffer_sufficient and ps < 0.5:
            ps = 0.5
            raw_pred["degraded"] = True
        results[mk] = {
            "post_mean": float(pm),
            "post_std": float(ps),
            "raw": raw_pred,
        }
    results["_feature_metadata"] = {
        "pressure_kwargs": jsonable(pressure_kw),
        "wind_kwargs": jsonable(wind_kw),
        "nowcast_features": jsonable(nc_features),
        "forecast_age_minutes": forecast_age_minutes,
        "forecast_lead_days": forecast_lead_days,
        "input_status": status_bundle,
        "status_contract_version": status_bundle["status_contract_version"],
        "numeric_policy": status_bundle["numeric_policy"],
        "status_policy": status_bundle["status_policy"],
        "decision_timestamp": status_bundle["decision_timestamp"],
        "weather_input_status": status_bundle["weather_input_status"],
        "wind_input_status": status_bundle["wind_input_status"],
        "pressure_input_status": status_bundle["pressure_input_status"],
        "forecast_input_status": status_bundle["forecast_input_status"],
        "observation_buffer_status": status_bundle["observation_buffer_status"],
        "rain_input_status": status_bundle["rain_input_status"],
        "nowcast_input_status": status_bundle["nowcast_input_status"],
    }
    try:
        lineage = _model_2a_lineage_metadata()
        results["_feature_metadata"]["model_lineage"] = lineage
        results["_feature_metadata"]["feature_spec"] = {
            key: value["feature_spec_path"] for key, value in lineage.items()
        }
    except Exception as _lineage_error:
        logger.warning("Could not collect Model 2A lineage metadata: %s", _lineage_error)
        results["_feature_metadata"]["model_lineage"] = {
            "error": str(_lineage_error)
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
    prob_max_reached: float | None = None,
    prob_min_reached: float | None = None,
) -> dict[str, float]:
    """Compute bucket probabilities via predict_bucket_probabilities.

    Returns a dict keyed by **bucket label** (e.g. "32-33") so that
    downstream UI components (bucket_bars, recommendation_table) can
    look up probabilities by the same canonical keys they use for
    market_prices.

    ``prob_max_reached`` / ``prob_min_reached`` are the zero-inflated point
    masses from the upside_zero / downside_zero classifiers; when supplied
    (intraday, today only) the mapper applies the mixture model (see
    ``models.inference.predict_bucket_probabilities``).
    """
    try:
        from models.inference import predict_bucket_probabilities

        raw_probs = predict_bucket_probabilities(
            mean, std, markets,
            max_since_midnight=max_sf,
            min_since_midnight=min_sf,
            is_today=is_today,
            is_min_temp=is_min_temp,
            prob_max_reached=prob_max_reached,
            prob_min_reached=prob_min_reached,
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
    forecast_max: float | None = None,
    forecast_min: float | None = None,
    is_today: bool = True,
    input_status: dict | None = None,
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
    if state and isinstance(state, dict) and "time_now" in state and markets:
        intra_preds = predict_intraday_all(
            target_date_str, is_min_temp, state, rain_kwargs,
            forecast_max=forecast_max, forecast_min=forecast_min,
            input_status=input_status,
        )
        if intra_preds:
            if "_error" in intra_preds:
                output["_intraday_error"] = intra_preds["_error"]
            else:
                for mk, ip in intra_preds.items():
                    if mk in ("_feature_metadata", "_error"):
                        continue
                    # zero-inflated point mass from the upside_zero /
                    # downside_zero classifier (carried in ip["raw"]).
                    _raw = ip.get("raw") or {}
                    _pmr = _raw.get("prob_max_reached")
                    _pnr = _raw.get("prob_min_reached")
                    ip["probs"] = compute_bucket_probs(
                        ip["post_mean"], ip["post_std"], markets,
                        is_today, is_min_temp,
                        max_sf=state.get("max_so_far"),
                        min_sf=state.get("min_so_far"),
                        prob_max_reached=_pmr,
                        prob_min_reached=_pnr,
                    )
                    ip["mean"] = ip["post_mean"]
                    ip["std"] = ip["post_std"]
                    ip["source"] = f"🔥 {mk}"
                    if ip.get("degraded"):
                        ip["degraded"] = True
                    output[mk] = ip

            _fm = intra_preds.pop("_feature_metadata", {})
            if _fm:
                output["_feature_metadata"] = _fm
    return output


def calculate_kelly(model_prob: float, market_price: float, kelly_frac: float) -> float:
    """Simple single-outcome Kelly fraction."""
    market_price = min(market_price, 0.99)
    edge = model_prob - market_price
    if edge > 0.01:
        f = edge / (1.0 - market_price)
        return min(f * kelly_frac, 0.10)
    return 0.0
