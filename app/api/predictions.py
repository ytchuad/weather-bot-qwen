from __future__ import annotations

import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException

from app.api.cache import prediction_cache
from app.api.schemas import ModelPrediction, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


def _svc_date(d: str) -> str:
    """Convert ISO date YYYY-MM-DD to service format YYYYMMDD."""
    return d.replace("-", "")


def _get_intraday_state_for(target_date: str):
    from app.services.weather_service import get_intraday_state

    state = get_intraday_state(_svc_date(target_date))
    # Ensure we don't cache None or invalid state
    if state is None or not isinstance(state, dict):
        return None
    return state


def _compute_rain_kwargs_for(target_date: str):
    from app.services.weather_service import compute_rain_kwargs, hkt_now

    return compute_rain_kwargs(_svc_date(target_date), hkt_now())


def _parse_date(d: str) -> date_type:
    try:
        return date_type.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date: {d}. Use YYYY-MM-DD.")


@router.get("", response_model=PredictionResponse)
@prediction_cache
def get_predictions(
    date: str,
    is_min_temp: bool = False,
    bias: float = 0.0,
    std_mult: float = 1.0,
):
    from app.services.market_service import fetch_event_markets, _resolve_today_event
    from app.services.model_service import run_all_models
    from app.services.weather_service import fetch_hko_data, hkt_now

    target_date = _parse_date(date)
    target_date_str = date

    hkt = hkt_now()
    is_today = target_date == hkt.date()

    hko = fetch_hko_data(target_date_str)
    state = _get_intraday_state_for(target_date_str)
    rain_kwargs = _compute_rain_kwargs_for(target_date_str)

    markets = []
    ev = _resolve_today_event(target_date_str, is_min_temp)
    if ev and "slug" in ev:
        markets = fetch_event_markets(ev["slug"], is_min_temp=is_min_temp)

    forecast_key = "forecast_min" if is_min_temp else "forecast_max"
    forecast_aws = hko.get(forecast_key) if hko else None

    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=is_min_temp,
        bias=bias,
        std_mult=std_mult,
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=forecast_aws,
        forecast_max=hko.get("forecast_max") if hko else None,
        forecast_min=hko.get("forecast_min") if hko else None,
        is_today=is_today,
    )

    models_out = {}
    for mk, pred in results.items():
        if mk == "_intraday_error":
            continue
        models_out[mk] = ModelPrediction(
            mean=pred.get("mean", 0.0),
            std=pred.get("std", 0.0),
            source=pred.get("source", mk),
            probs=pred.get("probs"),
        )

    return PredictionResponse(date=date, models=models_out)
