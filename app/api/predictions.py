from __future__ import annotations

import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException

from app.api.cache import prediction_cache, weather_cache
from app.api.schemas import ModelPrediction, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


def _svc_date(d: str) -> str:
    """Convert ISO date YYYY-MM-DD to service format YYYYMMDD."""
    return d.replace("-", "")


def _get_intraday_state_for(target_date: str):
    from app.services.weather_service import get_intraday_state

    return get_intraday_state(_svc_date(target_date))


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
    from app.services.market_service import fetch_event_markets
    from app.services.model_service import run_all_models
    from app.services.weather_service import fetch_hko_data, hkt_now

    target_date = _parse_date(date)
    target_date_str = date

    hkt = hkt_now()
    is_today = target_date == hkt.date()

    hko = fetch_hko_data(target_date_str)
    state = _get_intraday_state_for(target_date_str)
    rain_kwargs = _compute_rain_kwargs_for(target_date_str)

    markets = fetch_event_markets(target_date_str, is_min_temp=is_min_temp)

    forecast_aws = hko.get("forecast") if hko else None

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
