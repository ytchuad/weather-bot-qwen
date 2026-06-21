from __future__ import annotations

import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException

from app.api.cache import prediction_cache
from app.api.schemas import Suggestion, SuggestRequest, SuggestResponse
from app.services.model_service import calculate_kelly, run_all_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["Strategies"])


@router.post("/suggest", response_model=SuggestResponse)
@prediction_cache
def suggest_strategy(req: SuggestRequest):
    from app.services.market_service import fetch_event_markets
    from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now

    target_date = _parse_date(req.date)
    target_date_str = req.date

    hkt = hkt_now()
    is_today = target_date == hkt.date()

    hko = fetch_hko_data(target_date_str)
    _sd = target_date_str.replace("-", "")
    state = get_intraday_state(_sd)
    from app.services.weather_service import compute_rain_kwargs

    rain_kwargs = compute_rain_kwargs(_sd, hkt_now())
    markets = fetch_event_markets(target_date_str, is_min_temp=req.is_min_temp)

    forecast_key = "forecast_min" if req.is_min_temp else "forecast_max"
    forecast_aws = hko.get(forecast_key) if hko else None

    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=req.is_min_temp,
        bias=0.0,
        std_mult=1.0,
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=forecast_aws,
        is_today=is_today,
    )

    suggestions = []
    for mk, pred in results.items():
        if mk == "_intraday_error" or not pred.get("probs"):
            continue
        for bucket_name, model_prob in pred["probs"].items():
            market_price = _find_market_price(markets, bucket_name)
            if market_price is None:
                continue
            edge = model_prob - market_price
            if edge > 0.01:
                kelly = calculate_kelly(model_prob, market_price, req.kelly_fraction)
                action = "buy_yes" if model_prob > market_price else "buy_no"
            else:
                kelly = 0.0
                action = "pass"

            suggestions.append(
                Suggestion(
                    bucket=bucket_name,
                    market_price=market_price,
                    model_prob=model_prob,
                    edge=edge,
                    kelly_fraction=kelly,
                    action=action,
                )
            )

    suggestions.sort(key=lambda s: s.edge, reverse=True)
    return SuggestResponse(date=req.date, suggestions=suggestions)


def _parse_date(d: str) -> date_type:
    try:
        return date_type.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date: {d}. Use YYYY-MM-DD.")


def _find_market_price(markets: list[dict], bucket_name: str) -> float | None:
    for m in markets:
        if m.get("bucket") == bucket_name or m.get("name") == bucket_name:
            return m.get("yes_price")
    return None
