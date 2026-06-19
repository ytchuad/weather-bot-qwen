from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.cache import weather_cache
from app.api.schemas import WeatherNow
from app.services.weather_service import (
    fetch_hko_data,
    fetch_live_hko_temp_rh,
    hkt_now,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weather", tags=["Weather"])


@router.get("/now", response_model=WeatherNow)
@weather_cache
def weather_now(date: str | None = None):
    hkt = hkt_now()
    target_date = date or hkt.date()
    temp_rh = fetch_live_hko_temp_rh()
    hko = fetch_hko_data(str(target_date).replace("-", ""))

    temp = rh = None
    if temp_rh:
        _, temp, rh = temp_rh

    return WeatherNow(
        date=str(target_date),
        temp=temp,
        humidity=rh,
        max_today=hko.get("max_since_midnight"),
        min_today=hko.get("min_since_midnight"),
        forecast=hko.get("forecast_max"),
        aws_temp=hko.get("aws_temp"),
        source="HKO API",
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/intraday")
@weather_cache
def weather_intraday(date: str):
    from app.services.weather_service import get_intraday_state

    state = get_intraday_state(date.replace("-", ""))
    if state is None:
        return {"error": "No intraday data available", "date": date}
    return {
        "date": date,
        "temp_now": state.get("temp_now"),
        "rh_now": state.get("rh_now"),
        "max_so_far": state.get("max_so_far"),
        "min_so_far": state.get("min_so_far"),
        "time_now": state.get("time_now").isoformat() if state.get("time_now") else None,
        "temp_60m_ago": state.get("temp_60m_ago"),
    }


@router.get("/rain")
@weather_cache
def weather_rain(date: str):
    from app.services.weather_service import compute_rain_kwargs

    kwargs = compute_rain_kwargs(date.replace("-", ""), hkt_now())
    return {"date": date, **kwargs}
