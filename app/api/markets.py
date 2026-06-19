from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.cache import market_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/markets", tags=["Markets"])


@router.get("/buckets")
def get_buckets(type: str = "tmax"):
    from app.config import TMAX_BUCKETS, TMIN_BUCKETS

    if type == "tmin":
        return {"type": "tmin", "buckets": TMIN_BUCKETS}
    return {"type": "tmax", "buckets": TMAX_BUCKETS}


@router.get("/events")
@market_cache
def search_events(date: str | None = None):
    from app.services.market_service import search_events

    events = search_events(date)
    return {"events": events}


@router.get("/event/{slug}")
@market_cache
def get_event(slug: str, is_min_temp: bool = False):
    from app.services.market_service import fetch_event_markets, search_events

    markets = fetch_event_markets(slug, is_min_temp=is_min_temp)
    if not markets:
        events = search_events(str(None))
        match = [e for e in events if e.get("slug") == slug]
        if not match:
            raise HTTPException(status_code=404, detail=f"Event not found: {slug}")
        return {"slug": slug, "markets": [], "prices": {}}

    prices = {}
    for m in markets:
        name = m.get("name", "")
        bucket = m.get("bucket", name)
        price = m.get("yes_price")
        if price is not None:
            price = float(price)
            if price == float("-inf") or price == float("inf"):
                price = 0.0
            prices[bucket] = price

    safe_markets = []
    for m in markets:
        safe_m = dict(m)
        for k, v in list(safe_m.items()):
            if isinstance(v, float) and (v == float("-inf") or v == float("inf")):
                safe_m[k] = 0.0
        safe_markets.append(safe_m)

    return {
        "slug": slug,
        "markets": safe_markets,
        "prices": prices,
    }


def _resolve_today_event(date_str: str | None = None):
    """Resolve an HK temperature event for the given date, without Streamlit."""
    from datetime import date as date_type
    from app.services.market_service import search_events, parse_date_from_event
    from app.services.weather_service import hkt_now

    today = hkt_now().date()
    target = date_str or today.isoformat()
    try:
        target_date = date_type.fromisoformat(target) if isinstance(target, str) else today
    except Exception:
        target_date = today

    try:
        events = search_events("hong-kong-temperature")
    except Exception:
        return None

    def is_min_temp(ev):
        title = (ev.get("title") or "").lower()
        slug = (ev.get("slug") or "").lower()
        return "lowest" in title or "lowest" in slug

    by_date: dict[date_type, dict] = {}
    for ev in events:
        d = parse_date_from_event(ev.get("title", ""), ev.get("slug", ""))
        if d is None:
            continue
        kind = "tmin" if is_min_temp(ev) else "tmax"
        by_date.setdefault(d, {})[kind] = ev

    bucket = by_date.get(target_date, {})
    ev = bucket.get("tmax") or bucket.get("tmin")
    if ev is None:
        future_dates = sorted([d for d in by_date if d > target_date])
        if future_dates:
            ev = by_date[future_dates[0]].get("tmax") or by_date[future_dates[0]].get("tmin")
        if ev is None:
            past_dates = sorted([d for d in by_date if d < target_date], reverse=True)
            if past_dates:
                ev = by_date[past_dates[0]].get("tmax") or by_date[past_dates[0]].get("tmin")
    return ev


@router.get("/today-event")
@market_cache
def get_today_event(date: str | None = None):
    """Return today's HK temperature event (Tmax preferred) with event details."""
    ev = _resolve_today_event(date)
    if ev is None:
        raise HTTPException(status_code=404, detail="No temperature event found for date")
    return {"event": ev}
