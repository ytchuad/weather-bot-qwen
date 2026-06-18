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
        if m.get("yes_price") is not None:
            prices[bucket] = m["yes_price"]

    return {
        "slug": slug,
        "markets": markets,
        "prices": prices,
    }
