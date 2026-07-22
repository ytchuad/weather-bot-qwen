# app/services/today_event_resolver.py
"""Resolve today's Polymarket event for the HK temperature market.

Used by the Hub on first render to auto-load an event so the user lands on
a useful dashboard rather than an empty page.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from cachetools import TTLCache, cached

from ..config import CACHE_TTL_MEDIUM
from .market_service import parse_date_from_event, search_events

logger = __import__("logging").getLogger(__name__)

_medium_cache = TTLCache(maxsize=32, ttl=CACHE_TTL_MEDIUM)


def _is_min_temp(event: dict) -> bool:
    title = (event.get("title") or "").lower()
    slug = (event.get("slug") or "").lower()
    return "lowest" in title or "lowest" in slug


@cached(_medium_cache)
def _cached_search(query: str) -> list[dict]:
    """Cache-wrapped Polymarket search."""
    return search_events(query)


def resolve_today_event(
    metric: Literal["auto", "tmax", "tmin"] = "auto",
    today: date | None = None,
) -> dict | None:
    """Return today's HK-temperature Polymarket event, with sensible fallbacks.

    Resolution order:
      1. Search "hong-kong-temperature", parse dates from each title.
      2. Pick an event whose parsed date == `today` matching `metric`.
      3. If today's event missing → nearest future event for `metric`.
      4. If still nothing → nearest past event for `metric`.
      5. metric='auto' prefers Tmax over Tmin when both exist for a date.

    Returns:
        Event dict with at least {title, slug, ...}, or None if no event found.
    """
    from .weather_service import hkt_now  # local import avoids cycle

    today = today or hkt_now().date()
    preferred = (
        metric if metric in ("tmax", "tmin") else None
    )  # auto → resolved below

    try:
        events = _cached_search("hong-kong-temperature")
    except Exception:
        return None
    if not events:
        return None

    # Bucket events by (date, metric); choose one within selection window.
    by_date: dict[date, dict[str, dict]] = {}
    for ev in events:
        d = parse_date_from_event(ev.get("title", ""), ev.get("slug", ""))
        if d is None:
            continue
        kind = "tmin" if _is_min_temp(ev) else "tmax"
        by_date.setdefault(d, {})[kind] = ev

    def _pick_for(d: date) -> dict | None:
        bucket = by_date.get(d, {})
        if preferred == "tmax":
            return bucket.get("tmax")
        if preferred == "tmin":
            return bucket.get("tmin")
        # auto
        return bucket.get("tmax") or bucket.get("tmin")

    # 1. exact match today
    ev = _pick_for(today)
    if ev is not None:
        return ev

    # 2. nearest future
    future_dates = sorted(d for d in by_date if d > today)
    if future_dates:
        ev = _pick_for(future_dates[0])
        if ev is not None:
            return ev

    # 3. nearest past
    past_dates = sorted((d for d in by_date if d < today), reverse=True)
    if past_dates:
        ev = _pick_for(past_dates[0])
        if ev is not None:
            return ev

    return None
