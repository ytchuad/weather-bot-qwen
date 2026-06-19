# app/services/market_service.py
"""Polymarket event/market data fetching.

All functions use cachetools TTLCache for caching.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
from cachetools import TTLCache, cached

from ..config import (
    PM_GAMMA_API,
    PM_SEARCH_URL,
    PM_EVENTS_URL,
    TMAX_BUCKETS,
    TMIN_BUCKETS,
    CACHE_TTL_MEDIUM,
    CACHE_TTL_SHORT,
)

logger = logging.getLogger(__name__)

_medium_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_MEDIUM)
_short_cache = TTLCache(maxsize=128, ttl=CACHE_TTL_SHORT)

logger = logging.getLogger(__name__)


# ── public helpers ────────────────────────────────────────────────────

def parse_date_from_event(title: str, slug: str) -> date | None:
    """Extract a target date from event title / slug strings."""
    months = (
        "January|February|March|April|May|June|"
        "July|August|September|October|November|December"
    )
    # numeric "6-13" / "6/13" style
    m = re.search(r"(\d{1,2})[—\-/](\d{1,2})", title.replace("'", ""))
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = datetime.now().year
        d = date(year, month, day)
        if d < date.today():
            d = date(year + 1, month, day)
        return d

    # "on-june-13-2026" from slug
    if slug:
        from calendar import month_abbr as _ma
        mmap = {n.lower(): i for i, n in enumerate(_ma) if n}
        sm = re.search(r"on-([a-z]+)-(\d{1,2})-(\d{4})$", slug)
        if sm:
            mn = mmap.get(sm.group(1)[:3])
            if mn:
                return date(int(sm.group(3)), mn, int(sm.group(2)))

    # "on June 13" from title
    from calendar import month_name as _mn
    mmap2 = {n.lower(): i for i, n in enumerate(_mn) if n}
    tm = re.search(r"on\s+([A-Za-z]+)\s+(\d{1,2})", title)
    if tm:
        mn2 = mmap2.get(tm.group(1).lower())
        if mn2:
            year = datetime.now().year
            d = date(year, mn2, int(tm.group(2)))
            if d < date.today():
                d = date(year + 1, mn2, int(tm.group(2)))
            return d
    return None


def _parse_outcome_prices(market: dict) -> tuple[float | None, float | None]:
    """Extract Yes/No prices from a market dict (supports both old/new API shape)."""
    prices: dict[str, float] = {}

    op = market.get("outcomePrices")
    outcomes = market.get("outcomes", "")
    try:
        ol = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
    except Exception:
        ol = []
    if op and ol:
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except Exception:
                op = []
        if isinstance(op, list) and isinstance(ol, list):
            for name, ps in zip(ol, op):
                try:
                    prices[name] = float(ps)
                except (ValueError, TypeError):
                    pass

    if not prices:
        for token in market.get("tokens", []):
            outcome = token.get("outcome", "")
            price = token.get("price", None)
            if price is not None:
                try:
                    prices[outcome] = float(price)
                except (ValueError, TypeError):
                    pass

    yes = prices.get("Yes")
    no = prices.get("No")

    if yes is None:
        for t in market.get("tokens", []):
            if t.get("outcome", "").lower() == "yes":
                try:
                    yes = float(t.get("price", 0))
                except (ValueError, TypeError):
                    yes = 0.0
    if no is None:
        for t in market.get("tokens", []):
            if t.get("outcome", "").lower() == "no":
                try:
                    no = float(t.get("price", 0))
                except (ValueError, TypeError):
                    no = 0.0
    return yes, no


def _bucket_bounds(bucket_label: str) -> tuple[float, float]:
    """Map canonical bucket label to (lower, upper) float bounds.

    Tmin: <12 → (-inf, 12), 12-14 → (12, 14), …, >=28 → (28, inf)
    Tmax: <23 → (-inf, 23), 23-24 → (23, 24), …, >=34 → (34, inf)
    """
    stripped = bucket_label.strip()
    if stripped.startswith("<"):
        return (-np.inf, float(stripped[1:]))
    if stripped.startswith(">="):
        return (float(stripped[2:]), np.inf)
    if "-" in stripped:
        parts = stripped.split("-")
        return (float(parts[0]), float(parts[1]))
    # Fallback: treat as single value (unlikely but safe)
    try:
        v = float(stripped)
        return (v, v + 1.0)
    except ValueError:
        return (-np.inf, np.inf)


def _market_question_to_bucket(question: str, group_item_title: str, is_min_temp: bool) -> str | None:
    """Map Polymarket market question text to our bucket label."""
    source = group_item_title or question
    m = re.search(r"(\d+)\s*°?C", source, re.IGNORECASE)
    if not m:
        return None

    temp_val = int(m.group(1))
    lower = any(kw in source.lower() for kw in ["or below", "less than", "below"])
    upper = any(kw in source.lower() for kw in ["or higher", "greater than", "higher", "above"])

    if is_min_temp:
        if lower:
            if temp_val <= 12:
                return "<12"
            for lo in range(12, 28, 2):
                hi = lo + 2
                if temp_val <= hi:
                    return f"{lo}-{hi}"
            return ">=28"
        if upper:
            if temp_val >= 28:
                return ">=28"
            for lo in range(12, 28, 2):
                hi = lo + 2
                if temp_val >= lo:
                    return f"{lo}-{hi}"
            return "<12"
        for lo in range(12, 28, 2):
            hi = lo + 2
            if lo <= temp_val < hi:
                return f"{lo}-{hi}"
        if temp_val >= 28:
            return ">=28"
        return "<12"
    else:
        if lower:
            if temp_val <= 23:
                return "<23"
            for lo in range(23, 34):
                if temp_val <= lo:
                    return f"{lo}-{lo + 1}"
            return ">=34"
        if upper:
            if temp_val >= 34:
                return ">=34"
            for lo in range(23, 34):
                if temp_val >= lo:
                    return f"{lo}-{lo + 1}"
            return "<23"
        for lo in range(23, 34):
            if temp_val == lo:
                return f"{lo}-{lo + 1}"
        if temp_val >= 34:
            return ">=34"
        return "<23"


# ── cached fetchers ───────────────────────────────────────────────────

@cached(_medium_cache)
def search_events(query: str = "hong-kong-temperature") -> list[dict]:
    """Search Polymarket public events. Returns list of {title, slug}."""
    try:
        r = requests.get(PM_SEARCH_URL, params={"q": query}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("events", [])
        if not isinstance(data, list):
            return []
        return [
            {"title": e.get("title", ""), "slug": e.get("slug", "")}
            for e in data
            if isinstance(e, dict) and e.get("title") and e.get("slug")
        ]
    except requests.RequestException as e:
        logger.warning("Polymarket search failed: %s", e)
        return []


@cached(_medium_cache)
def fetch_event_by_slug(slug: str) -> dict | None:
    """Get full event object from Gamma API by slug."""
    if not slug:
        return None
    try:
        r = requests.get(f"{PM_EVENTS_URL}?slug={slug}", timeout=15)
        r.raise_for_status()
        data = r.json()
        if data and len(data) > 0:
            return data[0]
    except requests.RequestException as e:
        logger.warning("fetch_event_by_slug(%s) failed: %s", slug, e)
    return None


@cached(_medium_cache)
def fetch_event_markets(slug: str, is_min_temp: bool = False) -> list[dict]:
    """Return parsed bucket markets [{bucket, name, lower, upper, yes_price, no_price}]."""
    event = fetch_event_by_slug(slug)
    if event is None:
        # fallback: return default buckets
        buckets = TMIN_BUCKETS if is_min_temp else TMAX_BUCKETS
        return [
            _stamped_market(b, b, 0.5, 0.5) for b in buckets
        ]
    return _parse_markets_from_event(event, is_min_temp)


def _stamped_market(bucket: str, name: str, yes_price: float, no_price: float) -> dict:
    """Produce a market dict with bounds derived from the bucket label."""
    lo, hi = _bucket_bounds(bucket)
    return {
        "bucket": bucket,
        "name": name,
        "lower": lo,
        "upper": hi,
        "yes_price": yes_price,
        "no_price": no_price,
    }


def _parse_markets_from_event(event: dict, is_min_temp: bool) -> list[dict]:
    """Inner parser separated so it can be called without caching."""
    markets_out: list[dict] = []
    seen: set[str] = set()
    for m in event.get("markets", []):
        question = m.get("question", "").strip()
        group_item_title = m.get("groupItemTitle", "")
        bucket = _market_question_to_bucket(question, group_item_title, is_min_temp)
        if bucket is None or bucket in seen:
            continue
        seen.add(bucket)
        yes_price, no_price = _parse_outcome_prices(m)
        if yes_price is None:
            yes_price = 0.5
        if no_price is None:
            no_price = 0.5
        markets_out.append(_stamped_market(
            bucket,
            group_item_title or question or bucket,
            yes_price,
            no_price,
        ))

    if not markets_out:
        buckets = TMIN_BUCKETS if is_min_temp else TMAX_BUCKETS
        markets_out = [
            _stamped_market(b, b, 0.5, 0.5) for b in buckets
        ]

    # sort to our canonical order
    bucket_order = TMIN_BUCKETS if is_min_temp else TMAX_BUCKETS
    order_map = {b: i for i, b in enumerate(bucket_order)}
    markets_out.sort(key=lambda x: order_map.get(x["bucket"], 999))
    return markets_out


def bucket_for_temp(temp: float, is_min_temp: bool) -> str:
    """Return bucket label for a given temperature value."""
    if is_min_temp:
        if temp < 12: return "<12"
        if temp < 14: return "12-14"
        if temp < 16: return "14-16"
        if temp < 18: return "16-18"
        if temp < 20: return "18-20"
        if temp < 22: return "20-22"
        if temp < 24: return "22-24"
        if temp < 26: return "24-26"
        if temp < 28: return "26-28"
        return ">=28"
    if temp < 23: return "<23"
    if temp < 24: return "23-24"
    if temp < 25: return "24-25"
    if temp < 26: return "25-26"
    if temp < 27: return "26-27"
    if temp < 28: return "27-28"
    if temp < 29: return "28-29"
    if temp < 30: return "29-30"
    if temp < 31: return "30-31"
    if temp < 32: return "31-32"
    if temp < 33: return "32-33"
    if temp < 34: return "33-34"
    return ">=34"


@cached(_short_cache)
def fetch_today_event(target_date_str: str) -> dict | None:
    """Fetch today's temperature event for the given date (YYYY-MM-DD format)."""
    events = search_events("hong-kong-temperature")
    if not events:
        return None
    for ev in events:
        ev_date = parse_date_from_event(ev["title"], ev["slug"])
        if ev_date and ev_date.strftime("%Y-%m-%d") == target_date_str:
            return {"slug": ev["slug"], "title": ev["title"]}
    return None
