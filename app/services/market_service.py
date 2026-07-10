# app/services/market_service.py
"""Polymarket event/market data fetching.

All functions use cachetools TTLCache for caching.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
import numpy as np
import requests
from cachetools import TTLCache, cached

from ..config import (
    PM_SEARCH_URL,
    PM_EVENTS_URL,
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

    Tmin: 23 or below → (-inf, 23), 24C → (24, 25), …, 33 or higher → (33, inf)
    Tmax: <23 → (-inf, 23), 23-24 → (23, 24), …, >=34 → (34, inf)
    """
    stripped = bucket_label.strip()
    if " or below" in stripped.lower():
        val = float(re.search(r"(\d+)", stripped).group(1))
        return (-np.inf, val)
    if " or higher" in stripped.lower():
        val = float(re.search(r"(\d+)", stripped).group(1))
        return (val, np.inf)
    if stripped.startswith("<"):
        return (-np.inf, float(stripped[1:]))
    if stripped.startswith(">="):
        return (float(stripped[2:]), np.inf)
    if "-" in stripped:
        parts = stripped.split("-")
        return (float(parts[0]), float(parts[1]))
    # Single degree buckets like "24C" or "24°C"
    m = re.search(r"(\d+)", stripped)
    if m:
        v = float(m.group(1))
        return (v, v + 1.0)
    return (-np.inf, np.inf)


def _bucket_sort_key(label: str) -> tuple[float, int]:
    """Sort key for canonical bucket ordering.

    Lower tails (<N / "N or below") sort first, then individual-degree /
    range buckets by lower bound, then upper tails (>=N / "N or higher") last.
    """
    lo, hi = _bucket_bounds(label)
    if label.startswith("<"):
        return (hi, 0)
    if label.startswith(">="):
        return (lo, 2)
    lower_label = label.lower()
    if " or below" in lower_label:
        return (hi, 0)
    if " or higher" in lower_label:
        return (lo, 2)
    if any(c in label for c in ("C", "°")):
        return (lo, 1)
    return (lo, 1)


def _market_question_to_bucket(question: str, group_item_title: str, is_min_temp: bool) -> str | None:
    """Map Polymarket market question text to a bucket label dynamically.

    Exceedance markets (or below / or higher) get their own label,
    e.g. ``>=36``, ``<26``.  Exact-degree TMAX markets are turned into
    a range label (e.g. ``34`` → ``34-35``).
    TMIN markets use single-degree labels (e.g. ``24C``).
    """
    source = group_item_title or question
    m = re.search(r"(\d+)\s*°?C", source, re.IGNORECASE)
    if not m:
        return None

    temp_val = int(m.group(1))
    lower = "or below" in source.lower()
    upper = "or higher" in source.lower()

    if is_min_temp:
        # TMIN: single-degree buckets — 23 or below, 24C, …, 33 or higher
        if lower:
            return f"{temp_val} or below"
        if upper:
            return f"{temp_val} or higher"
        return f"{temp_val}C"

    # TMAX exceedance markets keep their own label
    if lower:
        return f"<{temp_val}"
    if upper:
        return f">={temp_val}"

    # TMAX exact-range: dynamic — "34" → "34-35"
    return f"{temp_val}-{temp_val + 1}"


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
        return []
    return _parse_markets_from_event(event, is_min_temp)


def _stamped_market(
    bucket: str, name: str, yes_price: float, no_price: float,
    **extras,
) -> dict:
    """Produce a market dict with bounds derived from the bucket label.
    
    *extras* (e.g. token_id, conditionId, bestAsk, spread) are merged
    directly into the result.
    """
    lo, hi = _bucket_bounds(bucket)
    return {
        "bucket": bucket,
        "name": name,
        "lower": lo,
        "upper": hi,
        "yes_price": yes_price,
        "no_price": no_price,
        **extras,
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

        # Extract CLOB token IDs and extra Gamma market data
        extras = {}
        clob_ids = m.get("clobTokenIds", [])
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                clob_ids = []
        if clob_ids and len(clob_ids) >= 2:
            extras["token_id"] = clob_ids[0]
            extras["no_token_id"] = clob_ids[1]
        for k in ("conditionId", "bestAsk", "spread", "lastTradePrice",
                  "liquidityClob", "volume24hrClob"):
            v = m.get(k)
            if v is not None:
                extras[k] = v
        if "bestAsk" in extras and "spread" in extras:
            try:
                extras["bestBid"] = max(0.0, float(extras["bestAsk"]) - float(extras["spread"]))
            except (ValueError, TypeError):
                pass

        markets_out.append(_stamped_market(
            bucket,
            group_item_title or question or bucket,
            yes_price,
            no_price,
            **extras,
        ))

    # sort buckets dynamically
    markets_out.sort(key=lambda x: _bucket_sort_key(x["bucket"]))
    return markets_out


def bucket_for_temp(temp: float, is_min_temp: bool) -> str:
    """Return bucket label for a given temperature value (dynamic)."""
    if is_min_temp:
        if temp < 23:
            return "23 or below"
        for t in range(24, 33):
            if t <= temp < t + 1:
                return f"{t}C"
        return "33 or higher"
    # TMAX: integer degree ranges
    t = int(temp)
    if t < 23:
        return f"<{t + 1}" if t + 1 < 23 else "<23"
    if t < 36:
        return f"{t}-{t + 1}"
    return f">={t}"


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
