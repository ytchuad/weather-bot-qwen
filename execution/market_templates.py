# execution/market_templates.py
"""Market template resolver — turn a named template into a Polymarket event slug.

Each strategy declares a ``market_template`` (e.g. ``"hk-tmax"``).  On each
execution cycle the system resolves that template into a concrete event slug
like ``highest-temperature-in-hong-kong-on-june-15-2026`` by injecting today's
(or the strategy's target) date — no manual "event discovery" step needed.
"""

from __future__ import annotations

import re
import logging
from datetime import date, datetime, timezone, timedelta
from typing import Callable

logger = logging.getLogger(__name__)

# ── Template → slug builders ──────────────────────────────────────────


def _hk_tmax_slug(dt: date) -> str:
    month = dt.strftime("%B").lower()
    return f"highest-temperature-in-hong-kong-on-{month}-{dt.day}-{dt.year}"


def _hk_tmin_slug(dt: date) -> str:
    month = dt.strftime("%B").lower()
    return f"lowest-temperature-in-hong-kong-on-{month}-{dt.day}-{dt.year}"


# ── Registry ──────────────────────────────────────────────────────────

BUILTIN_TEMPLATES: dict[str, Callable[[date], str]] = {
    "hk-tmax": _hk_tmax_slug,
    "hk-tmin": _hk_tmin_slug,
}

# Pattern used to extract date parts from a generated slug so we can
# verify a slug matches today's date.
_SLUG_DATE_PATTERN = re.compile(
    r"(?:highest|lowest)-temperature-in-hong-kong-on-"
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)-(\d{1,2})-(\d{4})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ── Public API ────────────────────────────────────────────────────────


def resolve_slug(template: str, target_date: date | None = None) -> str:
    """Resolve a *template* to a Polymarket event slug for *target_date*.

    Parameters
    ----------
    template : str
        One of ``"hk-tmax"``, ``"hk-tmin"``, or a custom slug regex pattern.
        Custom patterns may include ``{month}``, ``{day}``, ``{year}``
        placeholders.
    target_date : date | None
        Date to use in the slug.  Defaults to *today in HKT* (UTC+8).

    Returns
    -------
    str
        The resolved event slug.

    Raises
    ------
    ValueError
        If *template* is unknown.
    """
    if target_date is None:
        target_date = _hkt_today()

    # 1. Built-in template
    builder = BUILTIN_TEMPLATES.get(template)
    if builder is not None:
        return builder(target_date)

    # 2. Custom pattern with placeholders
    if "{month}" in template or "{day}" in template or "{year}" in template:
        return (
            template
            .replace("{month}", target_date.strftime("%B").lower())
            .replace("{day}", str(target_date.day))
            .replace("{year}", str(target_date.year))
        )

    # 3. Unknown — assume it's already a literal slug
    logger.warning("Unknown market template '%s' — treating as literal slug", template)
    return template


def parse_slug_date(slug: str) -> date | None:
    """Extract the event date from an HK temperature slug.

    Returns ``None`` if the slug doesn't match the expected pattern.
    """
    m = _SLUG_DATE_PATTERN.match(slug)
    if not m:
        return None
    month_name = m.group(1).lower()
    day = int(m.group(2))
    year = int(m.group(3))
    month = _MONTH_MAP.get(month_name)
    if month is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def is_today_event(slug: str) -> bool:
    """Check whether *slug* refers to today's event (HKT)."""
    slug_date = parse_slug_date(slug)
    if slug_date is None:
        return False
    today = _hkt_today()
    return slug_date == today


def _hkt_today() -> date:
    """Return today's date in HKT (UTC+8)."""
    return (datetime.now(timezone.utc) + timedelta(hours=8)).date()


def register_template(name: str, builder: Callable[[date], str]) -> None:
    """Register a custom template at runtime."""
    BUILTIN_TEMPLATES[name] = builder