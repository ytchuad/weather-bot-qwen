"""Shared timing helpers for independent one-minute Layer A stores."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

HKT = timezone(timedelta(hours=8))


def get_minute_partition_minutes(value: int | float | str | None = None) -> int:
    """Return the configured close interval, constrained to the contract range."""
    raw = value
    if raw is None:
        raw = os.getenv("LAYER_A_MINUTE_PARTITION_MINUTES", "10")
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        parsed = 10
    return max(5, min(15, parsed))


def minute_partition_start(value: datetime, minutes: int | float | str | None = None) -> datetime:
    """Floor an instant to a local Hong Kong minute partition boundary."""
    interval = get_minute_partition_minutes(minutes)
    local = value if value.tzinfo is not None else value.replace(tzinfo=HKT)
    local = local.astimezone(HKT)
    floored = (local.minute // interval) * interval
    return local.replace(minute=floored, second=0, microsecond=0)


def partition_is_due(
    start: datetime,
    *,
    now: datetime,
    minutes: int | float | str | None = None,
) -> bool:
    interval = get_minute_partition_minutes(minutes)
    start_local = start if start.tzinfo is not None else start.replace(tzinfo=HKT)
    now_local = now if now.tzinfo is not None else now.replace(tzinfo=HKT)
    return start_local.astimezone(HKT) + timedelta(minutes=interval) <= now_local.astimezone(HKT)


def partition_start_from_directory(directory: Path) -> datetime | None:
    """Read a date/hour/minute partition path without reading its payload."""
    text = str(directory).replace("\\", "/")
    date_match = re.search(r"date=(\d{4}-\d{2}-\d{2})", text)
    hour_match = re.search(r"hour=(\d{2})", text)
    minute_match = re.search(r"minute=(\d{2})", text)
    if not date_match or not hour_match:
        return None
    minute = int(minute_match.group(1)) if minute_match else 0
    try:
        return datetime.strptime(
            f"{date_match.group(1)} {hour_match.group(1)}:{minute:02d}",
            "%Y-%m-%d %H:%M",
        ).replace(tzinfo=HKT)
    except ValueError:
        return None


__all__ = [
    "get_minute_partition_minutes",
    "minute_partition_start",
    "partition_is_due",
    "partition_start_from_directory",
]
