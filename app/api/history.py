"""Rebuild-safe Layer A historical read APIs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from layer_a.historical_store import get_default_historical_store

router = APIRouter(prefix="/api/history", tags=["Layer A History"])


def _split_filters(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _envelope(store: Any, records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "remote_history": store.health_summary(),
        "records": records,
        key: records,
        "count": len(records),
    }


@router.get("/minute")
def minute_history(
    date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    bucket: list[str] | None = Query(default=None),
    model: list[str] | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    store = get_default_historical_store()
    rows = store.minute_history(
        date_value=date,
        start=start,
        end=end,
        bucket_filters=_split_filters(bucket),
        model_filters=_split_filters(model),
        limit=limit,
    )
    result = _envelope(store, rows, "minutes")
    result["join_contract"] = {
        "weather": "same-minute or latest prior valid snapshot",
        "market": "same-minute snapshot",
        "model": "latest cycle timestamp <= minute timestamp",
    }
    return result


@router.get("/model-cycles")
def model_cycles(
    date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    model: list[str] | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    store = get_default_historical_store()
    records = store.records("model", date_value=date, start=start, end=end)
    if model:
        wanted = set(_split_filters(model))
        records = [
            record
            for record in records
            if any(str(item.get("model_name")) in wanted for item in record.get("models", []) if isinstance(item, dict))
        ]
    records = records[:limit]
    return _envelope(store, records, "model_cycles")


@router.get("/market-snapshots")
def market_snapshots(
    date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    bucket: list[str] | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    store = get_default_historical_store()
    records = store.records("market", date_value=date, start=start, end=end)
    wanted = set(_split_filters(bucket))
    if wanted:
        records = [
            record
            for record in records
            if any(str(item.get("bucket")) in wanted for item in record.get("market_identity", []) if isinstance(item, dict))
        ]
    records = records[:limit]
    return _envelope(store, records, "market_snapshots")


@router.get("/weather-snapshots")
def weather_snapshots(
    date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    store = get_default_historical_store()
    records = store.records("weather", date_value=date, start=start, end=end, limit=limit)
    return _envelope(store, records, "weather_snapshots")


__all__ = ["router"]
