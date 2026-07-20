"""Read-only minute projection joining weather, market and model history."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from .schema import _parse_datetime, jsonable

HKT = timezone(timedelta(hours=8))


def _time(value: Any) -> datetime | None:
    return _parse_datetime(value, naive_timezone=timezone.utc)


def _minute(value: Any) -> datetime | None:
    parsed = _time(value)
    return parsed.astimezone(HKT).replace(second=0, microsecond=0) if parsed else None


def _age_seconds(source_timestamp: Any, decision_timestamp: datetime) -> float | None:
    source = _time(source_timestamp)
    if source is None:
        return None
    age = (decision_timestamp.astimezone(timezone.utc) - source).total_seconds()
    return age if age >= 0 else None


def _status_value(status: Mapping[str, Any] | None, *, at: datetime) -> tuple[Any, dict[str, Any]]:
    status = dict(status or {})
    value = status.get("value")
    source_timestamp = status.get("source_timestamp")
    age = _age_seconds(source_timestamp, at)
    status["age_seconds"] = age
    status["age_minutes"] = age / 60.0 if age is not None else None
    if status.get("is_missing") or value is None:
        quality = "missing"
    elif status.get("is_fallback"):
        quality = "fallback"
    elif status.get("is_stale"):
        quality = "stale"
    else:
        quality = status.get("raw_status") or "observed"
    status["quality_status"] = quality
    return value, status


def _weather_valid(snapshot: Mapping[str, Any]) -> bool:
    statuses = snapshot.get("observation_status")
    if not isinstance(statuses, Mapping):
        return snapshot.get("temperature_current") is not None
    temp = statuses.get("temperature_current")
    return isinstance(temp, Mapping) and not temp.get("is_missing", False) and temp.get("value") is not None


def _select_weather(snapshots: Sequence[Mapping[str, Any]], at: datetime) -> Mapping[str, Any] | None:
    candidates = [
        snapshot
        for snapshot in snapshots
        if (_time(snapshot.get("snapshot_timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
        <= at.astimezone(timezone.utc)
        and _weather_valid(snapshot)
    ]
    return max(candidates, key=lambda item: _time(item.get("snapshot_timestamp")) or datetime.min.replace(tzinfo=timezone.utc), default=None)


def _weather_projection(snapshot: Mapping[str, Any] | None, at: datetime) -> dict[str, Any]:
    if snapshot is None:
        return {
            "actual_temperature": None,
            "max_so_far": None,
            "min_so_far": None,
            "relative_humidity": None,
            "pressure": None,
            "dew_point": None,
            "rain_current": None,
            "weather_source_timestamp": None,
            "weather_age_seconds": None,
            "weather_quality_status": "missing",
            "weather_snapshot_id": None,
            "weather_observations": {},
        }
    statuses = snapshot.get("observation_status") or {}
    values: dict[str, Any] = {}
    projected_status: dict[str, Any] = {}
    for field, output in (
        ("temperature_current", "actual_temperature"),
        ("max_so_far", "max_so_far"),
        ("min_so_far", "min_so_far"),
        ("relative_humidity", "relative_humidity"),
        ("pressure", "pressure"),
        ("dew_point", "dew_point"),
        ("rain_current", "rain_current"),
    ):
        status = statuses.get(field) if isinstance(statuses, Mapping) else None
        if not isinstance(status, Mapping):
            value = snapshot.get(field)
            status = {
                "value": value,
                "source_timestamp": None,
                "is_missing": value is None,
                "is_stale": False,
                "is_fallback": False,
                "raw_status": "missing" if value is None else "observed_missing_timestamp",
            }
        value, clean_status = _status_value(status, at=at)
        values[output] = value
        projected_status[field] = clean_status
    temperature_status = projected_status["temperature_current"]
    return {
        **values,
        "weather_source_timestamp": temperature_status.get("source_timestamp"),
        "weather_age_seconds": temperature_status.get("age_seconds"),
        "weather_quality_status": temperature_status.get("quality_status"),
        "weather_snapshot_id": snapshot.get("weather_snapshot_id"),
        "weather_observations": projected_status,
    }


def _best_prices(snapshot: Mapping[str, Any], at: datetime) -> dict[str, Any]:
    identity = snapshot.get("market_identity") or []
    books = snapshot.get("clob_books") or []
    books_by_key = {
        (str(book.get("bucket")), str(book.get("token_side", "")).upper()): book
        for book in books
        if isinstance(book, Mapping)
    }
    gamma = snapshot.get("gamma_reference_prices") or {}
    result: dict[str, Any] = {}
    buckets = {str(item.get("bucket")) for item in identity if item.get("bucket")}
    buckets.update(str(key) for key in gamma.keys())
    buckets.update(str(book.get("bucket")) for book in books if book.get("bucket"))
    for bucket in sorted(buckets):
        reference = gamma.get(bucket)
        if isinstance(reference, Mapping):
            reference = reference.get("yes", reference.get("YES"))
        yes = books_by_key.get((bucket, "YES"), {})
        bids = yes.get("bids") or []
        asks = yes.get("asks") or []
        bid_values = [float(item.get("price")) for item in bids if item.get("price") is not None]
        ask_values = [float(item.get("price")) for item in asks if item.get("price") is not None]
        best_bid = max(bid_values) if bid_values else None
        best_ask = min(ask_values) if ask_values else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        book_timestamp = yes.get("book_timestamp") or yes.get("timestamp")
        book_age = _age_seconds(book_timestamp, at)
        result[bucket] = {
            "gamma_reference_price": reference,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "book_timestamp": book_timestamp,
            "book_age_seconds": book_age,
            "validation_status": yes.get("validation_status") or "missing",
        }
    return result


def _market_projection(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    at: datetime,
    bucket_filters: set[str] | None,
) -> dict[str, Any]:
    same_minute = [
        snapshot
        for snapshot in snapshots
        if _minute(snapshot.get("decision_timestamp")) == at.astimezone(HKT).replace(second=0, microsecond=0)
    ]
    by_kind: dict[str, Mapping[str, Any]] = {}
    for snapshot in same_minute:
        kind = str(snapshot.get("market_kind") or "unknown")
        previous = by_kind.get(kind)
        if previous is None or str(snapshot.get("decision_timestamp")) > str(previous.get("decision_timestamp")):
            by_kind[kind] = snapshot
    markets_by_kind: dict[str, Any] = {}
    for kind, snapshot in by_kind.items():
        prices = _best_prices(snapshot, at)
        if bucket_filters is not None:
            prices = {key: value for key, value in prices.items() if key in bucket_filters}
        markets_by_kind[kind] = {
            "market_snapshot_id": snapshot.get("market_snapshot_id"),
            "market_book_timestamp": max(
                (value.get("book_timestamp") for value in prices.values() if value.get("book_timestamp")),
                default=None,
            ),
            "market": prices,
        }
    first = next(iter(markets_by_kind.values()), None)
    first_market = (first or {}).get("market", {})
    return {
        "market_prices": {key: value.get("gamma_reference_price") for key, value in first_market.items()},
        "best_bid": {key: value.get("best_bid") for key, value in first_market.items()},
        "best_ask": {key: value.get("best_ask") for key, value in first_market.items()},
        "spread": {key: value.get("spread") for key, value in first_market.items()},
        "market_book_timestamp": {key: value.get("book_timestamp") for key, value in first_market.items()},
        "market_book_age_seconds": {key: value.get("book_age_seconds") for key, value in first_market.items()},
        "market": first_market,
        "markets_by_kind": markets_by_kind,
        "market_snapshot_id": (first or {}).get("market_snapshot_id"),
        "market_validation_status": "valid" if same_minute else "missing",
    }


def _model_projection(
    cycles: Sequence[Mapping[str, Any]],
    *,
    at: datetime,
    model_filters: set[str] | None,
) -> dict[str, Any]:
    eligible = [
        cycle
        for cycle in cycles
        if (_time(cycle.get("decision_timestamp")) or datetime.max.replace(tzinfo=timezone.utc))
        <= at.astimezone(timezone.utc)
    ]
    latest = max(
        eligible,
        key=lambda item: _time(item.get("decision_timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        default=None,
    )
    if latest is None:
        return {
            "latest_model_cycle_timestamp": None,
            "model_cycle_timestamp": None,
            "model_cycle_id": None,
            "model_age_seconds": None,
            "model_predictions": {},
            "model_probabilities": {},
            "models": {},
        }
    cycle_time = _time(latest.get("decision_timestamp"))
    age = (at.astimezone(timezone.utc) - cycle_time).total_seconds() if cycle_time else None
    predictions: dict[str, Any] = {}
    probabilities: dict[str, Any] = {}
    for model in latest.get("models", []):
        if not isinstance(model, Mapping):
            continue
        name = str(model.get("model_name") or model.get("model_key") or "unknown")
        if model_filters is not None and name not in model_filters:
            continue
        predictions[name] = {
            "point_prediction": model.get("point_prediction"),
            "q10": model.get("q10"),
            "q25": model.get("q25"),
            "q50": model.get("q50"),
            "q75": model.get("q75"),
            "q90": model.get("q90"),
        }
        probabilities[name] = model.get("full_bucket_probabilities") or {}
    timestamp = latest.get("decision_timestamp")
    return {
        "latest_model_cycle_timestamp": timestamp,
        "model_cycle_timestamp": timestamp,
        "model_cycle_id": latest.get("decision_cycle_id"),
        "model_age_seconds": age if age is None or age >= 0 else None,
        "model_predictions": predictions,
        "model_probabilities": probabilities,
        "models": predictions,
    }


def build_minute_view(
    model_cycles: Iterable[Mapping[str, Any]],
    market_snapshots: Iterable[Mapping[str, Any]],
    weather_snapshots: Iterable[Mapping[str, Any]],
    *,
    start: Any = None,
    end: Any = None,
    bucket_filters: Sequence[str] | None = None,
    model_filters: Sequence[str] | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Join only same-minute/prior weather and same-minute market data.

    Model cycles are always joined backward as-of.  The eligible cycle set is
    filtered before selecting ``max`` so a later cycle can never leak into an
    earlier minute row.
    """
    model_list = [dict(item) for item in model_cycles if isinstance(item, Mapping)]
    market_list = [dict(item) for item in market_snapshots if isinstance(item, Mapping)]
    weather_list = [dict(item) for item in weather_snapshots if isinstance(item, Mapping)]
    keys: set[datetime] = set()
    for record, field in (
        (model_list, "decision_timestamp"),
        (market_list, "decision_timestamp"),
        (weather_list, "snapshot_timestamp"),
    ):
        keys.update(
            _minute(item.get(field))
            for item in record
            if _minute(item.get(field)) is not None
        )
    start_dt = _minute(start) if start else None
    end_dt = _minute(end) if end else None
    if start_dt:
        keys = {key for key in keys if key >= start_dt}
    if end_dt:
        keys = {key for key in keys if key <= end_dt}
    bucket_set = set(str(item) for item in bucket_filters) if bucket_filters else None
    model_set = set(str(item) for item in model_filters) if model_filters else None
    rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        weather = _select_weather(weather_list, key)
        row = {
            "timestamp": key.isoformat(),
            **_weather_projection(weather, key),
            **_market_projection(market_list, at=key, bucket_filters=bucket_set),
            **_model_projection(model_list, at=key, model_filters=model_set),
        }
        rows.append(jsonable(row))
    return rows[: max(0, int(limit))]


__all__ = ["build_minute_view"]
