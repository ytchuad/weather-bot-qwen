"""Read-only minute projection joining weather, market and model history.

The projection has two deliberately separate weather views:

* raw actual-temperature chart points are emitted only at their
  ``observation_timestamp``; and
* replay/lineage fields select the latest weather version available at the
  row's timestamp (``first_seen_timestamp <= row timestamp``).

Keeping those views separate prevents a delayed HKO observation from being
relabeled as a later collector/model minute, while retaining point-in-time
availability for replay.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from .schema import _parse_datetime, jsonable

HKT = ZoneInfo("Asia/Hong_Kong")
UTC = timezone.utc
ALLOWED_CLOCK_SKEW = timedelta(minutes=5)


def _time(value: Any) -> datetime | None:
    # Layer A and legacy dashboard timestamps use HKT wall-clock semantics
    # when an old value has no explicit offset. New records are written with
    # an offset, so this branch is only for compatibility.
    return _parse_datetime(value, naive_timezone=HKT)


def _minute(value: Any) -> datetime | None:
    parsed = _time(value)
    return parsed.astimezone(HKT).replace(second=0, microsecond=0) if parsed else None


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _weather_observation_time(snapshot: Mapping[str, Any]) -> datetime | None:
    return _time(snapshot.get("observation_timestamp") or snapshot.get("snapshot_timestamp"))


def _weather_first_seen_time(snapshot: Mapping[str, Any]) -> datetime | None:
    # Older Layer A records predate the explicit availability field. Their
    # capture time is the safest retained approximation; only if it is absent
    # do we fall back to the observation timestamp for legacy readability.
    return _time(
        snapshot.get("first_seen_timestamp")
        or snapshot.get("capture_timestamp")
        or snapshot.get("observation_timestamp")
        or snapshot.get("snapshot_timestamp")
    )


def _capture_time(snapshot: Mapping[str, Any]) -> datetime | None:
    return _time(snapshot.get("capture_timestamp"))


def _age_seconds(source_timestamp: Any, decision_timestamp: datetime) -> float | None:
    source = _time(source_timestamp)
    if source is None:
        return None
    age = (_as_utc(decision_timestamp) - source).total_seconds()
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


def _weather_is_fallback(snapshot: Mapping[str, Any]) -> bool:
    statuses = snapshot.get("observation_status")
    temp = statuses.get("temperature_current") if isinstance(statuses, Mapping) else None
    return bool(isinstance(temp, Mapping) and temp.get("is_fallback"))


def _weather_is_future_corrupt(snapshot: Mapping[str, Any]) -> bool:
    observation = _weather_observation_time(snapshot)
    capture = _capture_time(snapshot)
    return bool(observation and capture and observation > capture + ALLOWED_CLOCK_SKEW)


def _weather_identity(snapshot: Mapping[str, Any]) -> tuple[str, str, str, str]:
    observation = _weather_observation_time(snapshot)
    source_status = snapshot.get("source_status")
    source_status = source_status if isinstance(source_status, Mapping) else {}
    return (
        str(snapshot.get("weather_observation_id") or (observation.isoformat() if observation else "")),
        str(snapshot.get("location") or ""),
        str(source_status.get("observation_source") or snapshot.get("observation_source") or ""),
        str(source_status.get("station") or snapshot.get("station") or ""),
    )


def _weather_preference(snapshot: Mapping[str, Any]) -> tuple[int, datetime, datetime, str]:
    return (
        0 if _weather_is_fallback(snapshot) else 1,
        _weather_first_seen_time(snapshot) or datetime.min.replace(tzinfo=UTC),
        _capture_time(snapshot) or datetime.min.replace(tzinfo=UTC),
        str(snapshot.get("weather_snapshot_id") or ""),
    )


def select_weather_as_of(snapshots: Sequence[Mapping[str, Any]], at: datetime) -> Mapping[str, Any] | None:
    """Return the weather version that was usable at one decision time.

    Both the observation itself and the version's availability must predate
    ``at``.  The latter is what prevents a later HKO correction from being
    applied to an earlier model cycle during replay.
    """
    at_utc = _as_utc(at)
    candidates = [
        snapshot
        for snapshot in snapshots
        if (observation := _weather_observation_time(snapshot)) is not None
        and observation <= at_utc
        and (first_seen := _weather_first_seen_time(snapshot)) is not None
        and first_seen <= at_utc
        and _weather_valid(snapshot)
    ]
    # Observation time is the primary as-of selector. A correction is only
    # preferred within the same observation time after it becomes available.
    return max(
        candidates,
        key=lambda item: (
            _weather_observation_time(item) or datetime.min.replace(tzinfo=UTC),
            _weather_preference(item),
        ),
        default=None,
    )


def _select_weather(snapshots: Sequence[Mapping[str, Any]], at: datetime) -> Mapping[str, Any] | None:
    """Private compatibility alias for the projection implementation."""
    return select_weather_as_of(snapshots, at)


def _temperature_status(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    statuses = snapshot.get("observation_status")
    status = statuses.get("temperature_current") if isinstance(statuses, Mapping) else None
    if isinstance(status, Mapping):
        return status
    value = snapshot.get("temperature_current")
    return {
        "value": value,
        "source_timestamp": None,
        "is_missing": value is None,
        "is_stale": False,
        "is_fallback": False,
        "raw_status": "missing" if value is None else "observed_missing_timestamp",
    }


def _weather_projection(
    snapshot: Mapping[str, Any] | None,
    at: datetime,
    *,
    actual_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        values = {
            "max_so_far": None,
            "min_so_far": None,
            "relative_humidity": None,
            "pressure": None,
            "dew_point": None,
            "rain_current": None,
        }
        projected_status: dict[str, Any] = {}
        temperature_status: Mapping[str, Any] = {}
    else:
        statuses = snapshot.get("observation_status") or {}
        values = {}
        projected_status = {}
        for field, output in (
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
        temperature_status = _temperature_status(snapshot)

    # ``actual_snapshot`` is intentionally separate from the as-of weather
    # snapshot. Passing it only when the row timestamp equals the observation
    # timestamp is what prevents raw actual values from being forward-filled.
    actual_value = None
    actual_status: dict[str, Any] | None = None
    if actual_snapshot is not None:
        actual_value, actual_status = _status_value(_temperature_status(actual_snapshot), at=at)
        projected_status["temperature_current"] = actual_status

    weather_observation = _weather_observation_time(snapshot) if snapshot is not None else None
    weather_first_seen = _weather_first_seen_time(snapshot) if snapshot is not None else None
    actual_observation = _weather_observation_time(actual_snapshot) if actual_snapshot is not None else None
    actual_first_seen = _weather_first_seen_time(actual_snapshot) if actual_snapshot is not None else None
    actual_source_release = (
        _time(actual_snapshot.get("source_release_timestamp")) if actual_snapshot is not None else None
    )
    return {
        "actual_temperature": actual_value,
        **values,
        "weather_source_timestamp": temperature_status.get("source_timestamp"),
        "weather_age_seconds": _age_seconds(weather_observation, at),
        "weather_quality_status": (
            _status_value(temperature_status, at=at)[1].get("quality_status") if temperature_status else "missing"
        ),
        "weather_snapshot_id": snapshot.get("weather_snapshot_id") if snapshot is not None else None,
        "weather_observation_timestamp": weather_observation.isoformat() if weather_observation else None,
        "weather_first_seen_timestamp": weather_first_seen.isoformat() if weather_first_seen else None,
        "weather_capture_timestamp": _capture_time(snapshot).isoformat() if snapshot is not None and _capture_time(snapshot) else None,
        "weather_source_release_timestamp": snapshot.get("source_release_timestamp") if snapshot is not None else None,
        "weather_data_through": weather_observation.isoformat() if weather_observation else None,
        "actual_weather_snapshot_id": actual_snapshot.get("weather_snapshot_id") if actual_snapshot is not None else None,
        "actual_observation_timestamp": actual_observation.isoformat() if actual_observation else None,
        "actual_first_seen_timestamp": actual_first_seen.isoformat() if actual_first_seen else None,
        "actual_source_release_timestamp": actual_source_release.isoformat() if actual_source_release else None,
        "actual_release_lag_seconds": (
            _age_seconds(actual_source_release, actual_first_seen) if actual_source_release and actual_first_seen else None
        ),
        "actual_temperature_quality_status": actual_status.get("quality_status") if actual_status else "missing",
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
    exact_cycle_only: bool,
) -> dict[str, Any]:
    at_minute = at.astimezone(HKT).replace(second=0, microsecond=0)
    eligible = [
        cycle
        for cycle in cycles
        if (timestamp := _time(cycle.get("decision_timestamp"))) is not None
        and (
            _minute(timestamp) == at_minute
            if exact_cycle_only
            else timestamp <= _as_utc(at)
        )
    ]
    latest = max(
        eligible,
        key=lambda item: _time(item.get("decision_timestamp")) or datetime.min.replace(tzinfo=UTC),
        default=None,
    )
    if latest is None:
        return {
            "latest_model_cycle_timestamp": None,
            "model_cycle_timestamp": None,
            "model_cycle_id": None,
            "model_age_seconds": None,
            "model_cycle_is_real": False,
            "model_predictions": {},
            "model_probabilities": {},
            "models": {},
        }
    cycle_time = _time(latest.get("decision_timestamp"))
    age = (_as_utc(at) - cycle_time).total_seconds() if cycle_time else None
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
        "model_cycle_is_real": bool(cycle_time and _minute(cycle_time) == at_minute),
        "model_predictions": predictions,
        "model_probabilities": probabilities,
        "models": predictions,
    }


def _selected_hkt_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _in_hkt_date(timestamp: Any, selected_date: date | None) -> bool:
    if selected_date is None:
        return True
    parsed = _time(timestamp)
    return bool(parsed and parsed.astimezone(HKT).date() == selected_date)


def _prepare_weather(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    selected_date: date | None,
    as_of: datetime,
) -> tuple[list[Mapping[str, Any]], dict[datetime, Mapping[str, Any]], dict[str, Any]]:
    scoped = [
        snapshot
        for snapshot in snapshots
        if _in_hkt_date(snapshot.get("observation_timestamp") or snapshot.get("snapshot_timestamp"), selected_date)
    ]
    excluded_future = sum(1 for snapshot in scoped if _weather_is_future_corrupt(snapshot))
    valid = [
        snapshot
        for snapshot in scoped
        if _weather_observation_time(snapshot) is not None
        and _weather_first_seen_time(snapshot) is not None
        and not _weather_is_future_corrupt(snapshot)
        and _weather_valid(snapshot)
    ]

    version_ids = [str(snapshot.get("weather_snapshot_id") or "") for snapshot in valid]
    duplicate_versions = sum(count - 1 for key, count in Counter(version_ids).items() if key and count > 1)
    unique_versions: dict[str, Mapping[str, Any]] = {}
    anonymous: list[Mapping[str, Any]] = []
    for snapshot in valid:
        version_id = str(snapshot.get("weather_snapshot_id") or "")
        if not version_id:
            anonymous.append(snapshot)
            continue
        previous = unique_versions.get(version_id)
        if previous is None or _weather_preference(snapshot) > _weather_preference(previous):
            unique_versions[version_id] = snapshot
    valid = [*unique_versions.values(), *anonymous]

    chart_versions = [snapshot for snapshot in valid if _weather_first_seen_time(snapshot) <= _as_utc(as_of)]
    by_observation: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for snapshot in chart_versions:
        identity = _weather_identity(snapshot)
        previous = by_observation.get(identity)
        if previous is None or _weather_preference(snapshot) > _weather_preference(previous):
            by_observation[identity] = snapshot
    actual_by_minute: dict[datetime, Mapping[str, Any]] = {}
    for snapshot in by_observation.values():
        minute = _minute(_weather_observation_time(snapshot))
        if minute is None:
            continue
        previous = actual_by_minute.get(minute)
        if previous is None or _weather_preference(snapshot) > _weather_preference(previous):
            actual_by_minute[minute] = snapshot

    latest = max(
        chart_versions,
        key=lambda item: _weather_observation_time(item) or datetime.min.replace(tzinfo=UTC),
        default=None,
    )
    diagnostics = {
        "excluded_future_weather_records": excluded_future,
        "duplicate_observation_versions": duplicate_versions,
        "latest_weather_observation_timestamp": (
            _weather_observation_time(latest).isoformat() if latest and _weather_observation_time(latest) else None
        ),
        "latest_weather_first_seen_timestamp": (
            _weather_first_seen_time(latest).isoformat() if latest and _weather_first_seen_time(latest) else None
        ),
    }
    return valid, actual_by_minute, diagnostics


def build_minute_projection(
    model_cycles: Iterable[Mapping[str, Any]],
    market_snapshots: Iterable[Mapping[str, Any]],
    weather_snapshots: Iterable[Mapping[str, Any]],
    *,
    start: Any = None,
    end: Any = None,
    date_value: str | date | None = None,
    as_of: Any = None,
    bucket_filters: Sequence[str] | None = None,
    model_filters: Sequence[str] | None = None,
    exact_model_cycles: bool = False,
    limit: int = 1000,
) -> dict[str, Any]:
    """Build a chart/replay projection without inventing observations.

    ``exact_model_cycles`` is used by the trajectory API so a prediction is
    present only at a real decision cycle. Replay keeps its existing as-of
    model join by leaving it false.
    """
    as_of_dt = _time(as_of) if as_of is not None else datetime.now(UTC)
    as_of_dt = as_of_dt or datetime.now(UTC)
    selected_date = _selected_hkt_date(date_value)
    empty_diagnostics = {
        "excluded_future_weather_records": 0,
        "duplicate_observation_versions": 0,
        "latest_weather_observation_timestamp": None,
        "latest_weather_first_seen_timestamp": None,
    }
    if date_value is not None and selected_date is None:
        return {"rows": [], "diagnostics": empty_diagnostics}
    if selected_date is not None and selected_date > as_of_dt.astimezone(HKT).date():
        return {"rows": [], "diagnostics": empty_diagnostics}

    model_list = [
        dict(item)
        for item in model_cycles
        if isinstance(item, Mapping)
        and _in_hkt_date(item.get("decision_timestamp"), selected_date)
        and (timestamp := _time(item.get("decision_timestamp"))) is not None
        and timestamp <= _as_utc(as_of_dt)
    ]
    market_list = [
        dict(item)
        for item in market_snapshots
        if isinstance(item, Mapping)
        and _in_hkt_date(item.get("decision_timestamp"), selected_date)
        and (timestamp := _time(item.get("decision_timestamp"))) is not None
        and timestamp <= _as_utc(as_of_dt)
    ]
    weather_list = [dict(item) for item in weather_snapshots if isinstance(item, Mapping)]
    weather_list, actual_by_minute, diagnostics = _prepare_weather(
        weather_list,
        selected_date=selected_date,
        as_of=as_of_dt,
    )
    weather_list = [
        snapshot
        for snapshot in weather_list
        if (observation := _weather_observation_time(snapshot)) is not None and observation <= _as_utc(as_of_dt)
    ]
    actual_by_minute = {
        minute: snapshot
        for minute, snapshot in actual_by_minute.items()
        if minute <= _minute(as_of_dt)
    }

    keys: set[datetime] = set(actual_by_minute)
    for records, field in ((model_list, "decision_timestamp"), (market_list, "decision_timestamp")):
        keys.update(_minute(item.get(field)) for item in records if _minute(item.get(field)) is not None)
    start_dt = _minute(start) if start else None
    end_dt = _minute(end) if end else None
    max_dt = _minute(as_of_dt)
    if start_dt:
        keys = {key for key in keys if key >= start_dt}
    if end_dt:
        keys = {key for key in keys if key <= end_dt}
    keys = {key for key in keys if key <= max_dt and _in_hkt_date(key, selected_date)}
    bucket_set = set(str(item) for item in bucket_filters) if bucket_filters else None
    model_set = set(str(item) for item in model_filters) if model_filters else None
    rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        weather = _select_weather(weather_list, key)
        row = {
            "timestamp": key.isoformat(),
            **_weather_projection(weather, key, actual_snapshot=actual_by_minute.get(key)),
            **_market_projection(market_list, at=key, bucket_filters=bucket_set),
            **_model_projection(
                model_list,
                at=key,
                model_filters=model_set,
                exact_cycle_only=exact_model_cycles,
            ),
        }
        rows.append(jsonable(row))
    return {"rows": rows[: max(0, int(limit))], "diagnostics": diagnostics}


def build_minute_view(
    model_cycles: Iterable[Mapping[str, Any]],
    market_snapshots: Iterable[Mapping[str, Any]],
    weather_snapshots: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible rows-only façade for minute projection callers."""
    return build_minute_projection(model_cycles, market_snapshots, weather_snapshots, **kwargs)["rows"]


__all__ = ["ALLOWED_CLOCK_SKEW", "build_minute_projection", "build_minute_view", "select_weather_as_of"]
