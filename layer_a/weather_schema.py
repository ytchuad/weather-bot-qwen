"""Strategy-independent one-minute weather observation contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any

from features.input_status import DEFAULT_STALE_AFTER_MINUTES, InputStatus

from .schema import _iso_datetime, _parse_datetime, jsonable

WEATHER_SCHEMA_VERSION = "layer_a.weather.v1"
HKT = timezone(timedelta(hours=8))

WEATHER_OBSERVATION_FIELDS = (
    "temperature_current",
    "max_so_far",
    "min_so_far",
    "relative_humidity",
    "pressure",
    "dew_point",
    "rain_current",
)

_PROHIBITED_FIELDS = {
    "account",
    "account_id",
    "cash",
    "cash_balance",
    "capital",
    "current_paper_positions",
    "fill",
    "fills",
    "paper_position",
    "paper_positions",
    "pnl",
    "realized_pnl",
    "simulated_fills",
    "strategy",
    "strategy_id",
    "strategy_key",
    "target_order",
    "target_orders",
    "unrealized_pnl",
}


class WeatherSnapshotSchemaError(ValueError):
    """Raised when a weather snapshot violates the immutable contract."""


_MISSING = object()


def make_weather_snapshot_id(
    snapshot_timestamp: Any,
    *,
    event_date: str | date,
    location: str,
    cadence_minutes: int = 1,
) -> str:
    """Create a deterministic minute identity independent of model/account state."""
    parsed = _parse_datetime(snapshot_timestamp, naive_timezone=timezone.utc)
    if parsed is None:
        raise WeatherSnapshotSchemaError("snapshot_timestamp is required for weather identity")
    cadence = max(1, int(cadence_minutes))
    local = parsed.astimezone(HKT)
    slot_minute = (local.minute // cadence) * cadence
    slot = local.replace(minute=slot_minute, second=0, microsecond=0)
    date_value = event_date.isoformat() if isinstance(event_date, date) else str(event_date)
    material = "|".join((date_value, str(location), slot.isoformat()))
    return f"ws-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return _MISSING


def _status_candidates(weather: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    aliases = {
        "temperature_current": ("temperature_current", "temp_current", "temperature", "temp_now"),
        "max_so_far": ("max_so_far", "max_since_midnight"),
        "min_so_far": ("min_so_far", "min_since_midnight"),
        "relative_humidity": ("relative_humidity", "rh_current", "humidity", "rh_now"),
        "pressure": ("pressure", "pressure_current"),
        "dew_point": ("dew_point", "dew_point_current"),
        "rain_current": ("rain_current", "rain_now", "rainfall_current"),
    }
    result: list[Mapping[str, Any]] = []
    for container_name in (
        "observation_status",
        "status",
        "observation_buffer_status",
        "weather_input_status",
        "rain_input_status",
    ):
        container = weather.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for alias in aliases[field]:
            value = container.get(alias)
            if isinstance(value, Mapping):
                result.append(value)
    return result


def _value_for(weather: Mapping[str, Any], field: str) -> Any:
    aliases = {
        "temperature_current": ("temperature_current", "temp_current", "temperature", "temp_now"),
        "max_so_far": ("max_so_far", "max_since_midnight"),
        "min_so_far": ("min_so_far", "min_since_midnight"),
        "relative_humidity": ("relative_humidity", "humidity", "rh_current", "rh_now"),
        "pressure": ("pressure", "pressure_current"),
        "dew_point": ("dew_point", "dew_point_current"),
        "rain_current": ("rain_current", "rain_now", "rainfall_current"),
    }
    direct = _first(weather, aliases[field])
    if direct is not _MISSING:
        return direct
    observations = weather.get("observations")
    if isinstance(observations, Mapping):
        nested_aliases = {
            "temperature_current": ("temperature_current", "temperature", "temp", "temp_now"),
            "max_so_far": ("max_so_far",),
            "min_so_far": ("min_so_far",),
            "relative_humidity": ("relative_humidity", "humidity", "rh"),
            "pressure": ("pressure",),
            "dew_point": ("dew_point",),
            "rain_current": ("rain_current", "rain", "rainfall"),
        }
        nested = _first(observations, nested_aliases[field])
        if nested is not _MISSING:
            return nested
    for status in _status_candidates(weather, field):
        if "value" in status:
            return status["value"]
    return None


def _status_for(
    weather: Mapping[str, Any],
    field: str,
    value: Any,
    *,
    decision_timestamp: Any,
) -> dict[str, Any]:
    candidate = next(iter(_status_candidates(weather, field)), None)
    if candidate is None:
        source_timestamp = weather.get("source_timestamp")
        return InputStatus.from_value(
            value,
            source_timestamp=source_timestamp,
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs",
            stale_after_minutes=DEFAULT_STALE_AFTER_MINUTES,
            is_missing=value is None,
            raw_status=None,
            observation_method="direct_observation" if value is not None else "insufficient_history",
        ).to_dict()
    # Rebuild through InputStatus so age is always calculated from the supplied
    # source and decision timestamps.  A numeric zero is intentionally passed
    # unchanged and is not treated as missing.
    return InputStatus.from_value(
        candidate.get("value", value),
        source_timestamp=candidate.get("source_timestamp"),
        decision_timestamp=decision_timestamp,
        source_name=candidate.get("source_name") or "hko_weather_obs",
        stale_after_minutes=DEFAULT_STALE_AFTER_MINUTES,
        quality_flags=list(candidate.get("quality_flags") or []),
        raw_status=candidate.get("raw_status"),
        is_missing=candidate.get("is_missing"),
        is_fallback=bool(candidate.get("is_fallback", False)),
        fallback_method=candidate.get("fallback_method"),
        observation_method=candidate.get("observation_method"),
    ).to_dict()


def _model_age(decision_timestamp: Any, model_timestamp: Any) -> tuple[str | None, float | None]:
    model_iso = _iso_datetime(model_timestamp, naive_timezone=timezone.utc)
    if model_iso is None:
        return None, None
    decision = _parse_datetime(decision_timestamp, naive_timezone=timezone.utc)
    model = _parse_datetime(model_iso, naive_timezone=timezone.utc)
    if decision is None or model is None:
        return model_iso, None
    age = (decision - model).total_seconds()
    return (model_iso, age) if age >= 0 and math.isfinite(age) else (None, None)


def build_weather_snapshot(
    *,
    snapshot_timestamp: Any,
    event_date: str | date,
    location: str,
    weather_state: Mapping[str, Any] | None = None,
    latest_model_cycle_id: str | None = None,
    model_cycle_timestamp: Any = None,
    capture_timestamp: Any = None,
    source_status: Mapping[str, Any] | None = None,
    weather_snapshot_id: str | None = None,
    **values: Any,
) -> dict[str, Any]:
    """Normalize one observation minute while retaining Phase 2A lineage."""
    state: dict[str, Any] = dict(weather_state or {})
    for key, value in values.items():
        if key not in state:
            state[key] = value
    snapshot_iso = _iso_datetime(snapshot_timestamp, naive_timezone=timezone.utc)
    if snapshot_iso is None:
        raise WeatherSnapshotSchemaError("snapshot_timestamp is invalid")
    capture_iso = _iso_datetime(
        capture_timestamp or datetime.now(timezone.utc),
        naive_timezone=timezone.utc,
    )
    event_date_iso = event_date.isoformat() if isinstance(event_date, date) else str(event_date)
    model_iso, age = _model_age(snapshot_iso, model_cycle_timestamp)
    if model_iso is None:
        latest_model_cycle_id = None
    snapshot_id = weather_snapshot_id or make_weather_snapshot_id(
        snapshot_iso,
        event_date=event_date_iso,
        location=location,
    )
    statuses = {
        field: _status_for(
            state,
            field,
            _value_for(state, field),
            decision_timestamp=snapshot_iso,
        )
        for field in WEATHER_OBSERVATION_FIELDS
    }
    observation_values = {
        field: statuses[field].get("value") for field in WEATHER_OBSERVATION_FIELDS
    }
    optional = {}
    for key in ("wind", "uv", "wind_state", "uv_state", "extra_observations"):
        if key in state and state[key] is not None:
            optional[key] = jsonable(state[key])
    record = {
        "weather_snapshot_id": str(snapshot_id),
        "schema_version": WEATHER_SCHEMA_VERSION,
        "snapshot_timestamp": snapshot_iso,
        "capture_timestamp": capture_iso,
        "event_date": event_date_iso,
        "location": str(location),
        "latest_model_cycle_id": latest_model_cycle_id,
        "model_cycle_timestamp": model_iso,
        "model_age_seconds": age,
        **observation_values,
        "observations": observation_values,
        "observation_status": statuses,
        "source_status": jsonable(source_status or {}),
        "optional_observations": optional,
    }
    validate_weather_snapshot(record)
    return jsonable(record)


def validate_weather_snapshot(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise WeatherSnapshotSchemaError("weather snapshot must be a mapping")
    required = (
        "weather_snapshot_id",
        "schema_version",
        "snapshot_timestamp",
        "capture_timestamp",
        "event_date",
        "location",
        "latest_model_cycle_id",
        "model_cycle_timestamp",
        "model_age_seconds",
        *WEATHER_OBSERVATION_FIELDS,
        "observation_status",
        "source_status",
    )
    for field in required:
        if field not in record:
            raise WeatherSnapshotSchemaError(f"required weather snapshot field is missing: {field}")
    if record.get("schema_version") != WEATHER_SCHEMA_VERSION:
        raise WeatherSnapshotSchemaError(f"unsupported weather snapshot schema: {record.get('schema_version')!r}")
    status = record.get("observation_status")
    if not isinstance(status, Mapping):
        raise WeatherSnapshotSchemaError("observation_status must be a mapping")
    status_fields = (
        "value",
        "source_timestamp",
        "age_seconds",
        "age_minutes",
        "is_missing",
        "is_stale",
        "is_fallback",
    )
    for field in WEATHER_OBSERVATION_FIELDS:
        if not isinstance(status.get(field), Mapping):
            raise WeatherSnapshotSchemaError(f"observation status is missing: {field}")
        missing_status_fields = [key for key in status_fields if key not in status[field]]
        if missing_status_fields:
            raise WeatherSnapshotSchemaError(
                f"truthful status fields are missing: {field}:{','.join(missing_status_fields)}"
            )

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                field = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if field in _PROHIBITED_FIELDS:
                    raise WeatherSnapshotSchemaError(
                        f"strategy/account field is not allowed in weather snapshot: {child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(record)
    try:
        json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WeatherSnapshotSchemaError(f"weather snapshot is not JSON serializable: {exc}") from exc


__all__ = [
    "WEATHER_OBSERVATION_FIELDS",
    "WEATHER_SCHEMA_VERSION",
    "WeatherSnapshotSchemaError",
    "build_weather_snapshot",
    "make_weather_snapshot_id",
    "validate_weather_snapshot",
]
