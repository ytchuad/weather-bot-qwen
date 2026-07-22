"""Independent one-minute weather observation capture loop.

The collector reads the existing cached HKO intraday state and links the
latest already-persisted five-minute model cycle.  It never invokes model
inference, canonical-cycle construction, Gamma market discovery, or CLOB
depth fetching.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from execution.market_templates import resolve_slug
from app.services.weather_service import get_intraday_state
from features.input_status import DEFAULT_STALE_AFTER_MINUTES, InputStatus

from .schema import jsonable
from .storage import LayerAStore, get_default_store
from .weather_schema import build_weather_snapshot
from .weather_storage import WeatherSnapshotStore, get_default_weather_store

logger = logging.getLogger(__name__)
HKT = timezone(timedelta(hours=8))


@dataclass
class WeatherCollectionRun:
    snapshot_timestamp: str
    latest_model_cycle_id: str | None = None
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    capture_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_timestamp": self.snapshot_timestamp,
            "latest_model_cycle_id": self.latest_model_cycle_id,
            "snapshot_count": len(self.snapshots),
            "captured_count": sum(item.get("status") == "captured" for item in self.capture_results),
            "duplicate_count": sum(item.get("status") == "duplicate" for item in self.capture_results),
            "errors": jsonable(self.errors),
            "weather_snapshot_ids": [item.get("weather_snapshot_id") for item in self.snapshots],
        }


def _now_utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=HKT).astimezone(timezone.utc)
    return current.astimezone(timezone.utc)


def _latest_model(
    model_store: LayerAStore,
    *,
    target_date: date,
    event_slug: str,
    before: datetime,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    event_slugs = {str(event_slug)}
    try:
        event_slugs.update(
            {
                resolve_slug("hk-tmax", target_date),
                resolve_slug("hk-tmin", target_date),
            }
        )
    except Exception:
        pass
    for market_kind in ("highest_temperature", "lowest_temperature"):
        for slug in event_slugs:
            try:
                record = model_store.latest_completed_model_record(
                    event_date=target_date.isoformat(),
                    event_slug=slug,
                    market_kind=market_kind,
                    before_timestamp=before,
                )
            except Exception:
                continue
            if isinstance(record, Mapping):
                candidates.append(dict(record))
    if not candidates:
        return None
    candidates.sort(key=lambda record: str(record.get("decision_timestamp") or ""))
    return candidates[-1]


def _finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _source_timestamp(value: Any) -> datetime | None:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return None
    return _now_utc(value)


def _csv_status(value: Any, timestamp: datetime) -> dict[str, Any]:
    return InputStatus.from_value(
        value,
        source_timestamp=timestamp,
        decision_timestamp=timestamp,
        source_name="hko_aws_csv",
        stale_after_minutes=DEFAULT_STALE_AFTER_MINUTES,
        observation_method="direct_observation",
    ).to_dict()


def _fallback_status(value: Any, *, timestamp: datetime, method: str, source_name: str) -> dict[str, Any]:
    return InputStatus.fallback(
        value,
        fallback_method=method,
        decision_timestamp=timestamp,
        source_name=source_name,
        raw_status="synthetic_fallback",
        observation_method="fallback",
    ).to_dict()


def _dew_point_celsius(temperature: float | None, humidity: float | None) -> float | None:
    if temperature is None or humidity is None or humidity <= 0:
        return None
    try:
        import math as _math

        a, b = 17.625, 243.04
        gamma = _math.log(humidity / 100.0) + (a * temperature) / (b + temperature)
        return float((b * gamma) / (a - gamma))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _status_from_state(state: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    for container_name in ("observation_buffer_status", "weather_input_status", "status"):
        container = state.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def _status_timestamp(status: Mapping[str, Any] | None) -> datetime | None:
    return _source_timestamp((status or {}).get("source_timestamp"))


def _ensure_weather_fields(
    state: Mapping[str, Any],
    *,
    observation_timestamp: datetime,
) -> dict[str, Any]:
    """Ensure fields used by models/regime analysis have truthful statuses."""
    enriched = dict(state)
    observations = dict(enriched.get("observations") or {}) if isinstance(enriched.get("observations"), Mapping) else {}
    observation_status = dict(enriched.get("observation_buffer_status") or {}) if isinstance(enriched.get("observation_buffer_status"), Mapping) else {}
    weather_status = dict(enriched.get("weather_input_status") or {}) if isinstance(enriched.get("weather_input_status"), Mapping) else {}

    temperature = _finite_number(enriched.get("temp_now"))
    if temperature is None:
        temperature = _finite_number(observations.get("temperature"))
    humidity = _finite_number(enriched.get("rh_now"))
    if humidity is None:
        humidity = _finite_number(observations.get("humidity"))
    source_timestamp = _source_timestamp(enriched.get("time_now")) or observation_timestamp

    pressure = _finite_number(enriched.get("pressure_current", enriched.get("pressure")))
    pressure_status = _status_from_state(enriched, "pressure_current", "pressure")
    if pressure is None:
        pressure = 1010.0
        pressure_status = _fallback_status(
            pressure,
            timestamp=observation_timestamp,
            method="climatological_default",
            source_name="hko_pressure",
        )
    elif pressure_status is None:
        pressure_status = _csv_status(pressure, source_timestamp)

    dew_point = _finite_number(enriched.get("dew_point_current", enriched.get("dew_point")))
    dew_status = _status_from_state(enriched, "dew_point_current", "dew_point")
    if dew_point is None:
        dew_point = _dew_point_celsius(temperature, humidity)
        dew_status = (
            InputStatus.from_value(
                dew_point,
                source_timestamp=source_timestamp if dew_point is not None else None,
                decision_timestamp=observation_timestamp,
                source_name="hko_weather_obs",
                stale_after_minutes=DEFAULT_STALE_AFTER_MINUTES,
                observation_method="derived_dew_point" if dew_point is not None else "insufficient_history",
                is_missing=dew_point is None,
            ).to_dict()
        )

    rain_current = _finite_number(enriched.get("rain_current", enriched.get("rain_now")))
    rain_status = _status_from_state(enriched, "rain_current", "rain_now")
    if rain_current is None:
        rain_current = 0.0
        rain_status = _fallback_status(
            rain_current,
            timestamp=observation_timestamp,
            method="model_compat_zero",
            source_name="i-lens_rain_obs",
        )
    elif rain_status is None:
        rain_status = _csv_status(rain_current, source_timestamp)

    observations.update(
        {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "dew_point": dew_point,
            "rain_current": rain_current,
        }
    )
    observation_status.update(
        {
            "pressure_current": dict(pressure_status or {}),
            "dew_point_current": dict(dew_status or {}),
            "rain_current": dict(rain_status or {}),
        }
    )
    weather_status.update(
        {
            "pressure_current": dict(pressure_status or {}),
            "dew_point_current": dict(dew_status or {}),
            "rain_current": dict(rain_status or {}),
        }
    )
    enriched.update(
        {
            "pressure_current": pressure,
            "dew_point_current": dew_point,
            "rain_current": rain_current,
            "observations": observations,
            "observation_buffer_status": observation_status,
            "weather_input_status": weather_status,
        }
    )
    return enriched


def _enrich_live_weather_state(state: Mapping[str, Any], *, target: date, decision: datetime) -> dict[str, Any]:
    """Attach pressure/rain source status for the independent collector."""
    enriched = dict(state)
    observation_time = _source_timestamp(state.get("time_now")) or decision
    try:
        from app.services.weather_service import compute_pressure_kwargs, compute_rain_kwargs

        decision_hkt = decision.astimezone(HKT).replace(tzinfo=None)
        pressure_kwargs = compute_pressure_kwargs(decision_timestamp=decision_hkt)
        for key in ("pressure_current", "pressure_30m_ago", "pressure_change_60m", "pressure_change_180m"):
            if pressure_kwargs.get(key) is not None:
                enriched[key] = pressure_kwargs[key]
        pressure_status = pressure_kwargs.get("_input_status")
        if isinstance(pressure_status, Mapping):
            enriched["weather_input_status"] = {
                **dict(enriched.get("weather_input_status") or {}),
                "pressure_current": pressure_status.get("pressure_current"),
            }
            enriched["observation_buffer_status"] = {
                **dict(enriched.get("observation_buffer_status") or {}),
                "pressure_current": pressure_status.get("pressure_current"),
            }
        drop_from_max = float(state.get("max_so_far", 0.0) or 0.0) - float(state.get("temp_now", 0.0) or 0.0)
        rain_kwargs = compute_rain_kwargs(
            target.strftime("%Y%m%d"),
            decision_hkt,
            drop_from_max=drop_from_max,
            temp_change_60m=float(state.get("temp_change_60m", 0.0) or 0.0),
        )
        if isinstance(rain_kwargs, Mapping):
            enriched.update({key: rain_kwargs[key] for key in ("rain_current", "rain_60m", "rain_120m") if key in rain_kwargs})
            rain_status = rain_kwargs.get("_input_status")
            if isinstance(rain_status, Mapping):
                enriched["weather_input_status"] = {
                    **dict(enriched.get("weather_input_status") or {}),
                    "rain_current": rain_status.get("rain_current"),
                }
                enriched["observation_buffer_status"] = {
                    **dict(enriched.get("observation_buffer_status") or {}),
                    "rain_current": rain_status.get("rain_current"),
                }
    except Exception:
        logger.exception("Independent weather enrichment failed")
    return _ensure_weather_fields(enriched, observation_timestamp=observation_time)


def _buffered_weather_snapshots(
    state: Mapping[str, Any],
    *,
    target: date,
    decision: datetime,
    model_record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize every valid minute present in the HKO intraday CSV buffer."""
    frame = state.get("df_today")
    if frame is None or not hasattr(frame, "iterrows"):
        return []
    try:
        frame = frame.sort_values("datetime")
    except Exception:
        return []

    running_temperatures: list[float] = []
    snapshots: list[dict[str, Any]] = []
    model_timestamp = _source_timestamp((model_record or {}).get("decision_timestamp"))
    model_id = (model_record or {}).get("decision_cycle_id")
    seen_minutes: set[datetime] = set()
    latest_frame_timestamp = max(
        (_source_timestamp(value) for value in frame.get("datetime", [])),
        default=None,
    )

    for _, row in frame.iterrows():
        timestamp = _source_timestamp(row.get("datetime"))
        temperature = _finite_number(row.get("temp"))
        if timestamp is None or temperature is None:
            continue
        minute = timestamp.replace(second=0, microsecond=0)
        if minute in seen_minutes:
            continue
        seen_minutes.add(minute)
        running_temperatures.append(temperature)
        maximum = max(running_temperatures)
        minimum = min(running_temperatures)
        humidity = _finite_number(row.get("rh"))
        is_latest = latest_frame_timestamp is not None and minute == latest_frame_timestamp.replace(second=0, microsecond=0)

        def row_number(*keys: str) -> float | None:
            for key in keys:
                value = _finite_number(row.get(key))
                if value is not None:
                    return value
            return None

        def latest_state_value(*keys: str) -> tuple[float | None, Mapping[str, Any] | None]:
            if not is_latest:
                return None, None
            for key in keys:
                value = _finite_number(state.get(key))
                status = _status_from_state(state, key)
                status_time = _status_timestamp(status)
                if value is not None and (status_time is None or status_time <= timestamp):
                    return value, status
            return None, None

        pressure = row_number("pressure", "pressure_current")
        pressure_status: Mapping[str, Any] | None = _csv_status(pressure, timestamp) if pressure is not None else None
        if pressure is None:
            pressure, pressure_status = latest_state_value("pressure_current", "pressure")
        if pressure is None:
            pressure = 1010.0
            pressure_status = _fallback_status(
                pressure,
                timestamp=timestamp,
                method="climatological_default",
                source_name="hko_pressure",
            )

        dew_point = row_number("dew_point", "dew_point_current")
        dew_point_status: Mapping[str, Any] | None = _csv_status(dew_point, timestamp) if dew_point is not None else None
        if dew_point is None:
            dew_point, dew_point_status = latest_state_value("dew_point_current", "dew_point")
        if dew_point is None:
            dew_point = _dew_point_celsius(temperature, humidity)
            dew_point_status = InputStatus.from_value(
                dew_point,
                source_timestamp=timestamp if dew_point is not None else None,
                decision_timestamp=timestamp,
                source_name="hko_weather_obs",
                stale_after_minutes=DEFAULT_STALE_AFTER_MINUTES,
                is_missing=dew_point is None,
                observation_method="derived_dew_point" if dew_point is not None else "insufficient_history",
            ).to_dict()

        rain_current = row_number("rain_current", "rainfall", "rain")
        rain_status: Mapping[str, Any] | None = _csv_status(rain_current, timestamp) if rain_current is not None else None
        if rain_current is None:
            rain_current, rain_status = latest_state_value("rain_current", "rain_now")
        if rain_current is None:
            rain_current = 0.0
            rain_status = _fallback_status(
                rain_current,
                timestamp=timestamp,
                method="model_compat_zero",
                source_name="i-lens_rain_obs",
            )
        weather_state = {
            "observations": {
                "temperature": temperature,
                "humidity": humidity,
                "pressure": pressure,
                "dew_point": dew_point,
                "rain_current": rain_current,
            },
            "max_so_far": maximum,
            "min_so_far": minimum,
            "pressure": pressure,
            "dew_point": dew_point,
            "rain_current": rain_current,
            "source_timestamp": timestamp,
            "status": {
                "temperature": _csv_status(temperature, timestamp),
                "humidity": _csv_status(humidity, timestamp),
                "max_so_far": _csv_status(maximum, timestamp),
                "min_so_far": _csv_status(minimum, timestamp),
            },
            "observation_status": {
                "temperature_current": _csv_status(temperature, timestamp),
                "relative_humidity": _csv_status(humidity, timestamp),
                "max_so_far": _csv_status(maximum, timestamp),
                "min_so_far": _csv_status(minimum, timestamp),
                "pressure": dict(pressure_status or {}),
                "dew_point": dict(dew_point_status or {}),
                "rain_current": dict(rain_status or {}),
            },
        }
        row_model_id = model_id if model_timestamp is not None and model_timestamp <= timestamp else None
        row_model_timestamp = model_timestamp if row_model_id is not None else None
        snapshots.append(
            build_weather_snapshot(
                snapshot_timestamp=timestamp,
                event_date=target,
                location="Hong Kong",
                weather_state=weather_state,
                latest_model_cycle_id=row_model_id,
                model_cycle_timestamp=row_model_timestamp,
                capture_timestamp=decision,
                source_status={
                    "collector": "layer_a_weather_only",
                    "observation_source": "hko_aws_csv_buffer",
                    "buffer_backfill": True,
                    "model_link_source": "local_completed_layer_a_cycle" if row_model_id else None,
                    "clob_fetched": False,
                },
            )
        )
    return snapshots


def collect_weather_snapshots_once(
    *,
    target_date: date | None = None,
    event_slug: str | None = None,
    decision_timestamp: datetime | None = None,
    weather_store: WeatherSnapshotStore | None = None,
    model_store: LayerAStore | None = None,
    state_provider: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> WeatherCollectionRun:
    """Capture one weather minute without running any models or fetching CLOB."""
    decision = _now_utc(decision_timestamp)
    target = target_date or decision.astimezone(HKT).date()
    run = WeatherCollectionRun(decision.isoformat())
    weather_store = weather_store or get_default_weather_store()
    model_store = model_store or get_default_store()
    slug = event_slug or resolve_slug("hk-tmax", target)
    use_live_enrichment = False
    try:
        if state_provider is None:
            # This existing state path fetches/uses the HKO AWS intraday CSV
            # and its local parquet cache.  It does not call model_service or
            # market_depth_service; the five-minute model builder keeps its
            # own cadence and remains untouched.
            state_provider = get_intraday_state
            use_live_enrichment = getattr(state_provider, "__module__", "") == "app.services.weather_service"
        state = state_provider(target.strftime("%Y%m%d"))
        if not isinstance(state, Mapping):
            run.errors.append({"stage": "weather_state", "error": "state_unavailable"})
            return run
        if use_live_enrichment:
            state = _enrich_live_weather_state(state, target=target, decision=decision)
        else:
            state = _ensure_weather_fields(state, observation_timestamp=decision)
    except Exception as exc:
        run.errors.append({"stage": "weather_state", "error": type(exc).__name__})
        return run

    model_record = _latest_model(
        model_store,
        target_date=target,
        event_slug=str(slug),
        before=decision,
    )
    run.latest_model_cycle_id = (model_record or {}).get("decision_cycle_id")
    try:
        run.snapshots = _buffered_weather_snapshots(
            state,
            target=target,
            decision=decision,
            model_record=model_record,
        )
        if not run.snapshots:
            run.snapshots.append(
                build_weather_snapshot(
                    snapshot_timestamp=decision,
                    event_date=target,
                    location="Hong Kong",
            weather_state=state,
                    latest_model_cycle_id=(model_record or {}).get("decision_cycle_id"),
                    model_cycle_timestamp=(model_record or {}).get("decision_timestamp"),
                    source_status={
                        "collector": "layer_a_weather_only",
                        "observation_source": "hko_intraday_state",
                        "model_link_source": "local_completed_layer_a_cycle",
                        "clob_fetched": False,
                    },
                )
            )
    except Exception as exc:
        run.errors.append({"stage": "weather_snapshot_schema", "error": type(exc).__name__})
        logger.exception("Weather-only snapshot normalization failed")
        return run

    try:
        results = weather_store.capture_many(run.snapshots)
        run.capture_results = [
            {
                "status": result.status,
                "weather_snapshot_id": result.weather_snapshot_id,
                "partition_id": result.partition_id,
            }
            for result in results
        ]
    except Exception as exc:
        run.errors.append({"stage": "weather_storage", "error": type(exc).__name__})
        logger.exception("Weather-only Layer A capture failed")
    return run


class WeatherSnapshotCollector:
    """Daemon collector aligned to a one-minute wall-clock cadence."""

    def __init__(self, *, interval_seconds: float = 60.0, **once_kwargs: Any) -> None:
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.once_kwargs = dict(once_kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._runs = 0
        self._failed_runs = 0
        self._last_tick: str | None = None
        self._last_success: str | None = None
        self._last_failure: str | None = None
        self._last_error: str | None = None
        self._last_run: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict[str, Any]:
        tick = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_tick = tick
        try:
            report = collect_weather_snapshots_once(**self.once_kwargs).as_dict()
        except Exception as exc:
            report = {
                "snapshot_count": 0,
                "captured_count": 0,
                "duplicate_count": 0,
                "errors": [{"stage": "collector", "error": type(exc).__name__}],
            }
            logger.exception("Weather-only collector run failed")
        with self._lock:
            self._runs += 1
            self._last_run = report
            errors = report.get("errors") or []
            self._last_error = "; ".join(
                str(item.get("error") or item.get("stage") or "unknown")
                for item in errors
                if isinstance(item, Mapping)
            ) or None
            if report.get("snapshot_count", 0) > 0 and report.get("captured_count", 0) > 0:
                self._last_success = datetime.now(timezone.utc).isoformat()
            elif report.get("snapshot_count", 0) > 0 and not report.get("errors"):
                self._last_success = datetime.now(timezone.utc).isoformat()
            else:
                self._failed_runs += 1
                self._last_failure = datetime.now(timezone.utc).isoformat()
        return report

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            if self._stop.is_set():
                break
            period = max(1.0, self.interval_seconds)
            now = time.time()
            next_boundary = (int(now // period) + 1) * period
            self._stop.wait(max(0.1, next_boundary - now))

    def start(self) -> None:
        if self.running:
            return
        try:
            store = self.once_kwargs.get("weather_store") or get_default_weather_store()
            startup_scan = getattr(store, "startup_scan", None)
            if callable(startup_scan):
                startup_scan()
        except Exception:
            logger.exception("Layer A weather collector startup recovery failed")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="layer-a-weather-collector")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def health_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "interval_seconds": self.interval_seconds,
                "runs": self._runs,
                "failed_runs": self._failed_runs,
                "last_tick": self._last_tick,
                "last_success": self._last_success,
                "last_failure": self._last_failure,
                "last_error": self._last_error,
                "last_run": jsonable(self._last_run),
            }


_DEFAULT_WEATHER_COLLECTOR: WeatherSnapshotCollector | None = None
_DEFAULT_WEATHER_COLLECTOR_LOCK = threading.Lock()


def get_default_weather_collector() -> WeatherSnapshotCollector:
    global _DEFAULT_WEATHER_COLLECTOR
    if _DEFAULT_WEATHER_COLLECTOR is None:
        with _DEFAULT_WEATHER_COLLECTOR_LOCK:
            if _DEFAULT_WEATHER_COLLECTOR is None:
                _DEFAULT_WEATHER_COLLECTOR = WeatherSnapshotCollector()
    return _DEFAULT_WEATHER_COLLECTOR


__all__ = [
    "WeatherCollectionRun",
    "WeatherSnapshotCollector",
    "collect_weather_snapshots_once",
    "get_default_weather_collector",
]
