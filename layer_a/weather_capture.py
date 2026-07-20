"""Independent one-minute weather observation capture loop.

The collector reads the existing cached HKO intraday state and links the
latest already-persisted five-minute model cycle.  It never invokes model
inference, canonical-cycle construction, Gamma market discovery, or CLOB
depth fetching.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from execution.market_templates import resolve_slug
from app.services.weather_service import get_intraday_state

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
    for market_kind in ("highest_temperature", "lowest_temperature"):
        try:
            record = model_store.latest_completed_model_record(
                event_date=target_date.isoformat(),
                event_slug=event_slug,
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
    try:
        if state_provider is None:
            # This existing state path fetches/uses the HKO AWS intraday CSV
            # and its local parquet cache.  It does not call model_service or
            # market_depth_service; the five-minute model builder keeps its
            # own cadence and remains untouched.
            state_provider = get_intraday_state
        state = state_provider(target.strftime("%Y%m%d"))
        if not isinstance(state, Mapping):
            run.errors.append({"stage": "weather_state", "error": "state_unavailable"})
            return run
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
        snapshot = build_weather_snapshot(
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
        run.snapshots.append(snapshot)
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
        self._last_success: str | None = None
        self._last_failure: str | None = None
        self._last_run: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict[str, Any]:
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
            started = time.monotonic()
            self.run_once()
            self._stop.wait(max(0.1, self.interval_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        if self.running:
            return
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
                "last_success": self._last_success,
                "last_failure": self._last_failure,
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
