"""Formal Layer A capture/replay quality contract and daily report."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .market_schema import DEFAULT_BOOK_FRESHNESS_SECONDS
from .schema import _parse_datetime, jsonable

logger = logging.getLogger(__name__)
HKT = timezone(timedelta(hours=8))
QUALITY_SCHEMA_VERSION = "layer_a.quality.v1"
QUALITY_THRESHOLDS = {
    "market_completeness_percentage": 95.0,
    "weather_completeness_percentage": 95.0,
    "canonical_clob_replay_eligibility_percentage": 95.0,
    "market_snapshot_linkage_percentage": 95.0,
    "weather_snapshot_linkage_percentage": 95.0,
    "model_replay_metadata_complete_percentage": 95.0,
    "weather_field_status_complete_percentage": 95.0,
    "duplicate_id_count": 0,
    "future_timestamp_count": 0,
    "unknown_token_mapping_count": 0,
    "yes_no_fetch_cycle_mismatch_count": 0,
    "cross_market_kind_identity_collision_count": 0,
}
_QUALITY_REPORT_LOCK = threading.Lock()


def _timestamp(value: Any) -> datetime | None:
    return _parse_datetime(value, naive_timezone=timezone.utc)


def _count_reason(reasons: Iterable[str], prefix: str) -> int:
    return sum(1 for reason in reasons if str(reason).startswith(prefix))


def validate_market_snapshot_for_replay(
    snapshot: Mapping[str, Any],
    *,
    max_book_age_seconds: float = DEFAULT_BOOK_FRESHNESS_SECONDS,
) -> list[str]:
    """Return contract violations for one canonical market snapshot.

    This validator deliberately requires the canonical ``book_timestamp``;
    an upstream legacy ``timestamp`` alone is not enough to enter replay.
    """
    reasons: list[str] = []

    def add(value: str) -> None:
        if value not in reasons:
            reasons.append(value)

    decision = _timestamp(snapshot.get("decision_timestamp"))
    capture = _timestamp(snapshot.get("capture_timestamp"))
    if decision is None:
        add("decision_timestamp_missing")
    if capture is None:
        add("capture_timestamp_missing")
    market_kind = str(snapshot.get("market_kind") or "")
    slug = str(snapshot.get("event_slug") or "").lower()
    slug_kind = (
        "highest_temperature" if slug.startswith("highest-temperature-")
        else "lowest_temperature" if slug.startswith("lowest-temperature-")
        else None
    )
    if slug_kind is not None and slug_kind != market_kind:
        add("market_kind_route_mismatch")

    identities = snapshot.get("market_identity")
    if not isinstance(identities, list) or not identities:
        add("market_identity_missing")
        identities = []
    by_bucket: dict[str, Mapping[str, Any]] = {}
    condition_ids: set[str] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping):
            add(f"market_identity[{index}]_invalid")
            continue
        bucket = str(identity.get("bucket") or "")
        if not bucket:
            add(f"market_identity[{index}].bucket_missing")
            continue
        by_bucket[bucket] = identity
        condition_id = str(identity.get("condition_id") or "")
        if condition_id and condition_id in condition_ids:
            add("condition_id_duplicate")
        if condition_id:
            condition_ids.add(condition_id)
        outcomes = identity.get("explicit_outcomes")
        if not isinstance(outcomes, list) or [str(item).lower() for item in outcomes] != ["yes", "no"]:
            add(f"market_identity[{bucket}].explicit_outcomes")
        for field in ("market_id", "condition_id", "yes_token_id", "no_token_id"):
            if identity.get(field) in (None, ""):
                add(f"market_identity[{bucket}].{field}")

    books = snapshot.get("clob_books")
    if not isinstance(books, list):
        books = []
    books_by_key = {
        (str(book.get("bucket") or ""), str(book.get("token_side") or "").upper()): book
        for book in books
        if isinstance(book, Mapping)
    }
    for bucket, identity in by_bucket.items():
        cycles: set[str] = set()
        for side in ("YES", "NO"):
            prefix = f"clob_books[{bucket}/{side}]"
            book = books_by_key.get((bucket, side))
            if book is None:
                add(prefix)
                add(f"{prefix}.missing")
                continue
            expected_token = identity.get("yes_token_id" if side == "YES" else "no_token_id")
            if not book.get("token_id") or not book.get("asset_id"):
                add(f"{prefix}.token_identity")
            if expected_token and (
                str(book.get("asset_id")) != str(expected_token)
                or str(book.get("token_id")) != str(expected_token)
            ):
                add(f"{prefix}.unknown_token_mapping")
            if book.get("validation_status") != "valid":
                add(f"{prefix}.validation_status")
            if not isinstance(book.get("bids"), list) or not isinstance(book.get("asks"), list):
                add(f"{prefix}.depth")
            book_timestamp = book.get("book_timestamp")
            book_dt = _timestamp(book_timestamp)
            if book_dt is None:
                add(f"{prefix}.missing_book_timestamp")
            else:
                if decision is not None and book_dt > decision:
                    add(f"{prefix}.future_book_timestamp")
                if capture is not None and book_dt > capture:
                    add(f"{prefix}.future_book_timestamp")
                if decision is not None:
                    age = (decision - book_dt).total_seconds()
                    if age < 0:
                        add(f"{prefix}.future_book_timestamp")
                    elif age > float(max_book_age_seconds):
                        add(f"{prefix}.stale_book")
                    declared_age = book.get("book_age_seconds")
                    if not isinstance(declared_age, (int, float)) or abs(float(declared_age) - age) > 1.0:
                        add(f"{prefix}.book_age_mismatch")
            cycle = book.get("fetch_cycle_id")
            if not cycle:
                add(f"{prefix}.fetch_cycle_id_missing")
            else:
                cycles.add(str(cycle))
        if len(cycles) > 1:
            add(f"yes_no_fetch_cycle_mismatch:{bucket}")

    return reasons


def _read_market_records(store: Any, date_value: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for info in store.scan(date_value=date_value):
        for snapshot in store.read_partition_snapshots(info):
            if snapshot.get("event_date") == date_value:
                records.append(snapshot)
    return records


def _read_weather_records(store: Any, date_value: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for info in store.scan(date_value=date_value):
        for snapshot in store.read_partition_snapshots(info):
            if snapshot.get("event_date") == date_value:
                records.append(snapshot)
    return records


def _unique_last(records: Iterable[Mapping[str, Any]], id_field: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get(id_field) or "")
        if key:
            by_id[key] = dict(record)
    return list(by_id.values())


def _expected_minutes(date_value: date, as_of: datetime) -> int:
    start = datetime.combine(date_value, datetime.min.time(), tzinfo=HKT).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    bounded = min(max(as_of.astimezone(timezone.utc), start), end)
    return max(1, int((bounded - start).total_seconds() // 60))


def _weather_snapshot_complete(snapshot: Mapping[str, Any]) -> bool:
    if not snapshot.get("weather_snapshot_id") or _timestamp(snapshot.get("snapshot_timestamp")) is None:
        return False
    status = snapshot.get("observation_status")
    temperature = snapshot.get("temperature_current")
    required_status_fields = (
        "value",
        "source_timestamp",
        "age_seconds",
        "age_minutes",
        "is_missing",
        "is_stale",
        "is_fallback",
    )
    return (
        temperature is not None
        and isinstance(status, Mapping)
        and all(
            isinstance(status.get(field), Mapping)
            and all(key in status[field] for key in required_status_fields)
            for field in (
                "temperature_current",
                "max_so_far",
                "min_so_far",
                "relative_humidity",
                "pressure",
                "dew_point",
                "rain_current",
            )
        )
    )


def _market_capture_complete(snapshot: Mapping[str, Any]) -> bool:
    identities = snapshot.get("market_identity")
    books = snapshot.get("clob_books")
    if not snapshot.get("market_snapshot_id") or not isinstance(identities, list) or not identities or not isinstance(books, list):
        return False
    by_key = {
        (str(book.get("bucket") or ""), str(book.get("token_side") or "").upper()): book
        for book in books
        if isinstance(book, Mapping)
    }
    return all(
        (str(identity.get("bucket")), side) in by_key
        and by_key[(str(identity.get("bucket")), side)].get("book_timestamp")
        for identity in identities
        if isinstance(identity, Mapping)
        for side in ("YES", "NO")
    )


def _model_metadata_complete(record: Mapping[str, Any]) -> bool:
    models = record.get("models")
    if not isinstance(models, list) or not models:
        return False
    fields = (
        "artifact_identity",
        "feature_spec",
        "numeric_features",
        "diagnostic_features",
    )
    return all(isinstance(model, Mapping) and all(model.get(field) not in (None, "", {}) for field in fields) for model in models)


def build_quality_report(
    *,
    date_value: str | date | None = None,
    as_of: datetime | None = None,
    model_store: Any = None,
    market_store: Any = None,
    weather_store: Any = None,
) -> dict[str, Any]:
    """Compute one deterministic report without changing strategy parameters."""
    from .market_storage import get_default_market_store
    from .storage import get_default_store
    from .weather_storage import get_default_weather_store

    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    target = date_value or now.astimezone(HKT).date()
    target_date = target if isinstance(target, date) else date.fromisoformat(str(target))
    target_iso = target_date.isoformat()
    model_store = model_store or get_default_store()
    market_store = market_store or get_default_market_store()
    weather_store = weather_store or get_default_weather_store()

    market_records = _read_market_records(market_store, target_iso)
    weather_records = _read_weather_records(weather_store, target_iso)
    model_records = [record for _info, record in model_store.iter_records(date_value=target_iso)]
    market_counts = Counter(str(item.get("market_snapshot_id") or "") for item in market_records)
    weather_counts = Counter(str(item.get("weather_snapshot_id") or "") for item in weather_records)
    duplicate_id_count = sum(max(0, count - 1) for key, count in (*market_counts.items(), *weather_counts.items()) if key)
    market_unique = _unique_last(market_records, "market_snapshot_id")
    weather_unique = _unique_last(weather_records, "weather_snapshot_id")
    expected_minutes = _expected_minutes(target_date, now)
    expected_market = expected_minutes * 2
    market_complete = sum(1 for item in market_unique if _market_capture_complete(item))
    weather_complete = sum(1 for item in weather_unique if _weather_snapshot_complete(item))

    replay_reasons: list[str] = []
    replay_eligible = 0
    future_timestamp_count = 0
    unknown_token_mapping_count = 0
    yes_no_fetch_cycle_mismatch_count = 0
    cross_market_kind_identity_collision_count = 0
    identity_owners: dict[tuple[str, str], str] = {}
    for snapshot in market_unique:
        snapshot_kind = str(snapshot.get("market_kind") or "")
        for field in ("decision_timestamp", "capture_timestamp", "latest_model_cycle_timestamp"):
            value = _timestamp(snapshot.get(field))
            if value is not None and value > now:
                future_timestamp_count += 1
        for identity in snapshot.get("market_identity", []) if isinstance(snapshot.get("market_identity"), list) else []:
            if not isinstance(identity, Mapping):
                continue
            for field in ("condition_id", "yes_token_id", "no_token_id"):
                value = identity.get(field)
                if value in (None, ""):
                    continue
                owner_key = (field, str(value))
                previous_kind = identity_owners.get(owner_key)
                if previous_kind is not None and previous_kind != snapshot_kind:
                    cross_market_kind_identity_collision_count += 1
                else:
                    identity_owners[owner_key] = snapshot_kind
        reasons = validate_market_snapshot_for_replay(snapshot)
        replay_reasons.extend(reasons)
        if not reasons and snapshot.get("latest_model_cycle_id"):
            replay_eligible += 1
        future_timestamp_count += _count_reason(reasons, "future_book_timestamp")
        unknown_token_mapping_count += sum(1 for reason in reasons if reason.endswith("unknown_token_mapping"))
        yes_no_fetch_cycle_mismatch_count += sum(
            1 for reason in reasons if reason.startswith("yes_no_fetch_cycle_mismatch")
        )
    # Count future weather source timestamps as well as future market books.
    for snapshot in weather_unique:
        capture = _timestamp(snapshot.get("capture_timestamp"))
        snapshot_time = _timestamp(snapshot.get("snapshot_timestamp"))
        for field in ("snapshot_timestamp", "capture_timestamp", "model_cycle_timestamp"):
            value = _timestamp(snapshot.get(field))
            if value is not None and value > now:
                future_timestamp_count += 1
        if capture is not None and snapshot_time is not None and snapshot_time > capture:
            future_timestamp_count += 1
        for status in (snapshot.get("observation_status") or {}).values() if isinstance(snapshot.get("observation_status"), Mapping) else ():
            if isinstance(status, Mapping):
                source = _timestamp(status.get("source_timestamp"))
                if source is not None and capture is not None and source > capture:
                    future_timestamp_count += 1

    for record in model_records:
        for field in ("decision_timestamp", "capture_timestamp"):
            value = _timestamp(record.get(field))
            if value is not None and value > now:
                future_timestamp_count += 1

    market_ids = {str(item.get("market_snapshot_id")) for item in market_unique if item.get("market_snapshot_id")}
    weather_ids = {str(item.get("weather_snapshot_id")) for item in weather_unique if item.get("weather_snapshot_id")}
    linked_market = sum(1 for record in model_records if str(record.get("market_snapshot_id") or "") in market_ids)
    linked_weather = sum(1 for record in model_records if str(record.get("weather_snapshot_id") or "") in weather_ids)
    metadata_complete = sum(1 for record in model_records if _model_metadata_complete(record))
    model_count = len(model_records)
    market_percentage = 100.0 * market_complete / expected_market if expected_market else 0.0
    weather_percentage = 100.0 * weather_complete / expected_minutes if expected_minutes else 0.0
    replay_percentage = 100.0 * replay_eligible / len(market_unique) if market_unique else 0.0
    metrics = {
        "expected_minute_slots": expected_minutes,
        "expected_market_snapshots": expected_market,
        "market_snapshots_seen": len(market_records),
        "market_snapshots_unique": len(market_unique),
        "market_snapshots_complete": market_complete,
        "market_completeness_percentage": round(market_percentage, 4),
        "weather_snapshots_seen": len(weather_records),
        "weather_snapshots_unique": len(weather_unique),
        "weather_snapshots_complete": weather_complete,
        "weather_completeness_percentage": round(weather_percentage, 4),
        "canonical_clob_replay_eligible": replay_eligible,
        "canonical_clob_replay_eligibility_percentage": round(replay_percentage, 4),
        "duplicate_id_count": duplicate_id_count,
        "future_timestamp_count": future_timestamp_count,
        "unknown_token_mapping_count": unknown_token_mapping_count,
        "yes_no_fetch_cycle_mismatch_count": yes_no_fetch_cycle_mismatch_count,
        "cross_market_kind_identity_collision_count": cross_market_kind_identity_collision_count,
        "model_cycles_seen": model_count,
        "market_snapshot_linkage_percentage": round(100.0 * linked_market / model_count, 4) if model_count else 0.0,
        "weather_snapshot_linkage_percentage": round(100.0 * linked_weather / model_count, 4) if model_count else 0.0,
        "model_replay_metadata_complete_percentage": round(100.0 * metadata_complete / model_count, 4) if model_count else 0.0,
        "weather_field_status_complete_percentage": round(
            100.0 * sum(
                1
                for item in weather_unique
                if isinstance(item.get("observation_status"), Mapping)
                and all(
                    isinstance(item["observation_status"].get(field), Mapping)
                    and all(
                        key in item["observation_status"].get(field, {})
                        for key in (
                            "value",
                            "source_timestamp",
                            "age_seconds",
                            "age_minutes",
                            "is_missing",
                            "is_stale",
                            "is_fallback",
                        )
                    )
                    for field in ("dew_point", "pressure", "rain_current")
                )
            ) / len(weather_unique),
            4,
        ) if weather_unique else 0.0,
    }
    checks = {
        key: (
            metrics.get(key, 0.0) > threshold
            if key.endswith("percentage")
            else metrics.get(key, 0) == threshold
        )
        for key, threshold in QUALITY_THRESHOLDS.items()
    }
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "report_date": target_iso,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": now.isoformat(),
        "thresholds": dict(QUALITY_THRESHOLDS),
        "metrics": metrics,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "replay_rejection_reason_counts": dict(Counter(replay_reasons)),
    }


def quality_report_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    try:
        from app.config import LAYER_A_QUALITY_DIR

        return Path(LAYER_A_QUALITY_DIR)
    except ImportError:
        return Path("data/layer_a_quality")


def write_daily_quality_report(report: Mapping[str, Any], *, root: Path | str | None = None) -> Path:
    with _QUALITY_REPORT_LOCK:
        output_root = quality_report_root(root)
        report_date = str(report.get("report_date") or "unknown-date")
        output_dir = output_root / f"date={report_date}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "quality_report.json"
        temporary = Path(f"{output_path}.tmp")
        temporary.write_text(
            json.dumps(jsonable(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        return output_path


def build_and_write_daily_quality_report(**kwargs: Any) -> dict[str, Any]:
    root = kwargs.pop("root", None)
    report = build_quality_report(**kwargs)
    path = write_daily_quality_report(report, root=root)
    return {**report, "report_path": str(path)}


class LayerAQualityWorker:
    """Periodically materialize the current day's formal quality report."""

    def __init__(self, *, interval_minutes: float | None = None) -> None:
        self.interval_minutes = max(
            1.0,
            float(interval_minutes or os.getenv("LAYER_A_QUALITY_INTERVAL_MINUTES", "15")),
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run: str | None = None
        self._last_report: dict[str, Any] | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict[str, Any]:
        try:
            report = build_and_write_daily_quality_report()
            self._last_report = {
                "report_date": report.get("report_date"),
                "gate_passed": bool(report.get("gate_passed", False)),
                "report_path": report.get("report_path"),
            }
            self._last_error = None
        except Exception as exc:
            self._last_report = None
            self._last_error = type(exc).__name__
            logger.exception("Layer A quality report generation failed")
        self._last_run = datetime.now(timezone.utc).isoformat()
        return {
            "last_run": self._last_run,
            "last_report": self._last_report,
            "last_error": self._last_error,
        }

    def _loop(self) -> None:
        self.run_once()
        while not self._stop.wait(self.interval_minutes * 60.0):
            self.run_once()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="layer-a-quality-worker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def health_summary(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interval_minutes": self.interval_minutes,
            "last_run": self._last_run,
            "last_report": self._last_report,
            "last_error": self._last_error,
        }


_DEFAULT_QUALITY_WORKER: LayerAQualityWorker | None = None
_DEFAULT_QUALITY_WORKER_LOCK = threading.Lock()


def get_default_quality_worker() -> LayerAQualityWorker:
    global _DEFAULT_QUALITY_WORKER
    if _DEFAULT_QUALITY_WORKER is None:
        with _DEFAULT_QUALITY_WORKER_LOCK:
            if _DEFAULT_QUALITY_WORKER is None:
                _DEFAULT_QUALITY_WORKER = LayerAQualityWorker()
    return _DEFAULT_QUALITY_WORKER


__all__ = [
    "QUALITY_SCHEMA_VERSION",
    "QUALITY_THRESHOLDS",
    "LayerAQualityWorker",
    "build_and_write_daily_quality_report",
    "build_quality_report",
    "get_default_quality_worker",
    "validate_market_snapshot_for_replay",
    "write_daily_quality_report",
]
