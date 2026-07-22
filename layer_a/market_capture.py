"""Independent one-minute market-only Layer A capture loop.

This module intentionally does not call the weather, model, canonical-cycle,
account, or strategy execution paths.  It fetches Gamma market references and
one coherent YES/NO CLOB batch, then appends the immutable snapshot to the
hourly market-only store.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from app.services.market_depth_service import fetch_market_depths_batch
from app.services.market_service import (
    fetch_event_markets,
    market_kind_from_slug,
    resolve_event_slug_for_kind,
)
from .market_schema import build_market_snapshot
from .market_storage import MarketSnapshotStore, get_default_market_store
from .schema import jsonable
from .storage import LayerAStore, get_default_store

logger = logging.getLogger(__name__)

HKT = timezone(timedelta(hours=8))
_MARKET_KINDS = ((False, "highest_temperature"), (True, "lowest_temperature"))


@dataclass
class MarketCollectionRun:
    decision_timestamp: str
    fetch_cycle_id: str
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    capture_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    event_slugs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_timestamp": self.decision_timestamp,
            "fetch_cycle_id": self.fetch_cycle_id,
            "snapshot_count": len(self.snapshots),
            "captured_count": sum(item.get("status") == "captured" for item in self.capture_results),
            "duplicate_count": sum(item.get("status") == "duplicate" for item in self.capture_results),
            "errors": jsonable(self.errors),
            "event_slugs": dict(self.event_slugs),
            "market_snapshot_ids": [item.get("market_snapshot_id") for item in self.snapshots],
        }


def _now_utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=HKT).astimezone(timezone.utc)
    return current.astimezone(timezone.utc)


def _market_kind(is_min_temp: bool) -> str:
    return "lowest_temperature" if is_min_temp else "highest_temperature"


def _market_token(market: Mapping[str, Any], *, no: bool) -> str | None:
    if no:
        value = market.get("no_token_id") or market.get("no_asset_id")
    else:
        value = market.get("token_id") or market.get("yes_token_id") or market.get("yes_asset_id")
    return str(value) if value not in (None, "") else None


def _gamma_reference_data(markets: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(market.get("bucket")): jsonable(dict(market))
        for market in markets
        if market.get("bucket")
    }


def _depth_maps(
    markets: Iterable[Mapping[str, Any]],
    market_kind: str,
    depth_by_key: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    yes: dict[str, Any] = {}
    no: dict[str, Any] = {}
    for market in markets:
        bucket = str(market.get("bucket") or "")
        if not bucket:
            continue
        yes_key = f"{market_kind}:{bucket}:YES"
        no_key = f"{market_kind}:{bucket}:NO"
        yes[bucket] = depth_by_key.get(yes_key)
        no[bucket] = depth_by_key.get(no_key)
    return yes, no


def collect_market_snapshots_once(
    *,
    target_date: date | None = None,
    event_slug: str | None = None,
    decision_timestamp: datetime | None = None,
    market_kinds: Iterable[bool] = (False, True),
    market_store: MarketSnapshotStore | None = None,
    model_store: LayerAStore | None = None,
    fetch_cycle_id: str | None = None,
) -> MarketCollectionRun:
    """Fetch and persist one market-only minute for all requested market kinds."""
    decision = _now_utc(decision_timestamp)
    target = target_date or decision.astimezone(HKT).date()
    cycle_id = fetch_cycle_id or f"market-{decision.strftime('%Y%m%dT%H%M%S.%fZ')}"
    run = MarketCollectionRun(decision.isoformat(), cycle_id)
    market_store = market_store or get_default_market_store()
    model_store = model_store or get_default_store()

    requested = {_market_kind(bool(value)) for value in market_kinds}
    markets_by_kind: dict[str, list[dict[str, Any]]] = {}
    slugs_by_kind: dict[str, str] = {}
    for is_min_temp, kind in _MARKET_KINDS:
        if kind not in requested:
            continue
        kind_slug = event_slug if market_kind_from_slug(event_slug) == kind else None
        if not kind_slug:
            try:
                kind_slug = resolve_event_slug_for_kind(target, is_min_temp=is_min_temp)
            except Exception as exc:
                run.errors.append(
                    {"stage": "gamma_event", "market_kind": kind, "error": type(exc).__name__}
                )
                kind_slug = None
        if not kind_slug:
            run.errors.append({"stage": "gamma_event", "market_kind": kind, "error": "event_slug_missing"})
            continue
        slugs_by_kind[kind] = str(kind_slug)
        run.event_slugs[kind] = str(kind_slug)
        try:
            markets = fetch_event_markets(str(kind_slug), is_min_temp=is_min_temp)
            markets_by_kind[kind] = [dict(item) for item in markets if isinstance(item, Mapping)]
            if not markets_by_kind[kind]:
                run.errors.append({"stage": "gamma_markets", "market_kind": kind, "error": "no_markets"})
        except Exception as exc:
            markets_by_kind[kind] = []
            run.errors.append(
                {"stage": "gamma_markets", "market_kind": kind, "error": type(exc).__name__}
            )

    # Fail closed when Gamma routes an identity from one event into another
    # market kind.  A shared condition/token is never safe to replay because
    # it makes YES/NO books indistinguishable across Tmax/Tmin.
    identity_owners: dict[tuple[str, str], tuple[str, str]] = {}
    invalid_kinds: set[str] = set()
    for kind, markets in markets_by_kind.items():
        expected_slug_kind = market_kind_from_slug(slugs_by_kind.get(kind))
        if expected_slug_kind is not None and expected_slug_kind != kind:
            invalid_kinds.add(kind)
            run.errors.append(
                {
                    "stage": "market_routing",
                    "market_kind": kind,
                    "error": "event_slug_kind_mismatch",
                    "event_slug": slugs_by_kind.get(kind),
                }
            )
        for market in markets:
            bucket = str(market.get("bucket") or "")
            for identity_field in ("conditionId", "condition_id", "token_id", "yes_token_id", "no_token_id"):
                value = market.get(identity_field)
                if value in (None, ""):
                    continue
                identity_key = (identity_field.replace("conditionId", "condition_id"), str(value))
                previous = identity_owners.get(identity_key)
                if previous is not None and previous[0] != kind:
                    invalid_kinds.update({kind, previous[0]})
                    run.errors.append(
                        {
                            "stage": "market_routing",
                            "market_kind": kind,
                            "error": "cross_market_kind_identity_collision",
                            "field": identity_key[0],
                            "value": identity_key[1],
                            "previous_market_kind": previous[0],
                            "previous_bucket": previous[1],
                            "bucket": bucket,
                        }
                    )
                else:
                    identity_owners[identity_key] = (kind, bucket)
    for kind in invalid_kinds:
        markets_by_kind[kind] = []

    token_map: dict[str, str] = {}
    for kind, markets in markets_by_kind.items():
        for market in markets:
            bucket = str(market.get("bucket") or "")
            if not bucket:
                continue
            for no in (False, True):
                token = _market_token(market, no=no)
                if token:
                    token_map[f"{kind}:{bucket}:{'NO' if no else 'YES'}"] = token

    depth_by_key: dict[str, Any] = {key: None for key in token_map}
    depth_fetch_errors: list[str] = []
    if token_map:
        try:
            fetched = fetch_market_depths_batch(token_map, fetch_cycle_id=cycle_id)
            if isinstance(fetched, Mapping):
                depth_by_key.update(fetched)
        except Exception as exc:
            error = type(exc).__name__
            depth_fetch_errors.append(error)
            run.errors.append({"stage": "clob_books", "error": error})
    else:
        depth_fetch_errors.append("no_token_ids")
        run.errors.append({"stage": "clob_books", "error": "no_token_ids"})

    for kind, markets in markets_by_kind.items():
        if not markets:
            continue
        yes_depth, no_depth = _depth_maps(markets, kind, depth_by_key)
        model_record = None
        try:
            model_record = model_store.latest_completed_model_record(
                event_date=target.isoformat(),
                event_slug=str(slugs_by_kind.get(kind)),
                market_kind=kind,
                before_timestamp=decision,
            )
        except Exception as exc:
            run.errors.append({"stage": "latest_model_cycle", "market_kind": kind, "error": type(exc).__name__})

        gamma_refs = _gamma_reference_data(markets)
        depth_errors = [
            f"{bucket}/{side}_book_missing"
            for market in markets
            for bucket in [str(market.get("bucket") or "")]
            for side, depths in (("YES", yes_depth), ("NO", no_depth))
            if bucket and depths.get(bucket) is None
        ]
        source_status = {
            "collector": "layer_a_market_only",
            "market_source": "polymarket_gamma_reference",
            "book_source": "polymarket_clob",
            "fetch_cycle_id": cycle_id,
            "metadata_status": "valid" if markets else "invalid",
            "depth_status": "valid" if not depth_errors else "incomplete",
            "errors": depth_errors + depth_fetch_errors,
        }
        try:
            snapshot = build_market_snapshot(
                decision_timestamp=decision,
                event_date=target,
                location="Hong Kong",
                event_slug=str(slugs_by_kind.get(kind)),
                market_kind=kind,
                markets=markets,
                market_depth=yes_depth,
                market_depth_no=no_depth,
                fetch_cycle_id=cycle_id,
                latest_model_cycle_id=(model_record or {}).get("decision_cycle_id"),
                latest_model_cycle_timestamp=(model_record or {}).get("decision_timestamp"),
                gamma_reference_prices={
                    bucket: {
                        "yes": market.get("yes_price"),
                        "no": market.get("no_price"),
                    }
                    for bucket, market in gamma_refs.items()
                },
                gamma_reference_data=gamma_refs,
                source_status=source_status,
            )
        except Exception as exc:
            run.errors.append({"stage": "market_snapshot_schema", "market_kind": kind, "error": type(exc).__name__})
            logger.exception("Market-only snapshot normalization failed for %s", kind)
            continue
        run.snapshots.append(snapshot)

    if run.snapshots:
        try:
            results = market_store.capture_many(run.snapshots)
            run.capture_results = [
                {
                    "status": result.status,
                    "market_snapshot_id": result.market_snapshot_id,
                    "partition_id": result.partition_id,
                }
                for result in results
            ]
        except Exception as exc:
            run.errors.append({"stage": "market_storage", "error": type(exc).__name__})
            logger.exception("Market-only Layer A capture failed")
    return run


class MarketSnapshotCollector:
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
            report = collect_market_snapshots_once(**self.once_kwargs).as_dict()
        except Exception as exc:
            report = {
                "snapshot_count": 0,
                "captured_count": 0,
                "duplicate_count": 0,
                "errors": [{"stage": "collector", "error": type(exc).__name__}],
            }
            logger.exception("Market-only collector run failed")
        with self._lock:
            self._runs += 1
            self._last_run = report
            errors = report.get("errors") or []
            self._last_error = "; ".join(
                str(item.get("error") or item.get("stage") or "unknown")
                for item in errors
                if isinstance(item, Mapping)
            ) or None
            if report.get("snapshot_count", 0) > 0 and not (
                report.get("errors") and report.get("captured_count", 0) == 0
            ):
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
            # Align every subsequent run to a wall-clock boundary.  A
            # sleep-after-work loop drifts by network latency and eventually
            # skips minute slots during slow Gamma/CLOB responses.
            period = max(1.0, self.interval_seconds)
            now = time.time()
            next_boundary = (int(now // period) + 1) * period
            self._stop.wait(max(0.1, next_boundary - now))

    def start(self) -> None:
        if self.running:
            return
        try:
            store = self.once_kwargs.get("market_store") or get_default_market_store()
            startup_scan = getattr(store, "startup_scan", None)
            if callable(startup_scan):
                startup_scan()
        except Exception:
            logger.exception("Layer A market collector startup recovery failed")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="layer-a-market-collector")
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


_DEFAULT_COLLECTOR: MarketSnapshotCollector | None = None
_DEFAULT_COLLECTOR_LOCK = threading.Lock()


def get_default_market_collector() -> MarketSnapshotCollector:
    global _DEFAULT_COLLECTOR
    if _DEFAULT_COLLECTOR is None:
        with _DEFAULT_COLLECTOR_LOCK:
            if _DEFAULT_COLLECTOR is None:
                _DEFAULT_COLLECTOR = MarketSnapshotCollector()
    return _DEFAULT_COLLECTOR


__all__ = [
    "MarketCollectionRun",
    "MarketSnapshotCollector",
    "collect_market_snapshots_once",
    "get_default_market_collector",
]
