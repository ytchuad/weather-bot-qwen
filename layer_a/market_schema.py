"""Strategy-independent one-minute market snapshot contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from .schema import (
    _iso_datetime,
    _normalise_book,
    _normalise_market,
    _parse_datetime,
    jsonable,
)


MARKET_SCHEMA_VERSION = "layer_a.market.v1"
DEFAULT_BOOK_FRESHNESS_SECONDS = 60.0
_PROHIBITED_FIELDS = {
    "account",
    "account_id",
    "cash",
    "cash_balance",
    "capital",
    "clob_would_trade",
    "current_paper_positions",
    "fill",
    "fills",
    "legacy_would_trade",
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


class MarketSnapshotSchemaError(ValueError):
    """Raised when a market-only snapshot violates its boundary."""


def make_market_snapshot_id(
    decision_timestamp: Any,
    *,
    event_date: str | date,
    location: str,
    event_slug: str,
    market_kind: str,
    cadence_minutes: int = 1,
) -> str:
    """Return an account-independent ID for one market sampling slot."""
    from .schema import make_decision_cycle_id

    cycle_id = make_decision_cycle_id(
        decision_timestamp,
        event_date=event_date,
        location=location,
        event_slug=event_slug,
        market_kind=market_kind,
        cadence_minutes=max(1, int(cadence_minutes)),
    )
    return f"ms-{cycle_id[3:]}"


def _add_reason(reasons: list[str], value: str) -> None:
    if value not in reasons:
        reasons.append(value)


def _assess_completeness(
    *,
    market_identity: Sequence[Mapping[str, Any]],
    books: Sequence[Mapping[str, Any]],
    latest_model_cycle_id: str | None,
    model_age_seconds: float | None,
    decision_timestamp: str | None,
    capture_timestamp: str | None,
    max_book_age_seconds: float = DEFAULT_BOOK_FRESHNESS_SECONDS,
    event_slug: str | None = None,
    market_kind: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    market_identity_complete = bool(market_identity)
    token_identity_complete = market_identity_complete
    for index, market in enumerate(market_identity):
        prefix = f"market_identity[{index}]"
        for field in (
            "market_id",
            "condition_id",
            "bucket",
            "explicit_outcomes",
            "tick_size",
            "minimum_order_size",
            "market_schema_version",
        ):
            if market.get(field) in (None, "", []):
                market_identity_complete = False
                _add_reason(reasons, f"{prefix}.{field}")
        outcomes = market.get("explicit_outcomes")
        normalized_outcomes = [str(item).lower() for item in outcomes] if isinstance(outcomes, list) else []
        if normalized_outcomes != ["yes", "no"]:
            market_identity_complete = False
            _add_reason(reasons, f"{prefix}.explicit_outcomes")
        for field in ("yes_token_id", "no_token_id"):
            if market.get(field) in (None, ""):
                token_identity_complete = False
                _add_reason(reasons, f"{prefix}.{field}")

    books_by_key = {
        (str(book.get("bucket")), str(book.get("token_side", "")).upper()): book
        for book in books
        if isinstance(book, Mapping)
    }
    depth_pair_complete = True
    book_timestamp_complete = True
    book_freshness_complete = True
    token_mapping_complete = True
    fetch_cycle_coherent = True
    yes_no_fetch_cycle_coherent = True
    source_complete = True
    fetch_cycles: set[str] = set()
    fetch_cycles_by_bucket: dict[str, set[str]] = {}
    expected_buckets = [str(item.get("bucket")) for item in market_identity]
    decision_dt = _parse_datetime(decision_timestamp, naive_timezone=timezone.utc)
    capture_dt = _parse_datetime(capture_timestamp, naive_timezone=timezone.utc)
    market_by_bucket = {
        str(item.get("bucket")): item
        for item in market_identity
        if isinstance(item, Mapping) and item.get("bucket")
    }
    for bucket in expected_buckets:
        for side in ("YES", "NO"):
            prefix = f"clob_books[{bucket}/{side}]"
            book = books_by_key.get((bucket, side))
            if book is None:
                depth_pair_complete = False
                book_timestamp_complete = False
                book_freshness_complete = False
                fetch_cycle_coherent = False
                source_complete = False
                _add_reason(reasons, prefix)
                continue
            if not book.get("token_id") or not book.get("asset_id"):
                token_identity_complete = False
                _add_reason(reasons, f"{prefix}.token_identity")
            expected_token = market_by_bucket.get(bucket, {}).get(
                "yes_token_id" if side == "YES" else "no_token_id"
            )
            if expected_token and (
                str(book.get("asset_id")) != str(expected_token)
                or str(book.get("token_id")) != str(expected_token)
            ):
                token_mapping_complete = False
                token_identity_complete = False
                _add_reason(reasons, f"{prefix}.unknown_token_mapping")
            if book.get("validation_status") != "valid":
                depth_pair_complete = False
                _add_reason(reasons, f"{prefix}.validation_status")
            if not isinstance(book.get("bids"), list) or not isinstance(book.get("asks"), list):
                depth_pair_complete = False
                _add_reason(reasons, f"{prefix}.depth")
            book_dt = _parse_datetime(book.get("book_timestamp"), naive_timezone=timezone.utc)
            book_age = book.get("book_age_seconds")
            timestamp_ok = bool(book_dt is not None)
            if decision_dt is not None and book_dt is not None:
                timestamp_ok = timestamp_ok and book_dt <= decision_dt
            if capture_dt is not None and book_dt is not None:
                timestamp_ok = timestamp_ok and book_dt <= capture_dt
            age_ok = isinstance(book_age, (int, float)) and math.isfinite(float(book_age)) and 0 <= float(book_age) <= max_book_age_seconds
            if not timestamp_ok:
                book_timestamp_complete = False
                if book_dt is not None and ((decision_dt is not None and book_dt > decision_dt) or (capture_dt is not None and book_dt > capture_dt)):
                    _add_reason(reasons, f"{prefix}.future_book_timestamp")
                else:
                    _add_reason(reasons, f"{prefix}.book_timestamp")
            if not age_ok:
                book_freshness_complete = False
                _add_reason(reasons, f"{prefix}.book_age_seconds")
            if timestamp_ok and decision_dt is not None and book_dt is not None and isinstance(book_age, (int, float)):
                expected_age = (decision_dt - book_dt).total_seconds()
                if abs(expected_age - float(book_age)) > 1.0:
                    book_freshness_complete = False
                    _add_reason(reasons, f"{prefix}.book_age_mismatch")
            if not book.get("source_name"):
                source_complete = False
                _add_reason(reasons, f"{prefix}.source_name")
            cycle = book.get("fetch_cycle_id")
            if not cycle:
                fetch_cycle_coherent = False
                _add_reason(reasons, f"{prefix}.fetch_cycle_id")
            else:
                fetch_cycles.add(str(cycle))
                fetch_cycles_by_bucket.setdefault(bucket, set()).add(str(cycle))
            if book.get("tick_size") in (None, ""):
                depth_pair_complete = False
                _add_reason(reasons, f"{prefix}.tick_size")
            if book.get("minimum_order_size") in (None, ""):
                depth_pair_complete = False
                _add_reason(reasons, f"{prefix}.minimum_order_size")

    if len(fetch_cycles) > 1:
        fetch_cycle_coherent = False
        _add_reason(reasons, "fetch_cycle_id_incoherent")
    for bucket, cycles in fetch_cycles_by_bucket.items():
        if len(cycles) > 1:
            yes_no_fetch_cycle_coherent = False
            fetch_cycle_coherent = False
            _add_reason(reasons, f"yes_no_fetch_cycle_mismatch:{bucket}")
    if event_slug:
        slug_kind = None
        slug_value = str(event_slug).lower()
        if slug_value.startswith("highest-temperature-"):
            slug_kind = "highest_temperature"
        elif slug_value.startswith("lowest-temperature-"):
            slug_kind = "lowest_temperature"
        if slug_kind is not None and market_kind and slug_kind != str(market_kind):
            market_identity_complete = False
            _add_reason(reasons, "event_slug_market_kind_mismatch")
    condition_ids = [str(item.get("condition_id")) for item in market_identity if item.get("condition_id")]
    if len(condition_ids) != len(set(condition_ids)):
        market_identity_complete = False
        _add_reason(reasons, "condition_id_duplicate")
    model_link_complete = bool(
        latest_model_cycle_id
        and isinstance(model_age_seconds, (int, float))
        and math.isfinite(float(model_age_seconds))
        and float(model_age_seconds) >= 0
    )
    if not model_link_complete:
        _add_reason(reasons, "latest_model_cycle_link")
    replay_eligible = bool(
        market_identity_complete
        and token_identity_complete
        and depth_pair_complete
        and book_timestamp_complete
        and book_freshness_complete
        and token_mapping_complete
        and fetch_cycle_coherent
        and yes_no_fetch_cycle_coherent
        and source_complete
        and model_link_complete
    )
    if not replay_eligible:
        _add_reason(reasons, "market_replay_requirements_not_met")
    return {
        "market_identity_complete": market_identity_complete,
        "token_identity_complete": token_identity_complete,
        "depth_pair_complete": depth_pair_complete,
        "book_timestamp_complete": book_timestamp_complete,
        "book_freshness_complete": book_freshness_complete,
        "token_mapping_complete": token_mapping_complete,
        "fetch_cycle_coherent": fetch_cycle_coherent,
        "yes_no_fetch_cycle_coherent": yes_no_fetch_cycle_coherent,
        "source_complete": source_complete,
        "model_link_complete": model_link_complete,
        "replay_eligible_for_market_replay": replay_eligible,
        "missing_fields": reasons,
        "rejection_reasons": list(reasons),
    }


def build_market_snapshot(
    *,
    decision_timestamp: Any,
    event_date: str | date,
    location: str,
    event_slug: str,
    market_kind: str,
    markets: Sequence[Mapping[str, Any]],
    market_depth: Mapping[str, Any],
    market_depth_no: Mapping[str, Any],
    fetch_cycle_id: str | None,
    latest_model_cycle_id: str | None = None,
    latest_model_cycle_timestamp: Any = None,
    gamma_reference_prices: Mapping[str, Any] | None = None,
    gamma_reference_data: Mapping[str, Any] | None = None,
    source_status: Mapping[str, Any] | None = None,
    market_snapshot_id: str | None = None,
    capture_timestamp: Any = None,
    max_book_age_seconds: float = DEFAULT_BOOK_FRESHNESS_SECONDS,
) -> dict[str, Any]:
    """Normalize one strategy-independent minute of market/depth state."""
    decision_iso = _iso_datetime(decision_timestamp, naive_timezone=timezone.utc)
    if decision_iso is None:
        raise MarketSnapshotSchemaError("decision_timestamp is invalid")
    capture_value = capture_timestamp
    if capture_value is None:
        capture_value = datetime.now(timezone.utc)
        decision_dt = _parse_datetime(decision_iso, naive_timezone=timezone.utc)
        if decision_dt is not None and capture_value < decision_dt:
            # Historical/test builders may intentionally construct a future
            # decision slot.  Never create a structurally impossible capture
            # timestamp before that decision; live callers pass the actual
            # capture time explicitly when this distinction matters.
            capture_value = decision_dt
    capture_iso = _iso_datetime(capture_value, naive_timezone=timezone.utc)
    if capture_iso is None:
        raise MarketSnapshotSchemaError("capture_timestamp is invalid")
    event_date_iso = event_date.isoformat() if isinstance(event_date, date) else str(event_date)
    market_records: list[dict[str, Any]] = []
    books: list[dict[str, Any]] = []
    for market in markets:
        bucket = str(market.get("bucket") or "")
        if not bucket:
            continue
        yes_book = market_depth.get(bucket) if isinstance(market_depth, Mapping) else None
        no_book = market_depth_no.get(bucket) if isinstance(market_depth_no, Mapping) else None
        normalized_market = _normalise_market(market, yes_book=yes_book, no_book=no_book)
        market_records.append(normalized_market)
        books.append(_normalise_book(bucket, "YES", normalized_market, yes_book, decision_iso=decision_iso))
        books.append(_normalise_book(bucket, "NO", normalized_market, no_book, decision_iso=decision_iso))

    model_timestamp_iso = _iso_datetime(latest_model_cycle_timestamp, naive_timezone=timezone.utc)
    model_age_seconds = None
    if model_timestamp_iso is not None:
        model_dt = _parse_datetime(model_timestamp_iso, naive_timezone=timezone.utc)
        decision_dt = _parse_datetime(decision_iso, naive_timezone=timezone.utc)
        if model_dt is not None and decision_dt is not None:
            age = (decision_dt - model_dt).total_seconds()
            if age >= 0:
                model_age_seconds = age
            else:
                latest_model_cycle_id = None
                model_timestamp_iso = None

    snapshot_id = market_snapshot_id or make_market_snapshot_id(
        decision_iso,
        event_date=event_date_iso,
        location=location,
        event_slug=event_slug,
        market_kind=market_kind,
    )
    record = {
        "market_snapshot_id": str(snapshot_id),
        "schema_version": MARKET_SCHEMA_VERSION,
        "decision_timestamp": decision_iso,
        "capture_timestamp": capture_iso,
        "event_date": event_date_iso,
        "location": str(location),
        "market_kind": str(market_kind),
        "event_slug": str(event_slug),
        "market_identity": market_records,
        "clob_books": books,
        "fetch_cycle_id": fetch_cycle_id,
        "latest_model_cycle_id": latest_model_cycle_id,
        "latest_model_cycle_timestamp": model_timestamp_iso,
        "model_age_seconds": model_age_seconds,
        "gamma_reference_prices": jsonable(gamma_reference_prices or {}),
        "gamma_reference_data": jsonable(gamma_reference_data or {}),
        "source_status": jsonable(source_status or {}),
    }
    record["completeness"] = _assess_completeness(
        market_identity=market_records,
        books=books,
        latest_model_cycle_id=latest_model_cycle_id,
        model_age_seconds=model_age_seconds,
        decision_timestamp=decision_iso,
        capture_timestamp=capture_iso,
        max_book_age_seconds=max_book_age_seconds,
        event_slug=event_slug,
        market_kind=market_kind,
    )
    validate_market_snapshot(record)
    return record


def validate_market_snapshot(record: Mapping[str, Any]) -> None:
    """Validate the market-only envelope and strategy-independent boundary."""
    if not isinstance(record, Mapping):
        raise MarketSnapshotSchemaError("market snapshot must be a mapping")
    for field in (
        "market_snapshot_id",
        "schema_version",
        "decision_timestamp",
        "capture_timestamp",
        "event_date",
        "location",
        "market_kind",
        "event_slug",
        "market_identity",
        "clob_books",
        "fetch_cycle_id",
        "latest_model_cycle_id",
        "latest_model_cycle_timestamp",
        "model_age_seconds",
        "gamma_reference_prices",
        "gamma_reference_data",
        "source_status",
        "completeness",
    ):
        if field not in record:
            raise MarketSnapshotSchemaError(f"required market snapshot field is missing: {field}")
    if record.get("schema_version") != MARKET_SCHEMA_VERSION:
        raise MarketSnapshotSchemaError(f"unsupported market snapshot schema: {record.get('schema_version')!r}")

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                field = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if field in _PROHIBITED_FIELDS:
                    raise MarketSnapshotSchemaError(
                        f"strategy/account field is not allowed in market snapshot: {child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(record)
    try:
        json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MarketSnapshotSchemaError(f"market snapshot is not JSON serializable: {exc}") from exc


__all__ = [
    "MARKET_SCHEMA_VERSION",
    "DEFAULT_BOOK_FRESHNESS_SECONDS",
    "MarketSnapshotSchemaError",
    "build_market_snapshot",
    "make_market_snapshot_id",
    "validate_market_snapshot",
]
