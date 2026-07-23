from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

from app.services.market_depth_service import fetch_market_depths_batch
from layer_a.market_capture import collect_market_snapshots_once
from layer_a.market_schema import build_market_snapshot
from layer_a.market_storage import MarketSnapshotStore
from layer_a.quality import (
    LayerAQualityWorker,
    validate_market_snapshot_for_replay,
)
from layer_a.replay import replay_model_cycle_minute_view


UTC = timezone.utc
DECISION = datetime(2026, 7, 20, 9, 3, tzinfo=UTC)


def _market(condition_id: str, yes_token: str, no_token: str) -> dict:
    return {
        "bucket": "30-31",
        "id": f"market-{condition_id}",
        "conditionId": condition_id,
        "outcomes": ["Yes", "No"],
        "token_id": yes_token,
        "no_token_id": no_token,
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 1.0,
        "market_schema_version": "gamma-market.v1",
    }


def _book(token: str, *, timestamp: datetime = DECISION - timedelta(seconds=2), cycle: str = "fetch-1") -> dict:
    return {
        "asset_id": token,
        "token_id": token,
        "book_timestamp": timestamp.isoformat(),
        "timestamp": timestamp.isoformat(),
        "source_name": "polymarket_clob",
        "fetch_cycle_id": cycle,
        "tick_size": 0.01,
        "minimum_order_size": 1.0,
        "bids": [{"price": 0.35, "size": 5.0}],
        "asks": [{"price": 0.40, "size": 4.0}],
        "validation_errors": [],
    }


def test_batch_depth_mapping_preserves_source_book_timestamp(monkeypatch):
    from app.services import market_depth_service

    monkeypatch.setattr(
        market_depth_service,
        "fetch_order_books_batch",
        lambda _tokens: [
            {**_book("token-b"), "asset_id": "token-b"},
            {**_book("token-a"), "asset_id": "token-a"},
        ],
    )
    result = fetch_market_depths_batch(
        {"bucket-a": "token-a", "bucket-b": "token-b"},
        fetch_cycle_id="cycle-1",
    )
    assert result["bucket-a"]["book_timestamp"] == (DECISION - timedelta(seconds=2)).isoformat()
    assert result["bucket-a"]["fetch_cycle_id"] == "cycle-1"
    assert result["bucket-b"]["asset_id"] == "token-b"


def test_market_capture_resolves_high_and_low_events_independently(monkeypatch, tmp_path):
    from layer_a import market_capture

    slugs = {
        "highest_temperature": "highest-temperature-in-hong-kong-on-july-20-2026",
        "lowest_temperature": "lowest-temperature-in-hong-kong-on-july-20-2026",
    }
    markets = {
        "highest_temperature": [_market("high-condition", "high-yes", "high-no")],
        "lowest_temperature": [_market("low-condition", "low-yes", "low-no")],
    }
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        market_capture,
        "resolve_event_slug_for_kind",
        lambda _target, *, is_min_temp: slugs["lowest_temperature" if is_min_temp else "highest_temperature"],
    )

    def fake_markets(slug: str, *, is_min_temp: bool = False):
        calls.append((slug, is_min_temp))
        return markets["lowest_temperature" if is_min_temp else "highest_temperature"]

    monkeypatch.setattr(market_capture, "fetch_event_markets", fake_markets)
    monkeypatch.setattr(
        market_capture,
        "fetch_market_depths_batch",
        lambda token_map, fetch_cycle_id=None: {
            key: _book(token, cycle=fetch_cycle_id or "cycle-1")
            for key, token in token_map.items()
        },
    )

    class ModelStore:
        def latest_completed_model_record(self, **_kwargs):
            return None

    run = collect_market_snapshots_once(
        target_date=date(2026, 7, 20),
        decision_timestamp=DECISION,
        market_store=MarketSnapshotStore(tmp_path),
        model_store=ModelStore(),
    )

    assert calls == [(slugs["highest_temperature"], False), (slugs["lowest_temperature"], True)]
    assert run.event_slugs == slugs
    assert {snapshot["market_kind"] for snapshot in run.snapshots} == {
        "highest_temperature",
        "lowest_temperature",
    }
    assert {snapshot["market_identity"][0]["condition_id"] for snapshot in run.snapshots} == {
        "high-condition",
        "low-condition",
    }
    assert not any(item.get("error") == "cross_market_kind_identity_collision" for item in run.errors)


def test_replay_validator_rejects_missing_future_and_mismatched_books():
    snapshot = build_market_snapshot(
        decision_timestamp=DECISION,
        capture_timestamp=DECISION,
        event_date="2026-07-20",
        location="Hong Kong",
        event_slug="highest-temperature-in-hong-kong-on-july-20-2026",
        market_kind="highest_temperature",
        markets=[_market("condition-1", "yes-token", "no-token")],
        market_depth={"30-31": _book("yes-token")},
        market_depth_no={"30-31": _book("no-token")},
        fetch_cycle_id="fetch-1",
        latest_model_cycle_id="cycle-1",
        latest_model_cycle_timestamp=DECISION,
    )
    assert validate_market_snapshot_for_replay(snapshot) == []

    missing = deepcopy(snapshot)
    missing["clob_books"][0]["book_timestamp"] = None
    assert any("missing_book_timestamp" in reason for reason in validate_market_snapshot_for_replay(missing))

    future = deepcopy(snapshot)
    future["clob_books"][0]["book_timestamp"] = (DECISION + timedelta(seconds=1)).isoformat()
    assert any("future_book_timestamp" in reason for reason in validate_market_snapshot_for_replay(future))

    mismatch = deepcopy(snapshot)
    mismatch["clob_books"][1]["fetch_cycle_id"] = "fetch-2"
    assert any("yes_no_fetch_cycle_mismatch" in reason for reason in validate_market_snapshot_for_replay(mismatch))


def test_quality_worker_materializes_report_without_upload_configuration(monkeypatch):
    from layer_a import quality

    monkeypatch.setattr(
        quality,
        "build_and_write_daily_quality_report",
        lambda: {
            "report_date": "2026-07-20",
            "gate_passed": False,
            "report_path": "quality/date=2026-07-20/quality_report.json",
        },
    )
    worker = LayerAQualityWorker(interval_minutes=1)
    result = worker.run_once()
    assert result["last_report"]["report_date"] == "2026-07-20"
    assert result["last_report"]["gate_passed"] is False
    assert worker.health_summary()["last_error"] is None


def test_strict_replay_does_not_use_unanchored_legacy_books():
    result = replay_model_cycle_minute_view(
        {
            "decision_cycle_id": "cycle-1",
            "decision_timestamp": DECISION.isoformat(),
            "event_date": "2026-07-20",
            "event_slug": "highest-temperature-in-hong-kong-on-july-20-2026",
            "market_kind": "highest_temperature",
            "market_snapshot_id": "missing-market-anchor",
            "weather_snapshot_id": "missing-weather-anchor",
        },
        [],
        [],
    )
    assert result["market_snapshot_linkage_ok"] is False
    assert result["weather_snapshot_linkage_ok"] is False
    assert not any(row.get("market_snapshot_id") for row in result["minute_rows"])
    assert not any(row.get("weather_snapshot_id") for row in result["minute_rows"])
