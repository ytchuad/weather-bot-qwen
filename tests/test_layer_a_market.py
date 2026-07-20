from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from layer_a.export import export_layer_a
from layer_a.market_capture import collect_market_snapshots_once
from layer_a.market_schema import build_market_snapshot
from layer_a.market_storage import MarketSnapshotStore
from layer_a.replay import load_market_snapshot_records, replay_market_signal
from layer_a.schema import build_layer_a_record
from layer_a.storage import LayerAStore
from layer_a.upload import DatasetUploader


UTC = timezone.utc
HKT = timezone(timedelta(hours=8))
SLUG = "highest-temperature-in-hong-kong-on-july-20-2026"


def _book(token: str, decision: datetime, cycle: str = "market-fetch-1") -> dict:
    return {
        "asset_id": token,
        "timestamp": (decision - timedelta(seconds=2)).isoformat(),
        "source_name": "polymarket_clob",
        "fetch_cycle_id": cycle,
        "tick_size": 0.01,
        "minimum_order_size": 1.0,
        "bids": [{"price": 0.35, "size": 5.0}, {"price": 0.30, "size": 10.0}],
        "asks": [{"price": 0.40, "size": 4.0}, {"price": 0.45, "size": 8.0}],
        "validation_errors": [],
    }


def _market() -> dict:
    return {
        "bucket": "30-31",
        "id": "market-1",
        "conditionId": "condition-1",
        "outcomes": ["Yes", "No"],
        "token_id": "yes-token-1",
        "no_token_id": "no-token-1",
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 1.0,
        "market_schema_version": "gamma-market.v1",
        "yes_price": 0.38,
    }


def _market_snapshot(
    decision: datetime,
    *,
    model_id: str = "model-cycle-0900",
    model_time: datetime | None = None,
    event_date: str = "2026-07-20",
) -> dict:
    model_time = model_time or datetime(2026, 7, 20, 9, tzinfo=UTC)
    return build_market_snapshot(
        decision_timestamp=decision,
        event_date=event_date,
        location="Hong Kong",
        event_slug=SLUG,
        market_kind="highest_temperature",
        markets=[_market()],
        market_depth={"30-31": _book("yes-token-1", decision)},
        market_depth_no={"30-31": _book("no-token-1", decision)},
        fetch_cycle_id="market-fetch-1",
        latest_model_cycle_id=model_id,
        latest_model_cycle_timestamp=model_time,
        gamma_reference_prices={"30-31": {"yes": 0.38, "no": 0.62}},
        gamma_reference_data={"30-31": {"source": "gamma-reference-only"}},
    )


def _model_record(decision: datetime = datetime(2026, 7, 20, 9, tzinfo=UTC)) -> dict:
    return build_layer_a_record(
        {
            "decision_timestamp": decision,
            "event_date": "2026-07-20",
            "event_slug": SLUG,
            "market_kind": "highest_temperature",
            "weather_state": {
                "observations": {"temperature": 30.0},
                "max_so_far": 30.0,
                "min_so_far": 25.0,
                "status": {"source": "test"},
            },
            "model_states": {
                "model_a": {
                    "model_name": "model_a",
                    "model_version": "v2",
                    "artifact_identity": "artifact-1",
                    "feature_spec": "features/model_a.json",
                    "numeric_features": {"temperature": 30.0},
                    "diagnostic_features": {"temperature": 30.0},
                    "point_prediction": 30.2,
                    "full_bucket_probabilities": {"30-31": 0.8},
                    "model_input_status_summary": {"status_contract_version": "test.v1"},
                }
            },
            "markets": [_market()],
            "market_depth": {"30-31": _book("yes-token-1", decision)},
            "market_depth_no": {"30-31": _book("no-token-1", decision)},
        }
    )


def test_market_snapshot_preserves_identity_books_and_model_age():
    snapshot = _market_snapshot(datetime(2026, 7, 20, 9, 3, tzinfo=UTC))
    assert snapshot["schema_version"] == "layer_a.market.v1"
    assert snapshot["latest_model_cycle_id"] == "model-cycle-0900"
    assert snapshot["model_age_seconds"] == 180.0
    assert snapshot["market_identity"][0]["condition_id"] == "condition-1"
    assert snapshot["market_identity"][0]["yes_token_id"] == "yes-token-1"
    assert {book["token_side"] for book in snapshot["clob_books"]} == {"YES", "NO"}
    assert all(len(book["bids"]) == 2 and len(book["asks"]) == 2 for book in snapshot["clob_books"])
    assert snapshot["completeness"]["replay_eligible_for_market_replay"] is True


def test_market_store_appends_one_hourly_partition_and_deduplicates(tmp_path):
    store = MarketSnapshotStore(tmp_path)
    base = datetime(2030, 7, 21, 1, 10, tzinfo=HKT).astimezone(UTC)
    first = _market_snapshot(base, event_date="2030-07-21")
    second = _market_snapshot(base + timedelta(minutes=1), event_date="2030-07-21")
    results = store.capture_many([first, second])
    assert [result.status for result in results] == ["captured", "captured"]
    assert len(list(tmp_path.rglob("*.jsonl.tmp"))) == 1
    assert not list(tmp_path.rglob("*.jsonl.zst"))
    assert store.capture(first).status == "duplicate"

    assert store.close_all() == {"closed": 1, "failed": 0}
    partitions = store.scan()
    assert len(partitions) == 1
    assert partitions[0].status == "complete"
    assert partitions[0].manifest["snapshot_count"] == 2
    assert len(store.read_snapshot_records()) == 2
    assert len(list(tmp_path.rglob("snapshots-*.jsonl.zst"))) == 1
    assert len(list(tmp_path.rglob("manifest-*.json"))) == 1


def test_latest_model_lookup_is_before_market_time(tmp_path):
    model_store = LayerAStore(tmp_path / "model")
    model = _model_record()
    model_store.capture(model)
    latest = model_store.latest_completed_model_record(
        event_date="2026-07-20",
        event_slug=SLUG,
        market_kind="highest_temperature",
        before_timestamp=datetime(2026, 7, 20, 9, 4, tzinfo=UTC),
    )
    assert latest is not None
    assert latest["decision_cycle_id"] == model["decision_cycle_id"]


def test_market_collector_uses_one_independent_clob_batch(monkeypatch, tmp_path):
    decision = datetime(2026, 7, 20, 9, 3, tzinfo=UTC)
    markets = [_market()]
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        "layer_a.market_capture.fetch_event_markets",
        lambda _slug, is_min_temp=False: markets if not is_min_temp else [],
    )

    def fake_fetch(token_map, fetch_cycle_id=None):
        calls.append(dict(token_map))
        return {
            key: _book(token, decision, cycle=fetch_cycle_id or "test")
            for key, token in token_map.items()
        }

    monkeypatch.setattr("layer_a.market_capture.fetch_market_depths_batch", fake_fetch)

    class FakeModelStore:
        def latest_completed_model_record(self, **_kwargs):
            return {
                "decision_cycle_id": "model-cycle-0900",
                "decision_timestamp": "2026-07-20T09:00:00+00:00",
            }

    run = collect_market_snapshots_once(
        target_date=date(2026, 7, 20),
        event_slug=SLUG,
        decision_timestamp=decision,
        market_kinds=(False,),
        market_store=MarketSnapshotStore(tmp_path),
        model_store=FakeModelStore(),
    )
    assert len(calls) == 1
    assert set(calls[0]) == {
        "highest_temperature:30-31:YES",
        "highest_temperature:30-31:NO",
    }
    assert len(run.snapshots) == 1
    assert run.snapshots[0]["latest_model_cycle_id"] == "model-cycle-0900"
    assert run.snapshots[0]["source_status"]["collector"] == "layer_a_market_only"


def test_one_model_signal_replays_each_minute_until_next_model_cycle():
    model = _model_record()
    snapshots = [
        _market_snapshot(
            datetime(2026, 7, 20, 9, minute, tzinfo=UTC),
            model_id=model["decision_cycle_id"],
        )
        for minute in (1, 2, 3, 4, 5)
    ]
    replay = replay_market_signal(
        model,
        snapshots,
        next_model_timestamp=datetime(2026, 7, 20, 9, 5, tzinfo=UTC),
        strategy_a_threshold=0.1,
        requested_shares=1.0,
    )
    assert replay["books_evaluated"] == 4
    assert replay["snapshot_ids"] == [snapshot["market_snapshot_id"] for snapshot in snapshots[:4]]
    assert replay["valid_clob_books_replayed"] == 8
    assert replay["all_linked_one_minute_books_replayed"] is True


def test_export_archive_contains_closed_market_partitions(tmp_path):
    model_store = LayerAStore(tmp_path / "model")
    market_store = MarketSnapshotStore(tmp_path / "market")
    model_store.capture(_model_record())
    market_store.capture(_market_snapshot(datetime(2026, 7, 20, 9, 1, tzinfo=UTC)))
    market_store.close_all()
    archive = tmp_path / "export.zip"
    result = export_layer_a(
        store=model_store,
        market_store=market_store,
        output=archive,
        date_value="2026-07-20",
        verify_checksums=True,
    )
    assert result["market_snapshot_count"] == 1
    assert result["market_partition_count"] == 1
    assert load_market_snapshot_records(archive)[0]["schema_version"] == "layer_a.market.v1"


def test_market_partition_upload_uses_separate_dataset_prefix(tmp_path):
    class Hub:
        def __init__(self):
            self.paths: set[str] = set()

        def file_exists(self, *, filename, **_kwargs):
            return filename in self.paths

        def upload_file(self, *, path_in_repo, **_kwargs):
            self.paths.add(path_in_repo)
            return object()

    hub = Hub()
    uploader = DatasetUploader("private/market", "token", api=hub, sleep_fn=lambda _seconds: None)
    store = MarketSnapshotStore(
        tmp_path,
        uploader=uploader,
        auto_upload=True,
        upload_interval_minutes=0,
    )
    snapshot = _market_snapshot(
        datetime(2030, 7, 21, 1, 10, tzinfo=HKT).astimezone(UTC),
        event_date="2030-07-21",
    )
    store.capture(snapshot)
    store.close_all()
    assert hub.paths
    assert all(path.startswith("layer_a_market/") for path in hub.paths)
