from __future__ import annotations

import zipfile
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.api.layer_a import _require_admin
from features.input_status import InputStatus
from layer_a.export import export_layer_a
from layer_a.historical_store import HistoricalStore
from layer_a.market_schema import build_market_snapshot
from layer_a.market_storage import MarketSnapshotStore
from layer_a.minute_view import build_minute_view
from layer_a.replay import replay_model_cycle_minute_view
from layer_a.schema import build_layer_a_record
from layer_a.storage import LayerAStore
from layer_a.weather_capture import WeatherSnapshotCollector, collect_weather_snapshots_once
from layer_a.weather_schema import build_weather_snapshot, make_weather_snapshot_id
from layer_a.weather_storage import WeatherSnapshotStore

UTC = timezone.utc
HKT = timezone(timedelta(hours=8))
EVENT_DATE = "2026-07-20"
SLUG = "highest-temperature-in-hong-kong-on-july-20-2026"


def _status(value, source_timestamp=None, *, fallback=False, missing=None):
    return InputStatus.from_value(
        value,
        source_timestamp=source_timestamp,
        decision_timestamp="2026-07-20T10:03:00+00:00",
        source_name="test-weather",
        is_missing=missing,
        is_fallback=fallback,
        fallback_method="unavailable" if fallback else None,
    ).to_dict()


def _weather(ts="2026-07-20T10:03:00+00:00", temp=30.0, source="2026-07-20T10:01:00+00:00"):
    return build_weather_snapshot(
        snapshot_timestamp=ts,
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state={
            "observations": {"temperature": temp, "humidity": 70.0},
            "max_so_far": temp,
            "min_so_far": 25.0,
            "status": {
                "temperature": _status(temp, source),
                "humidity": _status(70.0, source),
                "max_so_far": _status(temp, source),
                "min_so_far": _status(25.0, source),
            },
        },
        latest_model_cycle_id="model-0900",
        model_cycle_timestamp="2026-07-20T10:00:00+00:00",
    )


def _book(token, decision, cycle="fetch-1"):
    return {
        "asset_id": token,
        "timestamp": (decision - timedelta(seconds=2)).isoformat(),
        "source_name": "polymarket_clob",
        "fetch_cycle_id": cycle,
        "tick_size": 0.01,
        "minimum_order_size": 1.0,
        "bids": [{"price": 0.35, "size": 5.0}],
        "asks": [{"price": 0.40, "size": 5.0}],
        "validation_errors": [],
    }


def _market(ts="2026-07-20T10:01:00+00:00"):
    decision = datetime.fromisoformat(ts)
    market = {
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
    return build_market_snapshot(
        decision_timestamp=decision,
        event_date=EVENT_DATE,
        location="Hong Kong",
        event_slug=SLUG,
        market_kind="highest_temperature",
        markets=[market],
        market_depth={"30-31": _book("yes-token-1", decision)},
        market_depth_no={"30-31": _book("no-token-1", decision)},
        fetch_cycle_id="fetch-1",
        latest_model_cycle_id="model-0900",
        latest_model_cycle_timestamp="2026-07-20T10:00:00+00:00",
        gamma_reference_prices={"30-31": {"yes": 0.38, "no": 0.62}},
    )


def _model(ts="2026-07-20T10:00:00+00:00", cycle=None):
    decision = datetime.fromisoformat(ts)
    market = {
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
    return build_layer_a_record(
        {
            "decision_timestamp": decision,
            "decision_cycle_id": cycle or f"model-{decision.strftime('%H%M')}",
            "event_date": EVENT_DATE,
            "event_slug": SLUG,
            "market_kind": "highest_temperature",
            "weather_state": {"observations": {"temperature": 30.0}, "max_so_far": 30.0, "min_so_far": 25.0},
            "model_states": {
                "model_a": {
                    "model_name": "model_a",
                    "model_version": "test",
                    "artifact_identity": "test-artifact",
                    "feature_spec": "test.json",
                    "numeric_features": {"temperature": 30.0},
                    "point_prediction": 30.2,
                    "full_bucket_probabilities": {"30-31": 0.8},
                }
            },
            "markets": [market],
            "market_depth": {"30-31": _book("yes-token-1", decision)},
            "market_depth_no": {"30-31": _book("no-token-1", decision)},
        }
    )


def test_weather_snapshot_id_is_deterministic():
    args = dict(event_date=EVENT_DATE, location="Hong Kong")
    assert make_weather_snapshot_id("2026-07-20T10:03:59+00:00", **args) == make_weather_snapshot_id(
        "2026-07-20T10:03:01+00:00", **args
    )


def test_weather_zero_is_not_missing_and_missing_age_is_null():
    observed = _weather(temp=0.0)
    assert observed["temperature_current"] == 0.0
    assert observed["observation_status"]["temperature_current"]["is_missing"] is False
    missing = build_weather_snapshot(
        snapshot_timestamp="2026-07-20T10:03:00+00:00",
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state={"observations": {"temperature": 0.0}, "status": {"temperature": _status(0.0, None)}},
    )
    assert missing["observation_status"]["temperature_current"]["age_seconds"] is None
    assert missing["temperature_current"] == 0.0


def test_weather_source_age_and_repeated_observation_are_truthful():
    first = _weather(ts="2026-07-20T10:03:00+00:00", source="2026-07-20T10:01:00+00:00")
    second = _weather(ts="2026-07-20T10:04:00+00:00", source="2026-07-20T10:01:00+00:00")
    assert first["observation_status"]["temperature_current"]["age_seconds"] == 120.0
    assert second["observation_status"]["temperature_current"]["age_seconds"] == 180.0
    assert second["observation_status"]["temperature_current"]["source_timestamp"] == first["observation_status"]["temperature_current"]["source_timestamp"]


def test_weather_collector_reads_state_only_and_links_stored_model(monkeypatch, tmp_path):
    calls = []
    state = {"observations": {"temperature": 30.0}, "max_so_far": 30.0, "min_so_far": 25.0}
    monkeypatch.setattr("layer_a.weather_capture.get_intraday_state", lambda key: state)
    monkeypatch.setattr(
        "app.services.model_service.run_all_models",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("weather collector must not run models")),
    )
    monkeypatch.setattr(
        "app.services.market_depth_service.fetch_market_depths_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("weather collector must not fetch CLOB")),
    )

    class ModelStore:
        def latest_completed_model_record(self, **kwargs):
            calls.append(kwargs)
            return {"decision_cycle_id": "cycle-1", "decision_timestamp": "2026-07-20T10:00:00+00:00"}

    run = collect_weather_snapshots_once(
        target_date=date(2026, 7, 20),
        event_slug=SLUG,
        decision_timestamp=datetime(2026, 7, 20, 10, 3, tzinfo=UTC),
        weather_store=WeatherSnapshotStore(tmp_path),
        model_store=ModelStore(),
    )
    assert run.snapshots[0]["latest_model_cycle_id"] == "cycle-1"
    assert calls
    assert run.snapshots[0]["source_status"]["clob_fetched"] is False


def test_weather_collector_backfills_each_minute_from_hko_csv_buffer(tmp_path):
    state = {
        "df_today": pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2026-07-20 20:01",
                        "2026-07-20 20:02",
                        "2026-07-20 20:03",
                    ]
                ),
                "temp": [29.1, 29.2, 29.0],
                "rh": [81.0, 82.0, 83.0],
            }
        )
    }

    class ModelStore:
        def latest_completed_model_record(self, **_kwargs):
            return None

    run = collect_weather_snapshots_once(
        target_date=date(2026, 7, 20),
        event_slug=SLUG,
        decision_timestamp=datetime(2026, 7, 20, 12, 4, tzinfo=UTC),
        weather_store=WeatherSnapshotStore(tmp_path),
        model_store=ModelStore(),
        state_provider=lambda _key: state,
    )

    assert len(run.snapshots) == 3
    assert [item["snapshot_timestamp"] for item in run.snapshots] == [
        "2026-07-20T12:01:00+00:00",
        "2026-07-20T12:02:00+00:00",
        "2026-07-20T12:03:00+00:00",
    ]
    assert [item["temperature_current"] for item in run.snapshots] == [29.1, 29.2, 29.0]
    assert all(item["source_status"]["buffer_backfill"] is True for item in run.snapshots)
    assert [
        item["observation_status"]["temperature_current"]["source_timestamp"]
        for item in run.snapshots
    ] == [
        "2026-07-20T12:01:00+00:00",
        "2026-07-20T12:02:00+00:00",
        "2026-07-20T12:03:00+00:00",
    ]


def test_buffer_backfill_corrects_an_existing_capture_minute(tmp_path):
    weather_store = WeatherSnapshotStore(tmp_path / "weather")
    original = _weather(ts="2026-07-20T10:01:00+00:00", temp=30.0)
    corrected = _weather(ts="2026-07-20T10:01:00+00:00", temp=29.1)
    corrected["source_status"] = {"buffer_backfill": True}

    assert weather_store.capture(original).status == "captured"
    assert weather_store.close_all() == {"closed": 1, "failed": 0}
    assert weather_store.capture(corrected).status == "captured"

    history = HistoricalStore(
        local_store=LayerAStore(tmp_path / "model"),
        local_market_store=MarketSnapshotStore(tmp_path / "market"),
        local_weather_store=weather_store,
        cache_dir=tmp_path / "cache",
        auto_refresh=False,
    )
    rows = history.minute_history(date_value=EVENT_DATE)
    assert len(rows) == 1
    assert rows[0]["actual_temperature"] == 29.1


def test_weather_failure_isolated_and_collector_does_not_raise(tmp_path):
    collector = WeatherSnapshotCollector(
        state_provider=lambda _key: (_ for _ in ()).throw(RuntimeError("weather down")),
        weather_store=WeatherSnapshotStore(tmp_path),
    )
    report = collector.run_once()
    assert report["snapshot_count"] == 0
    assert report["errors"]


def test_market_failure_isolated_from_weather_collector(monkeypatch):
    from layer_a.market_capture import MarketSnapshotCollector

    monkeypatch.setattr(
        "layer_a.market_capture.collect_market_snapshots_once",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("clob down")),
    )
    report = MarketSnapshotCollector().run_once()
    assert report["snapshot_count"] == 0
    assert report["errors"]


def test_ten_minute_market_chunk_closes(tmp_path):
    store = MarketSnapshotStore(tmp_path)
    store.capture(_market("2026-07-20T10:01:00+00:00"))
    assert store.close_due(datetime(2026, 7, 20, 10, 10, tzinfo=UTC)) == {"closed": 1, "failed": 0}
    info = store.scan()[0]
    assert info.status == "complete"
    assert info.manifest["partition_minutes"] == 10
    assert info.checksum_valid is True


def test_ten_minute_weather_chunk_closes_atomically_with_checksum(tmp_path):
    store = WeatherSnapshotStore(tmp_path)
    store.capture(_weather(ts="2026-07-20T10:01:00+00:00"))
    assert list(tmp_path.rglob("*.jsonl.tmp"))
    result = store.close_due(datetime(2026, 7, 20, 10, 10, tzinfo=UTC))
    assert result == {"closed": 1, "failed": 0}
    info = store.scan()[0]
    assert info.status == "complete"
    assert info.checksum_valid is True
    assert info.manifest["partition_minutes"] == 10
    assert not list(tmp_path.rglob("*.jsonl.tmp"))


def test_weather_startup_recovery_keeps_incomplete_chunk(tmp_path):
    directory = tmp_path / "date=2026-07-20" / "hour=10" / "minute=00"
    directory.mkdir(parents=True)
    raw = directory / "snapshots-crash.jsonl.tmp"
    raw.write_text("not-json\n", encoding="utf-8")
    info = WeatherSnapshotStore(tmp_path).scan()[0]
    assert info.status == "incomplete"
    assert raw.exists()


def test_backward_asof_model_join_and_minute_output():
    models = [_model("2026-07-20T10:00:00+00:00", "cycle-0900"), _model("2026-07-20T10:05:00+00:00", "cycle-0905")]
    markets = [_market(f"2026-07-20T10:0{minute}:00+00:00") for minute in range(1, 5)]
    weather = [_weather(f"2026-07-20T10:0{minute}:00+00:00", 30 + minute) for minute in range(1, 5)]
    rows = build_minute_view(models, markets, weather)
    market_rows = [row for row in rows if row["market_snapshot_id"]]
    assert [row["model_cycle_id"] for row in market_rows] == ["cycle-0900"] * 4
    assert [row["actual_temperature"] for row in market_rows] == [31, 32, 33, 34]
    assert all(row["market"] for row in market_rows)
    assert all(row["model_age_seconds"] is not None for row in market_rows)


def test_replay_model_signal_uses_stored_output_and_delay():
    model = _model("2026-07-20T10:00:00+00:00", "cycle-0900")
    markets = [_market(f"2026-07-20T10:0{minute}:00+00:00") for minute in range(1, 5)]
    weather = [_weather(f"2026-07-20T10:0{minute}:00+00:00", 30 + minute) for minute in range(1, 5)]
    replay = replay_model_cycle_minute_view(model, markets, weather, next_model_timestamp="2026-07-20T10:05:00+00:00")
    delayed = replay_model_cycle_minute_view(model, markets, weather, next_model_timestamp="2026-07-20T10:05:00+00:00", entry_delay_minutes=2)
    assert replay["model_inference_runs"] == 0
    assert replay["books_evaluated"] == 4
    assert delayed["books_evaluated"] == 3
    assert replay["future_model_leakage"] is False


class _RemoteFiles:
    def __init__(self, mapping):
        self.mapping = mapping
        self.uploads = []

    def list_repo_files(self, **_kwargs):
        return sorted(self.mapping)

    def download_file(self, *, filename, **_kwargs):
        return self.mapping[filename]


def _remote_mapping(tmp_path):
    model_root = tmp_path / "source-model"
    market_root = tmp_path / "source-market"
    weather_root = tmp_path / "source-weather"
    LayerAStore(model_root).capture(_model())
    market_store = MarketSnapshotStore(market_root)
    market_store.capture(_market())
    market_store.close_all()
    weather_store = WeatherSnapshotStore(weather_root)
    weather_store.capture(_weather())
    weather_store.close_all()
    mapping = {}
    for prefix, root in (("layer_a", model_root), ("layer_a_market", market_root), ("layer_a_weather", weather_root)):
        for path in root.rglob("*"):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
                mapping[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    return mapping


def test_remote_history_cache_is_separate_and_merges_without_reupload(tmp_path):
    remote = _RemoteFiles(_remote_mapping(tmp_path))
    local_model = LayerAStore(tmp_path / "local-model")
    local_market = MarketSnapshotStore(tmp_path / "local-market")
    local_weather = WeatherSnapshotStore(tmp_path / "local-weather")
    local_weather.capture(_weather(temp=35.0))
    store = HistoricalStore(
        local_store=local_model,
        local_market_store=local_market,
        local_weather_store=local_weather,
        cache_dir=tmp_path / "remote-cache",
        repo_id="private/test",
        token="secret",
        api=remote,
        auto_refresh=False,
    )
    status = store.refresh(date_value=EVENT_DATE)
    assert status["status"] == "available"
    assert (tmp_path / "remote-cache" / "layer_a_weather").exists()
    assert not (tmp_path / "local-weather" / "layer_a_weather").exists()
    records = store.records("weather", date_value=EVENT_DATE)
    assert len(records) == 1
    assert records[0]["temperature_current"] == 35.0
    assert remote.uploads == []


def test_remote_failure_is_degraded_but_local_history_remains(tmp_path):
    local_weather = WeatherSnapshotStore(tmp_path / "local-weather")
    local_weather.capture(_weather())
    class Failing:
        def list_repo_files(self, **_kwargs):
            raise OSError("offline")
    store = HistoricalStore(
        local_weather_store=local_weather,
        cache_dir=tmp_path / "remote-cache",
        repo_id="private/test",
        token="secret",
        api=Failing(),
        auto_refresh=False,
    )
    assert store.refresh(date_value=EVENT_DATE)["status"] == "unavailable"
    assert store.records("weather", date_value=EVENT_DATE)


def test_export_contains_weather_stream_and_admin_requires_token(tmp_path, monkeypatch):
    model_store = LayerAStore(tmp_path / "model")
    weather_store = WeatherSnapshotStore(tmp_path / "weather")
    model_store.capture(_model())
    weather_store.capture(_weather())
    weather_store.close_all()
    archive = tmp_path / "export.zip"
    result = export_layer_a(store=model_store, weather_store=weather_store, output=archive, date_value=EVENT_DATE)
    assert result["weather_snapshot_count"] == 1
    with zipfile.ZipFile(archive) as handle:
        assert any(name.startswith("layer_a_weather/") for name in handle.namelist())
    monkeypatch.setenv("LAYER_A_ADMIN_TOKEN", "admin-secret")
    with pytest.raises(Exception):
        _require_admin(SimpleNamespace(headers={}))


def test_history_api_reads_merged_minute_view_after_rebuild(monkeypatch):
    from app.api import history as history_api

    class StubHistory:
        def minute_history(self, **_kwargs):
            return [{"timestamp": "2026-07-20T18:01:00+08:00", "actual_temperature": 31.0}]

        def health_summary(self):
            return {"status": "available", "files_cached": 3}

    monkeypatch.setattr(history_api, "get_default_historical_store", lambda: StubHistory())
    response = history_api.minute_history(
        date=EVENT_DATE,
        start=None,
        end=None,
        bucket=None,
        model=None,
        limit=100,
    )
    assert response["minutes"][0]["actual_temperature"] == 31.0
    assert response["remote_history"]["status"] == "available"
