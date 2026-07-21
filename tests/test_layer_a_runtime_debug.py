from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from layer_a.historical_store import HistoricalStore
from layer_a.market_schema import build_market_snapshot
from layer_a.market_storage import MarketSnapshotStore
from layer_a.minute_view import build_minute_view
from layer_a.schema import build_layer_a_record
from layer_a.storage import LayerAStore
from layer_a.weather_schema import build_weather_snapshot
from layer_a.weather_storage import WeatherSnapshotStore

UTC = timezone.utc
HKT = timezone(timedelta(hours=8))
EVENT_DATE = "2026-07-20"
SLUG = "highest-temperature-in-hong-kong-on-july-20-2026"


def _weather(ts: str = "2026-07-20T10:01:00+00:00") -> dict:
    return build_weather_snapshot(
        snapshot_timestamp=ts,
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state={
            "observations": {"temperature": 30.0, "humidity": 70.0},
            "max_so_far": 30.0,
            "min_so_far": 25.0,
        },
    )


def _book(token: str, timestamp: datetime) -> dict:
    return {
        "asset_id": token,
        "timestamp": timestamp.isoformat(),
        "source_name": "polymarket_clob",
        "fetch_cycle_id": "fetch-debug",
        "tick_size": 0.01,
        "minimum_order_size": 1.0,
        "bids": [{"price": 0.35, "size": 5.0}],
        "asks": [{"price": 0.40, "size": 5.0}],
        "validation_errors": [],
    }


def _market(ts: str = "2026-07-20T10:01:00+00:00") -> dict:
    decision = datetime.fromisoformat(ts)
    market = {
        "bucket": "30-31",
        "id": "market-debug",
        "conditionId": "condition-debug",
        "outcomes": ["Yes", "No"],
        "token_id": "yes-debug",
        "no_token_id": "no-debug",
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
        market_depth={"30-31": _book("yes-debug", decision)},
        market_depth_no={"30-31": _book("no-debug", decision)},
        fetch_cycle_id="fetch-debug",
        gamma_reference_prices={"30-31": {"yes": 0.38, "no": 0.62}},
    )


def _model(
    ts: str = "2026-07-20T10:00:00+00:00",
    *,
    cycle_id: str = "cycle-debug",
    market_kind: str = "highest_temperature",
) -> dict:
    decision = datetime.fromisoformat(ts)
    return build_layer_a_record(
        {
            "decision_timestamp": decision,
            "decision_cycle_id": cycle_id,
            "event_date": EVENT_DATE,
            "event_slug": SLUG,
            "market_kind": market_kind,
            "weather_state": {"observations": {"temperature": 30.0}},
            "model_states": {
                "model_debug": {
                    "model_name": "model_debug",
                    "model_version": "test",
                    "artifact_identity": "test-artifact",
                    "feature_spec": "test.json",
                    "numeric_features": {"temperature": 30.0},
                    "point_prediction": 30.2,
                    "full_bucket_probabilities": {"30-31": 0.8},
                }
            },
        }
    )


def _history_store(tmp_path: Path) -> HistoricalStore:
    return HistoricalStore(
        local_store=LayerAStore(tmp_path / "model"),
        local_market_store=MarketSnapshotStore(tmp_path / "market"),
        local_weather_store=WeatherSnapshotStore(tmp_path / "weather"),
        cache_dir=tmp_path / "remote-cache",
        legacy_csv_dir=tmp_path / "export",
        auto_refresh=False,
    )


def _patch_lifespan_dependencies(monkeypatch):
    from app.api import server
    from layer_a import canonical_capture, historical_store, market_capture, market_storage, storage, upload_worker, weather_capture, weather_storage

    class Service:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def startup_scan(self):
            return {}

        def health_summary(self):
            return {"running": self.started}

    class Remote(Service):
        auto_refresh = False
        repo_id = ""
        token = ""

        def start_background_refresh(self):
            self.started = True

        def stop_background_refresh(self):
            self.started = False

    class Upload(Service):
        enabled = False

    services = {
        "model": Service(),
        "market": Service(),
        "weather": Service(),
        "canonical": Service(),
        "remote": Remote(),
        "upload": Upload(),
    }
    monkeypatch.setattr(storage, "get_default_store", lambda: services["model"])
    monkeypatch.setattr(market_storage, "get_default_market_store", lambda: services["market"])
    monkeypatch.setattr(weather_storage, "get_default_weather_store", lambda: services["weather"])
    monkeypatch.setattr(market_capture, "get_default_market_collector", lambda: services["market"])
    monkeypatch.setattr(weather_capture, "get_default_weather_collector", lambda: services["weather"])
    monkeypatch.setattr(canonical_capture, "get_default_canonical_collector", lambda: services["canonical"])
    monkeypatch.setattr(historical_store, "get_default_historical_store", lambda: services["remote"])
    monkeypatch.setattr(upload_worker, "LayerAUploadWorker", lambda: services["upload"])
    monkeypatch.setattr(server.strategies, "start_scheduler", lambda: None)
    monkeypatch.setattr(server.strategies, "stop_scheduler", lambda: None)
    server._layer_a_canonical_collector = None
    server._layer_a_upload_worker = None
    return server, services


def test_actual_app_lifespan_starts_layer_a_services(monkeypatch):
    server, services = _patch_lifespan_dependencies(monkeypatch)
    with TestClient(server.app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert all(services[key].started for key in ("market", "weather", "canonical", "remote", "upload"))
        assert response.json()["layer_a"]["market_collector_running"] is True
    assert all(not services[key].started for key in ("market", "weather", "canonical", "remote", "upload"))


def test_actual_app_history_route_sees_open_chunks_and_no_close_duplicates(monkeypatch, tmp_path):
    server, _services = _patch_lifespan_dependencies(monkeypatch)
    store = _history_store(tmp_path)
    from app.api import history as history_api

    monkeypatch.setattr(history_api, "get_default_historical_store", lambda: store)
    with TestClient(server.app) as client:
        before = client.get(f"/api/history/minute?date={EVENT_DATE}")
        assert before.status_code == 200
        assert before.json()["count"] == 0
        store.local_weather_store.capture(_weather())
        store.local_market_store.capture(_market())
        visible = client.get(f"/api/history/minute?date={EVENT_DATE}")
        assert visible.status_code == 200
        assert visible.json()["count"] == 1
        assert visible.json()["minutes"][0]["source"] == "layer_a"
        store.local_weather_store.close_due(datetime(2026, 7, 20, 10, 10, tzinfo=UTC))
        store.local_market_store.close_due(datetime(2026, 7, 20, 10, 10, tzinfo=UTC))
        after_close = client.get(f"/api/history/minute?date={EVENT_DATE}")
        assert after_close.json()["count"] == 1


def test_partial_open_jsonl_tail_is_ignored(tmp_path):
    weather_store = WeatherSnapshotStore(tmp_path / "weather")
    weather_store.capture(_weather())
    raw_path = next((tmp_path / "weather").rglob("*.jsonl.tmp"))
    with raw_path.open("ab") as handle:
        handle.write(b'{"partial":')
    info = weather_store.scan()[0]
    assert len(weather_store.read_partition_snapshots(info)) == 1


def test_legacy_csv_hkt_fallback_and_utc_boundary(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    path = export_dir / f"{EVENT_DATE}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "snapshot_date", "model_key", "model_predicted_temp", "actual_temp", "max_so_far", "context_json", "all_model_predictions"],
        )
        writer.writeheader()
        writer.writerow({
            "timestamp": "2026-07-20T23:55:00",
            "snapshot_date": EVENT_DATE,
            "model_key": "model_debug",
            "model_predicted_temp": "30.2",
            "actual_temp": "29.5",
            "max_so_far": "29.5",
            "context_json": '{"min_so_far": 25.0}',
            "all_model_predictions": "{}",
        })
    next_day = export_dir / "2026-07-21.csv"
    next_day.write_text(
        "timestamp,snapshot_date,model_key,model_predicted_temp,actual_temp,max_so_far,context_json,all_model_predictions\n"
        "2026-07-20T16:00:00Z,2026-07-21,model_debug,30.2,29.5,29.5,{},{}\n",
        encoding="utf-8",
    )
    store = HistoricalStore(
        local_store=LayerAStore(tmp_path / "model"),
        local_market_store=MarketSnapshotStore(tmp_path / "market"),
        local_weather_store=WeatherSnapshotStore(tmp_path / "weather"),
        cache_dir=tmp_path / "cache",
        legacy_csv_dir=export_dir,
        auto_refresh=False,
    )
    hkt_rows = store.minute_history(date_value=EVENT_DATE)
    assert len(hkt_rows) == 1
    assert hkt_rows[0]["source"] == "legacy_csv"
    assert hkt_rows[0]["timestamp"].endswith("+08:00")
    assert hkt_rows[0]["market_snapshot_id"] is None
    assert store.minute_history(date_value="2026-07-21")[0]["timestamp"].startswith("2026-07-21T00:00")


def test_trajectory_chart_uses_minute_projection(monkeypatch):
    from app.api import charts

    class Store:
        def minute_history(self, **kwargs):
            assert kwargs == {
                "date_value": EVENT_DATE,
                "market_kind": "highest_temperature",
                "limit": 10000,
            }
            return [
                {
                    "timestamp": "2026-07-20T18:00:00+08:00",
                    "actual_temperature": 30.0,
                    "market_prices": {"30-31": 0.38},
                    "model_predictions": {"model_debug": {"point_prediction": 30.2}},
                },
                {
                    "timestamp": "2026-07-20T18:01:00+08:00",
                    "actual_temperature": 30.1,
                    "market_prices": {"30-31": 0.40},
                    "model_predictions": {"model_debug": {"point_prediction": 30.2}},
                },
            ]

    monkeypatch.setattr(charts, "get_default_historical_store", lambda: Store())
    monkeypatch.setattr(
        charts,
        "read_models_comparison",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("SQLite cycle data should not be used")),
    )

    payload = charts.get_models_comparison_chart(date=EVENT_DATE)

    assert payload["granularity"] == "minute"
    assert payload["data_source"] == "layer_a_minute_view"
    assert payload["timestamps"] == [
        "2026-07-20T18:00:00+08:00",
        "2026-07-20T18:01:00+08:00",
    ]
    assert payload["actual_temps"] == [30.0, 30.1]
    assert payload["models"]["model_debug"] == [30.2, 30.2]
    assert payload["market_temps"] == [30.5, 30.5]

    legacy_payload = charts._comparison_payload_from_minute_rows(
        [{
            "timestamp": "2026-07-20T18:00:00+08:00",
            "source": "legacy_csv",
            "actual_temperature": 30.0,
            "market_expected_temperature": 30.5,
            "model_predictions": {"model_debug": {"point_prediction": 30.2}},
        }]
    )
    assert legacy_payload["granularity"] == "strategy_cycle"
    assert legacy_payload["data_source"] == "legacy_csv"


def test_minute_history_reads_model_from_incomplete_partition(tmp_path):
    store = _history_store(tmp_path)
    store.local_store.capture(_model())

    info = store.local_store.scan(date_value=EVENT_DATE)[0]
    info.files["manifest"].unlink()

    rows = store.minute_history(
        date_value=EVENT_DATE,
        market_kind="highest_temperature",
    )

    assert len(rows) == 1
    assert rows[0]["model_cycle_id"] == "cycle-debug"
    assert rows[0]["model_predictions"]["model_debug"]["point_prediction"] == 30.2


def test_minute_history_does_not_use_other_temperature_market_cycles(tmp_path):
    store = _history_store(tmp_path)
    store.local_store.capture(_model(cycle_id="tmax-cycle"))
    store.local_store.capture(
        _model(
            "2026-07-20T10:01:00+00:00",
            cycle_id="tmin-cycle",
            market_kind="lowest_temperature",
        )
    )
    store.local_weather_store.capture(_weather("2026-07-20T10:02:00+00:00"))

    rows = store.minute_history(
        date_value=EVENT_DATE,
        market_kind="highest_temperature",
    )
    row = next(item for item in rows if item["timestamp"].startswith("2026-07-20T18:02"))
    assert row["model_cycle_id"] == "tmax-cycle"


def test_strategy_snapshot_sqlite_connection_waits_for_busy_writer(monkeypatch, tmp_path):
    from features import strategy_snapshot_logger as logger

    previous_conn = getattr(logger._LOCAL, "conn", None)
    logger._LOCAL.conn = None
    monkeypatch.setattr(logger, "DB_PATH", tmp_path / "snapshots.db")
    monkeypatch.setattr(logger, "EXPORT_DIR", tmp_path / "export")
    try:
        conn = logger._get_conn()
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
        conn.close()
    finally:
        logger._LOCAL.conn = previous_conn


def test_layer_a_preferred_over_legacy_csv(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / f"{EVENT_DATE}.csv").write_text(
        "timestamp,snapshot_date,actual_temp\n2026-07-20T10:01:00,2026-07-20,10\n",
        encoding="utf-8",
    )
    store = _history_store(tmp_path)
    store.legacy_csv_dir = export_dir
    store.local_weather_store.capture(_weather())
    rows = store.minute_history(date_value=EVENT_DATE)
    assert rows[0]["source"] == "layer_a"
    assert rows[0]["actual_temperature"] == 30.0


def test_naive_minute_view_timestamp_uses_hkt_wall_clock():
    rows = build_minute_view(
        [{"decision_timestamp": "2026-07-20T23:55:00", "decision_cycle_id": "cycle-naive", "models": []}],
        [],
        [],
    )
    assert rows[0]["timestamp"].endswith("+08:00")
    assert rows[0]["timestamp"].startswith("2026-07-20T23:55")


def test_routes_and_frontend_runtime_contract():
    from app.api.server import app

    paths = app.openapi()["paths"]
    assert "/api/history/minute" in paths
    assert "/api/history/model-cycles" in paths
    assert "/api/history/market-snapshots" in paths
    assert "/api/history/weather-snapshots" in paths
    assert "/admin/layer-a-history-refresh" in paths
    panel = Path("app/frontend/src/components/MinuteHistoryPanel.tsx").read_text(encoding="utf-8")
    client = Path("app/frontend/src/api/client.ts").read_text(encoding="utf-8")
    hub = Path("app/frontend/src/pages/Hub.tsx").read_text(encoding="utf-8")
    trajectory = Path("app/frontend/src/components/ModelsComparisonChart.tsx").read_text(encoding="utf-8")
    assert "refetchInterval: 60_000" in panel
    assert "refetchInterval: 60_000" in trajectory
    assert 'cache: "no-store"' in client
    assert "signal" in panel
    assert 'cache: "no-store"' in client
    assert "MinuteHistoryPanel" not in hub
    assert "Actual observations and execution books" not in panel


def test_dataset_remote_path_and_repo_type_contract(tmp_path):
    from layer_a.historical_store import HistoricalStore
    from layer_a.upload import DatasetUploader

    market_root = tmp_path / "market"
    market_store = MarketSnapshotStore(market_root)
    market_store.capture(_market())
    info = market_store.scan()[0]
    uploader = DatasetUploader(repo_id="owner/dataset", token="secret")
    assert uploader._remote_path(info, info.directory / "snapshots-test.jsonl.zst", market_root, "layer_a_market").startswith(
        "layer_a_market/date=2026-07-20/hour=18/minute=00/"
    )

    class Api:
        def __init__(self):
            self.list_kwargs = None

        def list_repo_files(self, **kwargs):
            self.list_kwargs = kwargs
            return []

    api = Api()
    store = HistoricalStore(
        local_store=LayerAStore(tmp_path / "local-model"),
        local_market_store=MarketSnapshotStore(tmp_path / "local-market"),
        local_weather_store=WeatherSnapshotStore(tmp_path / "local-weather"),
        cache_dir=tmp_path / "cache",
        repo_id="owner/dataset",
        token="secret",
        api=api,
        auto_refresh=False,
    )
    store.refresh(date_value=EVENT_DATE)
    assert api.list_kwargs["repo_type"] == "dataset"


def test_entrypoint_and_frontend_build_contract():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert '"uvicorn", "app.api.server:app"' in dockerfile
    assert "npm run build" in dockerfile
    assert "COPY --from=frontend-build" in dockerfile
    assets = list(Path("app/frontend/dist/assets").glob("*.js"))
    if assets:
        bundle_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in assets)
        assert "Historical Prediction Trajectory" in bundle_text
        assert "LAYER A · MINUTE HISTORY" not in bundle_text
    canonical = Path("layer_a/canonical_capture.py").read_text(encoding="utf-8")
    assert "interval_seconds: float = 300.0" in canonical
