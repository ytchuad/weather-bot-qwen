from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.services.canonical_cycle import _capture_linked_layer_a_snapshots
from layer_a.weather_schema import build_weather_snapshot
from layer_a.weather_storage import WeatherSnapshotStore


UTC = timezone.utc
EVENT_DATE = date(2026, 7, 23)


def _weather_state(temperature: float = 30.0) -> dict[str, object]:
    return {
        "observations": {"temperature": temperature, "humidity": 70.0},
        "max_so_far": max(temperature, 30.0),
        "min_so_far": 25.0,
    }


def _weather_version(
    *,
    observation: str = "2026-07-23T09:40:00+08:00",
    capture: str,
    temperature: float = 30.0,
) -> dict[str, object]:
    return build_weather_snapshot(
        snapshot_timestamp=observation,
        observation_timestamp=observation,
        capture_timestamp=capture,
        first_seen_timestamp=capture,
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state=_weather_state(temperature),
    )


def test_canonical_cycle_localizes_naive_hko_timestamp_as_hkt(monkeypatch):
    captured: dict[str, object] = {}

    class WeatherStore:
        def capture(self, snapshot):
            captured.update(snapshot)
            return SimpleNamespace(weather_snapshot_id=snapshot["weather_snapshot_id"])

    class MarketStore:
        def capture(self, snapshot):
            return SimpleNamespace(market_snapshot_id=snapshot["market_snapshot_id"])

    monkeypatch.setattr("layer_a.weather_storage.get_default_weather_store", lambda: WeatherStore())
    monkeypatch.setattr("layer_a.market_storage.get_default_market_store", lambda: MarketStore())

    _market_snapshot_id, weather_snapshot_id, lineage = _capture_linked_layer_a_snapshots(
        cycle_id="cycle-1",
        decision_timestamp=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
        target_date=EVENT_DATE,
        event_slug="highest-temperature-in-hong-kong-on-july-23-2026",
        is_min_temp=False,
        markets=[],
        state={"time_now": datetime(2026, 7, 23, 19, 0), **_weather_state()},
        rain_kwargs={},
        context_json={},
        market_depth={},
        market_depth_no={},
        depth_fetch_cycle_id=None,
        gamma_reference_prices={},
    )

    assert captured["observation_timestamp"] == "2026-07-23T11:00:00+00:00"
    assert captured["snapshot_timestamp"] == captured["observation_timestamp"]
    assert weather_snapshot_id == captured["weather_snapshot_id"]
    assert lineage == {
        "weather_snapshot_id": weather_snapshot_id,
        "weather_data_through": "2026-07-23T11:00:00+00:00",
        "weather_first_seen_timestamp": "2026-07-23T11:05:00+00:00",
        "weather_age_seconds": 300.0,
    }
    assert datetime.fromisoformat(str(captured["observation_timestamp"])).astimezone(
        ZoneInfo("Asia/Hong_Kong")
    ).strftime("%Y-%m-%d %H:%M") == "2026-07-23 19:00"


def test_timezone_aware_weather_timestamp_keeps_its_explicit_offset():
    snapshot = _weather_version(
        observation="2026-07-23T19:00:00+09:00",
        capture="2026-07-23T11:05:00+00:00",
    )

    assert snapshot["observation_timestamp"] == "2026-07-23T10:00:00+00:00"
    assert snapshot["snapshot_timestamp"] == "2026-07-23T10:00:00+00:00"


def test_repeated_observation_version_preserves_earliest_first_seen_timestamp(tmp_path):
    store = WeatherSnapshotStore(tmp_path)
    first = _weather_version(capture="2026-07-23T09:48:00+08:00")
    repeated = _weather_version(capture="2026-07-23T09:55:00+08:00")

    assert first["weather_snapshot_id"] == repeated["weather_snapshot_id"]
    assert store.capture(first).status == "captured"
    duplicate = store.capture(repeated)
    assert duplicate.status == "duplicate"
    assert duplicate.snapshot is not None
    assert duplicate.snapshot["first_seen_timestamp"] == "2026-07-23T01:48:00+00:00"

    records = store.read_partition_snapshots(store.scan()[0])
    assert len(records) == 1
    assert records[0]["first_seen_timestamp"] == "2026-07-23T01:48:00+00:00"
    assert records[0]["capture_timestamp"] == "2026-07-23T01:48:00+00:00"


def test_correction_has_a_separate_later_availability_time(tmp_path):
    store = WeatherSnapshotStore(tmp_path)
    original = _weather_version(capture="2026-07-23T09:48:00+08:00", temperature=30.0)
    correction = _weather_version(capture="2026-07-23T10:05:00+08:00", temperature=29.6)

    assert original["weather_observation_id"] == correction["weather_observation_id"]
    assert original["weather_snapshot_id"] != correction["weather_snapshot_id"]
    assert store.capture(original).status == "captured"
    assert store.capture(correction).status == "captured"

    records = sorted(
        [
            record
            for partition in store.scan()
            for record in store.read_partition_snapshots(partition)
        ],
        key=lambda record: record["first_seen_timestamp"],
    )
    assert [record["first_seen_timestamp"] for record in records] == [
        "2026-07-23T01:48:00+00:00",
        "2026-07-23T02:05:00+00:00",
    ]
