from __future__ import annotations

from layer_a.replay import replay_model_cycle_minute_view
from layer_a.schema import build_layer_a_record
from layer_a.weather_schema import build_weather_snapshot


EVENT_DATE = "2026-07-23"


def _weather(*, first_seen: str, temperature: float) -> dict:
    return build_weather_snapshot(
        snapshot_timestamp="2026-07-23T09:40:00+08:00",
        observation_timestamp="2026-07-23T09:40:00+08:00",
        first_seen_timestamp=first_seen,
        capture_timestamp=first_seen,
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state={
            "observations": {"temperature": temperature, "humidity": 70.0},
            "max_so_far": temperature,
            "min_so_far": 25.0,
        },
    )


def _model(timestamp: str, *, weather_snapshot_id: str | None = None, **lineage: object) -> dict:
    return {
        "decision_cycle_id": f"cycle-{timestamp[11:16]}",
        "decision_timestamp": timestamp,
        "event_date": EVENT_DATE,
        "weather_snapshot_id": weather_snapshot_id,
        **lineage,
    }


def test_delayed_weather_version_is_unavailable_until_its_first_seen_timestamp():
    delayed = _weather(first_seen="2026-07-23T09:48:00+08:00", temperature=30.0)

    unavailable = replay_model_cycle_minute_view(
        _model("2026-07-23T09:47:00+08:00"), [], [delayed]
    )
    available = replay_model_cycle_minute_view(
        _model("2026-07-23T09:48:00+08:00", weather_snapshot_id=delayed["weather_snapshot_id"]),
        [],
        [delayed],
    )

    assert unavailable["weather_lineage"] is None
    assert unavailable["minute_rows"][0]["weather_snapshot_id"] is None
    assert available["weather_snapshot_linkage_ok"] is True
    assert available["weather_snapshot_linkage_reason"] == "weather_anchor_resolved"
    assert available["weather_lineage"]["weather_snapshot_id"] == delayed["weather_snapshot_id"]
    assert available["weather_first_seen_timestamp"] == "2026-07-23T01:48:00+00:00"


def test_correction_only_applies_to_model_cycles_after_its_own_first_seen_timestamp():
    original = _weather(first_seen="2026-07-23T09:48:00+08:00", temperature=30.0)
    correction = _weather(first_seen="2026-07-23T10:05:00+08:00", temperature=29.6)
    versions = [original, correction]

    before_correction = replay_model_cycle_minute_view(
        _model("2026-07-23T09:50:00+08:00", weather_snapshot_id=original["weather_snapshot_id"]),
        [],
        versions,
    )
    after_correction = replay_model_cycle_minute_view(
        _model("2026-07-23T10:05:00+08:00", weather_snapshot_id=correction["weather_snapshot_id"]),
        [],
        versions,
    )

    assert before_correction["weather_lineage"]["weather_snapshot_id"] == original["weather_snapshot_id"]
    assert before_correction["minute_rows"][0]["weather_snapshot_id"] == original["weather_snapshot_id"]
    assert after_correction["weather_lineage"]["weather_snapshot_id"] == correction["weather_snapshot_id"]
    assert after_correction["minute_rows"][0]["weather_snapshot_id"] == correction["weather_snapshot_id"]


def test_model_cycle_lineage_dereferences_the_exact_point_in_time_weather_version():
    original = _weather(first_seen="2026-07-23T09:48:00+08:00", temperature=30.0)
    correction = _weather(first_seen="2026-07-23T10:05:00+08:00", temperature=29.6)
    record = build_layer_a_record(
        decision_cycle_id="cycle-0950",
        decision_timestamp="2026-07-23T09:50:00+08:00",
        capture_timestamp="2026-07-23T09:50:00+08:00",
        event_date=EVENT_DATE,
        weather_snapshot_id=original["weather_snapshot_id"],
        weather_data_through=original["observation_timestamp"],
        weather_first_seen_timestamp=original["first_seen_timestamp"],
    )

    replay = replay_model_cycle_minute_view(record, [], [original, correction])
    leaked_anchor = replay_model_cycle_minute_view(
        {**record, "weather_snapshot_id": correction["weather_snapshot_id"]},
        [],
        [original, correction],
    )

    assert record["weather_lineage"] == {
        "weather_snapshot_id": original["weather_snapshot_id"],
        "weather_data_through": "2026-07-23T01:40:00+00:00",
        "weather_first_seen_timestamp": "2026-07-23T01:48:00+00:00",
        "weather_age_seconds": 600.0,
    }
    assert replay["weather_snapshot_linkage_ok"] is True
    assert replay["weather_lineage"] == record["weather_lineage"]
    assert leaked_anchor["weather_snapshot_linkage_ok"] is False
    assert leaked_anchor["weather_snapshot_linkage_reason"] == "weather_anchor_not_point_in_time_selection"
