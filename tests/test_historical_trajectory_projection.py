from __future__ import annotations

from datetime import datetime, timedelta, timezone

from layer_a.minute_view import build_minute_projection
from layer_a.weather_schema import build_weather_snapshot


EVENT_DATE = "2026-07-23"


def _weather(
    *,
    observation: str,
    first_seen: str,
    capture: str,
    temperature: float = 30.0,
) -> dict:
    return build_weather_snapshot(
        snapshot_timestamp=observation,
        observation_timestamp=observation,
        first_seen_timestamp=first_seen,
        capture_timestamp=capture,
        event_date=EVENT_DATE,
        location="Hong Kong",
        weather_state={
            "observations": {"temperature": temperature, "humidity": 70.0},
            "max_so_far": temperature,
            "min_so_far": 25.0,
        },
    )


def _model(timestamp: str, cycle_id: str = "cycle-1") -> dict:
    return {
        "decision_timestamp": timestamp,
        "decision_cycle_id": cycle_id,
        "models": [{"model_name": "model_a", "point_prediction": 30.2}],
    }


def test_delayed_observation_stays_at_its_observation_timestamp_without_minute_fill():
    delayed = _weather(
        observation="2026-07-23T09:40:00+08:00",
        first_seen="2026-07-23T09:48:00+08:00",
        capture="2026-07-23T09:48:10+08:00",
    )
    markets = [
        {"decision_timestamp": f"2026-07-23T09:{minute:02d}:00+08:00"}
        for minute in range(41, 49)
    ]
    projection = build_minute_projection(
        [_model("2026-07-23T09:48:00+08:00")],
        markets,
        [delayed],
        date_value=EVENT_DATE,
        as_of="2026-07-23T09:48:30+08:00",
        exact_model_cycles=True,
    )
    rows = {row["timestamp"][11:16]: row for row in projection["rows"]}

    assert rows["09:40"]["actual_temperature"] == 30.0
    assert rows["09:40"]["actual_observation_timestamp"] == "2026-07-23T01:40:00+00:00"
    assert all(rows[f"09:{minute:02d}"]["actual_temperature"] is None for minute in range(41, 49))
    assert rows["09:48"]["model_cycle_id"] == "cycle-1"
    assert all(rows[f"09:{minute:02d}"]["model_cycle_id"] is None for minute in range(41, 48))


def test_hkt_calendar_date_filters_utc_boundary_and_future_dates():
    midnight_hkt = _weather(
        observation="2026-07-23T16:00:00+00:00",  # 24 Jul 00:00 HKT
        first_seen="2026-07-23T16:01:00+00:00",
        capture="2026-07-23T16:01:10+00:00",
    )

    previous_hkt_date = build_minute_projection(
        [], [], [midnight_hkt], date_value="2026-07-23", as_of="2026-07-24T00:05:00+08:00"
    )
    next_hkt_date = build_minute_projection(
        [], [], [midnight_hkt], date_value="2026-07-24", as_of="2026-07-24T00:05:00+08:00"
    )
    future = build_minute_projection(
        [_model("2026-07-25T09:00:00+08:00")],
        [],
        [],
        date_value="2026-07-25",
        as_of="2026-07-24T00:05:00+08:00",
        exact_model_cycles=True,
    )

    assert previous_hkt_date["rows"] == []
    assert next_hkt_date["rows"][0]["timestamp"].startswith("2026-07-24T00:00:00+08:00")
    assert future["rows"] == []


def test_future_corrupt_weather_is_rejected_with_diagnostics():
    corrupt = _weather(
        observation="2026-07-23T19:00:00+00:00",
        first_seen="2026-07-23T11:09:00+00:00",
        capture="2026-07-23T11:09:00+00:00",
    )
    projection = build_minute_projection(
        [], [], [corrupt], date_value="2026-07-24", as_of="2026-07-24T04:00:00+08:00"
    )

    assert projection["rows"] == []
    assert projection["diagnostics"]["excluded_future_weather_records"] == 1
    assert projection["diagnostics"]["latest_weather_observation_timestamp"] is None


def test_replay_weather_selection_respects_first_seen_and_correction_availability():
    original = _weather(
        observation="2026-07-23T09:40:00+08:00",
        first_seen="2026-07-23T09:48:00+08:00",
        capture="2026-07-23T09:48:10+08:00",
        temperature=30.0,
    )
    correction = _weather(
        observation="2026-07-23T09:40:00+08:00",
        first_seen="2026-07-23T10:05:00+08:00",
        capture="2026-07-23T10:05:10+08:00",
        temperature=29.6,
    )
    projection = build_minute_projection(
        [],
        [
            {"decision_timestamp": "2026-07-23T09:50:00+08:00"},
            {"decision_timestamp": "2026-07-23T10:05:00+08:00"},
        ],
        [original, correction],
        date_value=EVENT_DATE,
        as_of="2026-07-23T10:06:00+08:00",
    )
    rows = {row["timestamp"][11:16]: row for row in projection["rows"]}

    assert rows["09:50"]["weather_snapshot_id"] == original["weather_snapshot_id"]
    assert rows["10:05"]["weather_snapshot_id"] == correction["weather_snapshot_id"]
    assert rows["09:50"]["actual_temperature"] is None
    assert rows["10:05"]["actual_temperature"] is None


def test_chart_api_returns_raw_actual_points_and_projection_diagnostics(monkeypatch):
    from app.api import charts

    delayed = _weather(
        observation="2026-07-23T09:40:00+08:00",
        first_seen="2026-07-23T09:48:00+08:00",
        capture="2026-07-23T09:48:10+08:00",
    )
    projection = build_minute_projection(
        [_model("2026-07-23T09:48:00+08:00")],
        [{"decision_timestamp": "2026-07-23T09:41:00+08:00"}],
        [delayed],
        date_value=EVENT_DATE,
        as_of="2026-07-23T09:48:30+08:00",
        exact_model_cycles=True,
    )

    class Store:
        def minute_history_projection(self, **kwargs):
            assert kwargs == {
                "date_value": EVENT_DATE,
                "market_kind": "highest_temperature",
                "limit": 10000,
            }
            return {**projection, "has_layer_a_records": True}

    monkeypatch.setattr(charts, "get_default_historical_store", lambda: Store())
    payload = charts.get_models_comparison_chart(date=EVENT_DATE)

    points = dict(zip(payload["timestamps"], payload["actual_temps"]))
    assert points["2026-07-23T09:40:00+08:00"] == 30.0
    assert points["2026-07-23T09:41:00+08:00"] is None
    assert points["2026-07-23T09:48:00+08:00"] is None
    assert payload["diagnostics"] == projection["diagnostics"]


def test_chart_api_returns_no_rows_for_a_future_hkt_date(monkeypatch):
    from app.api import charts
    from app.services import weather_service

    hkt = timezone(timedelta(hours=8))
    monkeypatch.setattr(
        weather_service,
        "hkt_now",
        lambda: datetime(2026, 7, 23, 9, 48, tzinfo=hkt),
    )
    monkeypatch.setattr(
        charts,
        "get_default_historical_store",
        lambda: (_ for _ in ()).throw(AssertionError("future dates must not read trajectory storage")),
    )
    monkeypatch.setattr(
        charts,
        "read_models_comparison",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("future dates must not fall back to SQLite")),
    )

    payload = charts.get_models_comparison_chart(date="2026-07-24")

    assert payload["timestamps"] == []
    assert payload["actual_temps"] == []
    assert payload["models"] == {}
