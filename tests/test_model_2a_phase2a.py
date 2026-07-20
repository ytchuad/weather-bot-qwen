import json

import numpy as np
import pandas as pd
import pytest

from features.input_status import (
    InputStatus,
    build_forecast_input_status,
    build_observation_buffer_status,
    jsonable,
    make_status_bundle,
    serialize_status,
)
from features.model_2a_feature_builder import (
    build_model_2a_features,
    build_model_2a_input_status,
)
from features.model_2a_source_adapters import (
    standardize_forecast,
    standardize_weather_obs,
    standardize_wind_obs,
)


def _decision():
    return pd.Timestamp("2026-07-19 12:00:00")


def _canonical_inputs():
    decision = _decision()
    weather = standardize_weather_obs(
        pd.DataFrame(
            [
                {
                    "timestamp": decision - pd.Timedelta(minutes=60),
                    "temp_current": 29.0,
                    "rh_current": 70.0,
                    "pressure_current": 1010.0,
                    "dew_point_current": 23.0,
                },
                {
                    "timestamp": decision - pd.Timedelta(minutes=30),
                    "temp_current": 29.5,
                    "rh_current": 69.0,
                    "pressure_current": 1010.2,
                    "dew_point_current": 23.2,
                },
                {
                    "timestamp": decision,
                    "temp_current": 30.0,
                    "rh_current": 68.0,
                    "pressure_current": 1010.5,
                    "dew_point_current": 23.5,
                },
            ]
        ),
        "live",
    )
    wind = standardize_wind_obs(
        pd.DataFrame(
            [
                {
                    "timestamp": decision,
                    "station": "ref-1",
                    "station_type": "參考",
                    "wind_speed": 0.0,
                },
                {
                    "timestamp": decision,
                    "station": "offshore-1",
                    "station_type": "離岸及高地",
                    "wind_speed": 12.0,
                },
                {
                    "timestamp": decision,
                    "station": "京士柏",
                    "station_type": "離岸及高地",
                    "wind_speed": 10.0,
                },
            ]
        ),
        "live",
    )
    forecast = standardize_forecast(
        pd.DataFrame(
            [
                {
                    "forecast_issue_datetime": decision - pd.Timedelta(hours=1),
                    "target_date": decision.normalize(),
                    "forecast_max_temp": 32.0,
                    "forecast_min_temp": 27.0,
                }
            ]
        ),
        "live",
    )
    return decision, weather, wind, forecast


def test_status_age_is_derived_and_real_zero_is_not_missing():
    decision = _decision()
    status = InputStatus.from_value(
        0.0,
        source_timestamp=decision - pd.Timedelta(minutes=5),
        decision_timestamp=decision,
        source_name="test_observation",
    ).to_dict()

    assert status["value"] == 0.0
    assert status["age_seconds"] == pytest.approx(300.0)
    assert status["age_minutes"] == pytest.approx(5.0)
    assert status["is_missing"] is False
    assert status["is_fallback"] is False


def test_status_never_preserves_a_fabricated_age_without_source_timestamp():
    decision = _decision()
    status = InputStatus(
        value=0.0,
        source_timestamp=None,
        decision_timestamp=decision,
        age_minutes=8.0,
    ).to_dict()

    assert status["age_seconds"] is None
    assert status["age_minutes"] is None
    assert "missing_source_timestamp" in status["quality_flags"]


def test_status_distinguishes_stale_and_explicit_fallbacks():
    decision = _decision()
    stale = InputStatus.from_value(
        3.0,
        source_timestamp=decision - pd.Timedelta(hours=2),
        decision_timestamp=decision,
        stale_after_minutes=30,
    ).to_dict()
    compat_zero = InputStatus.fallback(
        0.0,
        fallback_method="model_compat_zero",
        decision_timestamp=decision,
        source_name="i-lens_wind_obs",
    ).to_dict()
    cached = InputStatus.fallback(
        4.0,
        fallback_method="cached_api_result",
        source_timestamp=decision - pd.Timedelta(minutes=12),
        decision_timestamp=decision,
        source_name="hko_pressure",
    ).to_dict()
    source_error = InputStatus.from_value(
        None,
        decision_timestamp=decision,
        source_name="hko_pressure",
        source_error=True,
    ).to_dict()

    assert stale["is_stale"] is True
    assert stale["is_fallback"] is False
    assert compat_zero["is_missing"] is True
    assert compat_zero["is_fallback"] is True
    assert compat_zero["fallback_method"] == "model_compat_zero"
    assert cached["is_missing"] is False
    assert cached["age_minutes"] == pytest.approx(12.0)
    assert source_error["raw_status"] == "source_error"


def test_status_bundle_and_nested_payload_are_json_safe():
    decision = _decision()
    bundle = make_status_bundle(
        {
            "weather_input_status": {
                "temp_current": InputStatus.from_value(
                    np.float64(30.0),
                    source_timestamp=decision,
                    decision_timestamp=decision,
                )
            }
        },
        decision_timestamp=decision,
    )

    encoded = serialize_status(bundle)
    decoded = json.loads(encoded)
    assert decoded["status_contract_version"] == "phase2a.v1"
    assert decoded["numeric_policy"] == "legacy_compatible"
    assert decoded["status_policy"] == "truthful"
    assert jsonable(bundle)["weather_input_status"]["temp_current"]["value"] == 30.0


def test_forecast_revision_history_preserves_three_values_and_current_revision():
    decision = _decision()
    forecast = pd.DataFrame(
        [
            {
                "forecast_issue_datetime": decision - pd.Timedelta(minutes=30),
                "target_date": decision.normalize(),
                "forecast_max_temp": 30.2,
                "forecast_min_temp": 25.0,
                "forecast_source": "hko",
            },
            {
                "forecast_issue_datetime": decision - pd.Timedelta(minutes=20),
                "target_date": decision.normalize(),
                "forecast_max_temp": 31.6,
                "forecast_min_temp": 25.2,
                "forecast_source": "hko",
            },
            {
                "forecast_issue_datetime": decision - pd.Timedelta(minutes=10),
                "target_date": decision.normalize(),
                "forecast_max_temp": 30.8,
                "forecast_min_temp": 25.1,
                "forecast_source": "i-lens",
            },
        ]
    )
    status = build_forecast_input_status(
        standardize_forecast(forecast, "live"),
        decision_timestamp=decision,
        target_date=decision,
    )

    current = status["forecast_max"]
    assert current["value"] == pytest.approx(30.8)
    assert current["previous_forecast_value"] == pytest.approx(31.6)
    assert current["revision_size"] == pytest.approx(-0.8)
    assert len(status["revision_history"]) == 3
    assert status["diagnostics"]["source_switching"] is True
    assert status["diagnostics"]["large_revision"] is True


def test_forecast_status_marks_target_mismatch_issue_regression_and_missing_issue():
    decision = _decision()
    forecast = pd.DataFrame(
        [
            {
                "forecast_issue_datetime": decision - pd.Timedelta(minutes=5),
                "target_date": decision - pd.Timedelta(days=1),
                "forecast_max_temp": 29.0,
                "forecast_min_temp": 24.0,
            },
            {
                "forecast_issue_datetime": decision - pd.Timedelta(minutes=15),
                "target_date": decision,
                "forecast_max_temp": 30.0,
                "forecast_min_temp": 24.5,
            },
            {
                "forecast_issue_datetime": pd.NaT,
                "target_date": decision,
                "forecast_max_temp": 30.5,
                "forecast_min_temp": 24.8,
            },
        ]
    )
    status = build_forecast_input_status(
        standardize_forecast(forecast, "live"),
        decision_timestamp=decision,
        target_date=decision,
    )

    assert status["diagnostics"]["target_date_mismatch"] is True
    assert status["diagnostics"]["issue_time_regression"] is True
    assert "missing_issue_timestamp" in status["forecast_max"]["continuity_anomaly"]


def test_observation_buffer_excludes_future_rows_and_marks_insufficient_history():
    decision = _decision()
    frame = pd.DataFrame(
        [
            {"timestamp": decision - pd.Timedelta(minutes=30), "temp": 29.0, "rh": 70.0},
            {"timestamp": decision - pd.Timedelta(minutes=5), "temp": 30.0, "rh": 68.0},
            {"timestamp": decision + pd.Timedelta(minutes=5), "temp": 31.0, "rh": 67.0},
        ]
    )
    status = build_observation_buffer_status(
        frame,
        decision_timestamp=decision,
        values={"temp_now": 30.0, "rh_now": 68.0},
    )

    assert status["temp_current"]["value"] == 30.0
    assert status["temp_current"]["source_timestamp"] == (
        decision - pd.Timedelta(minutes=5)
    ).isoformat()
    assert status["temp_30m_ago"]["value"] == 29.0
    assert status["temp_120m_ago"]["is_missing"] is True
    assert status["temp_120m_ago"]["fallback_method"] == "unavailable"
    assert status["obs_data_age_minutes"]["age_minutes"] == pytest.approx(5.0)


def test_v2_builder_keeps_offshore_highland_names_and_separates_status():
    decision, weather, wind, forecast = _canonical_inputs()
    spec = {
        "data_quality_rules": {"max_data_age_minutes": 30},
    }
    features = build_model_2a_features(
        decision, weather, wind, forecast, spec, "live"
    )
    status = build_model_2a_input_status(
        decision, weather, wind, forecast, spec, "live"
    )

    assert "wind_offshore_highland_mean" in features.columns
    assert "wind_highland_mean" not in features.columns
    assert "wind_offshore_highland_mean" in status["wind_input_status"]
    assert "wind_highland_mean" not in status["wind_input_status"]
    assert all("status" not in column for column in features.columns)


def test_v2_builder_records_observed_zero_separately_from_missing_group():
    decision, weather, wind, forecast = _canonical_inputs()
    status = build_model_2a_input_status(
        decision,
        weather,
        wind,
        forecast,
        {"data_quality_rules": {"max_data_age_minutes": 30}},
        "live",
    )

    ref = status["wind_input_status"]["wind_ref_mean"]
    assert ref["value"] == 0.0
    assert ref["is_missing"] is False
    assert ref["is_fallback"] is False


def test_legacy_model_log_keeps_numeric_features_unchanged(monkeypatch):
    import models.intraday_inference as inference

    feature_names = [
        "temp_current", "rh_current", "pressure_current", "dew_point_current",
        "dew_point_spread", "max_so_far", "min_so_far", "range_so_far",
        "drop_from_max", "time_since_max", "temp_change_30m", "temp_change_60m",
        "temp_slope_30m", "temp_slope_60m", "temp_acceleration_60m",
        "temp_volatility_60m", "rh_change_60m", "dew_point_change_60m",
        "dew_point_spread_change_60m", "pressure_change_60m", "pressure_change_180m",
        "forecast_min_temp", "forecast_max_temp", "forecast_range",
        "forecast_gap_from_max_so_far", "forecast_age_minutes", "forecast_lead_days",
        "wind_ref_mean", "wind_ref_max", "wind_victoria_harbour_mean",
        "wind_victoria_harbour_max", "wind_offshore_highland_mean",
        "wind_offshore_highland_max", "wind_all_change_60m",
        "wind_kings_park_current", "minutes_since_midnight", "month_sin",
        "month_cos", "day_sin", "day_cos", "is_morning", "is_afternoon",
        "is_evening", "obs_data_age_minutes", "wind_data_age_minutes",
    ]

    class FakeModel:
        def __init__(self, value):
            self.value = value

        def feature_name(self):
            return feature_names

        def predict(self, X, **kwargs):
            return np.array([self.value])

    inference.set_active_model("model_2a_v2")
    monkeypatch.setattr(
        inference,
        "_get_active",
        lambda: {
            "feature_cols": feature_names,
            "upside_q10": FakeModel(0.1),
            "upside_q25": FakeModel(0.2),
            "upside_q50": FakeModel(0.3),
            "upside_q75": FakeModel(0.4),
            "upside_q90": FakeModel(0.5),
            "upside_zero": None,
        },
    )
    kwargs = {
        "current_datetime": _decision(),
        "max_so_far": 30.0,
        "temp_now": 29.5,
        "humidity": 70.0,
        "min_so_far": 25.0,
        "temp_change_30m_pre": -0.5,
        "temp_change_60m_pre": -0.3,
        "forecast_tmax": 31.6,
        "forecast_tmin": 24.0,
        "wind_ref_mean": 0.0,
        "wind_offshore_highland_mean": 12.0,
        "temp_buffer": [29.5] * 61,
        "rh_buffer": [70.0] * 61,
    }
    base = inference.predict_intraday_tmax_model_2a_v2(**kwargs)
    status = make_status_bundle(
        {
            "wind_input_status": {
                "wind_ref_mean": InputStatus.fallback(
                    0.0,
                    fallback_method="model_compat_zero",
                    decision_timestamp=_decision(),
                ).to_dict()
            },
            "observation_buffer_status": {
                "obs_data_age_minutes": InputStatus.from_value(
                    2.0,
                    source_timestamp=_decision() - pd.Timedelta(minutes=2),
                    decision_timestamp=_decision(),
                ).to_dict()
            },
            "forecast_input_status": {
                "forecast_max": InputStatus.from_value(
                    31.6,
                    source_timestamp=_decision() - pd.Timedelta(minutes=10),
                    decision_timestamp=_decision(),
                ).to_dict()
            },
        },
        decision_timestamp=_decision(),
    )
    diagnostic = inference.predict_intraday_tmax_model_2a_v2(
        **kwargs, input_status=status
    )

    for key in (
        "remaining_upside_p50", "pred_tmax_p50", "prob_max_reached"
    ):
        assert diagnostic[key] == pytest.approx(base[key])
    assert diagnostic["_numeric_features"] == base["_features"]
    assert diagnostic["_numeric_features"]["wind_data_age_minutes"] == 8
    assert diagnostic["_features"]["wind_data_age_minutes"] is None
    assert diagnostic["_features"]["obs_data_age_minutes"] == pytest.approx(2.0)


def test_pressure_and_wind_fallback_status_has_no_legacy_wind_age(monkeypatch):
    from app.services import weather_service

    decision = _decision()
    monkeypatch.setattr(weather_service, "fetch_wind_live", lambda: pd.DataFrame())
    monkeypatch.setattr(weather_service, "_last_wind_kwargs", None)
    monkeypatch.setattr(weather_service, "_last_wind_status", None)
    result = weather_service.compute_wind_kwargs(decision_timestamp=decision)

    assert "wind_data_age_minutes" not in result
    assert result["_input_status"]["wind_ref_mean"]["fallback_method"] == "model_compat_zero"
    assert result["_input_status"]["wind_ref_mean"]["is_missing"] is True
    assert result["_input_status"]["groups"]["offshore_highland"]["mean"]["is_missing"] is True
    assert result["_input_status"]["wind_highland_mean"]["quality_flags"] == [
        "v1_semantics_unavailable",
        "missing_value",
    ]


def test_wind_observed_zero_has_timestamp_and_is_not_synthetic(monkeypatch):
    from app.services import weather_service

    decision = _decision()
    wind = pd.DataFrame(
        [
            {
                "timestamp": decision - pd.Timedelta(minutes=3),
                "group": "ref",
                "station": "ref-1",
                "wind_speed": 0.0,
            }
        ]
    )
    monkeypatch.setattr(weather_service, "fetch_wind_live", lambda: wind)
    result = weather_service.compute_wind_kwargs(decision_timestamp=decision)

    status = result["_input_status"]["wind_ref_mean"]
    assert result["wind_ref_mean"] == 0.0
    assert status["is_missing"] is False
    assert status["is_fallback"] is False
    assert status["age_minutes"] == pytest.approx(3.0)
    assert result["_input_status"]["wind_offshore_highland_mean"]["is_missing"] is True


def test_wind_cache_fallback_preserves_source_timestamp(monkeypatch):
    from app.services import weather_service

    decision = _decision()
    source_time = decision - pd.Timedelta(minutes=4)
    cached = {
        "wind_ref_mean": 2.0,
        "wind_ref_max": 3.0,
        "wind_victoria_harbour_mean": 4.0,
        "wind_victoria_harbour_max": 5.0,
        "wind_offshore_highland_mean": 6.0,
        "wind_offshore_highland_max": 7.0,
        "wind_all_change_60m": 1.0,
        "wind_kings_park_current": 2.0,
    }
    cached_status = {
        key: InputStatus.from_value(
            value,
            source_timestamp=source_time,
            decision_timestamp=decision,
        ).to_dict()
        for key, value in cached.items()
    }
    monkeypatch.setattr(weather_service, "_last_wind_kwargs", cached)
    monkeypatch.setattr(weather_service, "_last_wind_status", cached_status)
    monkeypatch.setattr(weather_service, "fetch_wind_live", lambda: pd.DataFrame())
    result = weather_service.compute_wind_kwargs(decision_timestamp=decision)

    status = result["_input_status"]["wind_ref_mean"]
    assert status["fallback_method"] == "cached_api_result"
    assert status["is_missing"] is False
    assert status["age_minutes"] == pytest.approx(4.0)
