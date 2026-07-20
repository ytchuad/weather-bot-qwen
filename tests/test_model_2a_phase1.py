import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from features.model_2a_source_adapters import (
    standardize_weather_obs,
    standardize_wind_obs,
)
from inference.model_2a_adapters import (
    Model2ALineageError,
    Model2AV1Adapter,
    Model2AV2Adapter,
    Model2AVersionError,
    get_model_2a_adapter,
    validate_model_2a_lineage,
)
from inference.model_2a_realtime_inference import (
    _update_missing_flags_from_canonical,
)


ROOT = Path(__file__).resolve().parents[1]
V1_ARTIFACT = ROOT / "models" / "intraday_minute_ml_model_2a"
V2_ARTIFACT = ROOT / "models" / "intraday_minute_ml_model_2a_v2"


def _artifact_manifest(directory: Path) -> dict:
    manifest = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[str(path.relative_to(directory))] = digest
    return manifest


def _copy_v2_adapter(tmp_path: Path) -> Model2AV2Adapter:
    artifact_dir = tmp_path / "model_2a_v2_artifact"
    shutil.copytree(V2_ARTIFACT, artifact_dir)

    spec = yaml.safe_load(
        (ROOT / "config" / "model_2a_feature_spec_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    spec["feature_list_path"] = str(artifact_dir / "feature_list.json")
    spec_path = tmp_path / "model_2a_v2_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    adapter = Model2AV2Adapter(spec_path)
    adapter.artifact_directory = artifact_dir.resolve()
    adapter.feature_list_path = adapter.artifact_directory / "feature_list.json"
    adapter.artifact_identity = str(adapter.artifact_directory)
    return adapter


def _synthetic_inputs():
    decision_time = pd.Timestamp("2026-07-19 12:00")
    weather = pd.DataFrame(
        [
            {
                "timestamp": decision_time - pd.Timedelta(minutes=60),
                "temp_current": 29.0,
                "rh_current": 70.0,
                "pressure_current": 1010.0,
                "dew_point_current": 23.0,
            },
            {
                "timestamp": decision_time - pd.Timedelta(minutes=30),
                "temp_current": 29.5,
                "rh_current": 69.0,
                "pressure_current": 1010.2,
                "dew_point_current": 23.2,
            },
            {
                "timestamp": decision_time,
                "temp_current": 30.0,
                "rh_current": 68.0,
                "pressure_current": 1010.5,
                "dew_point_current": 23.5,
            },
        ]
    )
    wind = pd.DataFrame(
        [
            {
                "timestamp": decision_time,
                "station": "ref-1",
                "station_type": "\u53c3\u8003",
                "wind_speed": 8.0,
            },
            {
                "timestamp": decision_time,
                "station": "offshore-1",
                "station_type": "\u96e2\u5cb8\u53ca\u9ad8\u5730",
                "wind_speed": 12.0,
            },
            {
                "timestamp": decision_time,
                "station": "\u4eac\u58eb\u67cf",
                "station_type": "\u96e2\u5cb8\u53ca\u9ad8\u5730",
                "wind_speed": 10.0,
            },
        ]
    )
    forecast = pd.DataFrame(
        [
            {
                "forecast_issue_datetime": decision_time - pd.Timedelta(hours=1),
                "target_date": decision_time.normalize(),
                "forecast_max_temp": 32.0,
                "forecast_min_temp": 27.0,
            }
        ]
    )
    return decision_time, weather, wind, forecast


def test_source_adapter_quality_flags_are_null_safe_and_int64():
    weather = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-07-19 10:00"),
                "temp_current": 30.0,
                "temp_anomaly_flag": False,
                "temp_spike_flag": False,
                "wind_missing_flag": True,
            },
            {
                "timestamp": pd.Timestamp("2026-07-19 10:01"),
                "temp_current": None,
                "temp_anomaly_flag": True,
                "temp_spike_flag": True,
                "wind_missing_flag": True,
            },
            {
                "timestamp": pd.Timestamp("2026-07-19 10:02"),
                "temp_current": 45.0,
                "temp_anomaly_flag": True,
                "temp_spike_flag": False,
                "wind_missing_flag": False,
            },
        ]
    )
    weather_result = standardize_weather_obs(weather, "live")
    assert weather_result["data_quality_flags"].tolist() == [4, 7, 1]
    assert str(weather_result["data_quality_flags"].dtype) == "int64"
    assert pd.isna(weather_result.loc[2, "temp_current_clean"])

    wind = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-07-19 10:00"),
                "station": "x",
                "station_type": 123,
                "wind_speed": None,
                "wind_missing_flag": True,
                "wind_anomaly_flag": True,
            }
        ]
    )
    wind_result = standardize_wind_obs(wind, "live")
    assert wind_result["data_quality_flags"].tolist() == [3]
    assert str(wind_result["data_quality_flags"].dtype) == "int64"


@pytest.mark.parametrize(
    ("adapter_cls", "version", "artifact_dir"),
    [
        (Model2AV1Adapter, "v1", V1_ARTIFACT),
        (Model2AV2Adapter, "v2", V2_ARTIFACT),
    ],
)
def test_model_2a_adapter_initialization_and_lineage(
    adapter_cls, version, artifact_dir
):
    adapter = adapter_cls()
    lineage = adapter.validate_lineage()
    assert adapter.model_version == version
    assert adapter.feature_version == version
    assert adapter.artifact_directory == artifact_dir.resolve()
    assert adapter.feature_list_path == artifact_dir.resolve() / "feature_list.json"
    assert len(adapter.ordered_feature_names) == 45
    assert lineage["validation"] == "passed"
    assert lineage["artifact_identity"] == str(artifact_dir.resolve())


def test_unknown_model_version_fails_closed_and_registry_has_no_default():
    with pytest.raises(Model2AVersionError):
        get_model_2a_adapter(None)
    with pytest.raises(Model2AVersionError):
        get_model_2a_adapter("v3")


def test_cross_version_spec_and_artifact_pairings_are_rejected():
    with pytest.raises(Model2ALineageError):
        validate_model_2a_lineage(
            "v1",
            ROOT / "config" / "model_2a_feature_spec.yaml",
            artifact_directory=V2_ARTIFACT,
        )
    with pytest.raises(Model2ALineageError):
        validate_model_2a_lineage(
            "v2",
            ROOT / "config" / "model_2a_feature_spec_v2.yaml",
            artifact_directory=V1_ARTIFACT,
        )


def test_reference_feature_groups_do_not_override_json_feature_order(tmp_path):
    adapter = _copy_v2_adapter(tmp_path)
    spec = adapter.load_spec()
    spec["feature_groups"] = list(reversed(spec["feature_groups"]))
    lineage = adapter.validate_lineage(spec=spec)
    assert lineage["ordered_feature_names"] == list(adapter.ordered_feature_names)


def test_feature_list_missing_extra_duplicate_and_reordered_features_fail(tmp_path):
    adapter = _copy_v2_adapter(tmp_path)
    feature_path = adapter.feature_list_path
    original = json.loads(feature_path.read_text(encoding="utf-8"))
    mutations = [
        original[:-1],
        original + ["unexpected_feature"],
        original[:-1] + [original[-2]],
        [original[1], original[0]] + original[2:],
    ]
    for mutated in mutations:
        feature_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(Model2ALineageError):
            adapter.validate_lineage()
    feature_path.write_text(json.dumps(original), encoding="utf-8")
    assert adapter.validate_lineage()["validation"] == "passed"


def test_threshold_metadata_and_quantile_availability_are_strict(tmp_path):
    adapter = _copy_v2_adapter(tmp_path)
    threshold_path = adapter.artifact_directory / "best_threshold.json"
    threshold_path.write_text(
        json.dumps({"upside_zero_threshold": 1.5}), encoding="utf-8"
    )
    with pytest.raises(Model2ALineageError):
        adapter.validate_lineage()

    threshold_path.write_text(
        json.dumps({"upside_zero_threshold": 0.4749080424799324}),
        encoding="utf-8",
    )
    q10_path = adapter.artifact_directory / "upside_q10.txt"
    q10_backup = tmp_path / "upside_q10.txt.backup"
    shutil.copy2(q10_path, q10_backup)
    q10_path.unlink()
    with pytest.raises(Model2ALineageError):
        adapter.validate_lineage()
    shutil.copy2(q10_backup, q10_path)
    assert adapter.validate_lineage()["validation"] == "passed"


def test_artifact_manifest_is_unchanged_by_lineage_validation():
    before_v1 = _artifact_manifest(V1_ARTIFACT)
    before_v2 = _artifact_manifest(V2_ARTIFACT)
    Model2AV1Adapter().validate_lineage()
    Model2AV2Adapter().validate_lineage()
    assert _artifact_manifest(V1_ARTIFACT) == before_v1
    assert _artifact_manifest(V2_ARTIFACT) == before_v2


def test_explicit_v2_synthetic_realtime_inference(monkeypatch):
    from inference import model_2a_realtime_inference as realtime

    decision_time, weather, wind, forecast = _synthetic_inputs()
    monkeypatch.setattr(realtime, "_write_inference_log", lambda *args: None)
    result = realtime.run_model_2a_inference(
        decision_time=decision_time,
        raw_weather=weather,
        raw_wind=wind,
        raw_forecast=forecast,
        model_version="v2",
        model_spec_path=str(ROOT / "config" / "model_2a_feature_spec_v2.yaml"),
    )
    assert result["model_version"] == "v2"
    assert result["feature_version"] == "v2"
    assert result["spec_path"].endswith("model_2a_feature_spec_v2.yaml")
    assert result["artifact_identity"].endswith("intraday_minute_ml_model_2a_v2")
    assert "error" not in result
    assert result["pred_tmax_q50"] == pytest.approx(
        result["max_so_far"] + result["upside_q50"]
    )


def test_explicit_v1_inference_returns_supported_deprecation_error(monkeypatch):
    from inference import model_2a_realtime_inference as realtime

    decision_time, weather, wind, forecast = _synthetic_inputs()
    monkeypatch.setattr(realtime, "_write_inference_log", lambda *args: None)
    result = realtime.run_model_2a_inference(
        decision_time=decision_time,
        raw_weather=weather,
        raw_wind=wind,
        raw_forecast=forecast,
        model_version="v1",
        model_spec_path=str(ROOT / "config" / "model_2a_feature_spec.yaml"),
    )
    assert result["model_version"] == "v1"
    assert result["warning_flags"] == ["unsupported_model_version"]
    assert "deprecated/unsupported" in result["error"]
    assert result["prediction"] is None


def test_realtime_missing_flags_function_accepts_canonical_sources():
    weather = pd.DataFrame(
        [{"temp_anomaly_flag": True, "temp_spike_flag": False}]
    )
    result = _update_missing_flags_from_canonical(
        weather,
        pd.DataFrame(),
        pd.DataFrame([{"forecast_missing_flag": False}]),
    )
    assert result == (True, False, True, True, False)


def test_monitoring_requires_version_and_uses_matching_metadata(tmp_path):
    from monitoring.model_2a_data_quality_checks import (
        run_model_2a_data_quality_checks,
    )
    from monitoring.model_2a_inference_parity_check import (
        run_model_2a_parity_check,
    )

    with pytest.raises(Model2AVersionError):
        run_model_2a_data_quality_checks(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), output_path=str(tmp_path)
        )

    checks = run_model_2a_data_quality_checks(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        model_version="v2",
        output_path=str(tmp_path),
    )
    assert not checks.empty
    assert set(checks["model_version"]) == {"v2"}
    assert set(checks["artifact_identity"]) == {str(V2_ARTIFACT.resolve())}

    with pytest.raises(Model2AVersionError):
        run_model_2a_parity_check(
            inference_log_path=str(tmp_path / "missing.parquet"),
            model_version="v1",
            output_path=str(tmp_path),
        )


def test_shadow_monitoring_rejects_ambiguous_or_mixed_versions():
    from monitoring.model_2a_daily_shadow_eval import (
        _validate_inference_log_version,
    )

    with pytest.raises(Model2AVersionError):
        _validate_inference_log_version(pd.DataFrame({"prediction": [1]}), "v2")
    with pytest.raises(Model2AVersionError):
        _validate_inference_log_version(
            pd.DataFrame({"model_version": ["v1", "v2"]}), "v2"
        )
    _validate_inference_log_version(
        pd.DataFrame({"model_version": ["v2"]}), "v2"
    )
