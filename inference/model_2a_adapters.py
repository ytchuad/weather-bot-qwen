# inference/model_2a_adapters.py
"""Explicit Model 2A lineage adapters.

Model 2A v1 and v2 are separate trained lineages.  This module is the only
version registry used by the new realtime and monitoring entry points.  It
does not translate feature names between versions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type, Union

from features.model_2a_feature_builder import build_model_2a_features
from inference.realtime_inference_base import load_model_spec


_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUANTILES = (10, 25, 50, 75, 90)

_FEATURE_PREFIX = (
    "temp_current",
    "rh_current",
    "pressure_current",
    "dew_point_current",
    "dew_point_spread",
    "max_so_far",
    "min_so_far",
    "range_so_far",
    "drop_from_max",
    "time_since_max",
    "temp_change_30m",
    "temp_change_60m",
    "temp_slope_30m",
    "temp_slope_60m",
    "temp_acceleration_60m",
    "temp_volatility_60m",
    "rh_change_60m",
    "dew_point_change_60m",
    "dew_point_spread_change_60m",
    "pressure_change_60m",
    "pressure_change_180m",
    "forecast_min_temp",
    "forecast_max_temp",
    "forecast_range",
    "forecast_gap_from_max_so_far",
    "forecast_age_minutes",
    "forecast_lead_days",
    "wind_ref_mean",
    "wind_ref_max",
    "wind_victoria_harbour_mean",
    "wind_victoria_harbour_max",
)

_FEATURE_SUFFIX = (
    "wind_all_change_60m",
    "wind_kings_park_current",
    "minutes_since_midnight",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "obs_data_age_minutes",
    "wind_data_age_minutes",
)

V1_FEATURE_NAMES = _FEATURE_PREFIX + (
    "wind_highland_mean",
    "wind_highland_max",
) + _FEATURE_SUFFIX

V2_FEATURE_NAMES = _FEATURE_PREFIX + (
    "wind_offshore_highland_mean",
    "wind_offshore_highland_max",
) + _FEATURE_SUFFIX


class Model2ALineageError(ValueError):
    """Raised when a Model 2A spec/artifact pairing is not exact."""


class Model2AVersionError(Model2ALineageError):
    """Raised when a Model 2A version is missing, unknown, or unsupported."""


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve repository-relative and absolute paths without fallback."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (_REPO_ROOT / candidate).resolve()


def _unsupported_v1_feature_builder(*args, **kwargs):
    raise Model2AVersionError(
        "Model 2A v1 realtime inference is deprecated/unsupported: "
        "the original v1 highland source semantics cannot be reconstructed "
        "safely by the current canonical v2 builder."
    )


class Model2AAdapter:
    """Version-specific contract for one Model 2A trained lineage."""

    model_version = ""
    feature_version = ""
    _artifact_relative = ""
    _default_spec_relative = ""
    _ordered_feature_names = ()
    _feature_builder: Callable[..., Any] = _unsupported_v1_feature_builder
    realtime_supported = False
    missingness_policy = {
        "numeric_inputs": "preserve_existing_nan_behavior",
        "quality_flags": "canonical_bitmask_only",
    }

    def __init__(self, model_spec_path: Optional[Union[str, Path]] = None):
        self.artifact_directory = (_REPO_ROOT / self._artifact_relative).resolve()
        self.feature_spec_path = _resolve_path(
            model_spec_path or (_REPO_ROOT / self._default_spec_relative)
        )
        self.feature_list_path = self.artifact_directory / "feature_list.json"
        # Read through the class so a plain function builder is not bound as
        # an instance method and receives an unexpected leading ``self``.
        self.feature_builder = self.__class__._feature_builder
        self.ordered_feature_names = tuple(self._ordered_feature_names)
        self.missingness_policy = dict(self.__class__.missingness_policy)
        self.classifier_metadata = {
            "artifact_path": str(self.artifact_directory / "upside_zero.txt"),
            "feature_names": list(self.ordered_feature_names),
            "required": True,
        }
        self.threshold_metadata = {
            "artifact_path": str(self.artifact_directory / "best_threshold.json"),
            "key": "upside_zero_threshold",
            "required": True,
        }
        self.artifact_identity = str(self.artifact_directory)

    @property
    def supports_realtime(self) -> bool:
        return bool(self.realtime_supported)

    def metadata(self) -> Dict[str, Any]:
        """Return JSON-friendly lineage metadata for logs and monitoring."""
        return {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "spec_path": str(self.feature_spec_path),
            "feature_spec_path": str(self.feature_spec_path),
            "artifact_directory": str(self.artifact_directory),
            "artifact_identity": self.artifact_identity,
            "feature_list_path": str(self.feature_list_path),
            "ordered_feature_names": list(self.ordered_feature_names),
            "missingness_policy": dict(self.missingness_policy),
            "classifier_metadata": dict(self.classifier_metadata),
            "threshold_metadata": dict(self.threshold_metadata),
            "realtime_supported": self.supports_realtime,
        }

    def load_spec(self) -> Dict[str, Any]:
        return load_model_spec(str(self.feature_spec_path))

    def validate_lineage(
        self,
        spec: Optional[Mapping[str, Any]] = None,
        artifact_directory: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        return _validate_adapter_lineage(
            adapter=self,
            spec=dict(spec) if spec is not None else self.load_spec(),
            artifact_directory=artifact_directory,
        )


class Model2AV1Adapter(Model2AAdapter):
    """Model 2A v1 artifact contract; realtime scoring is explicitly retired."""

    model_version = "v1"
    feature_version = "v1"
    _artifact_relative = "models/intraday_minute_ml_model_2a"
    _default_spec_relative = "config/model_2a_feature_spec.yaml"
    _ordered_feature_names = V1_FEATURE_NAMES
    _feature_builder = _unsupported_v1_feature_builder
    realtime_supported = False
    missingness_policy = {
        "numeric_inputs": "preserve_existing_nan_behavior",
        "quality_flags": "canonical_bitmask_only",
        "realtime": "deprecated_until_v1_highland_semantics_are_reconstructed",
    }


class Model2AV2Adapter(Model2AAdapter):
    """Preferred Model 2A v2 artifact/spec/builder contract."""

    model_version = "v2"
    feature_version = "v2"
    _artifact_relative = "models/intraday_minute_ml_model_2a_v2"
    _default_spec_relative = "config/model_2a_feature_spec_v2.yaml"
    _ordered_feature_names = V2_FEATURE_NAMES
    _feature_builder = build_model_2a_features
    realtime_supported = True
    missingness_policy = {
        "numeric_inputs": "preserve_existing_nan_behavior",
        "quality_flags": "canonical_bitmask_only",
        "wind_fields": "wind_offshore_highland_only",
    }


MODEL_2A_ADAPTER_REGISTRY: Dict[str, Type[Model2AAdapter]] = {
    "v1": Model2AV1Adapter,
    "v2": Model2AV2Adapter,
}


def get_model_2a_adapter(
    model_version: Optional[str],
    model_spec_path: Optional[Union[str, Path]] = None,
) -> Model2AAdapter:
    """Resolve exactly one explicit Model 2A version adapter."""
    normalized = model_version.strip().lower() if isinstance(model_version, str) else ""
    adapter_cls = MODEL_2A_ADAPTER_REGISTRY.get(normalized)
    if adapter_cls is None:
        raise Model2AVersionError(
            "Model 2A model_version must explicitly be 'v1' or 'v2'; "
            f"received {model_version!r}."
        )
    return adapter_cls(model_spec_path=model_spec_path)


def _validate_adapter_lineage(
    adapter: Model2AAdapter,
    spec: Mapping[str, Any],
    artifact_directory: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate one adapter/spec/artifact tuple before model scoring."""
    if spec.get("model_version") != adapter.model_version:
        raise Model2ALineageError(
            f"Requested {adapter.model_version} but YAML spec declares "
            f"{spec.get('model_version')!r}."
        )
    if spec.get("feature_version") != adapter.feature_version:
        raise Model2ALineageError(
            f"Model 2A {adapter.model_version} requires feature_version "
            f"{adapter.feature_version!r}; received {spec.get('feature_version')!r}."
        )

    expected_artifact_dir = adapter.artifact_directory
    actual_artifact_dir = (
        _resolve_path(artifact_directory)
        if artifact_directory is not None
        else expected_artifact_dir
    )
    if actual_artifact_dir != expected_artifact_dir:
        raise Model2ALineageError(
            f"Artifact directory identity mismatch for {adapter.model_version}: "
            f"expected {expected_artifact_dir}, received {actual_artifact_dir}."
        )
    if not actual_artifact_dir.is_dir():
        raise Model2ALineageError(
            f"Model 2A artifact directory does not exist: {actual_artifact_dir}"
        )

    spec_feature_list_raw = spec.get("feature_list_path")
    if not spec_feature_list_raw:
        raise Model2ALineageError("Model 2A YAML spec has no feature_list_path.")
    spec_feature_list_path = _resolve_path(spec_feature_list_raw)
    expected_feature_list_path = actual_artifact_dir / "feature_list.json"
    if spec_feature_list_path != expected_feature_list_path:
        raise Model2ALineageError(
            f"Spec/artifact identity mismatch for {adapter.model_version}: "
            f"spec points to {spec_feature_list_path}, expected "
            f"{expected_feature_list_path}."
        )

    required_paths = [
        expected_feature_list_path,
        actual_artifact_dir / "upside_zero.txt",
        actual_artifact_dir / "best_threshold.json",
    ] + [actual_artifact_dir / f"upside_q{q}.txt" for q in _QUANTILES]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise Model2ALineageError(
            f"Model 2A {adapter.model_version} artifacts are incomplete: "
            f"missing {missing_paths}."
        )

    try:
        with expected_feature_list_path.open("r", encoding="utf-8") as handle:
            feature_names = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Model2ALineageError(
            f"Cannot read feature list {expected_feature_list_path}: {exc}"
        ) from exc

    if not isinstance(feature_names, list) or not feature_names:
        raise Model2ALineageError("feature_list.json must be a non-empty JSON list.")
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise Model2ALineageError("feature_list.json contains an invalid feature name.")
    if len(feature_names) != len(set(feature_names)):
        raise Model2ALineageError("feature_list.json contains duplicate feature names.")
    if feature_names != list(adapter.ordered_feature_names):
        raise Model2ALineageError(
            f"{adapter.model_version} feature_list.json has missing, extra, or "
            "reordered features relative to its declared lineage."
        )

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise Model2ALineageError(
            "lightgbm is required for Model 2A lineage validation."
        ) from exc

    booster_feature_names: Dict[str, list] = {}
    for q in _QUANTILES:
        artifact_name = f"upside_q{q}"
        booster_path = actual_artifact_dir / f"{artifact_name}.txt"
        try:
            booster = lgb.Booster(model_file=str(booster_path))
            names = list(booster.feature_name())
        except Exception as exc:
            raise Model2ALineageError(
                f"Cannot load {adapter.model_version} quantile artifact "
                f"{booster_path}: {exc}"
            ) from exc
        if names != feature_names:
            raise Model2ALineageError(
                f"{artifact_name} feature names/order do not exactly match "
                "feature_list.json."
            )
        booster_feature_names[artifact_name] = names

    classifier_path = actual_artifact_dir / "upside_zero.txt"
    try:
        classifier = lgb.Booster(model_file=str(classifier_path))
        classifier_feature_names = list(classifier.feature_name())
    except Exception as exc:
        raise Model2ALineageError(
            f"Cannot load classifier artifact {classifier_path}: {exc}"
        ) from exc
    if classifier_feature_names != feature_names:
        raise Model2ALineageError(
            "upside_zero.txt feature names/order do not exactly match "
            "feature_list.json."
        )

    threshold_path = actual_artifact_dir / "best_threshold.json"
    try:
        with threshold_path.open("r", encoding="utf-8") as handle:
            threshold_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Model2ALineageError(
            f"Cannot read threshold metadata {threshold_path}: {exc}"
        ) from exc
    threshold = (
        threshold_metadata.get("upside_zero_threshold")
        if isinstance(threshold_metadata, dict)
        else None
    )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise Model2ALineageError(
            "best_threshold.json must contain numeric "
            "upside_zero_threshold in [0, 1]."
        )

    return {
        "model_version": adapter.model_version,
        "feature_version": adapter.feature_version,
        "spec_path": str(adapter.feature_spec_path),
        "feature_spec_path": str(adapter.feature_spec_path),
        "artifact_directory": str(actual_artifact_dir),
        "artifact_identity": adapter.artifact_identity,
        "feature_list_path": str(expected_feature_list_path),
        "ordered_feature_names": list(feature_names),
        "quantile_artifacts": {
            f"upside_q{q}": str(actual_artifact_dir / f"upside_q{q}.txt")
            for q in _QUANTILES
        },
        "classifier_metadata": {
            "artifact_path": str(classifier_path),
            "feature_names": classifier_feature_names,
        },
        "threshold_metadata": {
            "artifact_path": str(threshold_path),
            "key": "upside_zero_threshold",
            "value": float(threshold),
        },
        "booster_feature_names": booster_feature_names,
        "validation": "passed",
    }


def validate_model_2a_lineage(
    model_version: Optional[str],
    model_spec_path: Optional[Union[str, Path]] = None,
    artifact_directory: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Resolve and strictly validate an explicitly selected Model 2A version."""
    adapter = get_model_2a_adapter(model_version, model_spec_path=model_spec_path)
    return adapter.validate_lineage(artifact_directory=artifact_directory)


__all__ = [
    "Model2AAdapter",
    "Model2AV1Adapter",
    "Model2AV2Adapter",
    "Model2ALineageError",
    "Model2AVersionError",
    "MODEL_2A_ADAPTER_REGISTRY",
    "V1_FEATURE_NAMES",
    "V2_FEATURE_NAMES",
    "get_model_2a_adapter",
    "validate_model_2a_lineage",
]
