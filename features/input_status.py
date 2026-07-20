"""Truthful, JSON-safe status metadata for model inputs.

The legacy model feature vector is intentionally kept separate from this
module.  A caller may continue to provide a compatibility numeric value while
recording that the value was missing, stale, or synthesized.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


STATUS_CONTRACT_VERSION = "phase2a.v1"
NUMERIC_POLICY = "legacy_compatible"
STATUS_POLICY = "truthful"
DEFAULT_STALE_AFTER_MINUTES = 30.0
FORECAST_STALE_AFTER_MINUTES = 180.0
FORECAST_LARGE_REVISION_C = 1.0

FALLBACK_METHODS = frozenset(
    {
        "previous_observation",
        "cached_api_result",
        "model_compat_zero",
        "climatological_default",
        "unavailable",
    }
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    return False


def normalize_timestamp(value: Any) -> pd.Timestamp | None:
    """Return a usable timestamp or ``None`` without fabricating one."""
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp


def _align_timestamps(
    source_timestamp: pd.Timestamp,
    decision_timestamp: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Align naive/aware timestamps without changing their wall-clock meaning."""
    source_aware = source_timestamp.tzinfo is not None
    decision_aware = decision_timestamp.tzinfo is not None
    if source_aware and not decision_aware:
        source_timestamp = source_timestamp.tz_localize(None)
    elif decision_aware and not source_aware:
        source_timestamp = source_timestamp.tz_localize(decision_timestamp.tz)
    elif source_aware and decision_aware:
        source_timestamp = source_timestamp.tz_convert(decision_timestamp.tz)
    return source_timestamp, decision_timestamp


def calculate_age(
    source_timestamp: Any,
    decision_timestamp: Any,
) -> tuple[float | None, float | None]:
    """Calculate age from the two supplied timestamps.

    Missing timestamps deliberately return ``(None, None)``.  In particular,
    this function never substitutes a default age such as eight minutes.
    """
    source = normalize_timestamp(source_timestamp)
    decision = normalize_timestamp(decision_timestamp)
    if source is None or decision is None:
        return None, None
    source, decision = _align_timestamps(source, decision)
    seconds = float((decision - source).total_seconds())
    return seconds, seconds / 60.0


def _status_raw_name(
    *,
    missing: bool,
    stale: bool,
    fallback: bool,
    fallback_method: str | None,
    source_timestamp: Any,
    source_error: bool = False,
) -> str:
    if source_error:
        return "source_error"
    if fallback_method == "unavailable" and missing:
        return "unavailable"
    if fallback:
        if fallback_method == "cached_api_result":
            return "cached_fallback"
        if fallback_method == "previous_observation":
            return "previous_observation_fallback"
        if fallback_method in {"model_compat_zero", "climatological_default"}:
            return "synthetic_fallback"
        return "fallback"
    if missing:
        return "missing"
    if stale:
        return "stale"
    if normalize_timestamp(source_timestamp) is None:
        return "observed_missing_timestamp"
    return "observed"


@dataclass
class InputStatus:
    """Reusable status contract for one numeric or derived input value."""

    value: Any = None
    source_timestamp: Any = None
    decision_timestamp: Any = None
    age_seconds: float | None = None
    age_minutes: float | None = None
    is_missing: bool = False
    is_stale: bool = False
    is_fallback: bool = False
    fallback_method: str | None = None
    source_name: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    raw_status: str = "observed"
    observation_method: str | None = None

    def __post_init__(self) -> None:
        computed_seconds, computed_minutes = calculate_age(
            self.source_timestamp, self.decision_timestamp
        )
        # Age is derived from timestamps whenever both are available.  Never
        # preserve a caller-supplied fabricated age when the source timestamp
        # is absent.
        self.age_seconds = computed_seconds
        self.age_minutes = computed_minutes
        if not isinstance(self.quality_flags, list):
            self.quality_flags = list(self.quality_flags or [])
        if self.fallback_method and self.fallback_method not in FALLBACK_METHODS:
            self.quality_flags.append("unknown_fallback_method")
        if self.is_missing and "missing_value" not in self.quality_flags:
            self.quality_flags.append("missing_value")
        if self.is_stale and "stale_value" not in self.quality_flags:
            self.quality_flags.append("stale_value")
        if (
            not self.is_missing
            and normalize_timestamp(self.source_timestamp) is None
            and "missing_source_timestamp" not in self.quality_flags
        ):
            self.quality_flags.append("missing_source_timestamp")

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        source_timestamp: Any = None,
        decision_timestamp: Any = None,
        source_name: str | None = None,
        stale_after_minutes: float | None = None,
        quality_flags: list[str] | None = None,
        raw_status: str | None = None,
        is_missing: bool | None = None,
        is_fallback: bool = False,
        fallback_method: str | None = None,
        source_error: bool = False,
        observation_method: str | None = None,
    ) -> "InputStatus":
        missing = _is_missing(value) if is_missing is None else bool(is_missing)
        _, age_minutes = calculate_age(source_timestamp, decision_timestamp)
        stale = bool(
            not missing
            and stale_after_minutes is not None
            and age_minutes is not None
            and age_minutes > float(stale_after_minutes)
        )
        flags = list(quality_flags or [])
        if age_minutes is not None and age_minutes < 0:
            flags.append("future_source_timestamp")
        if raw_status is None:
            raw_status = _status_raw_name(
                missing=missing,
                stale=stale,
                fallback=is_fallback,
                fallback_method=fallback_method,
                source_timestamp=source_timestamp,
                source_error=source_error,
            )
        return cls(
            value=value,
            source_timestamp=source_timestamp,
            decision_timestamp=decision_timestamp,
            is_missing=missing,
            is_stale=stale,
            is_fallback=bool(is_fallback),
            fallback_method=fallback_method,
            source_name=source_name,
            quality_flags=flags,
            raw_status=raw_status,
            observation_method=observation_method,
        )

    @classmethod
    def fallback(
        cls,
        value: Any,
        *,
        fallback_method: str,
        decision_timestamp: Any = None,
        source_timestamp: Any = None,
        source_name: str | None = None,
        quality_flags: list[str] | None = None,
        raw_status: str | None = None,
        observation_method: str | None = None,
    ) -> "InputStatus":
        """Create an explicitly labelled fallback value."""
        return cls.from_value(
            value,
            source_timestamp=source_timestamp,
            decision_timestamp=decision_timestamp,
            source_name=source_name,
            quality_flags=quality_flags,
            raw_status=raw_status,
            is_missing=(
                _is_missing(value)
                or fallback_method in {
                    "model_compat_zero",
                    "climatological_default",
                    "unavailable",
                }
            ),
            is_fallback=True,
            fallback_method=fallback_method,
            observation_method=observation_method,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": jsonable(self.value),
            "source_timestamp": jsonable(self.source_timestamp),
            "decision_timestamp": jsonable(self.decision_timestamp),
            "age_seconds": jsonable(self.age_seconds),
            "age_minutes": jsonable(self.age_minutes),
            "is_missing": bool(self.is_missing),
            "is_stale": bool(self.is_stale),
            "is_fallback": bool(self.is_fallback),
            "fallback_method": self.fallback_method,
            "source_name": self.source_name,
            "quality_flags": list(self.quality_flags),
            "raw_status": self.raw_status,
            "observation_method": self.observation_method,
        }


def jsonable(value: Any) -> Any:
    """Convert status and common pandas/numpy values to JSON-safe values."""
    if isinstance(value, InputStatus):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _is_missing(value):
        return None
    return value


def serialize_status(value: Any) -> str:
    """Serialize a status object/map for parquet, SQLite, or CSV columns."""
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)


def make_status_bundle(
    statuses: Mapping[str, Any] | None = None,
    *,
    decision_timestamp: Any = None,
) -> dict[str, Any]:
    """Build the top-level Phase 2A status envelope."""
    bundle = {
        "status_contract_version": STATUS_CONTRACT_VERSION,
        "numeric_policy": NUMERIC_POLICY,
        "status_policy": STATUS_POLICY,
        "decision_timestamp": decision_timestamp,
    }
    if statuses:
        bundle.update(dict(statuses))
    return jsonable(bundle)


def _date_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return str(value) if value is not None else None


def _timestamp_sort_key(value: pd.Timestamp | None) -> tuple[int, int]:
    if value is None:
        return (0, -1)
    try:
        if value.tzinfo is not None:
            return (1, int(value.tz_convert("UTC").value))
        return (1, int(value.value))
    except (TypeError, ValueError, OverflowError):
        return (0, -1)


def _as_of(source_timestamp: Any, decision_timestamp: Any) -> bool:
    if source_timestamp is None:
        return True
    age_seconds, _ = calculate_age(source_timestamp, decision_timestamp)
    return age_seconds is None or age_seconds >= 0


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    if value is None and name == "forecast_max_temp":
        value = row.get("forecast_max")
    if value is None and name == "forecast_min_temp":
        value = row.get("forecast_min")
    return value


def _forecast_status_field(
    *,
    value: Any,
    issue_time: Any,
    decision_timestamp: Any,
    source_name: str | None,
    target_date: str | None,
    previous_value: Any,
    revision_size: float | None,
    fallback_source: str | None,
    continuity_anomaly: list[str],
    stale_after_minutes: float,
    is_fallback: bool = False,
    fallback_method: str | None = None,
    source_error: bool = False,
) -> dict[str, Any]:
    status = InputStatus.from_value(
        value,
        source_timestamp=issue_time,
        decision_timestamp=decision_timestamp,
        source_name=source_name,
        stale_after_minutes=stale_after_minutes,
        quality_flags=list(continuity_anomaly),
        is_fallback=is_fallback,
        fallback_method=fallback_method,
        source_error=source_error,
        observation_method="forecast_revision",
    ).to_dict()
    status.update(
        {
            "forecast_source": source_name,
            "forecast_issue_time": jsonable(issue_time),
            "forecast_target_date": target_date,
            "forecast_age": status.get("age_minutes"),
            "previous_forecast_value": jsonable(previous_value),
            "revision_size": jsonable(revision_size),
            "fallback_source": fallback_source,
            "continuity_anomaly": list(continuity_anomaly),
        }
    )
    return status


def build_forecast_status_from_values(
    *,
    forecast_max: Any = None,
    forecast_min: Any = None,
    decision_timestamp: Any,
    forecast_issue_time: Any = None,
    forecast_target_date: Any = None,
    forecast_source: str | None = None,
    previous_forecast_max: Any = None,
    previous_forecast_min: Any = None,
    fallback_source: str | None = None,
    continuity_anomaly: list[str] | None = None,
    stale_after_minutes: float = FORECAST_STALE_AFTER_MINUTES,
    source_error: bool = False,
) -> dict[str, Any]:
    """Build forecast status when the caller has scalar source payload data."""
    anomalies = list(continuity_anomaly or [])
    max_revision = None
    min_revision = None
    try:
        if not _is_missing(forecast_max) and not _is_missing(previous_forecast_max):
            max_revision = float(forecast_max) - float(previous_forecast_max)
            if abs(max_revision) >= FORECAST_LARGE_REVISION_C and "large_revision" not in anomalies:
                anomalies.append("large_revision")
    except (TypeError, ValueError):
        pass
    try:
        if not _is_missing(forecast_min) and not _is_missing(previous_forecast_min):
            min_revision = float(forecast_min) - float(previous_forecast_min)
    except (TypeError, ValueError):
        pass
    target_key = _date_key(forecast_target_date)
    max_fallback = "unavailable" if _is_missing(forecast_max) else None
    min_fallback = "unavailable" if _is_missing(forecast_min) else None
    result = {
        "forecast_max": _forecast_status_field(
            value=forecast_max,
            issue_time=forecast_issue_time,
            decision_timestamp=decision_timestamp,
            source_name=forecast_source,
            target_date=target_key,
            previous_value=previous_forecast_max,
            revision_size=max_revision,
            fallback_source=fallback_source,
            continuity_anomaly=anomalies,
            stale_after_minutes=stale_after_minutes,
            is_fallback=max_fallback is not None,
            fallback_method=max_fallback,
            source_error=source_error,
        ),
        "forecast_min": _forecast_status_field(
            value=forecast_min,
            issue_time=forecast_issue_time,
            decision_timestamp=decision_timestamp,
            source_name=forecast_source,
            target_date=target_key,
            previous_value=previous_forecast_min,
            revision_size=min_revision,
            fallback_source=fallback_source,
            continuity_anomaly=anomalies,
            stale_after_minutes=stale_after_minutes,
            is_fallback=min_fallback is not None,
            fallback_method=min_fallback,
            source_error=source_error,
        ),
        "revision_history": [],
        "diagnostics": {
            "target_date_mismatch": "target_date_mismatch" in anomalies,
            "issue_time_regression": "issue_time_regression" in anomalies,
            "large_revision": "large_revision" in anomalies,
            "source_switching": "source_switching" in anomalies,
            "source_error": source_error or "source_error" in anomalies,
            "stale_forecast_reuse": False,
            "missing_issue_timestamp": normalize_timestamp(forecast_issue_time) is None,
        },
        "decision_timestamp": jsonable(decision_timestamp),
    }
    return jsonable(result)


def build_forecast_input_status(
    forecast: pd.DataFrame | None,
    *,
    decision_timestamp: Any,
    target_date: Any,
    stale_after_minutes: float = FORECAST_STALE_AFTER_MINUTES,
    large_revision_threshold: float = FORECAST_LARGE_REVISION_C,
) -> dict[str, Any]:
    """Select an as-of forecast and expose revision/continuity diagnostics.

    The selected row is restricted to the requested target date and to rows
    available at or before the decision timestamp.  A row with a missing issue
    timestamp may be retained for compatibility, but its age remains null and
    it is explicitly flagged.
    """
    target_key = _date_key(target_date)
    if forecast is None or not isinstance(forecast, pd.DataFrame) or forecast.empty:
        return build_forecast_status_from_values(
            decision_timestamp=decision_timestamp,
            forecast_target_date=target_key,
            forecast_source=None,
            fallback_source="unavailable",
            continuity_anomaly=["missing_forecast"],
            stale_after_minutes=stale_after_minutes,
        )

    records: list[dict[str, Any]] = []
    for _, row in forecast.iterrows():
        issue = normalize_timestamp(
            row.get("forecast_issue_datetime", row.get("forecast_issue_time"))
        )
        available = normalize_timestamp(row.get("available_time", row.get("timestamp")))
        row_target = _date_key(row.get("target_date"))
        source = row.get("forecast_source") or row.get("source_system") or row.get("source_name")
        if _is_missing(source):
            source = None
        records.append(
            {
                "row": row,
                "issue": issue,
                "available": available,
                "target": row_target,
                "source": str(source) if source is not None else None,
                "as_of": _as_of(available or issue, decision_timestamp),
            }
        )

    matching = [r for r in records if r["as_of"] and r["target"] == target_key]
    wrong_target = [r for r in records if r["as_of"] and r["target"] != target_key]
    matching.sort(key=lambda r: _timestamp_sort_key(r["issue"] or r["available"]))
    selected = matching[-1] if matching else None
    previous = matching[-2] if len(matching) >= 2 else None

    anomalies: list[str] = []
    if wrong_target or any(r["target"] is None for r in records if r["as_of"]):
        anomalies.append("target_date_mismatch")
    as_of_targets = {r["target"] for r in records if r["as_of"] and r["target"] is not None}
    if len(as_of_targets) > 1:
        anomalies.append("unexpected_target_date_change")

    issue_values = [r["issue"] for r in records if r["issue"] is not None]
    if any(
        _timestamp_sort_key(current) < _timestamp_sort_key(previous_issue)
        for previous_issue, current in zip(issue_values, issue_values[1:])
    ):
        anomalies.append("issue_time_regression")

    if selected is not None and previous is not None and selected["source"] != previous["source"]:
        anomalies.append("source_switching")
    if selected is not None and selected["issue"] is None:
        anomalies.append("missing_issue_timestamp")
    if any(record["issue"] is None for record in matching):
        if "missing_issue_timestamp" not in anomalies:
            anomalies.append("missing_issue_timestamp")

    def _revision(field: str) -> float | None:
        if selected is None or previous is None:
            return None
        current_value = _row_value(selected["row"], field)
        previous_value = _row_value(previous["row"], field)
        if _is_missing(current_value) or _is_missing(previous_value):
            return None
        try:
            return float(current_value) - float(previous_value)
        except (TypeError, ValueError):
            return None

    revision_sizes = {field: _revision(field) for field in ("forecast_max_temp", "forecast_min_temp")}
    history_large_revision = False
    for previous_record, current_record in zip(matching, matching[1:]):
        for forecast_field in ("forecast_max_temp", "forecast_min_temp"):
            previous_value = _row_value(previous_record["row"], forecast_field)
            current_value = _row_value(current_record["row"], forecast_field)
            if _is_missing(previous_value) or _is_missing(current_value):
                continue
            try:
                if abs(float(current_value) - float(previous_value)) >= large_revision_threshold:
                    history_large_revision = True
            except (TypeError, ValueError):
                continue
    if history_large_revision or any(
        revision is not None and abs(revision) >= large_revision_threshold
        for revision in revision_sizes.values()
    ):
        anomalies.append("large_revision")

    def _status(field: str) -> dict[str, Any]:
        if selected is None:
            return _forecast_status_field(
                value=None,
                issue_time=None,
                decision_timestamp=decision_timestamp,
                source_name=None,
                target_date=target_key,
                previous_value=None,
                revision_size=None,
                fallback_source="unavailable",
                continuity_anomaly=anomalies + ["missing_forecast"],
                stale_after_minutes=stale_after_minutes,
                is_fallback=True,
                fallback_method="unavailable",
                source_error="source_error" in anomalies,
            )
        value = _row_value(selected["row"], field)
        previous_value = _row_value(previous["row"], field) if previous else None
        status = _forecast_status_field(
            value=value,
            issue_time=selected["issue"],
            decision_timestamp=decision_timestamp,
            source_name=selected["source"],
            target_date=selected["target"] or target_key,
            previous_value=previous_value,
            revision_size=revision_sizes[field],
            fallback_source=None,
            continuity_anomaly=anomalies,
            stale_after_minutes=stale_after_minutes,
            is_fallback=False,
            source_error="source_error" in anomalies,
        )
        if status.get("is_stale"):
            status["continuity_anomaly"] = list(status["continuity_anomaly"]) + [
                "stale_forecast_reuse"
            ]
            status["quality_flags"] = list(status["quality_flags"]) + [
                "stale_forecast_reuse"
            ]
        return status

    revision_history: list[dict[str, Any]] = []
    for record in matching:
        row = record["row"]
        revision_history.append(
            {
                "forecast_source": record["source"],
                "forecast_issue_time": jsonable(record["issue"]),
                "forecast_target_date": record["target"],
                "forecast_max_temp": jsonable(_row_value(row, "forecast_max_temp")),
                "forecast_min_temp": jsonable(_row_value(row, "forecast_min_temp")),
            }
        )

    diagnostics = {
        "target_date_mismatch": "target_date_mismatch" in anomalies,
        "unexpected_target_date_change": "unexpected_target_date_change" in anomalies,
        "issue_time_regression": "issue_time_regression" in anomalies,
        "large_revision": "large_revision" in anomalies,
        "source_switching": "source_switching" in anomalies,
        "source_error": "source_error" in anomalies,
        "stale_forecast_reuse": bool(
            selected
            and selected["issue"]
            and _status("forecast_max_temp").get("is_stale")
        ),
        "missing_issue_timestamp": "missing_issue_timestamp" in anomalies,
    }
    return jsonable(
        {
            "forecast_max": _status("forecast_max_temp"),
            "forecast_min": _status("forecast_min_temp"),
            "revision_history": revision_history,
            "diagnostics": diagnostics,
            "decision_timestamp": decision_timestamp,
        }
    )


def _source_column(frame: pd.DataFrame) -> str | None:
    for column in ("timestamp", "datetime", "available_time"):
        if column in frame.columns:
            return column
    return None


def _latest_source_timestamp(
    frame: pd.DataFrame,
    *,
    value_column: str | None,
    decision_timestamp: Any,
    selector: Any = None,
) -> pd.Timestamp | None:
    if frame is None or frame.empty:
        return None
    source_column = _source_column(frame)
    if source_column is None:
        return None
    subset = frame.copy()
    if selector is not None:
        subset = subset.loc[selector]
    if value_column and value_column in subset.columns:
        subset = subset.loc[~subset[value_column].map(_is_missing)]
    if subset.empty:
        return None
    timestamps = subset[source_column].map(normalize_timestamp)
    timestamps = [ts for ts in timestamps if ts is not None and _as_of(ts, decision_timestamp)]
    return max(timestamps, key=_timestamp_sort_key) if timestamps else None


def build_observation_buffer_status(
    frame: pd.DataFrame | None,
    *,
    decision_timestamp: Any,
    values: Mapping[str, Any] | None = None,
    stale_after_minutes: float = DEFAULT_STALE_AFTER_MINUTES,
) -> dict[str, Any]:
    """Describe current, lagged, extrema, and rolling buffer values."""
    values = dict(values or {})
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    source_column = _source_column(frame) if not frame.empty else None
    as_of_frame = frame
    if source_column is not None:
        as_of_frame = frame.loc[
            frame[source_column].map(lambda value: _as_of(value, decision_timestamp))
        ].copy()

    column_aliases = {
        "temp_current": ("temp_now", "temp", "temp_current"),
        "rh_current": ("rh_now", "rh", "rh_current"),
        "pressure_current": ("pressure_current", "pressure"),
        "dew_point_current": ("dew_point_current", "dew_point"),
    }

    def _value_for(name: str, aliases: tuple[str, ...]) -> Any:
        for alias in aliases:
            if alias in values:
                return values[alias]
        for alias in aliases[1:]:
            if alias in as_of_frame.columns:
                valid = as_of_frame.loc[~as_of_frame[alias].map(_is_missing)]
                if not valid.empty:
                    return valid.sort_values(source_column).iloc[-1][alias] if source_column else valid.iloc[-1][alias]
        return None

    statuses: dict[str, Any] = {}
    for name, aliases in column_aliases.items():
        value = _value_for(name, aliases)
        frame_column = next((alias for alias in aliases[1:] if alias in as_of_frame.columns), None)
        source = _latest_source_timestamp(
            as_of_frame,
            value_column=frame_column,
            decision_timestamp=decision_timestamp,
        )
        statuses[name] = InputStatus.from_value(
            value,
            source_timestamp=source,
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs" if name != "pressure_current" else "hko_pressure",
            stale_after_minutes=stale_after_minutes,
            observation_method="direct_observation" if source is not None else "insufficient_history",
        ).to_dict()

    for name, column in (("max_so_far", "temp"), ("min_so_far", "temp")):
        value = values.get(name)
        if value is None and column in as_of_frame.columns and not as_of_frame.empty:
            numeric = pd.to_numeric(as_of_frame[column], errors="coerce").dropna()
            if not numeric.empty:
                value = float(numeric.max() if name == "max_so_far" else numeric.min())
        source = None
        if column in as_of_frame.columns and value is not None:
            numeric = pd.to_numeric(as_of_frame[column], errors="coerce")
            selector = numeric.eq(float(value))
            source = _latest_source_timestamp(
                as_of_frame,
                value_column=column,
                decision_timestamp=decision_timestamp,
                selector=selector,
            )
        statuses[name] = InputStatus.from_value(
            value,
            source_timestamp=source,
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs",
            stale_after_minutes=stale_after_minutes,
            observation_method="direct_observation" if source is not None else "insufficient_history",
        ).to_dict()

    for minutes in (30, 60, 120):
        name = f"temp_{minutes}m_ago"
        value = values.get(name)
        source = None
        method = "insufficient_history"
        target = normalize_timestamp(decision_timestamp)
        if target is not None:
            target = target - pd.Timedelta(minutes=minutes)
        if source_column is not None and target is not None and "temp" in as_of_frame.columns:
            times = as_of_frame[source_column].map(normalize_timestamp)
            candidate = as_of_frame.loc[
                times.map(lambda ts: ts is not None and _as_of(ts, decision_timestamp) and ts <= target)
                & ~as_of_frame["temp"].map(_is_missing)
            ]
            if not candidate.empty:
                row = candidate.sort_values(source_column).iloc[-1]
                source = normalize_timestamp(row[source_column])
                if name not in values:
                    value = row["temp"]
                method = "direct_observation"
        statuses[name] = InputStatus.from_value(
            value,
            source_timestamp=source,
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs",
            stale_after_minutes=stale_after_minutes,
            is_fallback=method != "direct_observation",
            fallback_method="unavailable" if method != "direct_observation" else None,
            raw_status=method if method != "direct_observation" else None,
            observation_method=method,
        ).to_dict()

    for name in (
        "temp_change_30m",
        "temp_change_60m",
        "temp_volatility_60m",
        "temp_acceleration_60m",
        "rh_change_60m",
        "dew_point_change_60m",
        "dew_point_spread_change_60m",
        "time_since_max",
        "time_since_min",
    ):
        value = values.get(name)
        source = _latest_source_timestamp(
            as_of_frame,
            value_column="temp" if "temp" in as_of_frame.columns else None,
            decision_timestamp=decision_timestamp,
        )
        statuses[name] = InputStatus.from_value(
            value,
            source_timestamp=source,
            decision_timestamp=decision_timestamp,
            source_name="hko_weather_obs",
            stale_after_minutes=stale_after_minutes,
            observation_method="derived_rolling" if source is not None else "insufficient_history",
        ).to_dict()

    latest_source = _latest_source_timestamp(
        as_of_frame,
        value_column="temp" if "temp" in as_of_frame.columns else None,
        decision_timestamp=decision_timestamp,
    )
    _, observation_age = calculate_age(latest_source, decision_timestamp)
    statuses["obs_data_age_minutes"] = InputStatus.from_value(
        observation_age,
        source_timestamp=latest_source,
        decision_timestamp=decision_timestamp,
        source_name="hko_weather_obs",
        stale_after_minutes=stale_after_minutes,
        observation_method="source_age" if latest_source is not None else "insufficient_history",
    ).to_dict()
    return jsonable(statuses)


def attach_status_metadata_to_context(
    context_json: dict[str, Any],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy status/lineage metadata to snapshot context without flattening it."""
    if not metadata:
        return context_json
    clean = jsonable(metadata)
    context_json["feature_metadata"] = clean
    for key in (
        "status_contract_version",
        "numeric_policy",
        "status_policy",
        "decision_timestamp",
        "model_lineage",
        "feature_spec",
        "weather_input_status",
        "wind_input_status",
        "pressure_input_status",
        "forecast_input_status",
        "observation_buffer_status",
        "rain_input_status",
        "nowcast_input_status",
        "input_status",
    ):
        if key in clean:
            context_json[key] = clean[key]
    return context_json
