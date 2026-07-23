"""Charts API — read-only endpoints returning time-series data for frontend charts.

All endpoints read from pre-computed data stores (Layer A, CSV, SQLite).
No model loading or live API calls happen here.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import date as calendar_date
from typing import Any, Mapping

from fastapi import APIRouter

from layer_a.historical_store import get_default_historical_store
from features.strategy_snapshot_logger import read_models_comparison, read_snapshots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/charts", tags=["Charts"])

_RANGE_BUCKET = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")
_LOW_BUCKET = re.compile(r"^\s*(?:<|<=)\s*(-?\d+(?:\.\d+)?)\s*$")
_HIGH_BUCKET = re.compile(r"^\s*(?:>|>=)\s*(-?\d+(?:\.\d+)?)\s*$")


def _legacy_trajectory_fallback_enabled() -> bool:
    return os.getenv("ENABLE_LEGACY_TRAJECTORY_FALLBACK", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _bucket_midpoint(bucket: Any) -> float | None:
    label = str(bucket or "")
    match = _RANGE_BUCKET.match(label)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2.0
    match = _LOW_BUCKET.match(label)
    if match:
        return float(match.group(1)) - 0.5
    match = _HIGH_BUCKET.match(label)
    if match:
        return float(match.group(1)) + 0.5
    return None


def _market_expected_temperature(row: Mapping[str, Any]) -> float | None:
    legacy_value = row.get("market_expected_temperature")
    if isinstance(legacy_value, (int, float)) and legacy_value == legacy_value:
        return float(legacy_value)

    prices = row.get("market_prices")
    if not isinstance(prices, Mapping):
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for bucket, raw_price in prices.items():
        midpoint = _bucket_midpoint(bucket)
        if midpoint is None:
            continue
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        if price <= 0 or price != price:
            continue
        weighted_sum += midpoint * price
        total_weight += price
    return weighted_sum / total_weight if total_weight > 0 else None


def _point_prediction(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("point_prediction")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _expected_model_cycle_interval_seconds() -> float | None:
    """Expose the scheduler's configured cadence to trajectory clients.

    The chart uses this value only to decide whether two *real* model cycles
    may be visually joined.  Keeping it here avoids a frontend-only stale
    cutoff that silently diverges from the canonical collector configuration.
    """
    try:
        from layer_a.canonical_capture import get_default_canonical_collector

        interval = float(get_default_canonical_collector().interval_seconds)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    return interval if math.isfinite(interval) and interval > 0 else None


def _minute_comparison_projection(date: str, is_min_temp: bool) -> dict[str, Any]:
    """Build chart rows and timestamp diagnostics from the minute projection."""
    store = get_default_historical_store()
    try:
        kwargs = {
            "date_value": date,
            "market_kind": "lowest_temperature" if is_min_temp else "highest_temperature",
            "limit": 10000,
            "allow_legacy_fallback": _legacy_trajectory_fallback_enabled(),
        }
        projection_reader = getattr(store, "minute_history_projection", None)
        if callable(projection_reader):
            projection = projection_reader(**kwargs)
            return {
                "rows": list(projection.get("rows") or []),
                "diagnostics": dict(projection.get("diagnostics") or {}),
                "has_layer_a_records": bool(projection.get("has_layer_a_records")),
                "status": str(projection.get("status") or "ok"),
                "data_source": str(projection.get("data_source") or "layer_a_minute_view"),
            }
        minute_kwargs = {key: value for key, value in kwargs.items() if key != "allow_legacy_fallback"}
        return {
            "rows": store.minute_history(**minute_kwargs),
            "diagnostics": {},
            "has_layer_a_records": True,
            "status": "ok",
            "data_source": "layer_a_minute_view",
        }
    except Exception as exc:
        logger.exception("Layer A minute history unavailable for chart %s", date)
        return {
            "rows": [],
            "diagnostics": {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": "Layer A trajectory history is unavailable",
            },
            "has_layer_a_records": False,
            "status": "error",
            "data_source": "layer_a_minute_view",
        }


def _comparison_payload_from_minute_rows(
    rows: list[dict[str, Any]],
    *,
    diagnostics: Mapping[str, Any] | None = None,
    status: str = "ok",
    data_source: str | None = None,
) -> dict[str, Any]:
    timestamps = [str(row.get("timestamp")) for row in rows]
    market_temps = [_market_expected_temperature(row) for row in rows]
    actual_temps = [row.get("actual_temperature") for row in rows]
    point_metadata = [
        {
            "timestamp": row.get("timestamp"),
            "actual_observation_timestamp": row.get("actual_observation_timestamp"),
            "actual_first_seen_timestamp": row.get("actual_first_seen_timestamp"),
            "actual_source_release_timestamp": row.get("actual_source_release_timestamp"),
            "actual_release_lag_seconds": row.get("actual_release_lag_seconds"),
            "prediction_decision_timestamp": row.get("model_cycle_timestamp"),
            "weather_data_through": row.get("weather_data_through"),
            "weather_first_seen_timestamp": row.get("weather_first_seen_timestamp"),
            "weather_age_seconds": row.get("weather_age_seconds"),
            "weather_snapshot_id": row.get("weather_snapshot_id"),
            "model_cycle_id": row.get("model_cycle_id"),
            "model_age_seconds": row.get("model_age_seconds"),
            "model_cycle_is_real": row.get("model_cycle_is_real"),
        }
        for row in rows
    ]
    sources = {str(row.get("source") or "layer_a") for row in rows}
    legacy_only = sources == {"legacy_csv"}
    all_model_keys: set[str] = set()
    predictions_by_row: list[dict[str, float | None]] = []

    for row in rows:
        raw_models = row.get("model_predictions") or row.get("models") or {}
        predictions: dict[str, float | None] = {}
        if isinstance(raw_models, Mapping):
            for model_key, value in raw_models.items():
                parsed = _point_prediction(value)
                if parsed is not None:
                    key = str(model_key)
                    predictions[key] = parsed
                    all_model_keys.add(key)
        predictions_by_row.append(predictions)

    model_keys_sorted = sorted(
        all_model_keys,
        key=lambda key: (
            0 if key == "9d" else
            1 if key == "aws" else
            2 if key in ("baseline", "model_a") else
            3 if key.startswith("model_") else 9,
            key,
        ),
    )
    models = {
        key: [predictions.get(key) for predictions in predictions_by_row]
        for key in model_keys_sorted
    }
    return {
        "timestamps": timestamps,
        "market_temps": market_temps,
        "actual_temps": actual_temps,
        "models": models,
        "point_metadata": point_metadata,
        "expected_model_cycle_interval_seconds": _expected_model_cycle_interval_seconds(),
        "granularity": "strategy_cycle" if legacy_only else "minute",
        "data_source": data_source or ("legacy_csv" if legacy_only else "layer_a_minute_view"),
        "status": status,
        "diagnostics": dict(diagnostics or {}),
    }


@router.get("/models-comparison")
def get_models_comparison_chart(
    date: str | None = None,
    slug: str | None = None,
    is_min_temp: bool = False,
):
    """Return time-series data for the 'All Models vs Market' comparison chart.

    Reads the Layer A minute projection — no models or APIs are touched.
    Model values appear only at real decision-cycle timestamps. Legacy CSV or
    SQLite fallback is disabled by default and must be explicitly enabled with
    ``ENABLE_LEGACY_TRAJECTORY_FALLBACK=true``.
    """
    from app.services.weather_service import hkt_now as _hkt_now
    target_date = date or _hkt_now().strftime("%Y-%m-%d")

    try:
        future_date = calendar_date.fromisoformat(target_date) > _hkt_now().date()
    except ValueError:
        future_date = False
    if future_date:
        return {
            "timestamps": [],
            "market_temps": [],
            "actual_temps": [],
            "models": {},
            "point_metadata": [],
            "expected_model_cycle_interval_seconds": _expected_model_cycle_interval_seconds(),
            "granularity": "minute",
            "data_source": "layer_a_minute_view",
            "diagnostics": {
                "excluded_future_weather_records": 0,
                "excluded_cross_date_weather_records": 0,
                "duplicate_observation_versions": 0,
                "duplicate_weather_snapshot_id_records": 0,
                "weather_snapshot_id_content_collisions": 0,
                "latest_weather_observation_timestamp": None,
                "latest_weather_first_seen_timestamp": None,
            },
            "status": "empty",
        }

    minute_projection = _minute_comparison_projection(target_date, is_min_temp)
    minute_rows = minute_projection["rows"]
    if not slug or not _legacy_trajectory_fallback_enabled():
        return _comparison_payload_from_minute_rows(
            minute_rows,
            diagnostics=minute_projection["diagnostics"],
            status=minute_projection["status"],
            data_source=minute_projection["data_source"],
        )

    rows = read_models_comparison(date=target_date, slug=slug)

    if not rows:
        return {
            "timestamps": [],
            "market_temps": [],
            "actual_temps": [],
            "models": {},
            "point_metadata": [],
            "expected_model_cycle_interval_seconds": _expected_model_cycle_interval_seconds(),
            "granularity": "strategy_cycle",
            "data_source": "strategy_snapshot_sqlite",
            "status": "empty",
            "diagnostics": minute_projection["diagnostics"],
        }

    timestamps = []
    market_temps = []
    actual_temps = []
    all_model_keys: set[str] = set()

    for r in rows:
        timestamps.append(r["timestamp"])
        market_temps.append(r.get("pm_weighted_temp"))
        actual_temps.append(r.get("actual_temp"))
        all_model_keys.update(r.get("model_predictions", {}).keys())

    model_keys_sorted = sorted(
        all_model_keys,
        key=lambda k: (
            0 if k == "9d" else
            1 if k == "aws" else
            2 if k in ("baseline", "model_a") else
            3 if k.startswith("model_") else 9,
            k,
        ),
    )

    models: dict[str, list] = {mk: [] for mk in model_keys_sorted}
    for r in rows:
        preds = r.get("model_predictions", {})
        for mk in model_keys_sorted:
            models[mk].append(preds.get(mk))

    return {
        "timestamps": timestamps,
        "market_temps": market_temps,
        "actual_temps": actual_temps,
        "models": models,
        "point_metadata": [
            {
                "timestamp": row.get("timestamp"),
                "actual_observation_timestamp": row.get("actual_observation_timestamp"),
                "actual_first_seen_timestamp": row.get("actual_first_seen_timestamp"),
                "actual_source_release_timestamp": row.get("actual_source_release_timestamp"),
                "actual_release_lag_seconds": row.get("actual_release_lag_seconds"),
                "prediction_decision_timestamp": row.get("model_cycle_timestamp"),
                "weather_data_through": row.get("weather_data_through"),
                "weather_first_seen_timestamp": row.get("weather_first_seen_timestamp"),
                "weather_age_seconds": row.get("weather_age_seconds"),
                "weather_snapshot_id": row.get("weather_snapshot_id"),
                "model_cycle_id": row.get("model_cycle_id"),
                "model_age_seconds": row.get("model_age_seconds"),
                "model_cycle_is_real": row.get("model_cycle_is_real"),
            }
            for row in rows
        ],
        "expected_model_cycle_interval_seconds": _expected_model_cycle_interval_seconds(),
        "granularity": "strategy_cycle",
        "data_source": "strategy_snapshot_sqlite",
        "status": "legacy_fallback",
        "diagnostics": minute_projection["diagnostics"],
    }


@router.get("/bucket-probs")
def get_bucket_probs_chart(
    date: str | None = None,
    bucket: str | None = None,
):
    """Return per-timestamp model probabilities and market prices for a bucket.

    Reads from ``context_json['model_probs']`` and ``context_json['market_prices']``
    in the snapshot SQLite database.
    """
    from app.services.weather_service import hkt_now as _hkt_now
    target_date = date or _hkt_now().strftime("%Y-%m-%d")

    rows = read_snapshots(date=target_date)

    if not rows:
        return {"timestamps": [], "models": {}, "market_prices": []}

    # gather all buckets available across all snapshots
    all_buckets: set[str] = set()
    for r in rows:
        ctx = r.get("context_json") or {}
        mp = ctx.get("model_probs") or {}
        for mk, probs in mp.items():
            if isinstance(probs, dict):
                all_buckets.update(probs.keys())

    if not bucket:
        bucket = sorted(all_buckets)[0] if all_buckets else ""

    # Two-pass approach to ensure array alignment:
    #   1st pass — discover all model keys across all snapshots
    #   2nd pass — fill arrays at correct indices

    timestamps: list[str] = [r["timestamp"] for r in rows]
    all_model_keys: set[str] = set()

    for r in rows:
        ctx = r.get("context_json") or {}
        mp = ctx.get("model_probs") or {}
        for mk, probs in mp.items():
            if isinstance(probs, dict):
                all_model_keys.add(mk)

    sorted_model_keys = sorted(all_model_keys)

    models: dict[str, list[float | None]] = {
        mk: [None] * len(rows) for mk in sorted_model_keys
    }
    market_prices: list[float | None] = [None] * len(rows)

    for i, r in enumerate(rows):
        ctx = r.get("context_json") or {}
        mp = ctx.get("model_probs") or {}
        for mk, probs in mp.items():
            if isinstance(probs, dict) and mk in models:
                models[mk][i] = probs.get(bucket)

        mkt = ctx.get("market_prices") or {}
        market_prices[i] = mkt.get(bucket)

    return {
        "timestamps": timestamps,
        "models": models,
        "market_prices": market_prices,
        "bucket": bucket,
        "available_buckets": sorted(all_buckets),
    }
