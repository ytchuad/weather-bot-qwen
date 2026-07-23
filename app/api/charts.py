"""Charts API — read-only endpoints returning time-series data for frontend charts.

All endpoints read from pre-computed data stores (Layer A, CSV, SQLite).
No model loading or live API calls happen here.
"""

from __future__ import annotations

import logging
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


def _minute_comparison_projection(date: str, is_min_temp: bool) -> dict[str, Any]:
    """Build chart rows and timestamp diagnostics from the minute projection."""
    store = get_default_historical_store()
    try:
        kwargs = {
            "date_value": date,
            "market_kind": "lowest_temperature" if is_min_temp else "highest_temperature",
            "limit": 10000,
        }
        projection_reader = getattr(store, "minute_history_projection", None)
        if callable(projection_reader):
            projection = projection_reader(**kwargs)
            return {
                "rows": list(projection.get("rows") or []),
                "diagnostics": dict(projection.get("diagnostics") or {}),
                "has_layer_a_records": bool(projection.get("has_layer_a_records")),
            }
        return {"rows": store.minute_history(**kwargs), "diagnostics": {}, "has_layer_a_records": True}
    except Exception as exc:
        # Keep the existing SQLite path available for installations missing an
        # optional Layer A reader dependency.  Normal deployments should use
        # the minute projection above.
        logger.warning("Layer A minute history unavailable for chart %s: %s", date, exc)
        return {"rows": [], "diagnostics": {}, "has_layer_a_records": False}


def _comparison_payload_from_minute_rows(
    rows: list[dict[str, Any]],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamps = [str(row.get("timestamp")) for row in rows]
    market_temps = [_market_expected_temperature(row) for row in rows]
    actual_temps = [row.get("actual_temperature") for row in rows]
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
        "granularity": "strategy_cycle" if legacy_only else "minute",
        "data_source": "legacy_csv" if legacy_only else "layer_a_minute_view",
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
    Model values are joined backward as-of to each minute.  Repository-synced
    daily CSV is used by that projection after a rebuild; the legacy SQLite
    cycle path remains a compatibility fallback.
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
            "granularity": "minute",
            "data_source": "layer_a_minute_view",
            "diagnostics": {
                "excluded_future_weather_records": 0,
                "duplicate_observation_versions": 0,
                "latest_weather_observation_timestamp": None,
                "latest_weather_first_seen_timestamp": None,
            },
        }

    minute_projection = _minute_comparison_projection(target_date, is_min_temp)
    minute_rows = minute_projection["rows"]
    if not slug and (minute_rows or minute_projection["has_layer_a_records"]):
        return _comparison_payload_from_minute_rows(
            minute_rows,
            diagnostics=minute_projection["diagnostics"],
        )

    rows = read_models_comparison(date=target_date, slug=slug)

    if not rows:
        return {
            "timestamps": [],
            "market_temps": [],
            "actual_temps": [],
            "models": {},
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
        "granularity": "strategy_cycle",
        "data_source": "strategy_snapshot_sqlite",
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
