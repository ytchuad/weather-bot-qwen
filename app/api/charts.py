"""Charts API — read-only endpoints returning time-series data for frontend charts.

All endpoints read from pre-computed data stores (Layer A, CSV, SQLite).
No model loading or live API calls happen here.
"""

from __future__ import annotations

import logging
import re
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


def _minute_comparison_rows(date: str) -> list[dict[str, Any]]:
    """Build chart rows from the minute projection and its CSV fallback."""
    store = get_default_historical_store()
    try:
        return store.minute_history(date_value=date, limit=10000)
    except Exception as exc:
        # Keep the existing SQLite path available for installations missing an
        # optional Layer A reader dependency.  Normal deployments should use
        # the minute projection above.
        logger.warning("Layer A minute history unavailable for chart %s: %s", date, exc)
        return []


def _comparison_payload_from_minute_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    }


@router.get("/models-comparison")
def get_models_comparison_chart(
    date: str | None = None,
    slug: str | None = None,
):
    """Return time-series data for the 'All Models vs Market' comparison chart.

    Reads the Layer A minute projection — no models or APIs are touched.
    Model values are joined backward as-of to each minute.  Repository-synced
    daily CSV is used by that projection after a rebuild; the legacy SQLite
    cycle path remains a compatibility fallback.
    """
    from app.services.weather_service import hkt_now as _hkt_now
    target_date = date or _hkt_now().strftime("%Y-%m-%d")

    minute_rows = _minute_comparison_rows(target_date)
    if minute_rows and not slug:
        return _comparison_payload_from_minute_rows(minute_rows)

    rows = read_models_comparison(date=target_date, slug=slug)

    if not rows:
        return {
            "timestamps": [],
            "market_temps": [],
            "actual_temps": [],
            "models": {},
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
