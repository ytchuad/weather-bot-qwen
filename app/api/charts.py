"""Charts API — read-only endpoints returning time-series data for frontend charts.

All endpoints read from pre-computed data stores (SQLite, Parquet).
No model loading or live API calls happen here.
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException

from features.strategy_snapshot_logger import read_models_comparison, read_snapshots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/charts", tags=["Charts"])


@router.get("/models-comparison")
def get_models_comparison_chart(
    date: str | None = None,
    slug: str | None = None,
):
    """Return time-series data for the 'All Models vs Market' comparison chart.

    Reads from the snapshot SQLite database — no models or APIs are touched.
    Returns per-timestamp Polymarket weighted temp, actual temp, and every
    model's predicted temperature.
    """
    from app.services.weather_service import hkt_now as _hkt_now
    target_date = date or _hkt_now().strftime("%Y-%m-%d")

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

    timestamps: list[str] = []
    models: dict[str, list[float | None]] = {}
    market_prices: list[float | None] = []

    for r in rows:
        ts = r["timestamp"]
        timestamps.append(ts)
        ctx = r.get("context_json") or {}
        mp = ctx.get("model_probs") or {}
        for mk, probs in mp.items():
            if isinstance(probs, dict):
                if mk not in models:
                    models[mk] = []
                models[mk].append(probs.get(bucket))
        # ensure every model key has a value for this timestamp
        for mk in list(models.keys()):
            if len(models[mk]) < len(timestamps):
                models[mk].append(None)

        mkt = ctx.get("market_prices") or {}
        market_prices.append(mkt.get(bucket))

    sorted_model_keys = sorted(models.keys())
    sorted_models: dict[str, list[float | None]] = {k: models[k] for k in sorted_model_keys}

    return {
        "timestamps": timestamps,
        "models": sorted_models,
        "market_prices": market_prices,
        "bucket": bucket,
        "available_buckets": sorted(all_buckets),
    }
