"""Data Export API — read-only endpoint for downloading snapshot backup.

Used before redeploying HF Spaces, so historical data can be
re-imported via CSV on the next container restart.
"""

from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from features.strategy_snapshot_logger import read_snapshots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["Data Export"])


@router.get("/export-snapshots")
def export_snapshots(date: str | None = None):
    """Export all snapshots as JSON (for backup before redeploy).

    Returns every snapshot row with parsed ``context_json`` and
    ``all_model_predictions`` dicts, suitable for reconstructing the
    SQLite database via CSV import on the next startup.
    """
    import json
    import math
    from app.services.weather_service import hkt_now as _hkt_now

    rows = read_snapshots(date=date)

    # Sanitize numeric fields that may contain float('inf')/nan from SQLite
    FLOAT_FIELDS = {
        "pm_weighted_temp", "model_predicted_temp", "actual_temp",
        "max_so_far", "predicted_upside", "model_std",
        "position_size", "position_value",
    }
    for r in rows:
        for fld in FLOAT_FIELDS:
            v = r.get(fld)
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                r[fld] = None

    payload = {
        "version": 1,
        "exported_at": _hkt_now().isoformat(),
        "snapshot_count": len(rows),
        "snapshots": rows,
    }
    try:
        return JSONResponse(content=payload)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("export-snapshots serialization failed: %s\n%s", exc, tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "traceback": tb.split("\n"),
            },
        )


@router.get("/debug-response")
def debug_response(date: str):
    """Return count and sample rows for debugging the 500 error."""
    try:
        rows = read_snapshots(date=date)
        sample = []
        for r in rows[:3]:
            r2 = dict(r)
            for k in ("all_model_predictions", "context_json"):
                if isinstance(r2.get(k), dict):
                    r2[k] = {"_type": "dict", "_len": len(r2[k])}
            sample.append({
                "timestamp": r2.get("timestamp"),
                "pm_weighted_temp": r2.get("pm_weighted_temp"),
                "context_json_type": type(r.get("context_json")).__name__,
                "context_json_len": len(r.get("context_json")) if isinstance(r.get("context_json"), dict) else "N/A",
                "all_model_predictions_type": type(r.get("all_model_predictions")).__name__,
            })
        return {
            "total_rows": len(rows),
            "sample": sample,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc), "traceback": traceback.format_exc().split("\n")})
