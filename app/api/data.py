"""Data Export API — read-only endpoint for downloading snapshot backup.

Used before redeploying HF Spaces, so historical data can be
re-imported via CSV on the next container restart.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import Response

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
    import math
    from app.services.weather_service import hkt_now as _hkt_now

    rows = read_snapshots(date=date)

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
    json_bytes = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=True).encode("utf-8")
    return Response(content=json_bytes, media_type="application/json")
