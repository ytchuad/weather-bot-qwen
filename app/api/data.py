"""Data Export API — read-only endpoint for downloading snapshot backup.

Used before redeploying HF Spaces, so historical data can be
re-imported via CSV on the next container startup.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

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
    from app.services.weather_service import hkt_now as _hkt_now
    rows = read_snapshots(date=date)
    return {
        "version": 1,
        "exported_at": _hkt_now().isoformat(),
        "snapshot_count": len(rows),
        "snapshots": rows,
    }
