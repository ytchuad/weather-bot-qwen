"""Data Export API — read-only endpoint for downloading snapshot backup.

Used before redeploying HF Spaces, so historical data can be
re-imported via CSV on the next container restart.
"""

from __future__ import annotations

import json
import csv
import io
import logging
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from app.config import DATA_DIR, HF_SPACE_URL
from features.strategy_snapshot_logger import CSV_FIELDS, JSON_FIELDS, read_snapshots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["Data Export"])


def _normalise_export_date(date: str | None) -> str:
    if date is None:
        from app.services.weather_service import hkt_now
        date = hkt_now().strftime("%Y-%m-%d")
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must use YYYY-MM-DD format") from exc


def _get_remote_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "WeatherQuant/1.0"})
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _models_comparison_rows(payload: dict, date: str) -> list[dict]:
    """Build the same partial CSV rows used by download_snapshots.py fallback."""
    timestamps = payload.get("timestamps", [])
    if not timestamps:
        return []

    models = payload.get("models", {})
    market_temps = payload.get("market_temps", [])
    actual_temps = payload.get("actual_temps", [])
    rows: list[dict] = []
    for index, timestamp in enumerate(timestamps):
        all_predictions = {
            model_key: values[index]
            for model_key, values in models.items()
            if index < len(values) and values[index] is not None
        }
        for strategy_key in ("enhanced_v1_paper", "enhanced_v2_paper"):
            rows.append({
                "timestamp": timestamp,
                "snapshot_date": date,
                "slug": "",
                "strategy_key": strategy_key,
                "model_key": "",
                "pm_weighted_temp": market_temps[index] if index < len(market_temps) else None,
                "model_predicted_temp": None,
                "actual_temp": actual_temps[index] if index < len(actual_temps) else None,
                "max_so_far": None,
                "predicted_upside": None,
                "model_std": None,
                "position_size": 0,
                "position_value": 0,
                "all_model_predictions": all_predictions,
                "context_json": {},
            })
    return rows


def _fetch_remote_rows(date: str) -> tuple[list[dict] | None, str | None]:
    """Fetch one day's rows from HF, falling back to models-comparison."""
    base_url = HF_SPACE_URL.rstrip("/")
    if not base_url:
        return None, None

    query = urlencode({"date": date})
    export_url = f"{base_url}/api/data/export-snapshots?{query}"
    try:
        payload = _get_remote_json(export_url)
        snapshots = payload.get("snapshots")
        if isinstance(snapshots, list):
            return snapshots, "hf"
        logger.warning("HF export response did not contain snapshots: %s", export_url)
    except Exception as exc:
        logger.warning("HF snapshot export unavailable for %s: %s", date, exc)

    comparison_url = f"{base_url}/api/charts/models-comparison?{query}"
    try:
        payload = _get_remote_json(comparison_url)
        return _models_comparison_rows(payload, date), "hf-fallback"
    except Exception as exc:
        logger.warning("HF models-comparison fallback unavailable for %s: %s", date, exc)
        return None, None


def _csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for source in rows:
        row = dict(source)
        for field in JSON_FIELDS:
            if isinstance(row.get(field), (dict, list)):
                row[field] = json.dumps(row[field], ensure_ascii=False)
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


@router.get("/export-csv")
def export_snapshots_csv(date: str | None = None):
    """Download one day's snapshot export as a CSV attachment.

    HF is preferred so the button can retrieve snapshots that have not yet
    been synced into the local checkout. A local daily CSV is used as a
    fallback when HF is unavailable.
    """
    target_date = _normalise_export_date(date)
    remote_rows, source = _fetch_remote_rows(target_date)
    local_path = DATA_DIR / "export" / f"{target_date}.csv"
    filename = f"snapshots-{target_date}.csv"

    if remote_rows:
        return Response(
            content=_csv_bytes(remote_rows),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Snapshot-Source": source or "hf",
            },
        )

    if local_path.exists():
        return FileResponse(
            path=local_path,
            media_type="text/csv",
            filename=filename,
            headers={"X-Snapshot-Source": "local"},
        )

    if remote_rows == []:
        raise HTTPException(status_code=404, detail=f"No snapshots found for {target_date}")
    raise HTTPException(status_code=502, detail=f"Snapshot export unavailable for {target_date}")


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
