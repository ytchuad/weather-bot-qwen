"""Protected Layer A export and health endpoints."""

from __future__ import annotations

import hmac
import os
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from layer_a.export import export_layer_a
from layer_a.market_capture import get_default_market_collector
from layer_a.market_storage import get_default_market_store
from layer_a.storage import get_default_store

router = APIRouter(tags=["Layer A Admin"])


def _require_admin(request: Request) -> None:
    configured = os.getenv("LAYER_A_ADMIN_TOKEN", "")
    if not configured:
        raise HTTPException(status_code=503, detail="Layer A admin endpoint is disabled")
    supplied = request.headers.get("X-Layer-A-Admin-Token", "")
    if not supplied:
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="Layer A admin authentication required")


@router.get("/admin/export-layer-a", response_class=FileResponse)
def export_layer_a_endpoint(
    request: Request,
    date_value: str | None = Query(None, alias="date"),
    start: str | None = None,
    end: str | None = None,
    only_unuploaded: bool = False,
    verify_checksums: bool = False,
) -> FileResponse:
    _require_admin(request)
    store = get_default_store()
    from app.config import LAYER_A_EXPORT_DIR

    export_dir = LAYER_A_EXPORT_DIR
    output = export_dir / f"layer_a_export_{uuid.uuid4().hex}.zip"
    try:
        export_layer_a(
            store=store,
            output=output,
            date_value=date_value,
            start=start,
            end=end,
            only_unuploaded=only_unuploaded,
            verify_checksums=verify_checksums,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Layer A export failed") from exc
    return FileResponse(
        path=output,
        media_type="application/zip",
        filename=output.name,
    )


@router.get("/admin/layer-a-health")
def layer_a_health_endpoint(request: Request) -> dict:
    _require_admin(request)
    summary = get_default_store().health_summary()
    summary["market"] = get_default_market_store().health_summary()
    summary["market_collector"] = get_default_market_collector().health_summary()
    try:
        from layer_a.weather_capture import get_default_weather_collector
        from layer_a.weather_storage import get_default_weather_store

        summary["weather"] = get_default_weather_store().health_summary()
        summary["weather_collector"] = get_default_weather_collector().health_summary()
    except Exception:
        summary["weather"] = {"status": "unavailable"}
        summary["weather_collector"] = {"status": "unavailable"}
    try:
        from layer_a.historical_store import get_default_historical_store

        summary["remote_history"] = get_default_historical_store().health_summary()
    except Exception:
        summary["remote_history"] = {"status": "unavailable"}
    try:
        from layer_a.canonical_capture import get_default_canonical_collector

        summary["canonical_collector"] = get_default_canonical_collector().health_summary()
    except Exception:
        summary["canonical_collector"] = {"status": "unavailable"}
    weather_summary = summary.get("weather", {})
    weather_collector = summary.get("weather_collector", {})
    market_summary = summary.get("market", {})
    remote_summary = summary.get("remote_history", {})
    summary.update(
        {
            "last_weather_snapshot": weather_summary.get("last_weather_snapshot"),
            "weather_snapshots_today": weather_summary.get("weather_snapshots_today", 0),
            "weather_capture_failures": int(weather_summary.get("weather_capture_failures", 0)) + int(weather_collector.get("failed_runs", 0)),
            "last_market_snapshot": market_summary.get("last_market_snapshot", market_summary.get("last_successful_snapshot")),
            "market_snapshots_today": market_summary.get("market_snapshots_today", market_summary.get("market_snapshots_captured_today", 0)),
            "last_model_cycle": summary.get("last_model_cycle", summary.get("last_successful_cycle")),
            "model_cycles_today": summary.get("model_cycles_today", summary.get("cycles_captured_today", 0)),
            "remote_history_status": remote_summary.get("status", "unavailable"),
            "remote_history_last_refresh": remote_summary.get("last_refresh"),
            "remote_history_latest_timestamp": remote_summary.get("latest_timestamp"),
            "remote_history_files_cached": remote_summary.get("files_cached", 0),
            "remote_history_refresh_failures": remote_summary.get("refresh_failures", 0),
            "local_minute_chunks_open": int(market_summary.get("local_minute_chunks_open", 0)) + int(weather_summary.get("local_minute_chunks_open", 0)),
            "local_minute_chunks_closed": int(market_summary.get("local_minute_chunks_closed", 0)) + int(weather_summary.get("local_minute_chunks_closed", 0)),
            "oldest_unuploaded_chunk": min(
                [value for value in (market_summary.get("oldest_unuploaded_chunk"), weather_summary.get("oldest_unuploaded_chunk")) if value]
                or [None]
            ),
        }
    )
    return summary


@router.post("/admin/layer-a-history-refresh")
def layer_a_history_refresh_endpoint(
    request: Request,
    date_value: str | None = Query(None, alias="date"),
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Manually refresh the read-only remote history cache."""
    _require_admin(request)
    from layer_a.historical_store import get_default_historical_store

    return {
        "status": "ok",
        "remote_history": get_default_historical_store().manual_refresh(
            date_value=date_value,
            start=start,
            end=end,
        ),
    }
