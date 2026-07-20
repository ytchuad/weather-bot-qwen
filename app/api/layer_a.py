"""Protected Layer A export and health endpoints."""

from __future__ import annotations

import hmac
import os
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from layer_a.export import export_layer_a
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
    return get_default_store().health_summary()
