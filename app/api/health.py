from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
