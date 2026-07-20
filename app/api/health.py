from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health", tags=["System"])
def health():
    try:
        from layer_a.storage import get_default_store

        layer_a_summary = get_default_store().health_summary()
    except Exception:
        layer_a_summary = {"status": "unavailable"}
    try:
        from layer_a.market_capture import get_default_market_collector
        from layer_a.market_storage import get_default_market_store

        layer_a_summary["market"] = get_default_market_store().health_summary()
        layer_a_summary["market_collector"] = get_default_market_collector().health_summary()
    except Exception:
        layer_a_summary["market"] = {"status": "unavailable"}
        layer_a_summary["market_collector"] = {"status": "unavailable"}
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer_a": layer_a_summary,
    }
