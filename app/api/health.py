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
    try:
        from layer_a.weather_capture import get_default_weather_collector
        from layer_a.weather_storage import get_default_weather_store

        layer_a_summary["weather"] = get_default_weather_store().health_summary()
        layer_a_summary["weather_collector"] = get_default_weather_collector().health_summary()
    except Exception:
        layer_a_summary["weather"] = {"status": "unavailable"}
        layer_a_summary["weather_collector"] = {"status": "unavailable"}
    try:
        from layer_a.historical_store import get_default_historical_store

        layer_a_summary["remote_history"] = get_default_historical_store().health_summary()
    except Exception:
        layer_a_summary["remote_history"] = {"status": "unavailable"}
    try:
        from layer_a.upload_worker import LayerAUploadWorker

        layer_a_summary["upload_worker"] = LayerAUploadWorker().health_summary()
    except Exception:
        layer_a_summary["upload_worker"] = {"status": "unavailable"}
    try:
        from layer_a.canonical_capture import get_default_canonical_collector

        layer_a_summary["canonical_collector"] = get_default_canonical_collector().health_summary()
    except Exception:
        layer_a_summary["canonical_collector"] = {"status": "unavailable"}
    weather_summary = layer_a_summary.get("weather", {})
    weather_collector = layer_a_summary.get("weather_collector", {})
    market_summary = layer_a_summary.get("market", {})
    market_collector = layer_a_summary.get("market_collector", {})
    model_summary = layer_a_summary
    remote_summary = layer_a_summary.get("remote_history", {})
    layer_a_summary.update(
        {
            "last_weather_snapshot": weather_summary.get("last_weather_snapshot"),
            "weather_snapshots_today": weather_summary.get("weather_snapshots_today", 0),
            "weather_capture_failures": int(weather_summary.get("weather_capture_failures", 0)) + int(weather_collector.get("failed_runs", 0)),
            "market_collector_running": bool(market_collector.get("running", False)),
            "weather_collector_running": bool(weather_collector.get("running", False)),
            "market_last_tick": market_collector.get("last_tick"),
            "weather_last_tick": weather_collector.get("last_tick"),
            "market_last_success": market_collector.get("last_success"),
            "weather_last_success": weather_collector.get("last_success"),
            "market_last_error": market_collector.get("last_error"),
            "weather_last_error": weather_collector.get("last_error"),
            "last_market_snapshot": market_summary.get("last_market_snapshot", market_summary.get("last_successful_snapshot")),
            "market_snapshots_today": market_summary.get("market_snapshots_today", market_summary.get("market_snapshots_captured_today", 0)),
            "last_model_cycle": model_summary.get("last_model_cycle", model_summary.get("last_successful_cycle")),
            "model_cycles_today": model_summary.get("model_cycles_today", model_summary.get("cycles_captured_today", 0)),
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
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer_a": layer_a_summary,
    }
