from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import backtest, charts, data, diagnostics, health, history, layer_a, markets, predictions, strategies, weather

logger = logging.getLogger(__name__)
_layer_a_upload_worker = None
_layer_a_canonical_collector = None
_layer_a_quality_worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Weather Quant API starting")
    try:
        from layer_a.storage import get_default_store

        get_default_store().startup_scan()
    except Exception:
        logger.exception("Layer A startup scan failed")
    try:
        from layer_a.market_storage import get_default_market_store

        get_default_market_store().startup_scan()
    except Exception:
        logger.exception("Layer A market startup scan failed")
    try:
        from layer_a.weather_storage import get_default_weather_store

        get_default_weather_store().startup_scan()
    except Exception:
        logger.exception("Layer A weather startup scan failed")
    try:
        from layer_a.quality import build_and_write_daily_quality_report

        build_and_write_daily_quality_report()
    except Exception:
        logger.exception("Layer A quality report startup generation failed")
    try:
        from layer_a.quality import get_default_quality_worker

        global _layer_a_quality_worker
        _layer_a_quality_worker = get_default_quality_worker()
        _layer_a_quality_worker.start()
        logger.info("Layer A quality report worker started")
    except Exception:
        logger.exception("Layer A quality report worker start failed")
    try:
        from layer_a.market_capture import get_default_market_collector

        market_collector = get_default_market_collector()
        market_collector.start()
        logger.info("Layer A market collector started")
    except Exception:
        logger.exception("Layer A market collector start failed")
    try:
        from layer_a.weather_capture import get_default_weather_collector

        weather_collector = get_default_weather_collector()
        weather_collector.start()
        logger.info("Layer A weather collector started")
    except Exception:
        logger.exception("Layer A weather collector start failed")
    try:
        from layer_a.canonical_capture import get_default_canonical_collector

        global _layer_a_canonical_collector
        _layer_a_canonical_collector = get_default_canonical_collector()
        _layer_a_canonical_collector.start()
        logger.info("Layer A canonical-cycle service initialized")
    except Exception:
        logger.exception("Layer A canonical cycle collector start failed")
    try:
        from layer_a.historical_store import get_default_historical_store

        remote_history = get_default_historical_store()
        remote_history.start_background_refresh()
        if remote_history.auto_refresh and remote_history.repo_id and remote_history.token:
            logger.info("Layer A remote-history refresh started")
        else:
            logger.info("Layer A remote-history refresh disabled")
    except Exception:
        logger.exception("Layer A remote history refresh start failed")
    try:
        from layer_a.upload_worker import LayerAUploadWorker

        global _layer_a_upload_worker
        _layer_a_upload_worker = LayerAUploadWorker()
        _layer_a_upload_worker.start()
        logger.info("Layer A upload worker %s", "started" if _layer_a_upload_worker.enabled else "disabled")
    except Exception:
        logger.exception("Layer A upload worker start failed")
    strategies.start_scheduler()
    yield
    logger.info("Weather Quant API shutting down")
    try:
        from layer_a.market_capture import get_default_market_collector

        get_default_market_collector().stop()
    except Exception:
        logger.exception("Layer A market collector stop failed")
    try:
        from layer_a.weather_capture import get_default_weather_collector

        get_default_weather_collector().stop()
    except Exception:
        logger.exception("Layer A weather collector stop failed")
    if _layer_a_canonical_collector is not None:
        _layer_a_canonical_collector.stop()
    try:
        from layer_a.historical_store import get_default_historical_store

        get_default_historical_store().stop_background_refresh()
    except Exception:
        logger.exception("Layer A remote history refresh stop failed")
    if _layer_a_upload_worker is not None:
        _layer_a_upload_worker.stop()
    if _layer_a_quality_worker is not None:
        _layer_a_quality_worker.stop()
    strategies.stop_scheduler()


app = FastAPI(
    title="Weather Quant API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(weather.router)
app.include_router(predictions.router)
app.include_router(markets.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(diagnostics.router)
app.include_router(charts.router)
app.include_router(data.router)
app.include_router(layer_a.router)
app.include_router(history.router)

_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")
