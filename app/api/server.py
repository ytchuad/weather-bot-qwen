from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import backtest, charts, data, diagnostics, health, layer_a, markets, predictions, strategies, weather

logger = logging.getLogger(__name__)


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
        from layer_a.market_capture import get_default_market_collector

        get_default_market_collector().start()
    except Exception:
        logger.exception("Layer A market collector start failed")
    strategies.start_scheduler()
    yield
    logger.info("Weather Quant API shutting down")
    try:
        from layer_a.market_capture import get_default_market_collector

        get_default_market_collector().stop()
    except Exception:
        logger.exception("Layer A market collector stop failed")
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

_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")
