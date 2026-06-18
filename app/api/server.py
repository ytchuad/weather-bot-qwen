from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import backtest, health, markets, predictions, strategies, weather

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Weather Quant API starting")
    yield
    logger.info("Weather Quant API shutting down")


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


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
