from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    date: str
    is_min_temp: bool = False
    bias: float = 0.0
    std_mult: float = 1.0


class SuggestRequest(BaseModel):
    date: str
    is_min_temp: bool = False
    capital: float = 10000.0
    kelly_fraction: float = 0.25


class BacktestRequest(BaseModel):
    initial_capital: float = 10000.0
    kelly_fraction: float = 0.25


class BucketProbs(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModelPrediction(BaseModel):
    mean: float
    std: float
    source: str
    probs: BucketProbs | None = None


class PredictionResponse(BaseModel):
    date: str
    models: dict[str, ModelPrediction]


class Suggestion(BaseModel):
    bucket: str
    market_price: float
    model_prob: float
    edge: float
    kelly_fraction: float
    action: Literal["buy_yes", "buy_no", "pass"]


class SuggestResponse(BaseModel):
    date: str
    suggestions: list[Suggestion]


class BacktestTaskCreate(BaseModel):
    task_id: str


class BacktestStatus(BaseModel):
    task_id: str
    status: Literal["running", "done", "error"]
    progress: float = 0.0
    message: str = ""


class BacktestResult(BaseModel):
    summary: dict[str, Any]
    pnl_curve: list[dict[str, Any]]
    brier_scores: list[dict[str, Any]] | None = None


class EventMarket(BaseModel):
    slug: str
    title: str
    buckets: list[dict[str, Any]]
    prices: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class WeatherNow(BaseModel):
    date: str | None = None
    temp: float | None
    humidity: float | None
    max_today: float | None
    min_today: float | None
    forecast: float | None
    aws_temp: float | None
    source: str
    fetched_at: str
