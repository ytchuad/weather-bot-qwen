# Phase 1: FastAPI API Layer

## Objective

Build a RESTful API layer around the existing Python core (`app/services/*`, `models/*`, `execution/*`). The Streamlit app continues to run unchanged. This phase produces no frontend — only API endpoints consumed later by the React SPA (Phase 2) or testable via Swagger UI.

## Directory Structure

```
app/api/                    # NEW — FastAPI routes
├── __init__.py
├── server.py               # FastAPI app creation, CORS, lifespan
├── schemas.py              # All Pydantic request/response models
├── cache.py                # cachetools.TTLCache wrapper
├── weather.py              # Weather data endpoints
├── predictions.py          # Model prediction endpoints
├── markets.py              # Polymarket endpoints
├── strategies.py           # Strategy suggestion endpoint
├── backtest.py             # Backtest run/status/result endpoints
├── tasks.py                # Background task runner for backtests
└── health.py               # Health check

app/services/               # EXISTING — Unchanged
models/                     # EXISTING — Unchanged
execution/                  # EXISTING — Unchanged
```

## Endpoint Specifications

### 1. Health Check — `health.py`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Returns status, version, timestamp |

### 2. Weather — `weather.py`

| Method | Path | Parameters | Description |
|--------|------|------------|-------------|
| GET | `/api/weather/now` | — | Current temperature, humidity, conditions |
| GET | `/api/weather/intraday` | `date` (str, YYYY-MM-DD) | Full intraday state dict |
| GET | `/api/weather/rain` | `date` (str, YYYY-MM-DD) | Rainfall features |

### 3. Predictions — `predictions.py`

| Method | Path | Parameters | Description |
|--------|------|------------|-------------|
| GET | `/api/predictions` | `date`, `is_min_temp`, `bias`, `std_mult` | All model predictions + bucket probs |

Response shape:
```json
{
  "date": "2026-06-18",
  "models": {
    "9d": { "mean": 31.2, "std": 0.8, "source": "9-Day XGBoost", "probs": { "30-31": 0.25, "31-32": 0.40, ... } },
    "aws": { ... },
    "model_a": { ... }
  }
}
```

### 4. Markets — `markets.py`

| Method | Path | Parameters | Description |
|--------|------|------------|-------------|
| GET | `/api/markets/events` | `date` (str) | Search Polymarket events |
| GET | `/api/markets/event/{slug}` | — | Event details + all bucket markets + prices |
| GET | `/api/markets/buckets` | `type` (tmax|tmin) | Bucket definitions |

### 5. Strategies — `strategies.py`

| Method | Path | Request Body | Description |
|--------|------|-------------|-------------|
| POST | `/api/strategies/suggest` | `{ date, is_min_temp, capital, kelly_frac }` | Kelly-based trade suggestions |

### 6. Backtest — `backtest.py` (Async via BackgroundTasks)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/backtest/run` | Submit backtest task → returns `{ task_id }` |
| GET | `/api/backtest/status/{task_id}` | Query task progress (`running/done/error`) |
| GET | `/api/backtest/result/{task_id}` | Get completed backtest result |

## Pydantic Schema Design (`schemas.py`)

```python
# — Request models —
class PredictionRequest(BaseModel):
    date: str           # YYYY-MM-DD
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

# — Response models —
class BucketProbs(BaseModel):
    model_config = ConfigDict(extra="allow")

class ModelPrediction(BaseModel):
    mean: float
    std: float
    source: str
    probs: BucketProbs

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
    suggestions: list[Suggestion]

class TaskStatus(BaseModel):
    task_id: str
    status: Literal["running", "done", "error"]
    progress: float = 0.0

class BacktestResult(BaseModel):
    summary: dict
    pnl_curve: list[dict]
    # full result from evaluate_forward_test
```

## Caching Strategy (`cache.py`)

Replace Streamlit's `st.cache_data` with `cachetools.TTLCache`:

```python
from cachetools import TTLCache

_weather_cache = TTLCache(maxsize=32, ttl=60)    # 1 min for live weather
_prediction_cache = TTLCache(maxsize=32, ttl=300)  # 5 min for predictions
```

Usage: decorator that fetches from cache or calls original function.

## Async Backtest Flow (`tasks.py`)

```
POST /api/backtest/run
  → generate task_id (uuid4)
  → store { status: "running", progress: 0.0 } in _backtest_tasks dict
  → add BackgroundTask to run `_backtest_worker(task_id, params)`
  → return { task_id } immediately

GET /api/backtest/status/{task_id}
  → return { task_id, status, progress }

_backtest_worker(task_id, params):
  → run backtest_service.evaluate_forward_test(...)
  → store { status: "done", result: ... }
  → handle exceptions → store { status: "error", error: str }
```

## Server Setup (`server.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Weather Quant API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount routers
app.include_router(health.router)
app.include_router(weather.router, prefix="/api/weather")
app.include_router(predictions.router, prefix="/api/predictions")
app.include_router(markets.router, prefix="/api/markets")
app.include_router(strategies.router, prefix="/api/strategies")
app.include_router(backtest.router, prefix="/api/backtest")
```

## Dependencies (to add to `requirements.txt`)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
cachetools>=5.3.3
```

## Acceptance Criteria

- [ ] `GET /api/health` returns 200
- [ ] `GET /api/weather/now` returns real data
- [ ] `GET /api/predictions?date=...` returns model predictions with bucket probs
- [ ] `GET /api/markets/event/{slug}` returns market data
- [ ] `POST /api/backtest/run` → `GET /api/backtest/status/{id}` → `GET /api/backtest/result/{id}` works
- [ ] Swagger UI available at `/docs`
- [ ] Streamlit app still works at its original URL
- [ ] No errors in FastAPI startup or request handling
