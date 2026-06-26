# app/config.py
"""Dashboard constants, paths, colours, and bucket definitions."""
from pathlib import Path
from datetime import timedelta

# ---- App identity ----
APP_TITLE = "Weather Quant Dashboard"
APP_FAVICON = "🌦️"

# ---- Timezone ----
HKT_OFFSET = timedelta(hours=8)

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_DIR = ROOT_DIR / "config"

INTRADAY_10MIN_PATH = DATA_DIR / "intraday_hko_10min.parquet"
RAIN_15MIN_PATH = DATA_DIR / "hko_rainfall_15min.parquet"
FORWARD_TEST_LOG = DATA_DIR / "forward_test_log.parquet"
HISTORICAL_TEMP_PATH = DATA_DIR / "hko_tmax_historical.parquet"
PERF_LOG_PATH = DATA_DIR / "model_performance_log.parquet"
PNL_HISTORY_PATH = DATA_DIR / "pnl_history.json"
TRADE_AUDIT_PATH = DATA_DIR / "paper_trade_audit.parquet"
STRATEGY_CONFIG_PATH = CONFIG_DIR / "paper_strategies.json"
PORTFOLIO_CONFIG_PATH = CONFIG_DIR / "portfolios.json"
PORTFOLIO_STATE_PATH = DATA_DIR / "portfolio_state.json"

# ---- Model fallbacks ----
DEFAULT_TMAX_FORECAST_DELTA = 2.0
DEFAULT_TMIN_FORECAST_DELTA = -2.0
DEFAULT_TEMP = 28.0

# ---- Bucket definitions ----
TMAX_BUCKETS = [
    "<23", "23-24", "24-25", "25-26", "26-27", "27-28",
    "28-29", "29-30", "30-31", "31-32", "32-33", "33-34", ">=34",
]

TMIN_BUCKETS = [
    "23 or below", "24C", "25C", "26C", "27C", "28C",
    "29C", "30C", "31C", "32C", "33 or higher",
]

# ---- Model keys ----
MODEL_KEYS = ["9d", "aws", "baseline", "rain_nowcast", "rain_observed", "model_a", "model_b", "model_c", "model_d", "model_e", "model_f", "model_g"]
MODEL_LABELS = {
    "9d": "9-Day XGBoost",
    "aws": "AWS High-Freq",
    "baseline": "Baseline Intraday",
    "rain_nowcast": "Rain Nowcast",
    "rain_observed": "Rain Observed",
    "model_a": "Model A (Minute)",
    "model_b": "Model B (Rain)",
    "model_c": "Model C (Nowcast)",
    "model_d": "Model D",
    "model_e": "Model E",
    "model_f": "Model F (Gap)",
    "model_g": "Model G (Gap+Max)",
}

# ---- Strategy model mapping ----
STRATEGY_MODEL_ALIASES = {
    "baseline_paper": "baseline",
    "rain_observed_paper": "baseline",
    "rain_nowcast_paper": "rain_nowcast",
    "gated_ensemble_paper": "baseline",
    "model_a_paper": "model_a",
    "model_b_paper": "model_b",
    "model_c_paper": "model_c",
}

# ---- HKO API endpoints ----
HKO_RHRREAD_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
HKO_AWS_CSV_URL = "https://www.hko.gov.hk/wxinfo/awsgis/hko.csv"
HKO_MAXMIN_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv"
HKO_FORECAST_URL_TEMPLATE = "https://www.hko.gov.hk/wxinfo/awsgis/forecast/HKO.xml?_t={ts}"

# ---- Polymarket API ----
PM_GAMMA_API = "https://gamma-api.polymarket.com"
PM_SEARCH_URL = f"{PM_GAMMA_API}/public-search"
PM_EVENTS_URL = f"{PM_GAMMA_API}/events"

# ---- Colour palette ----
COLORS = {
    # Per-engine hues (kept for chart series)
    "baseline": "#636EFA",
    "rain_nowcast": "#EF553B",
    "rain_observed": "#00B4D8",
    "model_a": "#00CC96",
    "model_b": "#AB63FA",
    "model_c": "#FFA15A",
    "model_d": "#19D3F3",
    "model_e": "#FF6692",
    "model_f": "#00E5FF",
    "model_g": "#FFB86C",
    "9d": "#1f77b4",
    "aws": "#ff7f0e",
    # Semantic tokens
    "market": "#A78BFA",
    "positive": "#00CC96",
    "negative": "#EF553B",
    "neutral": "#B6B6B6",
    "temperature": "#F43F5E",
    # Crypto-app theme tokens
    "card_bg": "#14171F",
    "card_border": "#1F2330",
    "card_border_hi": "#2A3040",
    "bg": "#0B0E14",
    "text": "#E6E9EF",
    "text_muted": "#6B7280",
    "primary": "#00E5FF",
    "buy_yes": "#00D68F",
    "buy_no": "#FF4D6D",
    "hold": "#6B7280",
    "selected_glow": "rgba(0, 229, 255, 0.25)",
}

# ---- Auto-refresh defaults ----
DEFAULT_REFRESH_INTERVAL_MS = 600_000  # 10 minutes
CACHE_TTL_SHORT = 60       # 1 minute (live data)
CACHE_TTL_MEDIUM = 300     # 5 minutes (model data)
CACHE_TTL_LONG = 3600      # 1 hour (historical data)
