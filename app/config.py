# app/config.py
"""Dashboard constants, paths, colours, and bucket definitions."""
from pathlib import Path
from datetime import timedelta
import os

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

# ---- Layer A canonical capture (ephemeral local runtime state) ----
LAYER_A_DIR = Path(os.getenv("LAYER_A_DIR", str(DATA_DIR / "layer_a")))
LAYER_A_EXPORT_DIR = Path(os.getenv("LAYER_A_EXPORT_DIR", str(DATA_DIR / "layer_a_exports")))
LAYER_A_MARKET_DIR = Path(os.getenv("LAYER_A_MARKET_DIR", str(DATA_DIR / "layer_a_market")))
LAYER_A_WEATHER_DIR = Path(os.getenv("LAYER_A_WEATHER_DIR", str(DATA_DIR / "layer_a_weather")))
LAYER_A_MINUTE_PARTITION_MINUTES = int(os.getenv("LAYER_A_MINUTE_PARTITION_MINUTES", "10"))
LAYER_A_LEGACY_CSV_DIR = Path(os.getenv("LAYER_A_LEGACY_CSV_DIR", str(DATA_DIR / "export")))

# Read-only remote history is intentionally outside the writable capture roots.
LAYER_A_REMOTE_CACHE_DIR = Path(
    os.getenv("LAYER_A_REMOTE_CACHE_DIR", "/tmp/layer_a_remote_cache")
)
LAYER_A_HISTORY_LOOKBACK_DAYS = int(os.getenv("LAYER_A_HISTORY_LOOKBACK_DAYS", "7"))
LAYER_A_HISTORY_AUTO_REFRESH = os.getenv("LAYER_A_HISTORY_AUTO_REFRESH", "true").lower() in {
    "1", "true", "yes", "on"
}
LAYER_A_HISTORY_REFRESH_INTERVAL_MINUTES = float(
    os.getenv("LAYER_A_HISTORY_REFRESH_INTERVAL_MINUTES", "10")
)

# ---- Snapshot export source ----
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://shea-hilton-weather-prediction.hf.space")

# ---- Model fallbacks ----
DEFAULT_TMAX_FORECAST_DELTA = 2.0
DEFAULT_TMIN_FORECAST_DELTA = -2.0
DEFAULT_TEMP = 28.0

# Bucket labels are now derived dynamically from Polymarket event data.
# No hardcoded TMAX_BUCKETS / TMIN_BUCKETS lists — see market_service.py
# _market_question_to_bucket() and _bucket_sort_key() for the dynamic logic.

# ---- Model keys ----
MODEL_KEYS = ["9d", "aws", "baseline", "rain_nowcast", "rain_observed", "model_a", "model_b", "model_c", "model_d", "model_e", "model_g", "model_2a", "model_2a_v2", "model_2b"]
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
    "model_g": "Model G (Gap+Max)",
    "model_2a": "Model 2A (Core+Wind)",
    "model_2a_v2": "Model 2A v2 (Offshore+Highland)",
    "model_2b": "Model 2B (2A v2 + Rain)",
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
    "model_2a_paper": "model_2a",
    "model_2a_v2_paper": "model_2a_v2",
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
PM_CLOB_API = "https://clob.polymarket.com"

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
    "model_g": "#FFB86C",
    "model_2a": "#FF2C97",
    "model_2a_v2": "#E6007E",
    "model_2b": "#00B4D8",
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
