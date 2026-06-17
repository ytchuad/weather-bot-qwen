"""Backtesting snapshot builder.

Reconstructs historical feature snapshots from intraday_ml_train.parquet
for replaying strategy decisions through the paper-trader.

On the server (Streamlit Cloud), forward_test_log.parquet provides
real model predictions.  Locally, we use mock/synthetic predictions
so the framework is testable without the ML models installed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

HKT_OFFSET = timedelta(hours=8)

# ── Bucket definitions (from Polymarket temperature buckets) ──
BUCKET_DEFS = [
    ("<25°C",       -float("inf"), 25.0),
    ("25-26°C",      25.0, 26.0),
    ("26-27°C",      26.0, 27.0),
    ("27-28°C",      27.0, 28.0),
    ("28-29°C",      28.0, 29.0),
    ("29-30°C",      29.0, 30.0),
    ("30-31°C",      30.0, 31.0),
    ("31-32°C",      31.0, 32.0),
    ("32-33°C",      32.0, 33.0),
    ("33-34°C",      33.0, 34.0),
    ("34-35°C",      34.0, 35.0),
    (">=35°C",       35.0, float("inf")),
]


def _load_feature_df(data_dir: str = "data") -> pd.DataFrame:
    df = pd.read_parquet(Path(data_dir) / "intraday_ml_train.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _mock_probs(temp: float) -> dict:
    """Synthetic model predictions centered around the actual temp."""
    probs = {}
    total = 0.0
    for name, lo, hi in BUCKET_DEFS:
        if lo <= temp < hi:
            raw = 0.5
        elif hi <= temp:
            raw = 0.01 + 0.01 * (temp - hi)
        else:
            raw = 0.01 + 0.01 * (lo - temp)
        raw = max(0.001, min(1.0, raw))
        probs[name] = raw
        total += raw
    # Normalise
    for k in probs:
        probs[k] /= total
    return probs


def _mock_prices(temp: float) -> dict:
    """Synthetic market prices — higher near the actual temp."""
    prices = {}
    for name, lo, hi in BUCKET_DEFS:
        if lo <= temp < hi:
            prices[name] = 0.40
        else:
            dist = min(abs(temp - lo), abs(temp - hi)) if lo > -float("inf") and hi < float("inf") else 5.0
            prices[name] = max(0.01, 0.30 - 0.04 * dist)
    return prices


def build_snapshots(
    strategy_key: str = "enhanced_v1_paper",
    slug: str = "highest-temperature-in-hong-kong",
    date_from: str = None,
    date_to: str = None,
    data_dir: str = "data",
    use_mock: bool = True,
    sample_every_n: int = 1,
):
    """Yield snapshot dicts for backtesting.

    Each snapshot contains all inputs needed by ``compute_enhanced_orders()``.

    Parameters
    ----------
    strategy_key : str
        Strategy identifier (for metadata only).
    slug : str
        Polymarket event slug.
    date_from, date_to : str, optional
        Date range filters (YYYY-MM-DD).
    data_dir : str
        Path to data directory.
    use_mock : bool
        If True, use synthetic model probabilities and prices.
        Set False on the server where forward_test_log.parquet is readable.
    sample_every_n : int
        Take every Nth row to reduce snapshot count.
    """
    df = _load_feature_df(data_dir)
    if date_from:
        df = df[df["datetime"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["datetime"] <= pd.Timestamp(date_to)]
    df = df.iloc[::sample_every_n]

    features = [
        "temp", "max_so_far", "min_so_far", "drop_from_max",
        "rain_cooling_120m", "cooling_120m", "remaining_upside", "remaining_downside",
        "rolling_mean_120min", "rolling_std_120min",
        "hour", "minutes_since_midnight", "month", "day_of_year",
    ]
    available = [c for c in features if c in df.columns]

    for _, row in df.iterrows():
        dt = row["datetime"]
        temp = float(row.get("temp", 25.0))
        max_so_far = float(row.get("max_so_far", temp))
        min_so_far = float(row.get("min_so_far", temp))

        if use_mock:
            probs = _mock_probs(temp)
            prices = _mock_prices(temp)
        else:
            try:
                # Server path: read forward_test_log.parquet
                flog = pd.read_parquet(Path(data_dir) / "forward_test_log.parquet")
                mask = pd.to_datetime(flog["snapshot_time"]) == dt
                subset = flog[mask]
                probs = dict(zip(subset["bucket_name"], subset["model_prob"]))
                prices = dict(zip(subset["bucket_name"], subset["market_price"]))
            except Exception:
                logger.warning("falling back to mock at %s", dt)
                probs = _mock_probs(temp)
                prices = _mock_prices(temp)

        # Mock token IDs so apply_slippage_to_bets doesn't skip all buckets
        mock_token_ids = {}
        for bucket_name in probs:
            mock_token_ids[bucket_name] = (f"yes_{bucket_name}", f"no_{bucket_name}")

        ctx = {k: float(row[k]) for k in available if k in row}
        ctx.update({
            "slug": slug,
            "model_key": strategy_key,
            "capital": 10_000.0,
            "mock_slippage": True,
            "dt_now": dt.to_pydatetime().replace(tzinfo=timezone.utc) + HKT_OFFSET,
            "temp_now": temp,
            "max_so_far": max_so_far,
            "rain_regime": "no_rain",
            "model_std": 1.0,
            "recent_price_volatility": 0.0,
        })

        yield {
            "snapshot_time": dt,
            "slug": slug,
            "strategy_key": strategy_key,
            "target_probs": probs,
            "prices_dict": prices,
            "token_ids_dict": mock_token_ids,
            "context": ctx,
        }


def count_snapshots(date_from: str = None, date_to: str = None, sample_every_n: int = 1) -> int:
    df = _load_feature_df()
    if date_from:
        df = df[df["datetime"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["datetime"] <= pd.Timestamp(date_to)]
    return len(df.iloc[::sample_every_n])
