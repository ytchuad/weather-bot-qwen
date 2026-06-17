# features/forward_test_rain.py
"""
Forward-test logger for rain-aware model comparison.

Records both baseline and rain-aware predictions for live inference,
backfills actual outcomes after day-end, and produces summary reports.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTIONS_PATH = Path('data/forward_test_rain_predictions.parquet')
SUMMARY_JSON_PATH = Path('reports/forward_test_rain_summary.json')
SUMMARY_XLSX_PATH = Path('reports/forward_test_rain_summary.xlsx')
SUMMARY_CSV_PATH = Path('data/forward_test_rain_summary.csv')

BASELINE_MODEL_DIR = Path('models/intraday_ml')
_RAIN_CANDIDATE_PRIMARY = Path('models/intraday_ai_rain_nowcast_candidate')
_RAIN_CANDIDATE_FALLBACK = Path('models/intraday_ml_rain_nowcast_candidate')
_RAIN_CANDIDATE_LEGACY = Path('models/intraday_ml_rain_candidate')
RAIN_AWARE_CANDIDATE_BASE = (
    _RAIN_CANDIDATE_PRIMARY if _RAIN_CANDIDATE_PRIMARY.exists()
    else _RAIN_CANDIDATE_FALLBACK if _RAIN_CANDIDATE_FALLBACK.exists()
    else _RAIN_CANDIDATE_LEGACY
)


def get_latest_rain_aware_candidate() -> Path | None:
    """Find the latest rain-aware candidate model directory."""
    if not RAIN_AWARE_CANDIDATE_BASE.exists():
        return None
    candidates = sorted(RAIN_AWARE_CANDIDATE_BASE.iterdir(), key=lambda x: x.name, reverse=True)
    return candidates[0] if candidates else None


def load_models_from_dir(model_dir: Path):
    """Load LightGBM models from a directory."""
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f'upside_q{q}'] = lgb.Booster(model_file=str(model_dir / f'upside_q{q}.txt'))
        models[f'downside_q{q}'] = lgb.Booster(model_file=str(model_dir / f'downside_q{q}.txt'))
    return models


def load_feature_list(model_dir: Path):
    """Load feature list from a model directory."""
    with open(model_dir / 'feature_list.json', 'r') as f:
        return json.load(f)


def predict_upside(models, features: dict, feature_cols: list, rainfall_60m: float = 0.0):
    """Get upside prediction from a model."""
    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_cols}], columns=feature_cols)
    q10 = models['upside_q10'].predict(X)[0]
    q50 = models['upside_q50'].predict(X)[0]
    
    rain_cooling = features.get('rain_cooling_120m', features.get('rain_cooling_60m', 0))
    post_peak_flag = features.get('post_peak_rain_flag', 0)
    morning_peak_flag = features.get('morning_peak_rain_flag', 0)
    
    return {
        'remaining_upside_p10': max(0.0, q10),
        'remaining_upside_p50': max(0.0, q50),
    }


def get_current_rainfall_features():
    """Get current rainfall context for inference."""
    try:
        df_rain = pd.read_parquet('data/hko_rainfall_15min.parquet')
        latest = df_rain['rainfall'].iloc[-1] if len(df_rain) > 0 else 0.0
        rainfall_30m = df_rain['rainfall_interval_15m'].rolling(2, min_periods=1).sum().iloc[-1] if len(df_rain) >= 2 else 0.0
        rainfall_60m = df_rain['rainfall_interval_15m'].rolling(4, min_periods=1).sum().iloc[-1] if len(df_rain) >= 4 else 0.0
        rainfall_120m = df_rain['rainfall_interval_15m'].rolling(8, min_periods=1).sum().iloc[-1] if len(df_rain) >= 8 else 0.0
        return {
            'rainfall_60m_filled': rainfall_60m,
            'rainfall_120m_filled': rainfall_120m,
            'rainfall_30m_filled': rainfall_30m,
            'rain_cooling_120m': 0.0,
            'post_peak_rain_flag': 0,
            'morning_peak_rain_flag': 0,
        }
    except Exception:
        return {
            'rainfall_60m_filled': 0.0,
            'rainfall_120m_filled': 0.0,
            'rainfall_30m_filled': 0.0,
            'rain_cooling_120m': 0.0,
            'post_peak_rain_flag': 0,
            'morning_peak_rain_flag': 0,
        }


def log_predictions(timestamp: datetime, datetime_val, hour: int, max_so_far: float, min_so_far: float,
                  rainfall_60m_filled: float, rain_cooling_120m: float, post_peak_rain_flag: int,
                  morning_peak_rain_flag: int, temp_now: float, temp_60min_ago: float, temp_120m_ago: float,
                  forecast_tmax: float, forecast_tmin: float, time_since_max_so_far: float):
    """Log both baseline and rain-aware predictions."""
    
    bl_features = {
        'temp': temp_now,
        'max_so_far': max_so_far,
        'temp_change_30min': temp_now - temp_60min_ago if temp_60min_ago else 0.0,
        'temp_change_60min': temp_now - temp_60min_ago,
        'hour': hour,
        'minute': datetime_val.minute,
        'minutes_since_midnight': hour * 60 + datetime_val.minute,
        'month': datetime_val.month,
        'day_of_year': datetime_val.timetuple().tm_yday,
        'day_sin': np.sin(2 * np.pi * datetime_val.timetuple().tm_yday / 365.25),
        'day_cos': np.cos(2 * np.pi * datetime_val.timetuple().tm_yday / 365.25),
        'is_morning': 1 if 6 <= hour < 12 else 0,
        'is_afternoon': 1 if 12 <= hour < 18 else 0,
        'is_evening': 1 if 18 <= hour < 24 else 0,
        'is_night': 1 if 0 <= hour < 6 else 0,
        'max_bucket': (max_so_far // 0.5) * 0.5,
        'time_since_max_so_far': time_since_max_so_far,
        'forecast_tmax': forecast_tmax,
        'forecast_tmin': forecast_tmin,
        'range_so_far': max_so_far - min_so_far,
        'temp_change_120min': temp_now - temp_120m_ago if temp_120m_ago else 0.0,
        'rainfall_60m': rainfall_60m_filled,
        'rain_cooling_60m': rain_cooling_120m * 0.5,
        'post_peak_rain_flag': post_peak_rain_flag,
        'morning_peak_rain_flag': morning_peak_rain_flag,
        'rainfall_120m': rainfall_60m_filled,
        'rain_cooling_30m': rain_cooling_120m * 0.3,
        'drop_from_max': max_so_far - temp_now,
    }
    
    ra_features = bl_features.copy()
    ra_features.update({
        'rainfall_60m_filled': rainfall_60m_filled,
        'rainfall_60m_missing_flag': 0.0,
        'rainfall_120m_filled': rainfall_60m_filled,
        'rainfall_120m_missing_flag': 0.0,
        'rainfall_30m_filled': rainfall_60m_filled * 0.5,
        'rain_data_gap_flag': 0.0,
        'rainfall_data_age_minutes': 0.0,
        'rain_cooling_120m': rain_cooling_120m,
        'morning_peak_then_rain_flag': morning_peak_rain_flag,
    })
    
    try:
        bl_models = load_models_from_dir(BASELINE_MODEL_DIR)
        bl_feature_cols = load_feature_list(BASELINE_MODEL_DIR)
        bl_pred = predict_upside(bl_models, bl_features, bl_feature_cols)
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        bl_pred = {'remaining_upside_p10': np.nan, 'remaining_upside_p50': np.nan}
    
    try:
        ra_candidate_dir = get_latest_rain_aware_candidate()
        if ra_candidate_dir is None:
            raise RuntimeError("No rain-aware candidate found")
        ra_models = load_models_from_dir(ra_candidate_dir)
        ra_feature_cols = load_feature_list(ra_candidate_dir)
        ra_pred = predict_upside(ra_models, ra_features, ra_feature_cols, rainfall_60m_filled)
    except Exception as e:
        logger.error(f"Failed to load rain-aware model: {e}")
        ra_pred = {'remaining_upside_p10': np.nan, 'remaining_upside_p50': np.nan}
    
    record = {
        'timestamp': timestamp,
        'datetime': datetime_val,
        'hour': hour,
        'max_so_far': max_so_far,
        'min_so_far': min_so_far,
        'rainfall_60m_filled': rainfall_60m_filled,
        'rain_cooling_120m': rain_cooling_120m,
        'post_peak_rain_flag': post_peak_rain_flag,
        'morning_peak_rain_flag': morning_peak_rain_flag,
        'baseline_pred_remaining_upside_p50': bl_pred['remaining_upside_p50'],
        'rain_aware_pred_remaining_upside_p50': ra_pred['remaining_upside_p50'],
        'baseline_pred_tmax_p50': max_so_far + bl_pred['remaining_upside_p50'] if bl_pred['remaining_upside_p50'] else np.nan,
        'rain_aware_pred_tmax_p50': max_so_far + ra_pred['remaining_upside_p50'] if ra_pred['remaining_upside_p50'] else np.nan,
        'choice_of_model_switch': 'rain_aware' if rainfall_60m_filled > 0 else 'baseline',
        'actual_tmax_after_day_end': np.nan,
        'actual_remaining_upside_after_day_end': np.nan,
    }
    
    df = pd.DataFrame([record])
    if PREDICTIONS_PATH.exists():
        existing = pd.read_parquet(PREDICTIONS_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    
    combined.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info(f"Logged prediction for {datetime_val}")


def backfill_actual_outcomes():
    """Backfill actual daily outcomes for yesterday."""
    if not PREDICTIONS_PATH.exists():
        return
    
    df = pd.read_parquet(PREDICTIONS_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    yesterday = (datetime.now() - timedelta(days=1)).normalize()
    mask = df['datetime'].dt.normalize() == yesterday
    
    if not mask.any():
        return
    
    actual_tmax = df.loc[mask, 'max_so_far'].max() + df.loc[mask, 'baseline_pred_remaining_upside_p50'].max()
    actual_remaining_upside = df.loc[mask, 'max_so_far'].iloc[-1] + df.loc[mask, 'baseline_pred_remaining_upside_p50'].iloc[-1]
    
    try:
        from data.download_rainfall import download_live
        download_live()
        df_rain = pd.read_parquet('data/hko_rainfall_15min.parquet')
        rain_yesterday = df_rain[df_rain['datetime'].dt.normalize() == yesterday]
        if len(rain_yesterday) > 0:
            actual_tmax = rain_yesterday['rainfall'].max()
    except Exception:
        pass
    
    df.loc[mask, 'actual_tmax_after_day_end'] = actual_tmax
    df.loc[mask, 'actual_remaining_upside_after_day_end'] = actual_remaining_upside
    
    df.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info(f"Backfilled outcomes for {yesterday.date()}")


def generate_summary():
    """Generate forward test summary reports."""
    if not PREDICTIONS_PATH.exists():
        logger.warning("No predictions file found")
        return
    
    df = pd.read_parquet(PREDICTIONS_PATH)
    df = df.dropna(subset=['actual_remaining_upside_after_day_end'])
    
    if len(df) == 0:
        logger.warning("No completed predictions for summary")
        return
    
    def compute_mae(actual, pred):
        return np.mean(np.abs(actual - pred))
    
    def false_positive_rate(actual, pred, threshold=0.05):
        actual_zero = actual < threshold
        return np.mean((actual_zero) & (pred > 0.5))
    
    baseline_mae = compute_mae(df['actual_remaining_upside_after_day_end'], df['baseline_pred_remaining_upside_p50'])
    rain_aware_mae = compute_mae(df['actual_remaining_upside_after_day_end'], df['rain_aware_pred_remaining_upside_p50'])
    
    switch_mask = df['choice_of_model_switch'] == 'rain_aware'
    switch_mae = compute_mae(
        df.loc[switch_mask, 'actual_remaining_upside_after_day_end'],
        df.loc[switch_mask, 'rain_aware_pred_remaining_upside_p50']
    ) if switch_mask.any() else np.nan
    
    rain_cases = df[df['rainfall_60m_filled'] > 0]
    heavy_rain_cases = df[df['rainfall_60m_filled'] > 5]
    
    summary = {
        'count': int(len(df)),
        'rainy_case_count': int(len(rain_cases)),
        'heavy_rain_case_count': int(len(heavy_rain_cases)),
        'baseline_mae': float(baseline_mae),
        'rain_aware_mae': float(rain_aware_mae),
        'regime_switch_mae': float(switch_mae) if not np.isnan(switch_mae) else None,
        'baseline_false_negative_max_reached_rate': float(false_positive_rate(df['actual_remaining_upside_after_day_end'], df['baseline_pred_remaining_upside_p50'])),
        'rain_aware_false_negative_max_reached_rate': float(false_positive_rate(df['actual_remaining_upside_after_day_end'], df['rain_aware_pred_remaining_upside_p50'])),
        'switch_false_negative_max_reached_rate': float(false_positive_rate(df.loc[switch_mask, 'actual_remaining_upside_after_day_end'], df.loc[switch_mask, 'rain_aware_pred_remaining_upside_p50'])) if switch_mask.any() else None,
    }
    
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    df_summary = pd.DataFrame([summary])
    df_summary.to_excel(SUMMARY_XLSX_PATH, index=False)
    
    df.to_csv(SUMMARY_CSV_PATH, index=False)
    
    logger.info(f"Summary saved: {summary}")


def main():
    backfill_actual_outcomes()
    generate_summary()


if __name__ == '__main__':
    main()