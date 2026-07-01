# models/intraday_inference.py
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

BASELINE_DIR = Path('models/intraday_ml')
RAIN_NOWCAST_DIR = Path('models/intraday_ml_rain_nowcast')
MINUTE_MODEL_DIR = Path('models/intraday_minute_ml')
MINUTE_MODEL_B_DIR = Path('models/intraday_minute_ml_model_b')
MINUTE_MODEL_C_DIR = Path('models/intraday_minute_ml_model_c')
MINUTE_MODEL_A_TMIN_DIR = Path('models/intraday_minute_ml_tmin')
MINUTE_MODEL_B_TMIN_DIR = Path('models/intraday_minute_ml_model_b_tmin')
MINUTE_MODEL_C_TMIN_DIR = Path('models/intraday_minute_ml_model_c_tmin')
MINUTE_MODEL_D_TMIN_DIR = Path('models/intraday_minute_ml_model_d_tmin')
BASELINE_FL_PATH = BASELINE_DIR / 'feature_list.json'
RAIN_NOWCAST_FL_PATH = RAIN_NOWCAST_DIR / 'feature_list.json'
MINUTE_FL_PATH = MINUTE_MODEL_DIR / 'feature_list.json'
MINUTE_B_FL_PATH = MINUTE_MODEL_B_DIR / 'feature_list.json'
MINUTE_C_FL_PATH = MINUTE_MODEL_C_DIR / 'feature_list.json'
MINUTE_A_TMIN_FL_PATH = MINUTE_MODEL_A_TMIN_DIR / 'feature_list.json'
MINUTE_B_TMIN_FL_PATH = MINUTE_MODEL_B_TMIN_DIR / 'feature_list.json'
MINUTE_C_TMIN_FL_PATH = MINUTE_MODEL_C_TMIN_DIR / 'feature_list.json'
MINUTE_D_TMIN_FL_PATH = MINUTE_MODEL_D_TMIN_DIR / 'feature_list.json'
MINUTE_MODEL_E_MORNING_TMIN_DIR = Path('models/intraday_minute_ml_model_e_morning_tmin')
MINUTE_E_MORNING_TMIN_FL_PATH = MINUTE_MODEL_E_MORNING_TMIN_DIR / 'feature_list.json'
# Model G - forecast_gap + max_so_far model
MINUTE_MODEL_G_DIR = Path('models/intraday_minute_ml_model_g')
MINUTE_G_FL_PATH = MINUTE_MODEL_G_DIR / 'feature_list.json'
# Model 2A - core baseline with minute obs + forecast + wind
MINUTE_MODEL_2A_DIR = Path('models/intraday_minute_ml_model_2a')
MINUTE_2A_FL_PATH = MINUTE_MODEL_2A_DIR / 'feature_list.json'
RAIN_CALIBRATION_PATH = MINUTE_MODEL_D_TMIN_DIR / 'rain_calibration.json'
MORNING_E_CALIBRATION_PATH = MINUTE_MODEL_E_MORNING_TMIN_DIR / 'morning_calibration.json'

_rain_calib_cache = {}
_morning_e_cal_cache = None

# Hierarchical calibration for Model B (from calibration experiment §4.4.10)
# Applied when rainfall_60m > 0. Buckets with n < CALIB_MIN_N fall back to CALIB_RAIN_FALLBACK.
CALIB_RAIN_BY_HOUR = {
    (0, 6):   {"p10": -2.76, "p90": 1.30,  "n": 1149},
    (6, 12):  {"p10": -1.70, "p90": 1.14,  "n": 1557},
    (12, 18): {"p10": -0.16, "p90": 0.20,  "n": 1272},
}
CALIB_RAIN_FALLBACK = {"p10": -1.96, "p90": 0.55, "n": 4935}
CALIB_MIN_N = 500

_model_cache = {}
_active_model_key = 'baseline'



def _get_lgb():
    import lightgbm as _lgb
    return _lgb


def _load_single_model(model_dir):
    cache = {}
    fl_path = model_dir / 'feature_list.json'
    if not fl_path.exists():
        raise FileNotFoundError(f"feature_list.json not found: {fl_path}")
    with open(fl_path, 'r', encoding='utf-8') as f:
        cache['feature_cols'] = json.load(f)
    lgb = _get_lgb()
    for q in [10, 25, 50, 75, 90]:
        f_up = model_dir / f'upside_q{q}.txt'
        if f_up.exists():
            cache[f'upside_q{q}'] = lgb.Booster(model_file=str(f_up))
        f_down = model_dir / f'downside_q{q}.txt'
        if f_down.exists():
            cache[f'downside_q{q}'] = lgb.Booster(model_file=str(f_down))
    for clf_name in ['upside_zero', 'downside_zero']:
        clf_path = model_dir / f'{clf_name}.txt'
        if clf_path.exists():
            cache[clf_name] = lgb.Booster(model_file=str(clf_path))
        else:
            cache[clf_name] = None
    for clf_name in ['will_make_new_low_clf', 'is_downside_zero_clf', 'tmin_timing_clf']:
        clf_path = model_dir / f'{clf_name}.txt'
        if clf_path.exists():
            cache[clf_name] = lgb.Booster(model_file=str(clf_path))
        else:
            cache[clf_name] = None
    return cache

def _get_cached_models():
    """Cached model loading — returns a new dict, no module-level side effects."""
    result = {}
    result['baseline'] = _load_single_model(BASELINE_DIR)
    if RAIN_NOWCAST_DIR.exists() and RAIN_NOWCAST_FL_PATH.exists():
        try:
            result['rain_nowcast'] = _load_single_model(RAIN_NOWCAST_DIR)
        except Exception as e:
            logger.warning("Rain nowcast model failed to load: %s", e)
    else:
        logger.warning("Rain nowcast model not found at %s", RAIN_NOWCAST_DIR)
    if MINUTE_MODEL_DIR.exists() and MINUTE_FL_PATH.exists():
        try:
            result['model_a'] = _load_single_model(MINUTE_MODEL_DIR)
        except Exception as e:
            logger.warning("Model A (minute-level) failed to load: %s", e)
    else:
        logger.warning("Model A (minute-level) not found at %s", MINUTE_MODEL_DIR)
    if MINUTE_MODEL_B_DIR.exists() and MINUTE_B_FL_PATH.exists():
        try:
            result['model_b'] = _load_single_model(MINUTE_MODEL_B_DIR)
        except Exception as e:
            logger.warning("Model B (minute+rainfall) failed to load: %s", e)
    else:
        logger.warning("Model B (minute+rainfall) not found at %s", MINUTE_MODEL_B_DIR)
    if MINUTE_MODEL_C_DIR.exists() and MINUTE_C_FL_PATH.exists():
        try:
            result['model_c'] = _load_single_model(MINUTE_MODEL_C_DIR)
        except Exception as e:
            logger.warning("Model C (minute+rainfall+nowcast) failed to load: %s", e)
    else:
        logger.warning("Model C (minute+rainfall+nowcast) not found at %s", MINUTE_MODEL_C_DIR)
    if MINUTE_MODEL_A_TMIN_DIR.exists() and MINUTE_A_TMIN_FL_PATH.exists():
        try:
            result['model_a_tmin'] = _load_single_model(MINUTE_MODEL_A_TMIN_DIR)
        except Exception as e:
            logger.warning("Model A Tmin failed to load: %s", e)
    else:
        logger.warning("Model A Tmin not found at %s", MINUTE_MODEL_A_TMIN_DIR)
    if MINUTE_MODEL_B_TMIN_DIR.exists() and MINUTE_B_TMIN_FL_PATH.exists():
        try:
            result['model_b_tmin'] = _load_single_model(MINUTE_MODEL_B_TMIN_DIR)
        except Exception as e:
            logger.warning("Model B Tmin failed to load: %s", e)
    else:
        logger.warning("Model B Tmin not found at %s", MINUTE_MODEL_B_TMIN_DIR)
    if MINUTE_MODEL_C_TMIN_DIR.exists() and MINUTE_C_TMIN_FL_PATH.exists():
        try:
            result['model_c_tmin'] = _load_single_model(MINUTE_MODEL_C_TMIN_DIR)
        except Exception as e:
            logger.warning("Model C Tmin failed to load: %s", e)
    else:
        logger.warning("Model C Tmin not found at %s", MINUTE_MODEL_C_TMIN_DIR)
    if MINUTE_MODEL_D_TMIN_DIR.exists() and MINUTE_D_TMIN_FL_PATH.exists():
        try:
            result['model_d_tmin'] = _load_single_model(MINUTE_MODEL_D_TMIN_DIR)
        except Exception as e:
            logger.warning("Model D Tmin failed to load: %s", e)
    else:
        logger.warning("Model D Tmin not found at %s", MINUTE_MODEL_D_TMIN_DIR)
    if MINUTE_MODEL_E_MORNING_TMIN_DIR.exists() and MINUTE_E_MORNING_TMIN_FL_PATH.exists():
        try:
            result['model_e_morning_tmin'] = _load_single_model(MINUTE_MODEL_E_MORNING_TMIN_DIR)
        except Exception as e:
            logger.warning("Model E Morning Tmin failed to load: %s", e)
    else:
        logger.warning("Model E Morning Tmin not found at %s", MINUTE_MODEL_E_MORNING_TMIN_DIR)
    # Model G - forecast_gap + max_so_far model
    if MINUTE_MODEL_G_DIR.exists() and MINUTE_G_FL_PATH.exists():
        try:
            result['model_g'] = _load_single_model(MINUTE_MODEL_G_DIR)
        except Exception as e:
            logger.warning("Model G failed to load: %s", e)
    else:
        logger.warning("Model G not found at %s", MINUTE_MODEL_G_DIR)
    # Model 2A - core baseline with forecast + wind
    if MINUTE_MODEL_2A_DIR.exists() and MINUTE_2A_FL_PATH.exists():
        try:
            result['model_2a'] = _load_single_model(MINUTE_MODEL_2A_DIR)
        except Exception as e:
            logger.warning("Model 2A failed to load: %s", e)
    else:
        logger.warning("Model 2A not found at %s", MINUTE_MODEL_2A_DIR)
    return result


def _load_models():
    if 'baseline' not in _model_cache or 'rain_nowcast' not in _model_cache:
        loaded = _get_cached_models()
        _model_cache.update(loaded)
    # Lazy-load minute models if not already in cache (may fail silently)
    if 'model_a' not in _model_cache:
        if MINUTE_MODEL_DIR.exists() and MINUTE_FL_PATH.exists():
            try:
                _model_cache['model_a'] = _load_single_model(MINUTE_MODEL_DIR)
            except Exception as e:
                logger.warning("Model A (minute-level) lazy-load failed: %s", e)
    if 'model_b' not in _model_cache:
        if MINUTE_MODEL_B_DIR.exists() and MINUTE_B_FL_PATH.exists():
            try:
                _model_cache['model_b'] = _load_single_model(MINUTE_MODEL_B_DIR)
            except Exception as e:
                logger.warning("Model B (minute+rainfall) lazy-load failed: %s", e)
    if 'model_c' not in _model_cache:
        if MINUTE_MODEL_C_DIR.exists() and MINUTE_C_FL_PATH.exists():
            try:
                _model_cache['model_c'] = _load_single_model(MINUTE_MODEL_C_DIR)
            except Exception as e:
                logger.warning("Model C (minute+rainfall+nowcast) lazy-load failed: %s", e)
    if 'model_a_tmin' not in _model_cache:
        if MINUTE_MODEL_A_TMIN_DIR.exists() and MINUTE_A_TMIN_FL_PATH.exists():
            try:
                _model_cache['model_a_tmin'] = _load_single_model(MINUTE_MODEL_A_TMIN_DIR)
            except Exception as e:
                logger.warning("Model A Tmin lazy-load failed: %s", e)
    if 'model_b_tmin' not in _model_cache:
        if MINUTE_MODEL_B_TMIN_DIR.exists() and MINUTE_B_TMIN_FL_PATH.exists():
            try:
                _model_cache['model_b_tmin'] = _load_single_model(MINUTE_MODEL_B_TMIN_DIR)
            except Exception as e:
                logger.warning("Model B Tmin lazy-load failed: %s", e)
    if 'model_c_tmin' not in _model_cache:
        if MINUTE_MODEL_C_TMIN_DIR.exists() and MINUTE_C_TMIN_FL_PATH.exists():
            try:
                _model_cache['model_c_tmin'] = _load_single_model(MINUTE_MODEL_C_TMIN_DIR)
            except Exception as e:
                logger.warning("Model C Tmin lazy-load failed: %s", e)
    if 'model_d_tmin' not in _model_cache:
        if MINUTE_MODEL_D_TMIN_DIR.exists() and MINUTE_D_TMIN_FL_PATH.exists():
            try:
                _model_cache['model_d_tmin'] = _load_single_model(MINUTE_MODEL_D_TMIN_DIR)
            except Exception as e:
                logger.warning("Model D Tmin lazy-load failed: %s", e)
    if 'model_e_morning_tmin' not in _model_cache:
        if MINUTE_MODEL_E_MORNING_TMIN_DIR.exists() and MINUTE_E_MORNING_TMIN_FL_PATH.exists():
            try:
                _model_cache['model_e_morning_tmin'] = _load_single_model(MINUTE_MODEL_E_MORNING_TMIN_DIR)
            except Exception as e:
                logger.warning("Model E Morning Tmin lazy-load failed: %s", e)
    if 'model_g' not in _model_cache:
        if MINUTE_MODEL_G_DIR.exists() and MINUTE_G_FL_PATH.exists():
            try:
                _model_cache['model_g'] = _load_single_model(MINUTE_MODEL_G_DIR)
            except Exception as e:
                logger.warning("Model G lazy-load failed: %s", e)
    if 'model_2a' not in _model_cache:
        if MINUTE_MODEL_2A_DIR.exists() and MINUTE_2A_FL_PATH.exists():
            try:
                _model_cache['model_2a'] = _load_single_model(MINUTE_MODEL_2A_DIR)
            except Exception as e:
                logger.warning("Model 2A lazy-load failed: %s", e)
    return _model_cache


def set_active_model(model_key):
    """Switch active model: 'baseline', 'rain_nowcast', 'model_a', 'model_b', 'model_c', 'model_g', or Tmin variants."""
    global _active_model_key
    valid_keys = ('baseline', 'rain_nowcast', 'model_a', 'model_b', 'model_c',
                  'model_a_tmin', 'model_b_tmin', 'model_c_tmin', 'model_d_tmin',
                  'model_e_morning_tmin', 'model_g', 'model_2a')
    if model_key not in valid_keys:
        raise ValueError(f"Unknown model_key: {model_key}")
    _active_model_key = model_key
    logger.info("Active model switched to: %s", model_key)


def get_active_model_key():
    """Return current active model key."""
    return _active_model_key


def _get_active():
    """Return cache dict for the currently active model."""
    models = _load_models()
    return models.get(_active_model_key, models['baseline'])


def _build_features(current_datetime, max_so_far, min_so_far, temp_now, temp_60min_ago, temp_120m_ago,
                    forecast_tmax, forecast_tmin,
                    rainfall_60m_filled=0.0, rainfall_120m_filled=0.0,
                    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                    rainfall_30m_filled=0.0, rainfall_30m_missing_flag=1,
                    rainfall_data_age_minutes=0.0, rain_data_gap_flag=0,
                    temp_change_30min=0.0, temp_change_60min=0.0,
                    time_since_max_so_far=0.0, hour=None, minutes_since_midnight=None,
                    # Nowcast features (optional, point-in-time safe)
                    rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
                    rain_nc_any_0_120m=0.0, rain_nc_front_loaded_ratio=0.0,
                    rain_nc_heavy_0_120m=0.0, rain_nc_valid_horizon_count=0.0,
                    rain_nc_missing_flag=0, rain_nowcast_age_minutes=0.0,
                    rain_nowcast_missing_flag=0,
                    rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
                    rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
                    rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
                    rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
                    rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
                    rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
                    rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
                    rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
                    rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
                    rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
                    rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
                    rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
                    rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
                    rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
                    rain_cooling_120m=0.0, rise_from_min=0.0,
                    ):
    dt = current_datetime
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    hour = hour if hour is not None else dt.hour
    minute = dt.minute
    minutes_since_midnight = minutes_since_midnight if minutes_since_midnight is not None else hour * 60 + minute
    remaining_minutes_to_midnight = 1440 - minutes_since_midnight

    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    is_morning = 1 if 6 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 18 else 0
    is_evening = 1 if 18 <= hour < 24 else 0
    is_night = 1 if 0 <= hour < 6 else 0

    range_so_far = max_so_far - min_so_far if min_so_far is not None else 0.0
    max_bucket = (max_so_far // 0.5) * 0.5

    cooling_60m = max(temp_60min_ago - temp_now, 0) if temp_60min_ago is not None else 0.0
    cooling_30m = max(temp_60min_ago - temp_now, 0) if temp_60min_ago is not None else 0.0
    cooling_120m = max(temp_120m_ago - temp_now, 0) if temp_120m_ago is not None else 0.0
    rain_cooling_60m = cooling_60m if rainfall_60m_filled > 0 else 0.0
    rain_cooling_30m = cooling_30m if rainfall_60m_filled > 0 else 0.0
    rain_cooling_120m = cooling_120m if rainfall_120m_filled > 0 else 0.0
    drop_from_max = max_so_far - temp_now
    rise_from_min = temp_now - min_so_far if min_so_far is not None else 0.0

    RAIN_HEAVY_THRESHOLD = 5.0
    DROP_FROM_MAX_THRESHOLD = 0.5
    POST_PEAK_MINUTES_MIN = 30
    POST_PEAK_MINUTES_MAX = 240

    condition_post_peak = (
        (rainfall_60m_filled > RAIN_HEAVY_THRESHOLD) and
        (drop_from_max >= DROP_FROM_MAX_THRESHOLD) and
        (POST_PEAK_MINUTES_MIN <= time_since_max_so_far <= POST_PEAK_MINUTES_MAX)
    )
    post_peak_rain_flag = 1 if condition_post_peak else 0
    morning_peak_then_rain_flag = 1 if (condition_post_peak and 9 <= hour <= 14) else 0

    features = {
        'temp': temp_now,
        'max_so_far': max_so_far,
        'temp_change_30min': temp_change_30min,
        'temp_change_60min': temp_change_60min,
        'hour': hour,
        'minute': minute,
        'minutes_since_midnight': minutes_since_midnight,
        'month': month,
        'day_of_year': day_of_year,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'is_morning': is_morning,
        'is_afternoon': is_afternoon,
        'is_evening': is_evening,
        'is_night': is_night,
        'max_bucket': max_bucket,
        'time_since_max_so_far': time_since_max_so_far,
        'forecast_tmax': forecast_tmax if forecast_tmax is not None else max_so_far + 2.0,
        'forecast_tmin': forecast_tmin if forecast_tmin is not None else min_so_far - 2.0,
        'range_so_far': range_so_far,
        'temp_change_120min': temp_now - temp_120m_ago,
        'rainfall_60m_filled': rainfall_60m_filled,
        'rainfall_60m_missing_flag': rainfall_60m_missing_flag,
        'rainfall_120m_filled': rainfall_120m_filled,
        'rainfall_120m_missing_flag': rainfall_120m_missing_flag,
        'rainfall_30m_filled': rainfall_30m_filled,
        'rainfall_30m_missing_flag': rainfall_30m_missing_flag,
        'rainfall_data_age_minutes': rainfall_data_age_minutes,
        'rain_data_gap_flag': rain_data_gap_flag,
        'rain_cooling_60m': rain_cooling_60m,
        'rain_cooling_120m': rain_cooling_120m,
        'rain_cooling_30m': rain_cooling_30m,
        'post_peak_rain_flag': post_peak_rain_flag,
        'morning_peak_then_rain_flag': morning_peak_then_rain_flag,
        'drop_from_max': drop_from_max,
        'rise_from_min': rise_from_min,
        'rain_nc_sum_0_60m': rain_nc_sum_0_60m,
        'rain_nc_sum_0_120m': rain_nc_sum_0_120m,
        'rain_nc_any_0_120m': rain_nc_any_0_120m,
        'rain_nc_front_loaded_ratio': rain_nc_front_loaded_ratio,
        'rain_nc_heavy_0_120m': rain_nc_heavy_0_120m,
        'rain_nc_valid_horizon_count': rain_nc_valid_horizon_count,
        'rain_nc_missing_flag': rain_nc_missing_flag,
        'rain_nowcast_age_minutes': rain_nowcast_age_minutes,
        'rain_nowcast_missing_flag': rain_nowcast_missing_flag,
        'rain_nc_nearest_mm_sum_30m': rain_nc_nearest_mm_sum_30m,
        'rain_nc_nearest_mm_sum_60m': rain_nc_nearest_mm_sum_60m,
        'rain_nc_nearest_mm_sum_90m': rain_nc_nearest_mm_sum_90m,
        'rain_nc_nearest_mm_sum_120m': rain_nc_nearest_mm_sum_120m,
        'rain_nc_mean_r5km_sum_30m': rain_nc_mean_r5km_sum_30m,
        'rain_nc_mean_r5km_sum_60m': rain_nc_mean_r5km_sum_60m,
        'rain_nc_mean_r5km_sum_90m': rain_nc_mean_r5km_sum_90m,
        'rain_nc_mean_r5km_sum_120m': rain_nc_mean_r5km_sum_120m,
        'rain_nc_max_r5km_sum_30m': rain_nc_max_r5km_sum_30m,
        'rain_nc_max_r5km_sum_60m': rain_nc_max_r5km_sum_60m,
        'rain_nc_max_r5km_sum_90m': rain_nc_max_r5km_sum_90m,
        'rain_nc_max_r5km_sum_120m': rain_nc_max_r5km_sum_120m,
        'rain_nc_min_r5km_sum_30m': rain_nc_min_r5km_sum_30m,
        'rain_nc_min_r5km_sum_60m': rain_nc_min_r5km_sum_60m,
        'rain_nc_min_r5km_sum_90m': rain_nc_min_r5km_sum_90m,
        'rain_nc_min_r5km_sum_120m': rain_nc_min_r5km_sum_120m,
        'rain_nc_p90_r5km_sum_30m': rain_nc_p90_r5km_sum_30m,
        'rain_nc_p90_r5km_sum_60m': rain_nc_p90_r5km_sum_60m,
        'rain_nc_p90_r5km_sum_90m': rain_nc_p90_r5km_sum_90m,
        'rain_nc_p90_r5km_sum_120m': rain_nc_p90_r5km_sum_120m,
        'rain_nc_area_gt0_r5km_sum_30m': rain_nc_area_gt0_r5km_sum_30m,
        'rain_nc_area_gt0_r5km_sum_60m': rain_nc_area_gt0_r5km_sum_60m,
        'rain_nc_area_gt0_r5km_sum_90m': rain_nc_area_gt0_r5km_sum_90m,
        'rain_nc_area_gt0_r5km_sum_120m': rain_nc_area_gt0_r5km_sum_120m,
        'rain_nc_area_gt5_r5km_sum_30m': rain_nc_area_gt5_r5km_sum_30m,
        'rain_nc_area_gt5_r5km_sum_60m': rain_nc_area_gt5_r5km_sum_60m,
        'rain_nc_area_gt5_r5km_sum_90m': rain_nc_area_gt5_r5km_sum_90m,
        'rain_nc_area_gt5_r5km_sum_120m': rain_nc_area_gt5_r5km_sum_120m,
    }
    return features


def predict_intraday_tmax(current_datetime, max_so_far, temp_60min_ago, temp_now,
                          forecast_tmax=None, forecast_tmin=None, temp_120m_ago=None, min_so_far=None,
                          rainfall_60m_filled=0.0, rainfall_120m_filled=0.0,
                          rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                          rainfall_30m_filled=0.0, rainfall_30m_missing_flag=1,
                          rainfall_data_age_minutes=0.0, rain_data_gap_flag=0,
                          temp_change_30min=None, temp_change_60min=None,
                          time_since_max_so_far=None, hour=None, minutes_since_midnight=None,
                          # Nowcast features (optional)
                          rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
                          rain_nc_any_0_120m=0.0, rain_nc_front_loaded_ratio=0.0,
                          rain_nc_heavy_0_120m=0.0, rain_nc_valid_horizon_count=0.0,
                          rain_nc_missing_flag=0, rain_nowcast_age_minutes=0.0,
                          rain_nowcast_missing_flag=0,
                          rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
                          rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
                          rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
                          rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
                          rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
                          rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
                          rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
                          rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
                          rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
                          rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
                          rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
                          rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
                          rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
                          rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
                          rain_cooling_120m=0.0, rise_from_min=0.0,
                          ):
    active = _get_active()
    feature_cols = active['feature_cols']

    if min_so_far is None:
        min_so_far = max_so_far - 5.0
    if forecast_tmax is None:
        forecast_tmax = max_so_far + 2.0
    if forecast_tmin is None:
        forecast_tmin = min_so_far - 2.0
    if temp_120m_ago is None:
        temp_120m_ago = temp_60min_ago
    if temp_change_30min is None:
        temp_change_30min = temp_now - temp_60min_ago if temp_60min_ago is not None else 0
    if temp_change_60min is None:
        temp_change_60min = temp_now - temp_60min_ago
    if time_since_max_so_far is None:
        time_since_max_so_far = 0.0
    if hour is None:
        hour = current_datetime.hour
    if minutes_since_midnight is None:
        minutes_since_midnight = hour * 60 + current_datetime.minute

    feats = _build_features(
        current_datetime, max_so_far, min_so_far, temp_now, temp_60min_ago, temp_120m_ago,
        forecast_tmax, forecast_tmin,
        rainfall_60m_filled=rainfall_60m_filled, rainfall_120m_filled=rainfall_120m_filled,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag, rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        rainfall_30m_filled=rainfall_30m_filled, rainfall_30m_missing_flag=rainfall_30m_missing_flag,
        rainfall_data_age_minutes=rainfall_data_age_minutes, rain_data_gap_flag=rain_data_gap_flag,
        temp_change_30min=temp_change_30min, temp_change_60min=temp_change_60min,
        time_since_max_so_far=time_since_max_so_far, hour=hour, minutes_since_midnight=minutes_since_midnight,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m, rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m, rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m, rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m, rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m, rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m, rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m, rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m, rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m, rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m, rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m, rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m, rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m, rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m, rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
        rain_cooling_120m=rain_cooling_120m, rise_from_min=rise_from_min,
    )
    X = pd.DataFrame([feats], columns=feature_cols)
    model_features = active['upside_q50'].feature_name()
    X = X[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    if active['upside_zero'] is not None:
        zero_features = active['upside_zero'].feature_name()
        X_zero = X[zero_features]
        prob_max_reached = active['upside_zero'].predict(X_zero)[0]
    else:
        prob_max_reached = None

    hour = current_datetime.hour
    temp_decline = max_so_far - temp_now
    if prob_max_reached is None:
        prob_max_reached = 0.0
    if hour >= 18 and temp_decline > 1.0:
        prob_max_reached = max(prob_max_reached, 0.95)
    elif hour >= 16 and temp_decline > 2.0:
        prob_max_reached = max(prob_max_reached, 0.90)

    remaining_upside_p10 = max(0.0, remaining_upside_p10)
    remaining_upside_p25 = max(0.0, remaining_upside_p25)
    remaining_upside_p50 = max(0.0, remaining_upside_p50)
    remaining_upside_p75 = max(0.0, remaining_upside_p75)
    remaining_upside_p90 = max(0.0, remaining_upside_p90)

    pred_tmax_p10 = max(max_so_far, max_so_far + remaining_upside_p10)
    pred_tmax_p50 = max(max_so_far, max_so_far + remaining_upside_p50)
    pred_tmax_p90 = max(max_so_far, max_so_far + remaining_upside_p90)

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None
    }


def _build_minute_features(*, temp_current, rh_current, max_so_far, min_so_far,
                            time_since_max, time_since_min,
                            temp_buffer=None, rh_buffer=None,
                            current_datetime=None, hour=None, minute=None):
    """Build the 38-feature vector for Model A from raw minute-level inputs.

    Parameters
    ----------
    temp_current : float
    rh_current : float
    max_so_far : float
    min_so_far : float
    time_since_max : float (minutes)
    time_since_min : float (minutes)
    temp_buffer : list-like or None
        Recent temperature values (most recent last), at least ~60 entries for full feature set.
        Falls back to temp_current if None.
    rh_buffer : list-like or None
        Recent RH values (most recent last), at least ~60 entries.
        Falls back to rh_current if None.
    current_datetime : datetime or None
    hour : int or None
    minute : int or None
    """
    import numpy as np

    # Fallback buffers
    if temp_buffer is None or len(temp_buffer) == 0:
        temp_buffer = [temp_current]
    if rh_buffer is None or len(rh_buffer) == 0:
        rh_buffer = [rh_current]

    t_arr = np.array(list(temp_buffer))
    rh_arr = np.array(list(rh_buffer))
    idx_now_t = len(t_arr) - 1
    idx_now_rh = len(rh_arr) - 1

    def _safe_diff(offset):
        i = idx_now_t - offset
        return t_arr[idx_now_t] - t_arr[i] if i >= 0 else 0.0

    def _safe_rh_diff(offset):
        i = idx_now_rh - offset
        return rh_arr[idx_now_rh] - rh_arr[i] if i >= 0 else 0.0

    def _safe_rolling_std(window):
        start = max(0, idx_now_t - window + 1)
        return float(np.std(t_arr[start:idx_now_t + 1], ddof=1)) if (idx_now_t - start) >= 1 else 0.0

    def _safe_rh_rolling_mean(window):
        start = max(0, idx_now_rh - window + 1)
        return float(np.mean(rh_arr[start:idx_now_rh + 1])) if start <= idx_now_rh else rh_current

    def _safe_rh_rolling_std(window):
        start = max(0, idx_now_rh - window + 1)
        return float(np.std(rh_arr[start:idx_now_rh + 1], ddof=1)) if (idx_now_rh - start) >= 1 else 0.0

    # Temperature changes
    temp_change_5m = _safe_diff(5)
    temp_change_15m = _safe_diff(15)
    temp_change_30m = _safe_diff(30)
    temp_change_60m = _safe_diff(60)

    # Temperature acceleration (central difference: (t - 2*t.shift(15) + t.shift(30)) / 15)
    i15 = idx_now_t - 15
    i30 = idx_now_t - 30
    if i30 >= 0:
        temp_acceleration_30m = (t_arr[idx_now_t] - 2 * t_arr[i15] + t_arr[i30]) / 15.0
    else:
        temp_acceleration_30m = 0.0

    temp_std_30m = _safe_rolling_std(30)
    temp_std_60m = _safe_rolling_std(60)

    # RH changes
    rh_change_15m = _safe_rh_diff(15)
    rh_change_30m = _safe_rh_diff(30)
    rh_change_60m = _safe_rh_diff(60)

    rh_mean_30m = _safe_rh_rolling_mean(30)
    rh_mean_60m = _safe_rh_rolling_mean(60)
    rh_std_60m = _safe_rh_rolling_std(60)

    # Interactions
    temp_x_rh = temp_current * rh_current
    a, b = 17.27, 237.7
    gamma = (a * temp_current) / (b + temp_current) + np.log(max(rh_current, 0.01) / 100.0)
    dew_point_c = (b * gamma) / (a - gamma)
    dew_point_spread = temp_current - dew_point_c

    # Time features
    if current_datetime is not None:
        dt = current_datetime
        h = dt.hour if hour is None else hour
        m = dt.minute if minute is None else minute
        month = dt.month
        day_of_year = dt.timetuple().tm_yday
    else:
        h = hour if hour is not None else 12
        m = minute if minute is not None else 0
        month = 6
        day_of_year = 1
    minutes_since_midnight = h * 60 + m
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    is_morning = 1 if 6 <= h < 12 else 0
    is_afternoon = 1 if 12 <= h < 18 else 0
    is_evening = 1 if 18 <= h < 24 else 0
    is_night = 1 if 0 <= h < 6 else 0

    # Intraday state features (from minute history)
    range_so_far = max_so_far - min_so_far
    drop_from_max = max_so_far - temp_current
    rise_from_min = temp_current - min_so_far

    features = {
        "temp_current": temp_current,
        "rh_current": rh_current,
        "max_so_far_1m": max_so_far,
        "min_so_far_1m": min_so_far,
        "range_so_far_1m": range_so_far,
        "time_since_max_1m": time_since_max,
        "time_since_min_1m": time_since_min,
        "drop_from_max_1m": drop_from_max,
        "rise_from_min_1m": rise_from_min,
        "temp_change_5m": temp_change_5m,
        "temp_change_15m": temp_change_15m,
        "temp_change_30m": temp_change_30m,
        "temp_change_60m": temp_change_60m,
        "temp_acceleration_30m": temp_acceleration_30m,
        "temp_std_30m": temp_std_30m,
        "temp_std_60m": temp_std_60m,
        "rh_change_15m": rh_change_15m,
        "rh_change_30m": rh_change_30m,
        "rh_change_60m": rh_change_60m,
        "rh_mean_30m": rh_mean_30m,
        "rh_mean_60m": rh_mean_60m,
        "rh_std_60m": rh_std_60m,
        "temp_x_rh": temp_x_rh,
        "dew_point_c": dew_point_c,
        "dew_point_spread": dew_point_spread,
        "hour": h,
        "minute": m,
        "minutes_since_midnight": minutes_since_midnight,
        "month": month,
        "day_of_year": day_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_morning": is_morning,
        "is_afternoon": is_afternoon,
        "is_evening": is_evening,
        "is_night": is_night,
    }
    return features


def predict_intraday_tmax_model_a(
    current_datetime, max_so_far, temp_now,
    rh_current=50.0, min_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    force_clip=True,
):
    """Predict remaining upside for tmax using Model A (minute-level, temp + RH only).

    Parameters not documented are identical in meaning to `predict_intraday_tmax`.
    rh_current : float — current relative humidity (%), default 50.0 if unknown.
    temp_buffer : list-like, optional — recent temp values for rolling features.
    rh_buffer : list-like, optional — recent RH values for rolling features.
    """
    active = _get_active()
    if 'upside_q50' not in active:
        raise RuntimeError("model_a is not loaded — call _load_models() first or check models/intraday_minute_ml/")

    if min_so_far is None:
        min_so_far = max_so_far - 5.0

    feats = _build_minute_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime,
        hour=hour, minute=minute,
    )
    feature_cols = active['feature_cols']
    model_features = active['upside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('upside_zero') is not None:
        clf_features = active['upside_zero'].feature_name()
        prob_max_reached = active['upside_zero'].predict(X[clf_features])[0]
    else:
        prob_max_reached = None

    hour = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
    temp_decline = max_so_far - temp_now
    if prob_max_reached is None:
        prob_max_reached = 0.0
    if hour >= 18 and temp_decline > 1.0:
        prob_max_reached = max(prob_max_reached, 0.95)
    elif hour >= 16 and temp_decline > 2.0:
        prob_max_reached = max(prob_max_reached, 0.90)

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    pred_tmax_p10 = max(max_so_far, max_so_far + remaining_upside_p10)
    pred_tmax_p50 = max(max_so_far, max_so_far + remaining_upside_p50)
    pred_tmax_p90 = max(max_so_far, max_so_far + remaining_upside_p90)

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None,
    }


def predict_intraday_tmin_model_a(
    current_datetime, min_so_far, temp_now,
    rh_current=50.0, max_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    force_clip=True,
):
    active = _get_active()
    if 'downside_q50' not in active:
        raise RuntimeError("model_a_tmin is not loaded")

    if max_so_far is None:
        max_so_far = min_so_far + 5.0

    feats = _build_minute_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime,
        hour=hour, minute=minute,
    )
    feature_cols = active['feature_cols']
    model_features = active['downside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('downside_zero') is not None:
        clf_features = active['downside_zero'].feature_name()
        prob_min_reached = active['downside_zero'].predict(X[clf_features])[0]
    else:
        prob_min_reached = None

    if prob_min_reached is None:
        prob_min_reached = 0.0

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = quantiles

    pred_tmin_p50 = min(min_so_far, min_so_far - remaining_downside_p50)
    pred_tmin_p10 = min(min_so_far, min_so_far - remaining_downside_p90)
    pred_tmin_p90 = min(min_so_far, min_so_far - remaining_downside_p10)

    return {
        'remaining_downside_p10': remaining_downside_p10,
        'remaining_downside_p25': remaining_downside_p25,
        'remaining_downside_p50': remaining_downside_p50,
        'remaining_downside_p75': remaining_downside_p75,
        'remaining_downside_p90': remaining_downside_p90,
        'prob_min_reached': prob_min_reached,
        'pred_tmin_p50': pred_tmin_p50,
        'pred_tmin_p10': pred_tmin_p10,
        'pred_tmin_p90': pred_tmin_p90,
        'sample_count': None,
    }


RAIN_HEAVY_THRESHOLD = 5.0
DROP_FROM_MAX_THRESHOLD = 0.5
POST_PEAK_MINUTES_MIN = 30
POST_PEAK_MINUTES_MAX = 240


def _build_rainfall_minute_features(*, temp_current, rh_current, max_so_far, min_so_far,
                                     time_since_max, time_since_min,
                                     temp_buffer=None, rh_buffer=None,
                                     current_datetime=None, hour=None, minute=None,
                                     rainfall_60m=0.0, rainfall_120m=0.0,
                                     rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                                     temp_change_60m=0.0, drop_from_max=0.0):
    base = _build_minute_features(
        temp_current=temp_current, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
    )
    rain_cooling_60m = max(0, -temp_change_60m) if rainfall_60m > 0 else 0.0
    rain_cooling_120m = max(0, -temp_change_60m) if rainfall_120m > 0 else 0.0
    condition_post_peak = (
        (rainfall_60m > RAIN_HEAVY_THRESHOLD)
        and (drop_from_max >= DROP_FROM_MAX_THRESHOLD)
        and (POST_PEAK_MINUTES_MIN <= time_since_max <= POST_PEAK_MINUTES_MAX)
    )
    post_peak_rain_flag = 1 if condition_post_peak else 0
    h = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
    morning_peak_rain_flag = 1 if (condition_post_peak and 9 <= h <= 14) else 0

    base.update({
        "rainfall_60m": rainfall_60m,
        "rainfall_120m": rainfall_120m,
        "rainfall_60m_missing_flag": rainfall_60m_missing_flag,
        "rainfall_120m_missing_flag": rainfall_120m_missing_flag,
        "rain_cooling_60m": rain_cooling_60m,
        "rain_cooling_120m": rain_cooling_120m,
        "post_peak_rain_flag": post_peak_rain_flag,
        "morning_peak_rain_flag": morning_peak_rain_flag,
    })
    return base


def _build_model_c_features(*, temp_current, rh_current, max_so_far, min_so_far,
                             time_since_max, time_since_min,
                             temp_buffer=None, rh_buffer=None,
                             current_datetime=None, hour=None, minute=None,
                             rainfall_60m=0.0, rainfall_120m=0.0,
                             rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                             temp_change_60m=0.0, drop_from_max=0.0,
                             rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
                             rain_nc_any_0_120m=0, rain_nc_front_loaded_ratio=0.0,
                             rain_nc_heavy_0_120m=0, rain_nc_valid_horizon_count=0,
                             rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
                             rain_nowcast_missing_flag=0,
                             rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
                             rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
                             rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
                             rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
                             rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
                             rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
                             rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
                             rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
                             rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
                             rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
                             rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
                             rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
                             rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
                             rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0):
    base = _build_rainfall_minute_features(
        temp_current=temp_current, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
    )
    base.update({
        "rain_nc_sum_0_60m": rain_nc_sum_0_60m,
        "rain_nc_sum_0_120m": rain_nc_sum_0_120m,
        "rain_nc_any_0_120m": rain_nc_any_0_120m,
        "rain_nc_front_loaded_ratio": rain_nc_front_loaded_ratio,
        "rain_nc_heavy_0_120m": rain_nc_heavy_0_120m,
        "rain_nc_valid_horizon_count": rain_nc_valid_horizon_count,
        "rain_nc_missing_flag": rain_nc_missing_flag,
        "rain_nowcast_age_minutes": rain_nowcast_age_minutes,
        "rain_nowcast_missing_flag": rain_nowcast_missing_flag,
        "rain_nc_nearest_mm_sum_30m": rain_nc_nearest_mm_sum_30m,
        "rain_nc_nearest_mm_sum_60m": rain_nc_nearest_mm_sum_60m,
        "rain_nc_nearest_mm_sum_90m": rain_nc_nearest_mm_sum_90m,
        "rain_nc_nearest_mm_sum_120m": rain_nc_nearest_mm_sum_120m,
        "rain_nc_mean_r5km_sum_30m": rain_nc_mean_r5km_sum_30m,
        "rain_nc_mean_r5km_sum_60m": rain_nc_mean_r5km_sum_60m,
        "rain_nc_mean_r5km_sum_90m": rain_nc_mean_r5km_sum_90m,
        "rain_nc_mean_r5km_sum_120m": rain_nc_mean_r5km_sum_120m,
        "rain_nc_max_r5km_sum_30m": rain_nc_max_r5km_sum_30m,
        "rain_nc_max_r5km_sum_60m": rain_nc_max_r5km_sum_60m,
        "rain_nc_max_r5km_sum_90m": rain_nc_max_r5km_sum_90m,
        "rain_nc_max_r5km_sum_120m": rain_nc_max_r5km_sum_120m,
        "rain_nc_min_r5km_sum_30m": rain_nc_min_r5km_sum_30m,
        "rain_nc_min_r5km_sum_60m": rain_nc_min_r5km_sum_60m,
        "rain_nc_min_r5km_sum_90m": rain_nc_min_r5km_sum_90m,
        "rain_nc_min_r5km_sum_120m": rain_nc_min_r5km_sum_120m,
        "rain_nc_p90_r5km_sum_30m": rain_nc_p90_r5km_sum_30m,
        "rain_nc_p90_r5km_sum_60m": rain_nc_p90_r5km_sum_60m,
        "rain_nc_p90_r5km_sum_90m": rain_nc_p90_r5km_sum_90m,
        "rain_nc_p90_r5km_sum_120m": rain_nc_p90_r5km_sum_120m,
        "rain_nc_area_gt0_r5km_sum_30m": rain_nc_area_gt0_r5km_sum_30m,
        "rain_nc_area_gt0_r5km_sum_60m": rain_nc_area_gt0_r5km_sum_60m,
        "rain_nc_area_gt0_r5km_sum_90m": rain_nc_area_gt0_r5km_sum_90m,
        "rain_nc_area_gt0_r5km_sum_120m": rain_nc_area_gt0_r5km_sum_120m,
        "rain_nc_area_gt5_r5km_sum_30m": rain_nc_area_gt5_r5km_sum_30m,
        "rain_nc_area_gt5_r5km_sum_60m": rain_nc_area_gt5_r5km_sum_60m,
        "rain_nc_area_gt5_r5km_sum_90m": rain_nc_area_gt5_r5km_sum_90m,
        "rain_nc_area_gt5_r5km_sum_120m": rain_nc_area_gt5_r5km_sum_120m,
    })
    return base


def _build_model_d_features(*, temp_current, rh_current, max_so_far, min_so_far,
                             time_since_max, time_since_min,
                             temp_buffer=None, rh_buffer=None,
                             current_datetime=None, hour=None, minute=None,
                             rainfall_60m=0.0, rainfall_120m=0.0,
                             rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                             temp_change_60m=0.0, drop_from_max=0.0,
                             rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
                             rain_nc_any_0_120m=0.0, rain_nc_front_loaded_ratio=0.0,
                             rain_nc_heavy_0_120m=0.0, rain_nc_valid_horizon_count=0,
                             rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
                             rain_nowcast_missing_flag=0,
                             rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
                             rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
                             rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
                             rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
                             rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
                             rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
                             rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
                             rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
                             rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
                             rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
                             rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
                             rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
                             rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
                             rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
                             temp_buffer_long=None, rh_buffer_long=None,
                             prev_18_temp=0.0, prev_21_temp=0.0, prev_2359_temp=0.0,
                             prev_evening_temp_change=0.0, prev_evening_temp_min=0.0,
                             prev_evening_temp_range=0.0, prev_evening_temp_slope=0.0,
                             prev_evening_rh_mean=50.0, prev_evening_rh_max=50.0,
                             prev_evening_dew_point_mean=10.0,
                             prev_evening_rainfall_18_24=0.0, prev_evening_rain_flag=0):
    import numpy as np
    base = _build_model_c_features(
        temp_current=temp_current, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m,
        rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m,
        rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m,
        rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m,
        rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m,
        rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m,
        rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m,
        rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m,
        rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m,
        rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m,
        rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m,
        rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m,
        rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m,
        rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m,
        rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
    )

    t_long = np.array(list(temp_buffer_long)) if temp_buffer_long is not None and len(temp_buffer_long) > 0 else np.array([])
    rh_long = np.array(list(rh_buffer_long)) if rh_buffer_long is not None and len(rh_buffer_long) > 0 else np.array([])
    idx_now = len(t_long) - 1

    def _safe_diff_long(offset):
        i = idx_now - offset
        return t_long[idx_now] - t_long[i] if i >= 0 else np.nan

    def _safe_rolling_min(window):
        start = max(0, idx_now - window + 1)
        return float(np.min(t_long[start:idx_now + 1])) if start <= idx_now else np.nan

    def _safe_rolling_max(window):
        start = max(0, idx_now - window + 1)
        return float(np.max(t_long[start:idx_now + 1])) if start <= idx_now else np.nan

    def _safe_rolling_std_long(window):
        start = max(0, idx_now - window + 1)
        return float(np.std(t_long[start:idx_now + 1], ddof=1)) if (idx_now - start) >= 1 else np.nan

    def _safe_rh_rolling_mean_long(window):
        start = max(0, idx_now - window + 1)
        return float(np.mean(rh_long[start:idx_now + 1])) if start <= idx_now else np.nan

    def _safe_rh_rolling_max_long(window):
        start = max(0, idx_now - window + 1)
        return float(np.max(rh_long[start:idx_now + 1])) if start <= idx_now else np.nan

    def _dew_point_single(t, rh):
        a, b = 17.27, 237.7
        gamma = (a * t) / (b + t) + np.log(max(rh, 0.01) / 100.0)
        return (b * gamma) / (a - gamma)

    def _safe_dew_point_spread_min(window):
        start = max(0, idx_now - window + 1)
        if start <= idx_now:
            spreads = [temp_current - _dew_point_single(t_long[i], rh_long[i]) for i in range(start, idx_now + 1)]
            return float(min(spreads))
        return np.nan

    cross_temp_change_180m = _safe_diff_long(180)
    cross_temp_change_360m = _safe_diff_long(360)
    cross_temp_change_720m = _safe_diff_long(720)

    base.update({
        "temp_change_180m_crossday": cross_temp_change_180m,
        "temp_change_360m_crossday": cross_temp_change_360m,
        "temp_change_720m_crossday": cross_temp_change_720m,
        "temp_slope_360m_crossday": (cross_temp_change_360m / 6.0) if not np.isnan(cross_temp_change_360m) else np.nan,
        "temp_slope_720m_crossday": (cross_temp_change_720m / 12.0) if not np.isnan(cross_temp_change_720m) else np.nan,
        "temp_min_360m_crossday": _safe_rolling_min(360),
        "temp_min_720m_crossday": _safe_rolling_min(720),
        "temp_range_360m_crossday": (_safe_rolling_max(360) - _safe_rolling_min(360)) if idx_now >= 0 else np.nan,
        "temp_std_360m_crossday": _safe_rolling_std_long(360),
        "dew_point_spread_min_360m": _safe_dew_point_spread_min(360),
        "rh_mean_360m": _safe_rh_rolling_mean_long(360),
        "rh_max_360m": _safe_rh_rolling_max_long(360),
        "prev_18_temp": prev_18_temp,
        "prev_21_temp": prev_21_temp,
        "prev_2359_temp": prev_2359_temp,
        "prev_evening_temp_change": prev_evening_temp_change,
        "prev_evening_temp_min": prev_evening_temp_min,
        "prev_evening_temp_range": prev_evening_temp_range,
        "prev_evening_temp_slope": prev_evening_temp_slope,
        "prev_evening_rh_mean": prev_evening_rh_mean,
        "prev_evening_rh_max": prev_evening_rh_max,
        "prev_evening_dew_point_mean": prev_evening_dew_point_mean,
        "prev_evening_rainfall_18_24": prev_evening_rainfall_18_24,
        "prev_evening_rain_flag": prev_evening_rain_flag,
        "cooling_since_prev_18": prev_18_temp - temp_current,
        "cooling_since_prev_21": prev_21_temp - temp_current,
        "distance_to_prev_evening_min": temp_current - prev_evening_temp_min,
        "dew_point_floor_gap": temp_current - _dew_point_single(temp_current, rh_current),
        "is_before_evening_cooling_window": 1 if (hour if hour is not None else current_datetime.hour) < 18 else 0,
        "daytime_warming_so_far": max_so_far - min_so_far,
        "afternoon_temp_drop_60m": max(0, -temp_change_60m) if (hour if hour is not None else current_datetime.hour) >= 12 else 0.0,
        "afternoon_temp_drop_120m": max(0, -temp_change_60m) if (hour if hour is not None else current_datetime.hour) >= 12 else 0.0,
    })
    return base


def predict_intraday_tmax_model_b(
    current_datetime, max_so_far, temp_now,
    rh_current=50.0, min_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    force_clip=True,
):
    active = _get_active()
    if 'upside_q50' not in active:
        raise RuntimeError("model_b is not loaded")

    if min_so_far is None:
        min_so_far = max_so_far - 5.0

    feats = _build_rainfall_minute_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
    )
    feature_cols = active['feature_cols']
    model_features = active['upside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('upside_zero') is not None:
        clf_features = active['upside_zero'].feature_name()
        prob_max_reached = active['upside_zero'].predict(X[clf_features])[0]
    else:
        prob_max_reached = None

    hour = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
    temp_decline = max_so_far - temp_now
    if prob_max_reached is None:
        prob_max_reached = 0.0
    if hour >= 18 and temp_decline > 1.0:
        prob_max_reached = max(prob_max_reached, 0.95)
    elif hour >= 16 and temp_decline > 2.0:
        prob_max_reached = max(prob_max_reached, 0.90)

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    # ── Hierarchical calibration for rain rows (hour × rain regime) ──
    cal_used = None
    if rainfall_60m is not None and rainfall_60m > 0:
        h = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
        cal_bucket = None
        for (h_lo, h_hi), params in sorted(CALIB_RAIN_BY_HOUR.items()):
            if h_lo <= h < h_hi:
                cal_bucket = params
                break
        if cal_bucket is not None and cal_bucket["n"] < CALIB_MIN_N:
            cal_bucket = None
        if cal_bucket is None:
            cal_bucket = CALIB_RAIN_FALLBACK
        p10_cal, p90_cal = cal_bucket["p10"], cal_bucket["p90"]
        cal_q10 = min(remaining_upside_p10, remaining_upside_p50 + p10_cal)
        cal_q90 = min(remaining_upside_p90, remaining_upside_p50 + p90_cal)
        remaining_upside_p10 = cal_q10
        remaining_upside_p90 = cal_q90
        cal_used = {"n": cal_bucket["n"], "p10": p10_cal, "p90": p90_cal}

    pred_tmax_p10 = max(max_so_far, max_so_far + remaining_upside_p10)
    pred_tmax_p50 = max(max_so_far, max_so_far + remaining_upside_p50)
    pred_tmax_p90 = max(max_so_far, max_so_far + remaining_upside_p90)

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None,
        'cal_bucket': cal_used,
    }


def predict_intraday_tmin_model_b(
    current_datetime, min_so_far, temp_now,
    rh_current=50.0, max_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    force_clip=True,
):
    active = _get_active()
    if 'downside_q50' not in active:
        raise RuntimeError("model_b_tmin is not loaded")

    if max_so_far is None:
        max_so_far = min_so_far + 5.0

    feats = _build_rainfall_minute_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
    )
    feature_cols = active['feature_cols']
    model_features = active['downside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('downside_zero') is not None:
        clf_features = active['downside_zero'].feature_name()
        prob_min_reached = active['downside_zero'].predict(X[clf_features])[0]
    else:
        prob_min_reached = None

    if prob_min_reached is None:
        prob_min_reached = 0.0

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = quantiles

    # Hierarchical calibration for rain rows
    cal_used = None
    if rainfall_60m is not None and rainfall_60m > 0:
        h = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
        cal_bucket = None
        for (h_lo, h_hi), params in sorted(CALIB_RAIN_BY_HOUR.items()):
            if h_lo <= h < h_hi:
                cal_bucket = params
                break
        if cal_bucket is not None and cal_bucket["n"] < CALIB_MIN_N:
            cal_bucket = None
        if cal_bucket is None:
            cal_bucket = CALIB_RAIN_FALLBACK
        p10_cal, p90_cal = cal_bucket["p10"], cal_bucket["p90"]
        cal_q10 = min(remaining_downside_p10, remaining_downside_p50 + p10_cal)
        cal_q90 = min(remaining_downside_p90, remaining_downside_p50 + p90_cal)
        remaining_downside_p10 = cal_q10
        remaining_downside_p90 = cal_q90
        cal_used = {"n": cal_bucket["n"], "p10": p10_cal, "p90": p90_cal}

    pred_tmin_p50 = min(min_so_far, min_so_far - remaining_downside_p50)
    pred_tmin_p10 = min(min_so_far, min_so_far - remaining_downside_p90)
    pred_tmin_p90 = min(min_so_far, min_so_far - remaining_downside_p10)

    return {
        'remaining_downside_p10': remaining_downside_p10,
        'remaining_downside_p25': remaining_downside_p25,
        'remaining_downside_p50': remaining_downside_p50,
        'remaining_downside_p75': remaining_downside_p75,
        'remaining_downside_p90': remaining_downside_p90,
        'prob_min_reached': prob_min_reached,
        'pred_tmin_p50': pred_tmin_p50,
        'pred_tmin_p10': pred_tmin_p10,
        'pred_tmin_p90': pred_tmin_p90,
        'sample_count': None,
        'cal_bucket': cal_used,
    }


def predict_intraday_tmax_model_c(
    current_datetime, max_so_far, temp_now,
    rh_current=50.0, min_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
    rain_nc_any_0_120m=0, rain_nc_front_loaded_ratio=0.0,
    rain_nc_heavy_0_120m=0, rain_nc_valid_horizon_count=0,
    rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
    rain_nowcast_missing_flag=0,
    rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
    rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
    rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
    rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
    rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
    rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
    rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
    rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
    rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
    rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
    rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
    rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
    rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
    rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
    force_clip=True,
):
    active = _get_active()
    if 'upside_q50' not in active:
        raise RuntimeError("model_c is not loaded")

    if min_so_far is None:
        min_so_far = max_so_far - 5.0

    feats = _build_model_c_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m,
        rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m,
        rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m,
        rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m,
        rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m,
        rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m,
        rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m,
        rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m,
        rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m,
        rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m,
        rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m,
        rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m,
        rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m,
        rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m,
        rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
    )
    feature_cols = active['feature_cols']
    model_features = active['upside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('upside_zero') is not None:
        clf_features = active['upside_zero'].feature_name()
        prob_max_reached = active['upside_zero'].predict(X[clf_features])[0]
    else:
        prob_max_reached = None

    hour = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
    temp_decline = max_so_far - temp_now
    if prob_max_reached is None:
        prob_max_reached = 0.0
    if hour >= 18 and temp_decline > 1.0:
        prob_max_reached = max(prob_max_reached, 0.95)
    elif hour >= 16 and temp_decline > 2.0:
        prob_max_reached = max(prob_max_reached, 0.90)

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    # Model C uses Model B's hierarchical calibration for rain rows
    cal_used = None
    if rainfall_60m is not None and rainfall_60m > 0:
        h = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
        cal_bucket = None
        for (h_lo, h_hi), params in sorted(CALIB_RAIN_BY_HOUR.items()):
            if h_lo <= h < h_hi:
                cal_bucket = params
                break
        if cal_bucket is not None and cal_bucket["n"] < CALIB_MIN_N:
            cal_bucket = None
        if cal_bucket is None:
            cal_bucket = CALIB_RAIN_FALLBACK
        p10_cal, p90_cal = cal_bucket["p10"], cal_bucket["p90"]
        cal_q10 = min(remaining_upside_p10, remaining_upside_p50 + p10_cal)
        cal_q90 = min(remaining_upside_p90, remaining_upside_p50 + p90_cal)
        remaining_upside_p10 = cal_q10
        remaining_upside_p90 = cal_q90
        cal_used = {"n": cal_bucket["n"], "p10": p10_cal, "p90": p90_cal}

    pred_tmax_p10 = max(max_so_far, max_so_far + remaining_upside_p10)
    pred_tmax_p50 = max(max_so_far, max_so_far + remaining_upside_p50)
    pred_tmax_p90 = max(max_so_far, max_so_far + remaining_upside_p90)

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None,
        'cal_bucket': cal_used,
    }


def predict_intraday_tmin_model_c(
    current_datetime, min_so_far, temp_now,
    rh_current=50.0, max_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
    rain_nc_any_0_120m=0, rain_nc_front_loaded_ratio=0.0,
    rain_nc_heavy_0_120m=0, rain_nc_valid_horizon_count=0,
    rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
    rain_nowcast_missing_flag=0,
    rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
    rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
    rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
    rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
    rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
    rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
    rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
    rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
    rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
    rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
    rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
    rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
    rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
    rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
    force_clip=True,
):
    active = _get_active()
    if 'downside_q50' not in active:
        raise RuntimeError("model_c_tmin is not loaded")

    if max_so_far is None:
        max_so_far = min_so_far + 5.0

    feats = _build_model_c_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m,
        rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m,
        rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m,
        rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m,
        rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m,
        rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m,
        rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m,
        rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m,
        rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m,
        rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m,
        rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m,
        rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m,
        rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m,
        rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m,
        rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
    )
    feature_cols = active['feature_cols']
    model_features = active['downside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    if active.get('downside_zero') is not None:
        clf_features = active['downside_zero'].feature_name()
        prob_min_reached = active['downside_zero'].predict(X[clf_features])[0]
    else:
        prob_min_reached = None

    if prob_min_reached is None:
        prob_min_reached = 0.0

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = quantiles

    cal_used = None
    if rainfall_60m is not None and rainfall_60m > 0:
        h = hour if hour is not None else (current_datetime.hour if current_datetime is not None else 12)
        cal_bucket = None
        for (h_lo, h_hi), params in sorted(CALIB_RAIN_BY_HOUR.items()):
            if h_lo <= h < h_hi:
                cal_bucket = params
                break
        if cal_bucket is not None and cal_bucket["n"] < CALIB_MIN_N:
            cal_bucket = None
        if cal_bucket is None:
            cal_bucket = CALIB_RAIN_FALLBACK
        p10_cal, p90_cal = cal_bucket["p10"], cal_bucket["p90"]
        cal_q10 = min(remaining_downside_p10, remaining_downside_p50 + p10_cal)
        cal_q90 = min(remaining_downside_p90, remaining_downside_p50 + p90_cal)
        remaining_downside_p10 = cal_q10
        remaining_downside_p90 = cal_q90
        cal_used = {"n": cal_bucket["n"], "p10": p10_cal, "p90": p90_cal}

    pred_tmin_p50 = min(min_so_far, min_so_far - remaining_downside_p50)
    pred_tmin_p10 = min(min_so_far, min_so_far - remaining_downside_p90)
    pred_tmin_p90 = min(min_so_far, min_so_far - remaining_downside_p10)

    return {
        'remaining_downside_p10': remaining_downside_p10,
        'remaining_downside_p25': remaining_downside_p25,
        'remaining_downside_p50': remaining_downside_p50,
        'remaining_downside_p75': remaining_downside_p75,
        'remaining_downside_p90': remaining_downside_p90,
        'prob_min_reached': prob_min_reached,
        'pred_tmin_p50': pred_tmin_p50,
        'pred_tmin_p10': pred_tmin_p10,
        'pred_tmin_p90': pred_tmin_p90,
        'sample_count': None,
        'cal_bucket': cal_used,
    }


def predict_intraday_tmin_model_d(
    current_datetime, min_so_far, temp_now,
    rh_current=50.0, max_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
    rain_nc_any_0_120m=0, rain_nc_front_loaded_ratio=0.0,
    rain_nc_heavy_0_120m=0, rain_nc_valid_horizon_count=0,
    rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
    rain_nowcast_missing_flag=0,
    rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
    rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
    rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
    rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
    rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
    rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
    rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
    rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
    rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
    rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
    rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
    rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
    rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
    rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
    temp_buffer_long=None, rh_buffer_long=None,
    prev_18_temp=0.0, prev_21_temp=0.0, prev_2359_temp=0.0,
    prev_evening_temp_change=0.0, prev_evening_temp_min=0.0,
    prev_evening_temp_range=0.0, prev_evening_temp_slope=0.0,
    prev_evening_rh_mean=50.0, prev_evening_rh_max=50.0,
    prev_evening_dew_point_mean=10.0,
    prev_evening_rainfall_18_24=0.0, prev_evening_rain_flag=0,
    force_clip=True,
    new_low_shrink_threshold=0.3,
    new_low_shrink_factor=0.3,
):
    import json
    active = _get_active()
    if 'downside_q50' not in active:
        raise RuntimeError("model_d_tmin is not loaded")

    if max_so_far is None:
        max_so_far = min_so_far + 5.0

    feats = _build_model_d_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m,
        rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m,
        rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m,
        rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m,
        rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m,
        rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m,
        rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m,
        rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m,
        rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m,
        rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m,
        rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m,
        rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m,
        rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m,
        rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m,
        rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
        temp_buffer_long=temp_buffer_long, rh_buffer_long=rh_buffer_long,
        prev_18_temp=prev_18_temp, prev_21_temp=prev_21_temp, prev_2359_temp=prev_2359_temp,
        prev_evening_temp_change=prev_evening_temp_change,
        prev_evening_temp_min=prev_evening_temp_min,
        prev_evening_temp_range=prev_evening_temp_range,
        prev_evening_temp_slope=prev_evening_temp_slope,
        prev_evening_rh_mean=prev_evening_rh_mean,
        prev_evening_rh_max=prev_evening_rh_max,
        prev_evening_dew_point_mean=prev_evening_dew_point_mean,
        prev_evening_rainfall_18_24=prev_evening_rainfall_18_24,
        prev_evening_rain_flag=prev_evening_rain_flag,
    )
    feature_cols = active['feature_cols']
    model_features = active['downside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])

    # 00-06 rain: flag for special handling
    _is_rain_early = (hour is not None and hour < 6 and rainfall_60m > 0)

    prob_min_reached = None
    if active.get('is_downside_zero_clf') is not None:
        clf_features = active['is_downside_zero_clf'].feature_name()
        prob_min_reached = active['is_downside_zero_clf'].predict(X[clf_features])[0]
    elif active.get('downside_zero') is not None:
        clf_features = active['downside_zero'].feature_name()
        prob_min_reached = active['downside_zero'].predict(X[clf_features])[0]
    if prob_min_reached is None:
        prob_min_reached = 0.0

    prob_new_low = None
    if active.get('will_make_new_low_clf') is not None:
        nl_clf_features = active['will_make_new_low_clf'].feature_name()
        prob_new_low = active['will_make_new_low_clf'].predict(X[nl_clf_features])[0]
    if prob_new_low is None:
        prob_new_low = 0.0

    shrink = 1.0
    if prob_new_low < new_low_shrink_threshold:
        shrink = new_low_shrink_factor
    elif prob_new_low < 0.6:
        shrink = new_low_shrink_factor + (1.0 - new_low_shrink_factor) * (prob_new_low - new_low_shrink_threshold) / (0.6 - new_low_shrink_threshold)

    # 00-06 rain: disable shrinkage (COV80 already 64%, shrinking makes it worse)
    if _is_rain_early:
        shrink = 1.0

    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    quantiles = [v * shrink for v in quantiles]
    remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = quantiles

    # 00-06 rain: apply regime-specific residual calibration to widen intervals
    if _is_rain_early:
        global _rain_calib_cache
        if not _rain_calib_cache:
            try:
                if RAIN_CALIBRATION_PATH.exists():
                    _rain_calib_cache = json.loads(RAIN_CALIBRATION_PATH.read_text())
            except Exception:
                _rain_calib_cache = None
        if _rain_calib_cache and "00_06_rain_residual_p10" in _rain_calib_cache:
            p10_r = _rain_calib_cache["00_06_rain_residual_p10"]
            p90_r = _rain_calib_cache["00_06_rain_residual_p90"]
            remaining_downside_p10 = max(0.0, remaining_downside_p50 + p10_r)
            remaining_downside_p90 = remaining_downside_p50 + p90_r
            vals = sorted([remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90])
            remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = vals

    pred_tmin_p50 = min(min_so_far, min_so_far - remaining_downside_p50)
    pred_tmin_p10 = min(min_so_far, min_so_far - remaining_downside_p90)
    pred_tmin_p90 = min(min_so_far, min_so_far - remaining_downside_p10)

    timing_proba = None
    if active.get('tmin_timing_clf') is not None:
        t_clf_features = active['tmin_timing_clf'].feature_name()
        timing_proba = active['tmin_timing_clf'].predict(X[t_clf_features])
        timing_pred = int(np.argmax(timing_proba))
    else:
        timing_pred = -1

    return {
        'remaining_downside_p10': remaining_downside_p10,
        'remaining_downside_p25': remaining_downside_p25,
        'remaining_downside_p50': remaining_downside_p50,
        'remaining_downside_p75': remaining_downside_p75,
        'remaining_downside_p90': remaining_downside_p90,
        'prob_min_reached': prob_min_reached,
        'prob_new_low': prob_new_low,
        'pred_tmin_p50': pred_tmin_p50,
        'pred_tmin_p10': pred_tmin_p10,
        'pred_tmin_p90': pred_tmin_p90,
        'sample_count': None,
        'shrink_factor': shrink,
        'timing_pred': timing_pred,
    }


def predict_intraday_tmin_model_e_morning(
    current_datetime, min_so_far, temp_now,
    rh_current=50.0, max_so_far=None,
    time_since_max=0.0, time_since_min=0.0,
    temp_buffer=None, rh_buffer=None,
    hour=None, minute=None,
    rainfall_60m=0.0, rainfall_120m=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    temp_change_60m=0.0, drop_from_max=0.0,
    rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
    rain_nc_any_0_120m=0, rain_nc_front_loaded_ratio=0.0,
    rain_nc_heavy_0_120m=0, rain_nc_valid_horizon_count=0,
    rain_nc_missing_flag=0, rain_nowcast_age_minutes=0,
    rain_nowcast_missing_flag=0,
    rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
    rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
    rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
    rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
    rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
    rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
    rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
    rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
    rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
    rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
    rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
    rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
    rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
    rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
    temp_buffer_long=None, rh_buffer_long=None,
    prev_18_temp=0.0, prev_21_temp=0.0, prev_2359_temp=0.0,
    prev_evening_temp_change=0.0, prev_evening_temp_min=0.0,
    prev_evening_temp_range=0.0, prev_evening_temp_slope=0.0,
    prev_evening_rh_mean=50.0, prev_evening_rh_max=50.0,
    prev_evening_dew_point_mean=10.0,
    prev_evening_rainfall_18_24=0.0, prev_evening_rain_flag=0,
    force_clip=True,
):
    """Predict morning minimum (00:00-07:59 HKT) remaining downside.
    Only valid for hour < 8 (caller should filter). No shrinkage."""
    import json
    active = _get_active()
    if 'downside_q50' not in active:
        raise RuntimeError("model_e_morning_tmin is not loaded")

    if max_so_far is None:
        max_so_far = min_so_far + 5.0

    feats = _build_model_d_features(
        temp_current=temp_now, rh_current=rh_current,
        max_so_far=max_so_far, min_so_far=min_so_far,
        time_since_max=time_since_max, time_since_min=time_since_min,
        temp_buffer=temp_buffer, rh_buffer=rh_buffer,
        current_datetime=current_datetime, hour=hour, minute=minute,
        rainfall_60m=rainfall_60m, rainfall_120m=rainfall_120m,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag,
        rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        temp_change_60m=temp_change_60m, drop_from_max=drop_from_max,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m,
        rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m,
        rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m,
        rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m,
        rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m,
        rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m,
        rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m,
        rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m,
        rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m,
        rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m,
        rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m,
        rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m,
        rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m,
        rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m,
        rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
        temp_buffer_long=temp_buffer_long, rh_buffer_long=rh_buffer_long,
        prev_18_temp=prev_18_temp, prev_21_temp=prev_21_temp, prev_2359_temp=prev_2359_temp,
        prev_evening_temp_change=prev_evening_temp_change,
        prev_evening_temp_min=prev_evening_temp_min,
        prev_evening_temp_range=prev_evening_temp_range,
        prev_evening_temp_slope=prev_evening_temp_slope,
        prev_evening_rh_mean=prev_evening_rh_mean,
        prev_evening_rh_max=prev_evening_rh_max,
        prev_evening_dew_point_mean=prev_evening_dew_point_mean,
        prev_evening_rainfall_18_24=prev_evening_rainfall_18_24,
        prev_evening_rain_flag=prev_evening_rain_flag,
    )
    feature_cols = active['feature_cols']
    model_features = active['downside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([feats], columns=cols)[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    global _morning_e_cal_cache
    if _morning_e_cal_cache is None:
        _cal = {}
        try:
            if MORNING_E_CALIBRATION_PATH.exists():
                _cal = json.loads(MORNING_E_CALIBRATION_PATH.read_text())
        except Exception:
            pass
        _morning_e_cal_cache = _cal
    _CAL_MORNING = _morning_e_cal_cache

    cal_key = None
    if _CAL_MORNING:
        h = hour if hour is not None else current_datetime.hour
        if h < 2:
            cal_key = "00-02"
        elif h < 4:
            cal_key = "02-04"
        elif h < 6:
            cal_key = "04-06"
        elif h < 8:
            cal_key = "06-08"
        if cal_key is not None and cal_key not in _CAL_MORNING:
            cal_key = None

    if cal_key is not None:
        p10_off = _CAL_MORNING[cal_key]["p10"]
        p90_off = _CAL_MORNING[cal_key]["p90"]
        q10_cal = max(0.0, q50 + p10_off)
        q90_cal = max(q10_cal, q50 + p90_off)
        quantiles = sorted([q10_cal, q25, q50, q75, q90_cal])
    else:
        quantiles = sorted([q10, q25, q50, q75, q90])
    if force_clip:
        quantiles = [max(0.0, v) for v in quantiles]
    downsides = dict(zip(
        ['p10', 'p25', 'p50', 'p75', 'p90'], quantiles
    ))

    prob_low_reached = 0.0
    if active.get('morning_low_reached_clf') is not None:
        clf_features = active['morning_low_reached_clf'].feature_name()
        prob_low_reached = active['morning_low_reached_clf'].predict(X[clf_features])[0]

    prob_survives_day = 0.0
    if active.get('morning_low_survives_day_clf') is not None:
        clf_features = active['morning_low_survives_day_clf'].feature_name()
        prob_survives_day = active['morning_low_survives_day_clf'].predict(X[clf_features])[0]

    q50_down = downsides['p50']
    pred_morning_min_p50 = min_so_far - q50_down
    pred_morning_min_p10 = min_so_far - downsides['p90']
    pred_morning_min_p90 = min_so_far - downsides['p10']

    return {
        'remaining_morning_downside_p10': downsides['p10'],
        'remaining_morning_downside_p25': downsides['p25'],
        'remaining_morning_downside_p50': downsides['p50'],
        'remaining_morning_downside_p75': downsides['p75'],
        'remaining_morning_downside_p90': downsides['p90'],
        'pred_morning_min_p10': pred_morning_min_p10,
        'pred_morning_min_p50': pred_morning_min_p50,
        'pred_morning_min_p90': pred_morning_min_p90,
        'prob_morning_low_reached': prob_low_reached,
        'prob_morning_low_survives_day': prob_survives_day,
        'sample_count': None,
    }


def predict_intraday_tmin(current_datetime, min_so_far, temp_60min_ago, temp_now,
                          forecast_tmax=None, forecast_tmin=None, temp_120m_ago=None, max_so_far=None,
                          rainfall_60m_filled=0.0, rainfall_120m_filled=0.0,
                          rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                          rainfall_30m_filled=0.0, rainfall_30m_missing_flag=1,
                          rainfall_data_age_minutes=0.0, rain_data_gap_flag=0,
                          temp_change_30min=None, temp_change_60min=None,
                          time_since_min_so_far=None, hour=None, minutes_since_midnight=None,
                          # Nowcast features (optional)
                          rain_nc_sum_0_60m=0.0, rain_nc_sum_0_120m=0.0,
                          rain_nc_any_0_120m=0.0, rain_nc_front_loaded_ratio=0.0,
                          rain_nc_heavy_0_120m=0.0, rain_nc_valid_horizon_count=0.0,
                          rain_nc_missing_flag=0, rain_nowcast_age_minutes=0.0,
                          rain_nowcast_missing_flag=0,
                          rain_nc_nearest_mm_sum_30m=0.0, rain_nc_nearest_mm_sum_60m=0.0,
                          rain_nc_nearest_mm_sum_90m=0.0, rain_nc_nearest_mm_sum_120m=0.0,
                          rain_nc_mean_r5km_sum_30m=0.0, rain_nc_mean_r5km_sum_60m=0.0,
                          rain_nc_mean_r5km_sum_90m=0.0, rain_nc_mean_r5km_sum_120m=0.0,
                          rain_nc_max_r5km_sum_30m=0.0, rain_nc_max_r5km_sum_60m=0.0,
                          rain_nc_max_r5km_sum_90m=0.0, rain_nc_max_r5km_sum_120m=0.0,
                          rain_nc_min_r5km_sum_30m=0.0, rain_nc_min_r5km_sum_60m=0.0,
                          rain_nc_min_r5km_sum_90m=0.0, rain_nc_min_r5km_sum_120m=0.0,
                          rain_nc_p90_r5km_sum_30m=0.0, rain_nc_p90_r5km_sum_60m=0.0,
                          rain_nc_p90_r5km_sum_90m=0.0, rain_nc_p90_r5km_sum_120m=0.0,
                          rain_nc_area_gt0_r5km_sum_30m=0.0, rain_nc_area_gt0_r5km_sum_60m=0.0,
                          rain_nc_area_gt0_r5km_sum_90m=0.0, rain_nc_area_gt0_r5km_sum_120m=0.0,
                          rain_nc_area_gt5_r5km_sum_30m=0.0, rain_nc_area_gt5_r5km_sum_60m=0.0,
                          rain_nc_area_gt5_r5km_sum_90m=0.0, rain_nc_area_gt5_r5km_sum_120m=0.0,
                          rain_cooling_120m=0.0, rise_from_min=0.0,
                          ):
    active = _get_active()
    feature_cols = active['feature_cols']

    if max_so_far is None:
        max_so_far = min_so_far + 5.0
    if forecast_tmax is None:
        forecast_tmax = max_so_far + 2.0
    if forecast_tmin is None:
        forecast_tmin = min_so_far - 2.0
    if temp_120m_ago is None:
        temp_120m_ago = temp_60min_ago
    if temp_change_30min is None:
        temp_change_30min = temp_now - temp_60min_ago
    if temp_change_60min is None:
        temp_change_60min = temp_now - temp_60min_ago
    if time_since_min_so_far is None:
        time_since_min_so_far = 0.0
    if hour is None:
        hour = current_datetime.hour
    if minutes_since_midnight is None:
        minutes_since_midnight = hour * 60 + current_datetime.minute

    feats = _build_features(
        current_datetime, max_so_far, min_so_far, temp_now, temp_60min_ago, temp_120m_ago,
        forecast_tmax, forecast_tmin,
        rainfall_60m_filled=rainfall_60m_filled, rainfall_120m_filled=rainfall_120m_filled,
        rainfall_60m_missing_flag=rainfall_60m_missing_flag, rainfall_120m_missing_flag=rainfall_120m_missing_flag,
        rainfall_30m_filled=rainfall_30m_filled, rainfall_30m_missing_flag=rainfall_30m_missing_flag,
        rainfall_data_age_minutes=rainfall_data_age_minutes, rain_data_gap_flag=rain_data_gap_flag,
        temp_change_30min=temp_change_30min, temp_change_60min=temp_change_60min,
        time_since_max_so_far=time_since_min_so_far, hour=hour, minutes_since_midnight=minutes_since_midnight,
        rain_nc_sum_0_60m=rain_nc_sum_0_60m, rain_nc_sum_0_120m=rain_nc_sum_0_120m,
        rain_nc_any_0_120m=rain_nc_any_0_120m, rain_nc_front_loaded_ratio=rain_nc_front_loaded_ratio,
        rain_nc_heavy_0_120m=rain_nc_heavy_0_120m, rain_nc_valid_horizon_count=rain_nc_valid_horizon_count,
        rain_nc_missing_flag=rain_nc_missing_flag, rain_nowcast_age_minutes=rain_nowcast_age_minutes,
        rain_nowcast_missing_flag=rain_nowcast_missing_flag,
        rain_nc_nearest_mm_sum_30m=rain_nc_nearest_mm_sum_30m, rain_nc_nearest_mm_sum_60m=rain_nc_nearest_mm_sum_60m,
        rain_nc_nearest_mm_sum_90m=rain_nc_nearest_mm_sum_90m, rain_nc_nearest_mm_sum_120m=rain_nc_nearest_mm_sum_120m,
        rain_nc_mean_r5km_sum_30m=rain_nc_mean_r5km_sum_30m, rain_nc_mean_r5km_sum_60m=rain_nc_mean_r5km_sum_60m,
        rain_nc_mean_r5km_sum_90m=rain_nc_mean_r5km_sum_90m, rain_nc_mean_r5km_sum_120m=rain_nc_mean_r5km_sum_120m,
        rain_nc_max_r5km_sum_30m=rain_nc_max_r5km_sum_30m, rain_nc_max_r5km_sum_60m=rain_nc_max_r5km_sum_60m,
        rain_nc_max_r5km_sum_90m=rain_nc_max_r5km_sum_90m, rain_nc_max_r5km_sum_120m=rain_nc_max_r5km_sum_120m,
        rain_nc_min_r5km_sum_30m=rain_nc_min_r5km_sum_30m, rain_nc_min_r5km_sum_60m=rain_nc_min_r5km_sum_60m,
        rain_nc_min_r5km_sum_90m=rain_nc_min_r5km_sum_90m, rain_nc_min_r5km_sum_120m=rain_nc_min_r5km_sum_120m,
        rain_nc_p90_r5km_sum_30m=rain_nc_p90_r5km_sum_30m, rain_nc_p90_r5km_sum_60m=rain_nc_p90_r5km_sum_60m,
        rain_nc_p90_r5km_sum_90m=rain_nc_p90_r5km_sum_90m, rain_nc_p90_r5km_sum_120m=rain_nc_p90_r5km_sum_120m,
        rain_nc_area_gt0_r5km_sum_30m=rain_nc_area_gt0_r5km_sum_30m, rain_nc_area_gt0_r5km_sum_60m=rain_nc_area_gt0_r5km_sum_60m,
        rain_nc_area_gt0_r5km_sum_90m=rain_nc_area_gt0_r5km_sum_90m, rain_nc_area_gt0_r5km_sum_120m=rain_nc_area_gt0_r5km_sum_120m,
        rain_nc_area_gt5_r5km_sum_30m=rain_nc_area_gt5_r5km_sum_30m, rain_nc_area_gt5_r5km_sum_60m=rain_nc_area_gt5_r5km_sum_60m,
        rain_nc_area_gt5_r5km_sum_90m=rain_nc_area_gt5_r5km_sum_90m, rain_nc_area_gt5_r5km_sum_120m=rain_nc_area_gt5_r5km_sum_120m,
        rain_cooling_120m=rain_cooling_120m, rise_from_min=rise_from_min,
    )
    X = pd.DataFrame([feats], columns=feature_cols)
    model_features = active['downside_q50'].feature_name()
    X = X[model_features]

    q10 = active['downside_q10'].predict(X)[0]
    q25 = active['downside_q25'].predict(X)[0]
    q50 = active['downside_q50'].predict(X)[0]
    q75 = active['downside_q75'].predict(X)[0]
    q90 = active['downside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])
    remaining_downside_p10, remaining_downside_p25, remaining_downside_p50, remaining_downside_p75, remaining_downside_p90 = quantiles

    if active['downside_zero'] is not None:
        zero_features = active['downside_zero'].feature_name()
        X_zero = X[zero_features]
        prob_min_reached = active['downside_zero'].predict(X_zero)[0]
    else:
        prob_min_reached = None

    remaining_downside_p10 = max(0.0, remaining_downside_p10)
    remaining_downside_p25 = max(0.0, remaining_downside_p25)
    remaining_downside_p50 = max(0.0, remaining_downside_p50)
    remaining_downside_p75 = max(0.0, remaining_downside_p75)
    remaining_downside_p90 = max(0.0, remaining_downside_p90)

    pred_tmin_p10 = min(min_so_far, min_so_far - remaining_downside_p90)
    pred_tmin_p50 = min(min_so_far, min_so_far - remaining_downside_p50)
    pred_tmin_p90 = min(min_so_far, min_so_far - remaining_downside_p10)

    return {
        'remaining_downside_p10': remaining_downside_p10,
        'remaining_downside_p25': remaining_downside_p25,
        'remaining_downside_p50': remaining_downside_p50,
        'remaining_downside_p75': remaining_downside_p75,
        'remaining_downside_p90': remaining_downside_p90,
        'prob_min_reached': prob_min_reached,
        'pred_tmin_p50': pred_tmin_p50,
        'pred_tmin_p10': pred_tmin_p10,
        'pred_tmin_p90': pred_tmin_p90,
        'sample_count': None
    }


def predict_intraday_tmax_all(
    current_datetime, max_so_far, temp_60min_ago, temp_now,
    forecast_tmax=None, forecast_tmin=None, temp_120m_ago=None, min_so_far=None,
    rainfall_60m_filled=0.0, rainfall_120m_filled=0.0,
    rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
    rainfall_30m_filled=0.0, rainfall_30m_missing_flag=1,
    rainfall_data_age_minutes=0.0, rain_data_gap_flag=0,
    temp_change_30min=None, temp_change_60min=None,
    time_since_max_so_far=None, hour=None, minutes_since_midnight=None,
    rh_current=50.0, temp_buffer=None, rh_buffer=None,
    time_since_min_so_far=None,
    # Model 2A / Model G pressure data
    pressure_current=None, pressure_30m_ago=None, pressure_change_60m=0.0, pressure_change_180m=0.0,
    wind_ref_mean=0.0, wind_ref_max=0.0,
    wind_victoria_harbour_mean=0.0, wind_victoria_harbour_max=0.0,
    wind_highland_mean=0.0, wind_highland_max=0.0,
    wind_all_change_60m=0.0, wind_kings_park_current=0.0,
    forecast_age_minutes=None, forecast_lead_days=None,
    obs_data_age_minutes=None, wind_data_age_minutes=None,
    **rain_kwargs):
    _load_models()
    # Strip any Model A/D/E params that may have leaked into rain_kwargs (defensive)
    for _key in ('rh_current', 'temp_buffer', 'rh_buffer', 'time_since_min_so_far',
                 'temp_buffer_long', 'rh_buffer_long',
                 'prev_18_temp', 'prev_21_temp', 'prev_2359_temp',
                 'prev_evening_temp_change', 'prev_evening_temp_min',
                 'prev_evening_temp_range', 'prev_evening_temp_slope',
                 'prev_evening_rh_mean', 'prev_evening_rh_max',
                 'prev_evening_dew_point_mean',
                 'prev_evening_rainfall_18_24', 'prev_evening_rain_flag'):
        rain_kwargs.pop(_key, None)
    results = {}
    for model_key in ['baseline', 'rain_nowcast']:
        if model_key not in _model_cache:
            continue
        set_active_model(model_key)
        results[model_key] = predict_intraday_tmax(
            current_datetime, max_so_far, temp_60min_ago, temp_now,
            forecast_tmax=forecast_tmax, forecast_tmin=forecast_tmin,
            temp_120m_ago=temp_120m_ago, min_so_far=min_so_far,
            rainfall_60m_filled=rainfall_60m_filled, rainfall_120m_filled=rainfall_120m_filled,
            rainfall_60m_missing_flag=rainfall_60m_missing_flag, rainfall_120m_missing_flag=rainfall_120m_missing_flag,
            rainfall_30m_filled=rainfall_30m_filled, rainfall_30m_missing_flag=rainfall_30m_missing_flag,
            rainfall_data_age_minutes=rainfall_data_age_minutes, rain_data_gap_flag=rain_data_gap_flag,
            temp_change_30min=temp_change_30min, temp_change_60min=temp_change_60min,
            time_since_max_so_far=time_since_max_so_far, hour=hour,
            minutes_since_midnight=minutes_since_midnight,
            **rain_kwargs
        )
    if 'model_a' in _model_cache:
        set_active_model('model_a')
        try:
            results['model_a'] = predict_intraday_tmax_model_a(
                current_datetime, max_so_far, temp_now,
                rh_current=rh_current, min_so_far=min_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
            )
        except Exception as e:
            logger.warning("Model A prediction failed: %s", e)
            results['model_a'] = None
    if 'model_b' in _model_cache:
        set_active_model('model_b')
        try:
            results['model_b'] = predict_intraday_tmax_model_b(
                current_datetime, max_so_far, temp_now,
                rh_current=rh_current, min_so_far=min_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
            )
        except Exception as e:
            logger.warning("Model B prediction failed: %s", e)
            results['model_b'] = None
    if 'model_c' in _model_cache:
        set_active_model('model_c')
        try:
            # Extract nowcast params from rain_kwargs (forwarded from caller)
            nc_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('rain_nc_') or k.startswith('rain_nowcast_')}
            results['model_c'] = predict_intraday_tmax_model_c(
                current_datetime, max_so_far, temp_now,
                rh_current=rh_current, min_so_far=min_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
                **nc_kw,
            )
        except Exception as e:
            logger.warning("Model C prediction failed: %s", e)
            results['model_c'] = None
    if 'model_g' in _model_cache:
        set_active_model('model_g')
        try:
            results['model_g'] = predict_intraday_tmax_model_g(
                current_datetime, max_so_far, temp_now,
                humidity=rh_current, min_so_far=min_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour, minute=current_datetime.minute if current_datetime else None,
                forecast_tmax=forecast_tmax,
                pressure=pressure_current, pressure_30m_ago=pressure_30m_ago,
            )
        except Exception as e:
            logger.warning("Model G prediction failed: %s", e)
            results['model_g'] = None
    if 'model_2a' in _model_cache:
        set_active_model('model_2a')
        try:
            results['model_2a'] = predict_intraday_tmax_model_2a(
                current_datetime, max_so_far, temp_now,
                humidity=rh_current, min_so_far=min_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour, minute=current_datetime.minute if current_datetime else None,
                forecast_tmax=forecast_tmax, forecast_tmin=forecast_tmin,
                pressure_current=pressure_current,
                pressure_change_60m=pressure_change_60m,
                pressure_change_180m=pressure_change_180m,
                dew_point_current=None,
                forecast_age_minutes=forecast_age_minutes,
                forecast_lead_days=forecast_lead_days,
                wind_ref_mean=wind_ref_mean,
                wind_ref_max=wind_ref_max,
                wind_victoria_harbour_mean=wind_victoria_harbour_mean,
                wind_victoria_harbour_max=wind_victoria_harbour_max,
                wind_highland_mean=wind_highland_mean,
                wind_highland_max=wind_highland_max,
                wind_all_change_60m=wind_all_change_60m,
                wind_kings_park_current=wind_kings_park_current,
                obs_data_age_minutes=obs_data_age_minutes,
                wind_data_age_minutes=wind_data_age_minutes,
            )
        except Exception as e:
            logger.warning("Model 2A prediction failed: %s", e)
            results['model_2a'] = None
    set_active_model('baseline')
    return results


def predict_intraday_tmax_model_g(current_datetime, max_so_far, temp_now,
                                  humidity=50.0, min_so_far=None,
                                  time_since_max=0.0, time_since_min=0.0,
                                  temp_buffer=None, rh_buffer=None,
                                  hour=None, minute=None,
                                  forecast_tmax=None, pressure=None, pressure_30m_ago=None,
                                  **kwargs):
    """Predict remaining upside using Model G (forecast_gap + max_so_far model)."""

    h = hour if hour is not None else (current_datetime.hour if current_datetime else 12)
    m = minute if minute is not None else (current_datetime.minute if current_datetime else 0)
    month = current_datetime.month if current_datetime else 6

    # Build Model G features (17 features)
    temp_arr = np.array(list(temp_buffer) if temp_buffer else [temp_now])
    rh_arr = np.array(list(rh_buffer) if rh_buffer else [humidity])
    idx = len(temp_arr) - 1

    temp_change_30m = temp_now - (temp_arr[idx-30] if idx >= 30 else temp_arr[0])

    start_vol = max(0, idx - 29)
    temp_volatility_30m = float(np.std(temp_arr[start_vol:idx+1], ddof=1)) if (idx - start_vol) >= 1 else 0.0

    rh_change_30m = humidity - (rh_arr[idx-30] if idx >= 30 else rh_arr[0])

    forecast_gap = forecast_tmax - max_so_far if forecast_tmax is not None else 0.0
    pressure_delta = pressure - pressure_30m_ago if pressure and pressure_30m_ago else 0.0
    drop_from_max = max_so_far - temp_now if max_so_far is not None else 0.0

    features = {
        "value": temp_now,
        "humidity": humidity,
        "pressure": pressure if pressure is not None else 1010.0,
        "temp_slope_30min": temp_change_30m,
        "temp_volatility_30min": temp_volatility_30m,
        "humid_delta_30min": rh_change_30m,
        "pressure_delta_30min": pressure_delta,
        "forecast_gap": forecast_gap,
        "hour": h,
        "minute": m,
        "day_of_week": current_datetime.weekday() if current_datetime else 0,
        "month": month,
        "max_so_far": max_so_far if max_so_far is not None else temp_now,
        "drop_from_max": drop_from_max,
        "time_since_max": time_since_max,
    }

    active = _get_active()
    feature_cols = active['feature_cols']
    model_features = active['upside_q50'].feature_name()
    cols = [c for c in feature_cols if c in model_features]
    X = pd.DataFrame([features], columns=cols)[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    # upside_zero classifier
    prob_max_reached = 0.0
    if active.get('upside_zero') is not None:
        try:
            prob_max_reached = active['upside_zero'].predict(X)[0]
        except Exception:
            import json as _json
            import math
            thresh_path = Path('models/intraday_minute_ml_model_g/best_threshold.json')
            if thresh_path.exists():
                with open(thresh_path) as _f:
                    th = _json.load(_f).get('upside_zero_threshold', 0.5)
                prob_class = active['upside_zero'].predict(X, pred_contrib=False)[0]
                prob_max_reached = 1.0 / (1.0 + math.exp(-prob_class)) if isinstance(prob_class, float) else 0.0
                prob_max_reached = 1.0 if prob_max_reached > th else 0.0

    pred_tmax_p10 = max_so_far + remaining_upside_p10
    pred_tmax_p25 = max_so_far + remaining_upside_p25
    pred_tmax_p50 = max_so_far + remaining_upside_p50
    pred_tmax_p75 = max_so_far + remaining_upside_p75
    pred_tmax_p90 = max_so_far + remaining_upside_p90

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p25': pred_tmax_p25,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p75': pred_tmax_p75,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None,
    }


def predict_intraday_tmax_model_2a(
    current_datetime, max_so_far, temp_now,
    humidity=50.0, pressure_current=None, pressure_change_60m=0.0, pressure_change_180m=0.0,
    dew_point_current=None,
    min_so_far=None, time_since_max=0.0,
    temp_buffer=None, rh_buffer=None,
    forecast_tmax=None, forecast_tmin=None,
    forecast_age_minutes=None, forecast_lead_days=None,
    wind_ref_mean=None, wind_ref_max=None,
    wind_victoria_harbour_mean=None, wind_victoria_harbour_max=None,
    wind_highland_mean=None, wind_highland_max=None,
    wind_all_change_60m=None, wind_kings_park_current=None,
    obs_data_age_minutes=None, wind_data_age_minutes=None,
    hour=None, minute=None,
):
    """Predict remaining upside using Model 2A (core baseline with forecast + wind)."""
    h = hour if hour is not None else (current_datetime.hour if current_datetime else 12)
    m = minute if minute is not None else (current_datetime.minute if current_datetime else 0)
    dt = current_datetime

    temp_arr = np.array(list(temp_buffer) if temp_buffer else [temp_now])
    idx = len(temp_arr) - 1
    temp_change_30m = temp_now - (temp_arr[idx-3] if idx >= 3 else temp_arr[0])
    temp_change_60m = temp_now - (temp_arr[idx-6] if idx >= 6 else temp_arr[0])
    temp_slope_30m = temp_change_30m / 30.0
    temp_slope_60m = temp_change_60m / 60.0

    start_vol = max(0, idx - 5)
    temp_volatility_60m = float(np.std(temp_arr[start_vol:idx+1], ddof=1)) if (idx - start_vol) >= 1 else 0.0
    temp_acceleration_60m = temp_slope_30m - (temp_slope_30m - (
        temp_arr[idx-3] - (temp_arr[idx-6] if idx >= 6 else temp_arr[0])
    ) / 30.0)

    rh_arr = np.array(list(rh_buffer) if rh_buffer else [humidity])
    rh_idx = len(rh_arr) - 1
    rh_change_60m = humidity - (rh_arr[rh_idx-6] if rh_idx >= 6 else rh_arr[0])

    # Compute dew_point_current via Magnus formula if not provided
    if dew_point_current is None and humidity is not None and temp_now is not None:
        try:
            import math as _math
            _a = 17.625
            _b = 243.04
            _gamma = _math.log(humidity / 100.0) + (_a * temp_now) / (_b + temp_now)
            dew_point_current = (_b * _gamma) / (_a - _gamma)
        except Exception:
            dew_point_current = temp_now - 5
    elif dew_point_current is None:
        dew_point_current = temp_now - 5

    dew_point_spread = temp_now - dew_point_current
    # Compute dew_point deltas from buffer via Magnus formula (matching training diff(6))
    dew_point_change_60m = 0.0
    dew_point_spread_change_60m = 0.0
    if idx >= 6 and rh_idx >= 6 and dew_point_current is not None:
        try:
            import math as _m
            _a, _b = 17.625, 243.04
            _t6 = temp_arr[idx-6]
            _rh6 = rh_arr[rh_idx-6]
            _gamma6 = _m.log(_rh6 / 100.0) + (_a * _t6) / (_b + _t6)
            _dp6 = (_b * _gamma6) / (_a - _gamma6)
            dew_point_change_60m = dew_point_current - _dp6
            dew_point_spread_change_60m = (temp_now - dew_point_current) - (_t6 - _dp6)
        except Exception:
            pass

    forecast_gap = forecast_tmax - max_so_far if forecast_tmax is not None else 0.0
    forecast_range = forecast_tmax - forecast_tmin if forecast_tmax is not None and forecast_tmin is not None else 0.0

    mins_midnight = h * 60 + m
    doy = dt.timetuple().tm_yday if dt else 1
    month_sin = np.sin(2 * np.pi * dt.month / 12) if dt else 0
    month_cos = np.cos(2 * np.pi * dt.month / 12) if dt else 0
    day_sin = np.sin(2 * np.pi * doy / 365.25)
    day_cos = np.cos(2 * np.pi * doy / 365.25)
    is_morning = 1 if 6 <= h < 12 else 0
    is_afternoon = 1 if 12 <= h < 18 else 0
    is_evening = 1 if 18 <= h < 24 else 0

    features = {
        "temp_current": temp_now,
        "rh_current": humidity,
        "pressure_current": pressure_current if pressure_current is not None else 1010.0,
        "dew_point_current": dew_point_current if dew_point_current is not None else temp_now - 5,
        "dew_point_spread": dew_point_spread,
        "max_so_far": max_so_far if max_so_far is not None else temp_now,
        "min_so_far": min_so_far if min_so_far is not None else temp_now,
        "range_so_far": (max_so_far - min_so_far) if max_so_far is not None and min_so_far is not None else 0,
        "drop_from_max": (max_so_far - temp_now) if max_so_far is not None else 0,
        "time_since_max": time_since_max,
        "temp_change_30m": temp_change_30m,
        "temp_change_60m": temp_change_60m,
        "temp_slope_30m": temp_slope_30m,
        "temp_slope_60m": temp_slope_60m,
        "temp_acceleration_60m": temp_acceleration_60m,
        "temp_volatility_60m": temp_volatility_60m,
        "rh_change_60m": rh_change_60m,
        "dew_point_change_60m": dew_point_change_60m,
        "dew_point_spread_change_60m": dew_point_spread_change_60m,
        "pressure_change_60m": pressure_change_60m,
        "pressure_change_180m": pressure_change_180m,
        "forecast_min_temp": forecast_tmin if forecast_tmin is not None else 0,
        "forecast_max_temp": forecast_tmax if forecast_tmax is not None else 0,
        "forecast_range": forecast_range,
        "forecast_gap_from_max_so_far": forecast_gap,
        "forecast_age_minutes": forecast_age_minutes if forecast_age_minutes is not None else 0,
        "forecast_lead_days": forecast_lead_days if forecast_lead_days is not None else 0,
        "wind_ref_mean": wind_ref_mean if wind_ref_mean is not None else 0,
        "wind_ref_max": wind_ref_max if wind_ref_max is not None else 0,
        "wind_victoria_harbour_mean": wind_victoria_harbour_mean if wind_victoria_harbour_mean is not None else 0,
        "wind_victoria_harbour_max": wind_victoria_harbour_max if wind_victoria_harbour_max is not None else 0,
        "wind_highland_mean": wind_highland_mean if wind_highland_mean is not None else 0,
        "wind_highland_max": wind_highland_max if wind_highland_max is not None else 0,
        "wind_all_change_60m": wind_all_change_60m if wind_all_change_60m is not None else 0,
        "wind_kings_park_current": wind_kings_park_current if wind_kings_park_current is not None else 0,
        "minutes_since_midnight": mins_midnight,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_morning": is_morning,
        "is_afternoon": is_afternoon,
        "is_evening": is_evening,
        "obs_data_age_minutes": obs_data_age_minutes if obs_data_age_minutes is not None else 8,
        "wind_data_age_minutes": wind_data_age_minutes if wind_data_age_minutes is not None else 8,
    }

    active = _get_active()
    feature_cols = active['feature_cols']
    model_features = active['upside_q50'].feature_name()
    cols = [c for c in feature_cols if c in features and c in model_features]
    X = pd.DataFrame([features], columns=cols)[model_features]

    q10 = active['upside_q10'].predict(X)[0]
    q25 = active['upside_q25'].predict(X)[0]
    q50 = active['upside_q50'].predict(X)[0]
    q75 = active['upside_q75'].predict(X)[0]
    q90 = active['upside_q90'].predict(X)[0]

    quantiles = sorted([q10, q25, q50, q75, q90])
    remaining_upside_p10, remaining_upside_p25, remaining_upside_p50, remaining_upside_p75, remaining_upside_p90 = quantiles

    prob_max_reached = 0.0
    if active.get('upside_zero') is not None:
        try:
            clf_features = active['upside_zero'].feature_name()
            prob_max_reached = active['upside_zero'].predict(X[clf_features])[0]
        except Exception:
            import json as _json
            import math
            thresh_path = Path('models/intraday_minute_ml_model_2a/best_threshold.json')
            if thresh_path.exists():
                with open(thresh_path) as _f:
                    th = _json.load(_f).get('upside_zero_threshold', 0.5)
                prob_class = active['upside_zero'].predict(X, pred_contrib=False)[0]
                prob_max_reached = 1.0 / (1.0 + math.exp(-prob_class)) if isinstance(prob_class, float) else 0.0
                prob_max_reached = 1.0 if prob_max_reached > th else 0.0

    prob_not_reached = 1.0 - prob_max_reached
    remaining_upside_p10 *= prob_not_reached
    remaining_upside_p25 *= prob_not_reached
    remaining_upside_p50 *= prob_not_reached
    remaining_upside_p75 *= prob_not_reached
    remaining_upside_p90 *= prob_not_reached
    pred_tmax_p10 = max_so_far + remaining_upside_p10
    pred_tmax_p25 = max_so_far + remaining_upside_p25
    pred_tmax_p50 = max_so_far + remaining_upside_p50
    pred_tmax_p75 = max_so_far + remaining_upside_p75
    pred_tmax_p90 = max_so_far + remaining_upside_p90

    return {
        'remaining_upside_p10': remaining_upside_p10,
        'remaining_upside_p25': remaining_upside_p25,
        'remaining_upside_p50': remaining_upside_p50,
        'remaining_upside_p75': remaining_upside_p75,
        'remaining_upside_p90': remaining_upside_p90,
        'prob_max_reached': prob_max_reached,
        'pred_tmax_p10': pred_tmax_p10,
        'pred_tmax_p25': pred_tmax_p25,
        'pred_tmax_p50': pred_tmax_p50,
        'pred_tmax_p75': pred_tmax_p75,
        'pred_tmax_p90': pred_tmax_p90,
        'sample_count': None,
    }


def predict_intraday_tmin_all(current_datetime, min_so_far, temp_60min_ago, temp_now,
                               forecast_tmax=None, forecast_tmin=None, temp_120m_ago=None, max_so_far=None,
                               rainfall_60m_filled=0.0, rainfall_120m_filled=0.0,
                               rainfall_60m_missing_flag=1, rainfall_120m_missing_flag=1,
                               rainfall_30m_filled=0.0, rainfall_30m_missing_flag=1,
                               rainfall_data_age_minutes=0.0, rain_data_gap_flag=0,
                               temp_change_30min=None, temp_change_60min=None,
                               time_since_min_so_far=None, hour=None, minutes_since_midnight=None,
                               # Model A params (optional)
                               rh_current=50.0, temp_buffer=None, rh_buffer=None,
                               time_since_max_so_far=None,
                               **rain_kwargs):
    _load_models()
    for _key in ('rh_current', 'temp_buffer', 'rh_buffer', 'time_since_max_so_far'):
        rain_kwargs.pop(_key, None)
    # temp_buffer_long / rh_buffer_long / prev_* are consumed by D/E only
    for _key in ('temp_buffer_long', 'rh_buffer_long',
                 'prev_18_temp', 'prev_21_temp', 'prev_2359_temp',
                 'prev_evening_temp_change', 'prev_evening_temp_min',
                 'prev_evening_temp_range', 'prev_evening_temp_slope',
                 'prev_evening_rh_mean', 'prev_evening_rh_max',
                 'prev_evening_dew_point_mean',
                 'prev_evening_rainfall_18_24', 'prev_evening_rain_flag'):
        rain_kwargs.pop(_key, None)
    results = {}
    for model_key in ['baseline', 'rain_nowcast']:
        if model_key not in _model_cache:
            continue
        set_active_model(model_key)
        results[model_key] = predict_intraday_tmin(
            current_datetime, min_so_far, temp_60min_ago, temp_now,
            forecast_tmax=forecast_tmax, forecast_tmin=forecast_tmin,
            temp_120m_ago=temp_120m_ago, max_so_far=max_so_far,
            rainfall_60m_filled=rainfall_60m_filled, rainfall_120m_filled=rainfall_120m_filled,
            rainfall_60m_missing_flag=rainfall_60m_missing_flag, rainfall_120m_missing_flag=rainfall_120m_missing_flag,
            rainfall_30m_filled=rainfall_30m_filled, rainfall_30m_missing_flag=rainfall_30m_missing_flag,
            rainfall_data_age_minutes=rainfall_data_age_minutes, rain_data_gap_flag=rain_data_gap_flag,
            temp_change_30min=temp_change_30min, temp_change_60min=temp_change_60min,
            time_since_min_so_far=time_since_min_so_far, hour=hour,
            minutes_since_midnight=minutes_since_midnight,
            **rain_kwargs
        )
    if 'model_a_tmin' in _model_cache:
        set_active_model('model_a_tmin')
        try:
            results['model_a_tmin'] = predict_intraday_tmin_model_a(
                current_datetime, min_so_far, temp_now,
                rh_current=rh_current, max_so_far=max_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
            )
        except Exception as e:
            logger.warning("Model A Tmin prediction failed: %s", e)
            results['model_a_tmin'] = None
    if 'model_b_tmin' in _model_cache:
        set_active_model('model_b_tmin')
        try:
            results['model_b_tmin'] = predict_intraday_tmin_model_b(
                current_datetime, min_so_far, temp_now,
                rh_current=rh_current, max_so_far=max_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
            )
        except Exception as e:
            logger.warning("Model B Tmin prediction failed: %s", e)
            results['model_b_tmin'] = None
    if 'model_c_tmin' in _model_cache:
        set_active_model('model_c_tmin')
        try:
            nc_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('rain_nc_') or k.startswith('rain_nowcast_')}
            results['model_c_tmin'] = predict_intraday_tmin_model_c(
                current_datetime, min_so_far, temp_now,
                rh_current=rh_current, max_so_far=max_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
                **nc_kw,
            )
        except Exception as e:
            logger.warning("Model C Tmin prediction failed: %s", e)
            results['model_c_tmin'] = None
    if 'model_d_tmin' in _model_cache:
        set_active_model('model_d_tmin')
        try:
            nc_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('rain_nc_') or k.startswith('rain_nowcast_')}
            prev_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('prev_')}
            d_kw = {k: v for k, v in rain_kwargs.items() if k in ('temp_buffer_long', 'rh_buffer_long')}
            results['model_d_tmin'] = predict_intraday_tmin_model_d(
                current_datetime, min_so_far, temp_now,
                rh_current=rh_current, max_so_far=max_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
                **nc_kw,
                **prev_kw,
                **d_kw,
            )
        except Exception as e:
            logger.warning("Model D Tmin prediction failed: %s", e)
            results['model_d_tmin'] = None
    if 'model_e_morning_tmin' in _model_cache and hour is not None and hour < 8:
        set_active_model('model_e_morning_tmin')
        try:
            nc_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('rain_nc_') or k.startswith('rain_nowcast_')}
            prev_kw = {k: v for k, v in rain_kwargs.items() if k.startswith('prev_')}
            d_kw = {k: v for k, v in rain_kwargs.items() if k in ('temp_buffer_long', 'rh_buffer_long')}
            results['model_e_morning_tmin'] = predict_intraday_tmin_model_e_morning(
                current_datetime, min_so_far, temp_now,
                rh_current=rh_current, max_so_far=max_so_far,
                time_since_max=time_since_max_so_far or 0.0,
                time_since_min=time_since_min_so_far or 0.0,
                temp_buffer=temp_buffer, rh_buffer=rh_buffer,
                hour=hour,
                rainfall_60m=rainfall_60m_filled,
                rainfall_120m=rainfall_120m_filled,
                rainfall_60m_missing_flag=rainfall_60m_missing_flag,
                rainfall_120m_missing_flag=rainfall_120m_missing_flag,
                temp_change_60m=temp_change_60min or 0.0,
                drop_from_max=0.0,
                **nc_kw,
                **prev_kw,
                **d_kw,
            )
        except Exception as e:
            logger.warning("Model E Morning Tmin prediction failed: %s", e)
            results['model_e_morning_tmin'] = None
    # Backward-compatible aliases for dashboard loops expecting 'model_a' etc.
    for alias in ['model_a', 'model_b', 'model_c', 'model_d']:
        tmin_key = f"{alias}_tmin"
        if tmin_key in results and alias not in results:
            results[alias] = results[tmin_key]
    set_active_model('baseline')
    return results


def combine_with_prior(prior_mean, prior_std, intraday_pred, weight=0.0, std_scale=0.9):
    if weight <= 0.0 or intraday_pred is None:
        intra_mean = intraday_pred.get('pred_tmax_p50') or intraday_pred.get('pred_tmin_p50')
        if intra_mean is None:
            return prior_mean, prior_std
        if 'remaining_upside_p90' in intraday_pred:
            p90 = intraday_pred['remaining_upside_p90']
            p10 = intraday_pred['remaining_upside_p10']
        else:
            p90 = intraday_pred['remaining_downside_p90']
            p10 = intraday_pred['remaining_downside_p10']
        intra_std = max((p90 - p10) / 2.56 * std_scale, 0.2)
        return intra_mean, intra_std

    intra_mean = intraday_pred.get('pred_tmax_p50') or intraday_pred.get('pred_tmin_p50')
    if intra_mean is None:
        return prior_mean, prior_std
    if 'remaining_upside_p90' in intraday_pred:
        p90 = intraday_pred['remaining_upside_p90']
        p10 = intraday_pred['remaining_upside_p10']
    else:
        p90 = intraday_pred['remaining_downside_p90']
        p10 = intraday_pred['remaining_downside_p10']
    intra_std = max((p90 - p10) / 2.56 * std_scale, 0.2)
    post_mean = weight * prior_mean + (1 - weight) * intra_mean
    post_std = np.sqrt((weight * prior_std)**2 + ((1 - weight) * intra_std)**2)
    return post_mean, post_std
