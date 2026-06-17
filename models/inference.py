# models/inference.py
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
MODEL_DIR = Path('models')

# 全域快取，避免在非 Streamlit 環境中重複載入
_model_cache = {}

def _get_xgb():
    import xgboost as _xgb
    return _xgb


def _maybe_st_cache_resource(func):
    try:
        import streamlit as _st
        return _st.cache_resource(func)
    except ImportError:
        return func


@_maybe_st_cache_resource
def load_models():
    if 'config' in _model_cache:
        return _model_cache

    xgb = _get_xgb()
    config_path = MODEL_DIR / 'feature_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"找不到特徵配置: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    models = {
        'tmax_mean': xgb.XGBRegressor(),
        'tmax_std': xgb.XGBRegressor(),
        'tmin_mean': xgb.XGBRegressor(),
        'tmin_std': xgb.XGBRegressor()
    }

    for name, model in models.items():
        model_path = MODEL_DIR / f"xgb_{name}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        model.load_model(str(model_path))
        logger.info(f"✅ 成功載入模型: {model_path.name}")

    _model_cache['config'] = config
    _model_cache['models'] = models

    return _model_cache

def predict_distribution(features_dict, model_type='tmax', hko_spread=None, std_multiplier=1.15):
    """
    預測目標溫度的常態分佈參數 (Mean, Std)。
    
    Args:
        features_dict (dict): 對齊訓練格式的特徵字典。
        model_type (str): 'tmax' 或 'tmin'。
        hko_spread (float, optional): HKO P75 - P25 的差值。若提供，將用於校準 Std。
        std_multiplier (float): 標準差的校準擴張係數 (預設 1.15)。
        
    Returns:
        tuple: (mean, std)
    """
    cache = load_models()
    config = cache['config']
    models = cache['models']
    
    feature_cols = config['features']
    min_std = config.get('min_std', 0.5)
    
    # 嚴格按照訓練時的順序組裝特徵向量
    X = np.array([[features_dict.get(col, 0.0) for col in feature_cols]])
    
    mean_model = models[f'{model_type}_mean']
    std_model = models[f'{model_type}_std']
    
    pred_mean = mean_model.predict(X)[0]
    pred_abs_res = std_model.predict(X)[0]
    
    # 將絕對殘差轉換為標準差: sigma = E[|X-mu|] * sqrt(pi/2)
    conversion_factor = config.get('std_conversion_factor', np.sqrt(np.pi / 2))
    pred_std = pred_abs_res * conversion_factor
    
    # 強制最小標準差
    pred_std = max(pred_std, min_std)
    
    # 套用校準擴張係數
    pred_std = pred_std * std_multiplier
    
    # 若 HKO 提供了 Spread (P75 - P25)，將其隱含的 Std 作為下限
    # 常態分佈中，P75 - P25 約等於 1.35 * std
    if hko_spread is not None and hko_spread > 0:
        hko_implied_std = hko_spread / 1.35
        pred_std = max(pred_std, hko_implied_std)
        
    return float(pred_mean), float(pred_std)

# models/inference.py (節錄替換部分)

def predict_bucket_probabilities(mean, std, market_buckets, max_since_midnight=None, min_since_midnight=None, is_today=False, is_min_temp=False):
    """
    基於常態分佈計算每個市場桶的機率，並嚴格執行雙向物理截斷。
    """
    probs = {}
    from scipy.stats import norm as _norm

    for bucket in market_buckets:
        name = bucket['name']
        lower = bucket['lower']
        upper = bucket['upper']
        
        # === 核心風控：雙向即時物理截斷 ===
        if is_today:
            if not is_min_temp and max_since_midnight is not None:
                # 最高溫市場：最終最高溫不可能低於已出現的最高溫
                if upper <= max_since_midnight:
                    probs[name] = 0.0
                    continue
            elif is_min_temp and min_since_midnight is not None:
                # 最低溫市場：最終最低溫不可能高於已出現的最低溫
                if lower >= min_since_midnight:
                    probs[name] = 0.0
                    continue
            
        cdf_upper = _norm.cdf(upper, loc=mean, scale=std) if upper != np.inf else 1.0
        cdf_lower = _norm.cdf(lower, loc=mean, scale=std) if lower != -np.inf else 0.0
        
        probs[name] = max(0.0, cdf_upper - cdf_lower)
        
    # 歸一化
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    else:
        if market_buckets:
            # 極端防禦：若所有桶都被截斷，將機率分配給最邊緣的桶
            if is_min_temp:
                target_bucket = min(market_buckets, key=lambda x: x['lower'] if x['lower'] != -np.inf else 999)
            else:
                target_bucket = max(market_buckets, key=lambda x: x['upper'] if x['upper'] != np.inf else -999)
            probs[target_bucket['name']] = 1.0
            
    return probs