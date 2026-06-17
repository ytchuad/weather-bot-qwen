# features/live_feature_builder.py
import requests
import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

_HKT_OFFSET = timedelta(hours=8)
def _hkt_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + _HKT_OFFSET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=tc"
HISTORY_PATH = Path('data/live_forecast_history.parquet')
CONFIG_PATH = Path('models/feature_config.json')

_last_fetch_time = None

DIR_ANGLES = {
    '北': 0, '東北偏北': 22.5, '東北': 45, '東北偏東': 67.5,
    '東': 90, '東南偏東': 112.5, '東南': 135, '東南偏南': 157.5,
    '南': 180, '西南偏南': 202.5, '西南': 225, '西南偏西': 247.5,
    '西': 270, '西北偏西': 292.5, '西北': 315, '西北偏北': 337.5
}
WEATHER_KEYWORDS = ['雨', '雷暴', '驟雨', '多雲', '天晴', '酷熱', '寒冷', '潮濕', '霧', '煙霞']

def parse_hko_value(field):
    if isinstance(field, dict):
        val = field.get('value')
        try: return float(val) if val is not None else np.nan
        except (ValueError, TypeError): return np.nan
    try: return float(field)
    except (ValueError, TypeError): return np.nan

def parse_wind(text):
    if not isinstance(text, str): return np.nan, np.nan
    nums = re.findall(r'\d+', text)
    force = np.mean([float(n) for n in nums]) if nums else np.nan
    dir_text = re.sub(r'\d+至\d+級|\d+級|級|風', '', text).strip()
    angle = np.nan
    if '至' in dir_text:
        parts = dir_text.split('至')
        angles = [DIR_ANGLES.get(p.strip(), np.nan) for p in parts if p.strip() in DIR_ANGLES]
        if angles: angle = np.nanmean(angles)
    else:
        for k, v in sorted(DIR_ANGLES.items(), key=lambda x: len(x[0]), reverse=True):
            if k in dir_text: angle = v; break
    return angle, force

def extract_keywords(text):
    if not isinstance(text, str): text = ""
    return {f'kw_{kw}': 1 if kw in text else 0 for kw in WEATHER_KEYWORDS}

def fetch_percentiles():
    """抓取 HKO 9天預報百分位數 JS 檔案"""
    start_d = _hkt_now() + timedelta(days=1)
    end_d = start_d + timedelta(days=8)
    url = f"https://www.hko.gov.hk/wxinfo/currwx/climatjs/9day{start_d.strftime('%m%d')}_{end_d.strftime('%m%d')}.js?_={int(_hkt_now().timestamp()*1000)}"
    
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.hko.gov.hk/'}, timeout=5)
        resp.raise_for_status()
        text = resp.text
        
        # 使用正則提取 JSON 陣列結構 (兼容 var NineDayForecast = [...];)
        match = re.search(r'\[\s*\{.*?"ForecastDate".*?\}\s*\]', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            p_map = {}
            for item in data:
                date_str = str(item.get('ForecastDate', ''))
                # 兼容多種可能的 Key 命名 (MaxTempPS25, ps25, tmax_p25 等)
                p25 = item.get('MaxTempPS25') or item.get('ps25') or item.get('tmax_p25')
                p75 = item.get('MaxTempPS75') or item.get('ps75') or item.get('tmax_p75')
                if date_str and p25 is not None and p75 is not None:
                    p_map[date_str] = {'p25': float(p25), 'p75': float(p75)}
            return p_map
    except Exception as e:
        logging.warning(f"獲取百分位數失敗 (可能尚未更新): {e}")
    return {}

def update_forecast_database():
    global _last_fetch_time
    now = datetime.now()
    if _last_fetch_time is not None and (now - _last_fetch_time).total_seconds() < 60:
        return
    _last_fetch_time = now
    logging.info("正在獲取 HKO 9天預報 API...")
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.error(f"API 請求失敗: {e}")
        return

    # 獲取百分位數
    p_data = fetch_percentiles()
    if p_data:
        logging.info(f"✅ 成功獲取 {len(p_data)} 天的百分位數數據")

    fetch_time = datetime.now()
    records = []
    
    for day in data.get('weatherForecast', []):
        try:
            date_str = day['forecastDate']
            target_dt = pd.to_datetime(date_str, format='%Y%m%d')
            
            # 提取百分位數 (若無則填 NaN)
            p_info = p_data.get(date_str, {})
            
            records.append({
                'fetch_time': fetch_time,
                'target_date': target_dt,
                'predicted_max_temp': parse_hko_value(day.get('forecastMaxtemp')),
                'predicted_min_temp': parse_hko_value(day.get('forecastMintemp')),
                'predicted_min_rh': parse_hko_value(day.get('forecastMinrh')),
                'predicted_max_rh': parse_hko_value(day.get('forecastMaxrh')),
                'predicted_wind': day.get('forecastWind', ''),
                'predicted_weather': day.get('forecastWeather', ''),
                'pred_max_p25': p_info.get('p25', np.nan),
                'pred_max_p75': p_info.get('p75', np.nan)
            })
        except Exception as e:
            logging.warning(f"解析單日預報失敗: {e}")
            continue
            
    if not records: return
        
    new_df = pd.DataFrame(records)
    
    if HISTORY_PATH.exists():
        existing_df = pd.read_parquet(HISTORY_PATH)
        existing_df = existing_df[existing_df['fetch_time'] != fetch_time]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_parquet(HISTORY_PATH, index=False)
    logging.info(f"✅ 已更新預報歷史資料庫，當前總紀錄數: {len(combined_df)}")

def build_features_for_date(target_date):
    # [修正] 使用 pd.Timestamp 以支援 normalize()
    target_date = pd.to_datetime(target_date).normalize()
    
    if not HISTORY_PATH.exists():
        logging.error("找不到 live_forecast_history.parquet")
        return None, None
        
    df = pd.read_parquet(HISTORY_PATH)
    df['target_date'] = pd.to_datetime(df['target_date']).dt.normalize()
    df['fetch_time'] = pd.to_datetime(df['fetch_time'])
    
    g = df[df['target_date'] == target_date].sort_values('fetch_time', ascending=False)
    
    if g.empty:
        logging.warning(f"歷史資料庫中無 {target_date.date()} 的預報紀錄。")
        return None, None
        
    latest = g.iloc[0]
    pred_max_latest = latest['predicted_max_temp']
    pred_min_latest = latest['predicted_min_temp']
    
    today = pd.Timestamp(_hkt_now()).normalize()
    lead_days = (target_date - today).days
    lead_days = max(1, min(10, lead_days))
    
    if len(g) >= 2:
        prev1 = g.iloc[1]
        delta_max_1d = pred_max_latest - prev1['predicted_max_temp']
        delta_min_1d = pred_min_latest - prev1['predicted_min_temp']
    else:
        delta_max_1d, delta_min_1d = 0.0, 0.0
        
    target_pub = latest['fetch_time'] - pd.Timedelta(days=3)
    past = g[g['fetch_time'] <= target_pub]
    if not past.empty:
        prev3 = past.iloc[0]
        delta_max_3d = pred_max_latest - prev3['predicted_max_temp']
        delta_min_3d = pred_min_latest - prev3['predicted_min_temp']
    else:
        delta_max_3d, delta_min_3d = 0.0, 0.0
        
    angle, force = parse_wind(latest['predicted_wind'])
    wind_dir_sin = np.sin(np.radians(angle)) if not np.isnan(angle) else 0.0
    wind_dir_cos = np.cos(np.radians(angle)) if not np.isnan(angle) else 0.0
    wind_force = force if not np.isnan(force) else 0.0
    
    kw_flags = extract_keywords(latest['predicted_weather'])
    
    mean_rh = (latest['predicted_min_rh'] + latest['predicted_max_rh']) / 2
    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    raw_features = {
        'pred_max_latest': pred_max_latest, 'pred_min_latest': pred_min_latest,
        'lead_days': lead_days,
        'delta_max_1d': delta_max_1d, 'delta_min_1d': delta_min_1d,
        'delta_max_3d': delta_max_3d, 'delta_min_3d': delta_min_3d,
        'wind_dir_sin': wind_dir_sin, 'wind_dir_cos': wind_dir_cos, 'wind_force': wind_force,
        'mean_rh': mean_rh, 'month_sin': month_sin, 'month_cos': month_cos
    }
    raw_features.update(kw_flags)
    
    # 嚴格對齊 XGBoost 訓練特徵
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        required_features = config['features']
    except Exception:
        required_features = list(raw_features.keys())
        
    final_features = {feat: raw_features.get(feat, 0.0) for feat in required_features}
    
    # 提取百分位數 Spread 作為輔助元數據 (不喂給 XGBoost)
    p25 = latest.get('pred_max_p25', np.nan)
    p75 = latest.get('pred_max_p75', np.nan)
    hko_spread = (p75 - p25) if (pd.notna(p25) and pd.notna(p75)) else None
    
    meta_data = {
        'hko_spread': hko_spread,
        'pred_max_p50': pred_max_latest # 以確定性預報作為 P50 的近似
    }
    
    return final_features, meta_data

if __name__ == "__main__":
    update_forecast_database()
    
    # [修正] 使用 pd.Timestamp
    test_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    feats, meta = build_features_for_date(test_date)
    if feats:
        print(f"\n--- {test_date.date()} 特徵提取結果 ---")
        print(f"XGBoost 特徵數: {len(feats)}")
        print(f"輔助元數據 (Meta): {meta}")