# features/forward_test_logger.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import requests
import json
import re
import io

from features.live_feature_builder import build_features_for_date, update_forecast_database
from models.inference import predict_distribution, predict_bucket_probabilities

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_PATH = Path('data/forward_test_log.parquet')

def fetch_current_observations():
    """獲取 HKO 自午夜起的最高/最低氣溫"""
    max_since_midnight, min_since_midnight = None, None
    try:
        url = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8-sig')))
        max_col = next((c for c in df.columns if 'maximum' in c.lower()), None)
        min_col = next((c for c in df.columns if 'minimum' in c.lower()), None)
        station_col = next((c for c in df.columns if 'station' in c.lower() or 'place' in c.lower()), None)
        if max_col and min_col and station_col:
            for _, row in df.iterrows():
                if 'observatory' in str(row[station_col]).lower() or '天文台' in str(row[station_col]):
                    if pd.notna(row[max_col]): max_since_midnight = float(row[max_col])
                    if pd.notna(row[min_col]): min_since_midnight = float(row[min_col])
                    break
    except Exception as e:
        logging.warning(f"獲取即時觀測失敗: {e}")
    return max_since_midnight, min_since_midnight

def fetch_aws_daily_extreme(target_date_str):
    """獲取 AWS 針對特定日期的最高/最低預測"""
    forecast_max, forecast_min = None, None
    try:
        url = f"https://www.hko.gov.hk/wxinfo/awsgis/forecast/HKO.xml?_t={int(datetime.now().timestamp()*1000)}"
        headers = {'User-Agent': 'Mozilla/5.0', 'Cache-Control': 'no-cache'}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        hourly_max, hourly_min = -99.0, 99.0
        for entry in data.get('HourlyWeatherForecast', []):
            f_hour = str(entry.get('ForecastHour', ''))
            try:
                val = float(entry.get('ForecastTemperature', 0))
                if val < 10.0 or val > 45.0: continue
                if f_hour[:8] == target_date_str:
                    if val > hourly_max: hourly_max = val
                    if val < hourly_min: hourly_min = val
            except: continue
            
        if hourly_max > -99.0: forecast_max = hourly_max
        if hourly_min < 99.0: forecast_min = hourly_min
    except: pass
    return forecast_max, forecast_min

def fetch_polymarket_markets_snapshot(slug: str):
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        if not events: return None
        markets = events[0].get('markets', [])
        parsed = []
        for m in markets:
            title = m.get('groupItemTitle', '')
            match = re.search(r'(\d+)', title)
            if not match: continue
            val = float(match.group(1))
            title_lower = title.lower()
            if any(kw in title_lower for kw in ['below', 'lower', 'under']): lower, upper = -np.inf, val + 1.0
            elif any(kw in title_lower for kw in ['higher', 'above', 'over']): lower, upper = val, np.inf
            else: lower, upper = val, val + 1.0
            try:
                outcomes = json.loads(m.get('outcomes', '[]'))
                prices = json.loads(m.get('outcomePrices', '[]'))
                yes_idx = next((i for i, out in enumerate(outcomes) if out.lower() == 'yes'), 0)
                price_yes = float(prices[yes_idx]) if yes_idx < len(prices) else 0.01
            except: price_yes = 0.01
            parsed.append({'name': title, 'lower': lower, 'upper': upper, 'price_yes': price_yes})
        return parsed
    except: return None

def discover_and_log_all_markets():
    logging.info("=== 啟動雙引擎全市場前向測試掃描 ===")
    update_forecast_database()
    max_since_midnight, min_since_midnight = fetch_current_observations()
    
    all_records = []
    snapshot_time = datetime.now()
    
    for delta in range(4):
        target_date = datetime.now() + timedelta(days=delta)
        target_date_dt = pd.Timestamp(target_date).normalize()
        target_date_str = target_date_dt.strftime("%Y%m%d")
        is_today = (delta == 0)
        
        for market_type in ['highest', 'lowest']:
            is_min_temp = (market_type == 'lowest')
            model_type = 'tmin' if is_min_temp else 'tmax'
            
            month_str = target_date_dt.strftime("%B").lower()
            day_str = str(target_date_dt.day)
            year_str = str(target_date_dt.year)
            slug = f"{market_type}-temperature-in-hong-kong-on-{month_str}-{day_str}-{year_str}"
            
            markets = fetch_polymarket_markets_snapshot(slug)
            if not markets: continue
            
            features, meta = build_features_for_date(target_date_dt)
            if not features: continue
            
            # 1. 9-Day XGBoost 預測
            mean_9d, std_9d = predict_distribution(features, model_type, meta.get('hko_spread') if meta else None, 1.0)
            probs_9d = predict_bucket_probabilities(
                mean_9d, std_9d, markets, 
                max_since_midnight=max_since_midnight if is_today else None,
                min_since_midnight=min_since_midnight if is_today else None,
                is_today=is_today, is_min_temp=is_min_temp
            )
            
            # 2. AWS High-Freq 預測 (使用 AWS 錨點，並收縮 Std)
            aws_max, aws_min = fetch_aws_daily_extreme(target_date_str)
            aws_anchor = aws_min if is_min_temp else aws_max
            
            if aws_anchor is not None:
                mean_aws = aws_anchor
                std_aws = std_9d * 0.8
            else:
                mean_aws, std_aws = mean_9d, std_9d # Fallback
                
            probs_aws = predict_bucket_probabilities(
                mean_aws, std_aws, markets,
                max_since_midnight=max_since_midnight if is_today else None,
                min_since_midnight=min_since_midnight if is_today else None,
                is_today=is_today, is_min_temp=is_min_temp
            )
            
            # 組裝紀錄 (分別寫入 9d 與 aws)
            for m in markets:
                name = m['name']
                p_mkt = m['price_yes']
                
                # 寫入 9D 紀錄
                p_mod_9d = probs_9d.get(name, 0.0)
                all_records.append({
                    'snapshot_time': snapshot_time, 'target_date': target_date_dt,
                    'market_type': market_type, 'model_version': '9d',
                    'bucket_name': name, 'lower_bound': m['lower'], 'upper_bound': m['upper'],
                    'model_prob': p_mod_9d, 'market_price': p_mkt, 'edge': p_mod_9d - p_mkt,
                    'actual_outcome': np.nan
                })
                
                # 寫入 AWS 紀錄
                p_mod_aws = probs_aws.get(name, 0.0)
                all_records.append({
                    'snapshot_time': snapshot_time, 'target_date': target_date_dt,
                    'market_type': market_type, 'model_version': 'aws',
                    'bucket_name': name, 'lower_bound': m['lower'], 'upper_bound': m['upper'],
                    'model_prob': p_mod_aws, 'market_price': p_mkt, 'edge': p_mod_aws - p_mkt,
                    'actual_outcome': np.nan
                })

    if not all_records: return

    new_df = pd.DataFrame(all_records)
    if LOG_PATH.exists():
        existing_df = pd.read_parquet(LOG_PATH)
        existing_df = existing_df[existing_df['snapshot_time'] != snapshot_time]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(LOG_PATH, index=False)
    logging.info(f"✅ 成功記錄 {len(all_records)} 筆雙引擎快照至 {LOG_PATH}")

if __name__ == "__main__":
    discover_and_log_all_markets()