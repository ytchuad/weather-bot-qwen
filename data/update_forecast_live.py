# data/update_forecast_live.py
"""從 HKO API 抓取最新 9 天預報，更新 live_forecast_history.parquet"""
import pandas as pd
import requests
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIVE_FORECAST_PATH = Path('data/live_forecast_history.parquet')
API_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en"

def main():
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        records = []
        fetch_time = datetime.now()
        for day in data.get('weatherForecast', []):
            records.append({
                'fetch_time': fetch_time,
                'target_date': day.get('forecastDate'),
                'predicted_max_temp': float(day.get('forecastMaxtemp', {}).get('value', 0)),
                'predicted_min_temp': float(day.get('forecastMintemp', {}).get('value', 0)),
                'predicted_min_rh': float(day.get('forecastMinrh', {}).get('value', 0)),
                'predicted_max_rh': float(day.get('forecastMaxrh', {}).get('value', 0)),
                'predicted_wind': day.get('forecastWind', ''),
                'predicted_weather': day.get('forecastWeather', '')
            })
        df_new = pd.DataFrame(records)
        if LIVE_FORECAST_PATH.exists():
            df_old = pd.read_parquet(LIVE_FORECAST_PATH)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['fetch_time', 'target_date'], keep='last')
        else:
            df_combined = df_new
        df_combined.to_parquet(LIVE_FORECAST_PATH, index=False)
        logger.info(f"已更新預報，共 {len(df_new)} 天，總筆數: {len(df_combined)}")
    except Exception as e:
        logger.error(f"更新預報失敗: {e}")

if __name__ == '__main__':
    main()