# execution/live_bot.py
import sys
import os
import time
import logging
import yaml
import json
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
import re
# from scipy.stats import norm
import xarray as xr
from scipy.stats import t  # 改用 t 分佈處理極端高溫尾部風險
import io
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 將專案根目錄加入 sys.path 以便從根目錄執行時能正確導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from execution.kelly_betting import calculate_kelly_bets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """載入 config.yaml"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fetch_polymarket_prices_mock(config, calibrated_probs):
    """模擬 Polymarket 價格 (基於模型機率製造約 8% 價差供測試)"""
    logging.info("📡 正在使用模擬價格模式...")
    prices = {}
    for b, p in calibrated_probs.items():
        prices[b] = round(max(0.01, min(0.99, p * 0.92)), 2)
    return prices

def generate_polymarket_slug():
    """動態生成 Slug"""
    today = datetime.now()
    return f"highest-temperature-in-hong-kong-on-{today.strftime('%B').lower()}-{today.day}-{today.year}"

def parse_bucket_bounds(title):
    """
    [動態解析] 從 Polymarket 標題解析數學邊界
    "24°C or below" -> (-inf, 25.0)
    "25°C"          -> (25.0, 26.0)
    "34°C or higher"-> (34.0, inf)
    """
    title = title.strip()
    title_lower = title.lower()
    match = re.search(r'(\d+)', title)
    if not match: return None, None, None
    val = float(match.group(1))
    
    if any(kw in title_lower for kw in ['below', 'lower', 'under', 'less']):
        return title, -np.inf, val + 1.0
    elif any(kw in title_lower for kw in ['higher', 'above', 'over', 'greater', 'more']):
        return title, val, np.inf
    else:
        return title, val, val + 1.0

def fetch_polymarket_markets(config):
    """
    [API 驅動] 獲取市場結構、動態邊界與真實價格
    """
    slug = generate_polymarket_slug()
    logging.info(f"🔍 正在獲取 Polymarket 市場結構 (Slug: {slug})...")
    
    gamma_url = "https://gamma-api.polymarket.com/events"
    headers = {'User-Agent': 'WeatherBot/1.0'}
    
    try:
        resp = requests.get(gamma_url, params={'slug': slug}, headers=headers, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        if not events: raise ValueError(f"找不到 Event: {slug}")
            
        markets_data = events[0].get('markets', [])
        parsed_markets = []
        
        for m in markets_data:
            title = m.get('groupItemTitle', '')
            if not title: continue
            
            # 1. 動態解析邊界
            name, lower, upper = parse_bucket_bounds(title)
            if name is None: continue
            
            # 2. 解析真實價格
            price = 0.01
            try:
                outcomes = json.loads(m.get('outcomes', '[]'))
                outcome_prices = json.loads(m.get('outcomePrices', '[]'))
                yes_idx = next((i for i, out in enumerate(outcomes) if out.lower() == 'yes'), 0)
                if yes_idx < len(outcome_prices):
                    price = max(0.0001, min(0.9999, float(outcome_prices[yes_idx])))
            except Exception: pass
                
            parsed_markets.append({'name': name, 'lower': lower, 'upper': upper, 'price': price})
            
        if not parsed_markets: raise ValueError("未能解析任何市場桶")
            
        # 按溫度下界排序，確保日誌輸出美觀
        parsed_markets.sort(key=lambda x: x['lower'])
        logging.info(f"✅ 成功從 API 解析 {len(parsed_markets)} 個動態市場桶")
        return parsed_markets
        
    except Exception as e:
        logging.error(f"❌ 獲取市場結構失敗: {e}")
        return None

def fetch_hko_max_since_midnight():
    """
    [精準風控] 獲取「天文台總部」自午夜起最高氣溫
    包含詳細 Debug 日誌與整數 API 降級備案
    """
    url = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv"
    
    try:
        logging.debug(f"🔍 正在下載 CSV: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        # 1. 印出原始內容前 300 字元 (幫助確認是否為 HTML 錯誤頁面或格式變動)
        raw_text = resp.content.decode('utf-8-sig', errors='replace')
        logging.debug(f"📄 CSV 原始內容前 300 字元:\n{raw_text[:300]}")
        
        df = pd.read_csv(io.StringIO(raw_text))
        logging.debug(f"📊 CSV 解析出的欄位名稱: {df.columns.tolist()}")
        
        # 2. 動態匹配欄位
        max_col = next((c for c in df.columns if 'maximum' in c.lower() or '最高' in c), None)
        station_col = next((c for c in df.columns if 'station' in c.lower() or '站' in c.lower() or 'place' in c.lower()), None)
        
        logging.debug(f"🎯 匹配結果 -> max_col: '{max_col}', station_col: '{station_col}'")
        
        if not max_col or not station_col:
            logging.error("❌ [CSV 失敗] 無法找到 'maximum/最高' 或 'station/站' 欄位！")
            raise ValueError("Column mapping failed")
            
        # 3. 印出所有站點名稱 (幫助找出 HKO 的正確關鍵字)
        all_stations = df[station_col].unique().tolist()
        logging.debug(f"🏢 CSV 中包含的所有站點: {all_stations}")
            
        # 4. 尋找 HKO 總部
        target_keywords = ['observatory', '天文台']
        for _, row in df.iterrows():
            station = str(row[station_col]).lower()
            if any(kw in station for kw in target_keywords):
                temp = row[max_col]
                if pd.notna(temp):
                    temp = float(temp)
                    logging.info(f"🌡️ [成功] HKO 總部自午夜最高氣溫: {temp}°C (精確截斷)")
                    return temp
                    
        logging.error(f"❌ [CSV 失敗] 遍歷所有站點後，未找到包含 {target_keywords} 的紀錄。")
        raise ValueError("Station not found")
        
    except Exception as e:
        logging.warning(f"⚠️ CSV 獲取或解析失敗 ({type(e).__name__})，啟動降級備案...")
        
        # === 降級備案：使用整數 API 進行「保守截斷」 ===
        try:
            fallback_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
            logging.debug(f"🔄 正在呼叫降級 API: {fallback_url}")
            resp = requests.get(fallback_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            for record in data.get('temperature', {}).get('data', []):
                place = record.get('place', '').lower()
                if 'observatory' in place:
                    temp_int = float(record['value'])
                    # 保守策略：因為不知道小數點，假設最低可能是 .0
                    # 例如當前 30°C，我們只剔除上限 <= 29.9 的桶 (即 29°C 及以下)
                    # 這樣就不會誤殺 30.0~30.9 的可能性
                    conservative_temp = temp_int - 0.1 
                    logging.info(f"🌡️ [降級成功] HKO 即時整數氣溫: {temp_int}°C -> 保守截斷點: {conservative_temp}°C")
                    return conservative_temp
                    
            logging.error("❌ [降級失敗] rhrread API 中找不到 Observatory 站點。")
        except Exception as fallback_e:
            logging.error(f"❌ [降級失敗] rhrread API 請求異常: {fallback_e}")
            
    return None

def fetch_hko_aws_forecast():
    """
    [核心錨點] 從 HKO AWS 獲取今日「小時預報」的最高精確值
    完美還原 Google Apps Script 的成功邏輯：
    1. 使用 .xml 端點但解析為 JSON
    2. 加入防快取時間戳與 User-Agent
    """
    import time
    # 加入毫秒級時間戳避免快取，並偽裝 User-Agent 避免被 HKO 防火牆阻擋
    url = f"https://www.hko.gov.hk/wxinfo/awsgis/forecast/HKO.xml?_t={int(time.time() * 1000)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Cache-Control': 'no-cache'
    }
    today_str = datetime.now().strftime("%Y%m%d")
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        
        # 關鍵：雖然是 .xml 結尾，但內容實際上是 JSON (對應 GAS 的 JSON.parse)
        data = resp.json() 
        
        # 1. 優先從 HourlyWeatherForecast 提取今日最高精確值
        hourly_max = -99.0
        for entry in data.get('HourlyWeatherForecast', []):
            ft = str(entry.get('ForecastHour', ''))
            if len(ft) >= 8 and ft[:8] == today_str:
                try:
                    val = float(entry.get('ForecastTemperature', 0))
                    if val > hourly_max:
                        hourly_max = val
                except ValueError:
                    continue
                    
        if hourly_max > -99.0:
            logging.info(f"🎯 HKO AWS 小時預報今日最高: {hourly_max}°C")
            return hourly_max
            
    except Exception as e:
        logging.warning(f"⚠️ 獲取 HKO AWS 預報失敗: {e}")
        
    return None
def compute_real_ensemble_probabilities(config, markets, max_since_midnight):
    """計算真實機率 (移除人為微調，完全依賴數據)"""
    nc_path = Path(config['paths']['ecmwf_daily_tmax'])
    if not nc_path.exists(): raise FileNotFoundError(f"找不到 NWP 資料: {nc_path}")
        
    ds = xr.open_dataset(nc_path)
    if 0 not in ds['lead_day'].values: raise ValueError("Lead Day 0 不可用")
        
    members = ds.sel(lead_day=0)['tmax_daily'].values.flatten()
    
    hko_forecast_max = fetch_hko_aws_forecast()
    
    if hko_forecast_max is not None:
        # 修正：直接使用 HKO 精確預報，不再進行人為 +0.2 微調
        mean = hko_forecast_max
        logging.info(f"🧠 策略: 採用 HKO AWS 精確預報為錨點 (Mean={mean:.1f}°C)")
    else:
        mean = np.mean(members) + 3.5
        logging.info(f"⚠️ 降級: 使用 ECMWF (Mean={mean:.1f}°C)")

    std = max(np.std(members), 0.8) 
    logging.info(f"📈 最終模型: Mean={mean:.2f}°C, Std={std:.2f}°C")
    
    df = 3 
    poly_probs = {}
    eliminated_buckets = []
    
    for m in markets:
        # 核心風控：嚴格截斷
        if max_since_midnight is not None and m['upper'] <= max_since_midnight:
            poly_probs[m['name']] = 0.0
            eliminated_buckets.append(m['name'])
            continue
            
        z_upper = (m['upper'] - mean) / std if m['upper'] != np.inf else np.inf
        z_lower = (m['lower'] - mean) / std if m['lower'] != -np.inf else -np.inf
        
        cdf_upper = t.cdf(z_upper, df) if z_upper != np.inf else 1.0
        cdf_lower = t.cdf(z_lower, df) if z_lower != -np.inf else 0.0
        
        prob = max(0.0, cdf_upper - cdf_lower) # 允許 0.0
        poly_probs[m['name']] = prob
        
    if eliminated_buckets:
        logging.info(f"🚫 已基於 HKO 總部氣溫 ({max_since_midnight}°C) 剔除: {eliminated_buckets}")
        
    total = sum(poly_probs.values())
    if total > 0:
        poly_probs = {k: v/total for k, v in poly_probs.items()}
    else:
        max_key = max(poly_probs.keys(), key=lambda x: markets[[m['name'] for m in markets].index(x)]['lower'])
        poly_probs = {k: (1.0 if k == max_key else 0.0) for k in poly_probs}
        
    return poly_probs

def run_bot_loop(config):
    """主執行循環 (確保風控截斷確實執行)"""
    logging.info("=== 天氣預測交易機器人啟動 (終極實戰版) ===")
    
    cycle_count = 0
    sleep_interval = 3600

    while True:
        cycle_count += 1
        logging.info(f"\n--- 開始第 {cycle_count} 次執行循環 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
        try:
            # === 關鍵修正：確保這裡有被執行，否則無法剔除低溫桶 ===
            logging.info("1. 獲取 HKO 總部即時氣溫 (風控截斷)...")
            max_since_midnight = fetch_hko_max_since_midnight()
            if max_since_midnight is None:
                logging.warning("⚠️ 無法獲取即時氣溫，風控截斷將暫停。")

            # 2. 獲取市場結構
            logging.info("2. 獲取 Polymarket 市場結構...")
            markets = fetch_polymarket_markets(config)
            if not markets:
                time.sleep(sleep_interval)
                continue
                
            market_prices = {m['name']: m['price'] for m in markets}
            logging.info(f"💰 市場真實價格: {market_prices}")

            # 3. 計算機率 (傳入 max_since_midnight 進行截斷)
            logging.info("3. 計算 ECMWF 集合預報機率 (含小數點精度截斷)...")
            cal_probs = compute_real_ensemble_probabilities(config, markets, max_since_midnight)
            prob_str = ", ".join([f"{k}:{v:.3f}" for k, v in cal_probs.items() if v > 0.001])
            logging.info(f"📊 最終交易機率: [{prob_str}]")

            # 4. 凱利計算
            logging.info("4. 計算最佳下注金額...")
            capital = config['betting']['initial_capital']
            bets = calculate_kelly_bets(
                calibrated_probs=cal_probs,
                market_prices=market_prices,
                total_capital=capital,
                half_kelly=(config['betting']['kelly_fraction'] == 0.5),
                max_bucket_exposure=config['betting']['max_bucket_exposure']
            )

            # 5. 輸出訂單 (過濾噪音)
            logging.info("5. 📋 訂單建議:")
            active_bets = {b: amt for b, amt in bets.items() if amt >= 1.0 and cal_probs.get(b, 0) > 0.005}
            
            if not active_bets:
                logging.info("   💤 無具顯著正邊緣之選項，建議保持觀望。")
            else:
                for b, amt in sorted(active_bets.items(), key=lambda x: x[1], reverse=True):
                    edge = cal_probs.get(b, 0) - market_prices.get(b, 0)
                    logging.info(f"   🎯 桶 {b}: 買入 ${amt:.2f} "
                                 f"(機率: {cal_probs[b]:.3f}, 價格: {market_prices[b]:.2f}, 邊緣: {edge:+.3f})")

        except Exception as e:
            logging.error(f"❌ 循環執行發生未預期錯誤: {e}", exc_info=True)

        logging.info(f"⏳ 等待 {sleep_interval//60} 分鐘後執行下一次循環... (Ctrl+C 可手動停止)")
        time.sleep(sleep_interval)
if __name__ == "__main__":
    config = load_config()
    try:
        run_bot_loop(config)
    except KeyboardInterrupt:
        logging.info("\n🛑 機器人已手動停止。")