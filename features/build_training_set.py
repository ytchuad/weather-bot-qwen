# features/build_training_set.py
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FCST_PATH = Path('data/hko_historical_forecasts.parquet')
OBS_PATH = Path('data/hko_tmax_historical.parquet')
OUT_MAX_PATH = Path('data/training_set_max.parquet')
OUT_MIN_PATH = Path('data/training_set_min.parquet')

# 16 方位角度映射
DIR_ANGLES = {
    '北': 0, '東北偏北': 22.5, '東北': 45, '東北偏東': 67.5,
    '東': 90, '東南偏東': 112.5, '東南': 135, '東南偏南': 157.5,
    '南': 180, '西南偏南': 202.5, '西南': 225, '西南偏西': 247.5,
    '西': 270, '西北偏西': 292.5, '西北': 315, '西北偏北': 337.5
}

WEATHER_KEYWORDS = ['雨', '雷暴', '驟雨', '多雲', '天晴', '酷熱', '寒冷', '潮濕', '霧', '煙霞']

def parse_wind(text):
    """解析風向與風力"""
    if not isinstance(text, str):
        return np.nan, np.nan
    
    # 1. 提取風力 (數字)
    nums = re.findall(r'\d+', text)
    force = np.mean([float(n) for n in nums]) if nums else np.nan
    
    # 2. 提取風向 (移除風力字眼)
    dir_text = re.sub(r'\d+至\d+級|\d+級|級|風', '', text).strip()
    angle = np.nan
    
    if '至' in dir_text:
        parts = dir_text.split('至')
        angles = []
        for p in parts:
            for k, v in DIR_ANGLES.items():
                if k in p:
                    angles.append(v)
                    break
        if len(angles) == 2:
            angle = np.mean(angles) # 簡化處理：取平均角度
        elif len(angles) == 1:
            angle = angles[0]
    else:
        # 優先匹配長詞 (如 東北偏北)，再匹配短詞
        for k, v in sorted(DIR_ANGLES.items(), key=lambda x: len(x[0]), reverse=True):
            if k in dir_text:
                angle = v
                break
                
    return angle, force

def extract_keywords(text):
    """提取天氣關鍵字 Flags"""
    flags = {}
    if not isinstance(text, str): 
        text = ""
    for kw in WEATHER_KEYWORDS:
        flags[f'kw_{kw}'] = 1 if kw in text else 0
    return flags

def main():
    logging.info("載入預報與觀測數據...")
    fcst_df = pd.read_parquet(FCST_PATH)
    obs_df = pd.read_parquet(OBS_PATH)
    
    # 轉換日期格式
    fcst_df['target_date'] = pd.to_datetime(fcst_df['target_date'])
    fcst_df['publish_date'] = pd.to_datetime(fcst_df['publish_date'])
    obs_df['date'] = pd.to_datetime(obs_df['date'])
    
    # 合併數據
    merged = pd.merge(fcst_df, obs_df, left_on='target_date', right_on='date', how='inner')
    merged['lead_days'] = (merged['target_date'] - merged['publish_date']).dt.days
    
    # 過濾 Lead Days (1 到 10 天)
    merged = merged[(merged['lead_days'] >= 1) & (merged['lead_days'] <= 10)].copy()
    logging.info(f"合併後並過濾 Lead Days 1-10 的紀錄數: {len(merged)}")
    
    # 確保按日期排序
    merged = merged.sort_values(['target_date', 'publish_date'])
    
    logging.info("提取時間序列 Delta 特徵與最新預報...")
    records = []
    
    for target_date, g in merged.groupby('target_date'):
        latest = g.iloc[-1]
        
        pred_max_latest = latest['predicted_max_temp']
        pred_min_latest = latest['predicted_min_temp']
        lead_days = latest['lead_days']
        
        # Delta 1d (與前一次預報相比)
        if len(g) >= 2:
            prev1 = g.iloc[-2]
            delta_max_1d = pred_max_latest - prev1['predicted_max_temp']
            delta_min_1d = pred_min_latest - prev1['predicted_min_temp']
        else:
            delta_max_1d, delta_min_1d = 0.0, 0.0
            
        # Delta 3d (與約 3 天前的預報相比)
        target_pub = latest['publish_date'] - pd.Timedelta(days=3)
        past = g[g['publish_date'] <= target_pub]
        if not past.empty:
            prev3 = past.iloc[-1]
            delta_max_3d = pred_max_latest - prev3['predicted_max_temp']
            delta_min_3d = pred_min_latest - prev3['predicted_min_temp']
        else:
            delta_max_3d, delta_min_3d = 0.0, 0.0
            
        records.append({
            'target_date': latest['target_date'],
            'tmax': latest['tmax'],
            'tmin': latest['tmin'],
            'pred_max_latest': pred_max_latest,
            'pred_min_latest': pred_min_latest,
            'lead_days': lead_days,
            'delta_max_1d': delta_max_1d,
            'delta_min_1d': delta_min_1d,
            'delta_max_3d': delta_max_3d,
            'delta_min_3d': delta_min_3d,
            'predicted_wind': latest['predicted_wind'],
            'predicted_weather': latest['predicted_weather'],
            'predicted_min_rh': latest['predicted_min_rh'],
            'predicted_max_rh': latest['predicted_max_rh']
        })
        
    df = pd.DataFrame(records)
    
    logging.info("生成季節性、風向、天氣關鍵字與濕度特徵...")
    # 季節性
    month = df['target_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # 風向與風力
    wind_parsed = df['predicted_wind'].apply(parse_wind)
    df['wind_angle'] = wind_parsed.apply(lambda x: x[0])
    df['wind_force'] = wind_parsed.apply(lambda x: x[1])
    df['wind_dir_sin'] = np.sin(np.radians(df['wind_angle']))
    df['wind_dir_cos'] = np.cos(np.radians(df['wind_angle']))
    
    # 天氣關鍵字
    kw_df = df['predicted_weather'].apply(extract_keywords).apply(pd.Series)
    df = pd.concat([df, kw_df], axis=1)
    
    # 濕度
    df['mean_rh'] = (df['predicted_min_rh'] + df['predicted_max_rh']) / 2
    
    # 清理中間欄位
    cols_to_drop = ['predicted_wind', 'predicted_weather', 'wind_angle', 'predicted_min_rh', 'predicted_max_rh']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 分離 Max 與 Min 數據集並處理缺失值
    logging.info("清理缺失值並儲存最終訓練集...")
    
    # Max Temp 訓練集
    df_max = df.dropna(subset=['tmax', 'pred_max_latest']).copy()
    df_max = df_max.dropna() # 嚴格丟棄任何包含 NaN 的行
    df_max.to_parquet(OUT_MAX_PATH, index=False)
    
    # Min Temp 訓練集
    df_min = df.dropna(subset=['tmin', 'pred_min_latest']).copy()
    df_min = df_min.dropna()
    df_min.to_parquet(OUT_MIN_PATH, index=False)
    
    # 列印統計資訊
    logging.info(f"✅ 最高溫訓練集已儲存至 {OUT_MAX_PATH} (樣本數: {len(df_max)})")
    logging.info(f"✅ 最低溫訓練集已儲存至 {OUT_MIN_PATH} (樣本數: {len(df_min)})")
    
    logging.info("\n--- 最高溫 (tmax) 目標變數統計 ---")
    print(df_max['tmax'].describe())
    logging.info(f"日期範圍: {df_max['target_date'].min().date()} 至 {df_max['target_date'].max().date()}")
    
    logging.info("\n--- 最低溫 (tmin) 目標變數統計 ---")
    print(df_min['tmin'].describe())

if __name__ == "__main__":
    main()