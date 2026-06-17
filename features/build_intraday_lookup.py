# features/build_intraday_lookup.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INTRADAY_PATH = Path('data/intraday_hko_10min.parquet')
DAILY_PATH = Path('data/hko_tmax_historical.parquet')
OUT_UPSIDE = Path('data/lookup_upside.parquet')
OUT_DOWNSIDE = Path('data/lookup_downside.parquet')

def categorize_trend(val):
    if pd.isna(val): return np.nan
    if val > 0.3: return 'rising'
    if val < -0.3: return 'falling'
    return 'flat'

def build_lookup_cube(df_in, bucket_col, target_col, zero_col):
    """構建包含所有回退層級 (強制保留 hour 維度) 的聚合 Cube"""
    # [關鍵修復] 永遠不丟棄 hour，依序丟棄 trend -> bucket -> month
    levels = [
        ['month', 'hour', bucket_col, 'trend_60min'],
        ['month', 'hour', bucket_col],
        ['month', 'hour'],
        ['hour']
    ]
    res = []
    for lvl in levels:
        g = df_in.groupby(lvl).agg(
            count=(target_col, 'count'),
            p10=(target_col, lambda x: x.quantile(0.1)),
            p25=(target_col, lambda x: x.quantile(0.25)),
            p50=(target_col, 'median'),
            p75=(target_col, lambda x: x.quantile(0.75)),
            p90=(target_col, lambda x: x.quantile(0.9)),
            prob_zero=(zero_col, 'mean')
        ).reset_index()
        
        # 補齊缺失的維度為 'ALL'
        for col in ['month', 'hour', bucket_col, 'trend_60min']:
            if col not in g.columns:
                g[col] = 'ALL'
        
        res.append(g)
        
    cube = pd.concat(res).drop_duplicates()
    
    # 將所有維度欄位統一轉為字串，避免 PyArrow 寫入 Parquet 時崩潰
    for col in ['month', 'hour', bucket_col, 'trend_60min']:
        cube[col] = cube[col].astype(str)
        
    return cube

def main():
    logging.info("🚀 開始構建日內臨近預報經驗查詢表...")
    
    if not INTRADAY_PATH.exists() or not DAILY_PATH.exists():
        logging.error("❌ 找不到 intraday 或 daily 歷史數據 Parquet 檔。")
        return
        
    df = pd.read_parquet(INTRADAY_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    
    # 1. 合併每日官方極值
    df['date'] = df['datetime'].dt.date
    daily['date'] = pd.to_datetime(daily['date']).dt.date
    df = df.merge(daily[['date', 'tmax', 'tmin']], on='date', how='left')
    df = df.dropna(subset=['tmax', 'tmin'])
    
    # 2. 計算日內累積極值與趨勢
    df = df.sort_values('datetime')
    
    # 日內極值必須按天隔離 (不能繼承昨天的極值)
    df['max_so_far'] = df.groupby('date')['temp'].cummax()
    df['min_so_far'] = df.groupby('date')['temp'].cummin()
    
    # 60 分鐘前的溫度 (6 個 10 分鐘區間) - 填補 NaN 使凌晨可用
    df['temp_60m_ago'] = df.groupby('date')['temp'].shift(6)
    df['temp_60m_ago'] = df['temp_60m_ago'].fillna(df['temp'])
    df['temp_change_60min'] = df['temp'] - df['temp_60m_ago']
    df['trend_60min'] = df['temp_change_60min'].apply(categorize_trend)
    # 趨勢分類不再需要 dropna，因為已經沒有 NaN
    
    # 丟棄每天最開始的 6 個無效點 (因為它們連前一天的數據都沒有，屬於真正的數據起點)
    df = df.dropna(subset=['trend_60min']) 
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # 3. 計算剩餘空間與是否已觸頂/底
    df['remaining_upside'] = (df['tmax'] - df['max_so_far']).clip(lower=0)
    df['remaining_downside'] = (df['min_so_far'] - df['tmin']).clip(lower=0)
    
    df['is_upside_zero'] = (df['remaining_upside'] <= 0.1).astype(int)
    df['is_downside_zero'] = (df['remaining_downside'] <= 0.1).astype(int)
    
    # 4. 分桶 (0.5°C 寬度)
    df['max_bucket'] = (df['max_so_far'] // 0.5) * 0.5
    df['min_bucket'] = (df['min_so_far'] // 0.5) * 0.5
    
    # 5. 構建 Cube
    logging.info("⏳ 聚合 Upside (最高溫) 經驗分佈...")
    upside_cube = build_lookup_cube(df, 'max_bucket', 'remaining_upside', 'is_upside_zero')
    
    logging.info("⏳ 聚合 Downside (最低溫) 經驗分佈...")
    downside_cube = build_lookup_cube(df, 'min_bucket', 'remaining_downside', 'is_downside_zero')
    
    # 6. 儲存
    upside_cube.to_parquet(OUT_UPSIDE, index=False)
    downside_cube.to_parquet(OUT_DOWNSIDE, index=False)
    
    logging.info(f"✅ 成功儲存 Upside 查詢表 ({len(upside_cube)} 行) 至 {OUT_UPSIDE}")
    logging.info(f"✅ 成功儲存 Downside 查詢表 ({len(downside_cube)} 行) 至 {OUT_DOWNSIDE}")
    
    # 統計摘要
    base_lvl = upside_cube[upside_cube['trend_60min'] != 'ALL']
    logging.info(f"📊 最細粒度群組數: {len(base_lvl)}, 平均樣本數: {base_lvl['count'].mean():.1f}")

if __name__ == "__main__":
    main()