# data/build_hko_obs.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """載入 config.yaml 設定檔"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_hko_csv(filepath):
    """
    解析 HKO 歷史 CSV 檔案。
    跳過前兩行說明，使用第三行作為欄位名稱。
    """
    # 讀取第一行來判斷溫度指標 (使用 utf-8-sig 處理 BOM 字元)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline().lower()
    
    if '最高' in first_line or 'max' in first_line:
        metric = 'tmax'
    elif '最低' in first_line or 'min' in first_line:
        metric = 'tmin'
    elif '平均' in first_line or 'mean' in first_line or 'temp' in first_line:
        metric = 'tmean'
    else:
        raise ValueError(f"無法從 {filepath} 的第一行判斷溫度類型: {first_line}")
        
    # 讀取實際資料，使用 header=2 (第三行) 作為欄位名稱
    df = pd.read_csv(filepath, header=2)
    
    # 標準化欄位名稱 (處理 '年/Year' 這類中英文混雜的欄位)
    year_col = [c for c in df.columns if '年' in str(c) or 'year' in str(c).lower()][0]
    month_col = [c for c in df.columns if '月' in str(c) or 'month' in str(c).lower()][0]
    day_col = [c for c in df.columns if '日' in str(c) or 'day' in str(c).lower()][0]
    
    # 尋找數值欄位 (通常是 '數值/Value')
    val_col_candidates = [c for c in df.columns if '值' in str(c) or 'value' in str(c).lower()]
    val_col = val_col_candidates[0] if val_col_candidates else df.columns[3]

    # 提取需要的欄位
    temp_df = pd.DataFrame({
        'year': df[year_col],
        'month': df[month_col],
        'day': df[day_col],
        'value': df[val_col]
    })
    
    # 清理資料：轉換為數值，強制將 '***' 等無法轉換的字串設為 NaN
    temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce')
    temp_df = temp_df[temp_df['value'].notna()]
    temp_df = temp_df[temp_df['value'] > -100] # 移除 -999 等無效缺測值
    
    # 轉換為標準的 datetime 格式
    temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']], errors='coerce')
    temp_df = temp_df.dropna(subset=['date'])
    
    return temp_df[['date', 'value']].rename(columns={'value': metric})

def main():
    config = load_config()
    raw_dir = Path(config['paths']['raw_csv_dir'])
    out_path = Path(config['paths']['historical_obs'])
    
    # 尋找 raw 資料夾下所有的 CSV
    csv_files = list(raw_dir.glob('*.csv'))
    if not csv_files:
        logging.error(f"在 {raw_dir} 中找不到 CSV 檔案。請先下載並放入該資料夾。")
        return

    dfs = []
    for f in csv_files:
        logging.info(f"正在解析 {f.name}...")
        dfs.append(parse_hko_csv(f))
        
    # 合併 DataFrames
    logging.info("正在合併資料...")
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = pd.merge(combined_df, df, on='date', how='outer')
        
    # 排序並去除重複日期
    combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'])
    
    # 確保所有目標欄位都存在
    for col in ['tmax', 'tmin', 'tmean']:
        if col not in combined_df.columns:
            combined_df[col] = np.nan
            
    combined_df = combined_df[['date', 'tmax', 'tmin', 'tmean']]
    
    # 儲存為 Parquet 格式
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(out_path, index=False)
    logging.info(f"成功儲存歷史觀測資料至 {out_path}")
    
    # 輸出基本統計資訊
    logging.info(f"資料日期範圍: {combined_df['date'].min().date()} 至 {combined_df['date'].max().date()}")
    logging.info("基本統計數據 (°C):")
    print(combined_df[['tmax', 'tmin', 'tmean']].describe())

if __name__ == "__main__":
    main()