# data/build_intraday_obs.py
import zipfile
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOWNLOAD_DIR = Path(r"c:\Users\cyt\Downloads")
OUT_PATH = Path("data/intraday_hko_10min.parquet")
CUTOFF_DATE = pd.Timestamp("2021-12-30")

def process_zip(zip_path):
    """解析單個 Zip 檔案，具備自動分隔符嗅探與模糊匹配能力"""
    records = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.csv') and not file_info.is_dir():
                    try:
                        with z.open(file_info) as f:
                            # [修復 1] 使用 utf-8-sig 處理 BOM，使用 sep=None 自動嗅探分隔符 (逗號或 Tab)
                            df = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig')
                            
                            # 模糊匹配列名 (全部轉小寫並去除空白)
                            cols = {c.strip().lower(): c for c in df.columns}
                            
                            dt_col = next((cols[c] for c in cols if 'date' in c and 'time' in c), None)
                            if not dt_col:
                                dt_col = next((cols[c] for c in cols if 'date' in c), None)
                                
                            station_col = next((cols[c] for c in cols if 'station' in c or 'place' in c), None)
                            temp_col = next((cols[c] for c in cols if 'temp' in c), None)
                            
                            if not all([dt_col, station_col, temp_col]):
                                continue
                                
                            # [修復 2] 放寬站名匹配 (包含 Observatory 或 天文台，忽略大小寫)
                            mask = df[station_col].astype(str).str.contains('Observatory|天文台', case=False, na=False)
                            df_hko = df[mask].copy()
                            
                            if df_hko.empty:
                                continue
                                
                            # [修復 3] 時間解析雙重保險
                            df_hko['datetime'] = pd.to_datetime(df_hko[dt_col].astype(str), format='%Y%m%d%H%M', errors='coerce')
                            # 若全數解析失敗，嘗試讓 Pandas 自動推斷格式
                            if df_hko['datetime'].isna().all():
                                df_hko['datetime'] = pd.to_datetime(df_hko[dt_col], errors='coerce')
                                
                            df_hko['temp'] = pd.to_numeric(df_hko[temp_col], errors='coerce')
                            df_hko = df_hko.dropna(subset=['datetime', 'temp'])
                            
                            if not df_hko.empty:
                                records.append(df_hko[['datetime', 'temp']])
                    except Exception as e:
                        logging.debug(f"讀取 {file_info.filename} 失敗: {e}")
    except zipfile.BadZipFile:
        logging.warning(f"損壞的 Zip 檔案: {zip_path.name}")
    except Exception as e:
        logging.warning(f"處理 {zip_path.name} 時發生錯誤: {e}")
        
    return records

def main():
    logging.info("🚀 開始提取與清洗 HKO 分鐘級溫度數據 (終極魯棒版)...")
    
    # 1. 尋找目標 Zip 檔案 (自動適應檔名結尾)
    zip_files = sorted([f for f in DOWNLOAD_DIR.glob("c2c072ddf3c38c79*.zip")])
    
    if not zip_files:
        logging.warning(f"找不到特定前綴的 Zip，嘗試讀取 {DOWNLOAD_DIR} 中的所有 Zip...")
        zip_files = sorted(list(DOWNLOAD_DIR.glob("*.zip")))
        
    logging.info(f"找到 {len(zip_files)} 個 Zip 檔案待處理。")
    if zip_files:
        logging.info(f"第一個檔案: {zip_files[0].name}")
        logging.info(f"最後一個檔案: {zip_files[-1].name}")
    
    # 2. 遍歷並提取
    all_records = []
    for i, zp in enumerate(zip_files):
        logging.info(f"處理中 [{i+1}/{len(zip_files)}]: {zp.name}")
        recs = process_zip(zp)
        all_records.extend(recs)
        
    if not all_records:
        logging.error("❌ 未提取到任何數據。請確認下載資料夾路徑是否正確，或 CSV 結構是否發生重大變更。")
        return
        
    logging.info("📦 合併所有原始紀錄...")
    df = pd.concat(all_records, ignore_index=True)
    logging.info(f"原始紀錄總數: {len(df):,}")
    
    # 3. 過濾 2021-12-30 之前的髒數據
    df = df[df['datetime'] >= CUTOFF_DATE].copy()
    logging.info(f"過濾 {CUTOFF_DATE.date()} 之前的數據後剩餘: {len(df):,}")
    
    # 4. 排序與分鐘級去重
    logging.info("⏳ 執行分鐘級去重 (Floor to minute & Mean)...")
    df = df.sort_values('datetime')
    df['datetime'] = df['datetime'].dt.floor('min')
    df = df.groupby('datetime', as_index=False)['temp'].mean()
    
    # 5. 重採樣至 10 分鐘等距網格
    logging.info("⏳ 重採樣至 10 分鐘頻率...")
    df = df.set_index('datetime')
    df = df.resample('10min').mean() 
    
    # 6. 插值小間隙 (<= 30 分鐘)
    logging.info("🔧 插值微小間隙 (<= 30 mins)...")
    df['temp'] = df['temp'].interpolate(method='linear', limit=3)
    
    # 7. 丟棄大間隙
    initial_len = len(df)
    df = df.dropna().reset_index()
    logging.info(f"丟棄大間隙後移除 {initial_len - len(df):,} 行。")
    
    # 8. 儲存
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    
    # 9. 輸出統計
    logging.info(f"✅ 成功儲存 {len(df):,} 行至 {OUT_PATH}")
    logging.info(f"📅 日期範圍: {df['datetime'].min()} 至 {df['datetime'].max()}")
    logging.info(f"📊 數據範例 (前 5 行):\n{df.head()}")

if __name__ == "__main__":
    main()