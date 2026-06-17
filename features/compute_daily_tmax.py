# features/compute_daily_tmax.py
import logging
import yaml
import xarray as xr
import numpy as np
from pathlib import Path

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """載入 config.yaml 設定檔"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_daily_tmax(config):
    """從單點 ECMWF 集合預報中計算每日最高溫度"""
    nc_input = Path(config['paths']['ecmwf_ens_latest'])
    nc_output = Path(config['paths']['ecmwf_daily_tmax'])
    
    if not nc_input.exists():
        logging.error(f"找不到輸入檔案: {nc_input}")
        logging.error("請先執行 data/download_ecmwf_ens.py")
        return

    logging.info(f"正在讀取 {nc_input}...")
    ds = xr.open_dataset(nc_input)
    
    # 確認溫度變數名稱 (cfgrib 通常命名為 't2m'，有時可能是 '2t')
    temp_var = 't2m' if 't2m' in ds.data_vars else '2t'
    if temp_var not in ds.data_vars:
        raise ValueError(f"在資料集中找不到溫度變數。可用的變數: {list(ds.data_vars)}")
        
    logging.info(f"找到溫度變數: {temp_var}")
    
    # 1. 將開爾文 (Kelvin) 轉換為攝氏度 (Celsius)
    t2m_c = ds[temp_var] - 273.15
    
    # 2. 建立 lead_day 座標 (將 step 小時數除以 24 並取整數)
    lead_day = (ds['step'] // 24).astype(int)
    t2m_c = t2m_c.assign_coords(lead_day=lead_day)
    
    # 3. 計算每日最高溫度 (按 lead_day 分組，並在 step 維度上取最大值)
    daily_tmax = t2m_c.groupby('lead_day').max(dim='step')
    
    # 4. 計算集合平均與標準差 (沿著 'number' 成員維度計算)
    ens_mean = daily_tmax.mean(dim='number')
    ens_std = daily_tmax.std(dim='number')
    
    # 5. 移除多餘的 latitude/longitude 維度 (因為我們只取了一個點，大小為 1)
    daily_tmax = daily_tmax.squeeze(drop=True)
    ens_mean = ens_mean.squeeze(drop=True)
    ens_std = ens_std.squeeze(drop=True)
    
    # 6. 建構最終輸出的 Dataset
    out_ds = xr.Dataset({
        'tmax_daily': daily_tmax,
        'ens_mean': ens_mean,
        'ens_std': ens_std
    })
    
    # 設定屬性
    out_ds['tmax_daily'].attrs['units'] = 'Celsius'
    out_ds['tmax_daily'].attrs['description'] = 'Daily maximum 2m temperature per ensemble member'
    out_ds['ens_mean'].attrs['units'] = 'Celsius'
    out_ds['ens_mean'].attrs['description'] = 'Ensemble mean of daily maximum 2m temperature'
    out_ds['ens_std'].attrs['units'] = 'Celsius'
    out_ds['ens_std'].attrs['description'] = 'Ensemble standard deviation of daily maximum 2m temperature'
    
    # 7. 儲存為 NetCDF
    nc_output.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(nc_output)
    logging.info(f"成功儲存每日最高溫度資料至: {nc_output}")
    
    # 8. 列印基本資訊供驗證
    logging.info("--- 資料驗證 ---")
    # 修正：.dims 回傳的是維度名稱元組，需搭配 .shape 使用
    dims_info = dict(zip(out_ds['tmax_daily'].dims, out_ds['tmax_daily'].shape))
    logging.info(f"tmax_daily 維度: {dims_info} (預期包含 number 和 lead_day)")
    
    # 檢查是否有 'number' 維度 (代表集合成員)
    if 'number' in out_ds.dims:
        n_members = len(out_ds['number'])
    else:
        n_members = 1  # 可能只有控制成員
    logging.info(f"集合成員數量 (number): {n_members}")
    logging.info(f"預測天數 (lead_day): {sorted(out_ds['lead_day'].values.tolist())}")
    
    # 印出 lead_day 0 的集合平均溫度作為抽查
    if 0 in out_ds['lead_day'].values:
        mean_t0 = float(out_ds['ens_mean'].sel(lead_day=0).values)
        std_t0 = float(out_ds['ens_std'].sel(lead_day=0).values)
        logging.info(f"Lead Day 0 集合平均最高溫: {mean_t0:.2f} °C (標準差: {std_t0:.2f} °C)")

def main():
    config = load_config()
    compute_daily_tmax(config)

if __name__ == "__main__":
    main()