# data/download_ecmwf_ens.py
import os
import logging
import yaml
import xarray as xr
import cfgrib
from ecmwf.opendata import Client
from pathlib import Path

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """載入 config.yaml 設定檔"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_ecmwf_ens(config):
    """使用 ecmwf-opendata 從 AWS 下載 ECMWF 集合預報"""
    logging.info("正在初始化 ECMWF Client (來源: AWS)...")
    client = Client(source="aws")
    
    # 設定輸出路徑
    raw_dir = Path(config['paths']['ecmwf_ens_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_grib_path = raw_dir / "latest_ens_raw.grib"
    
    # 構建下載請求
    # date=-1 代表自動獲取最新可用的週期
    # time=0 代表 00Z (也可以改為 12 代表 12Z)
    # type="pf" 代表 Perturbed Forecast (擾動成員，共 50 個)
    request = {
        'class': 'od',
        'date': -1,
        'time': 0, 
        'step': config['forecast']['step_range'], # "0/to/144/by/3"
        'param': "2t",                            # 2公尺溫度
        'type': "pf",                             # 擾動成員
        'stream': "enfo",                         # 集合預報流
        'levtype': "sfc",                         # 地面層
        'target': str(raw_grib_path)
    }
    
    logging.info(f"開始下載 ECMWF 集合預報 (最新 00Z 週期，步階 0-144 小時)...")
    logging.info("這可能需要幾分鐘時間，請耐心等待...")
    
    try:
        client.retrieve(request)
        logging.info(f"GRIB 檔案下載完成，儲存於: {raw_grib_path}")
        return raw_grib_path
    except Exception as e:
        logging.error(f"下載 ECMWF 資料失敗: {e}")
        logging.error("請確認您的網路連線，或嘗試重新執行。")
        raise

def extract_single_point(grib_path, config):
    """從 GRIB 檔案中提取單一網格點並存為 NetCDF"""
    logging.info("正在讀取 GRIB 檔案並提取單一網格點...")
    
    nc_path = Path(config['paths']['ecmwf_ens_latest'])
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    
    lat = config['location']['lat']
    lon = config['location']['lon']
    
    # 使用 cfgrib 打開 GRIB
    # 因為 ECMWF 的 GRIB 經常因為 step 不同而被 cfgrib 拆分成多個 dataset，我們使用 open_datasets
    try:
        datasets = cfgrib.open_datasets(str(grib_path))
        if not datasets:
            raise ValueError("cfgrib 無法從 GRIB 檔案中讀取任何資料集。")
        
        # 嘗試合併拆分後的 datasets
        try:
            ds = xr.merge(datasets)
        except Exception as e:
            logging.warning(f"合併 datasets 時發生警告: {e}。將使用第一個 dataset。")
            ds = datasets[0]
            
    except Exception as e:
        logging.error(f"讀取 GRIB 檔案失敗: {e}")
        logging.error("請確保已正確安裝 cfgrib 和 eccodes: conda install -c conda-forge cfgrib eccodes")
        raise

    # 檢查是否有 'number' 維度 (代表集合成員)
    if 'number' not in ds.dims:
        logging.warning("警告: 未找到 'number' 維度。可能只下載了控制成員或解析異常。")
    else:
        logging.info(f"成功讀取 {len(ds['number'])} 個集合成員。")

    # ECMWF 的經度範圍可能是 0-360，而我們的 config 是 114.174 (0-180)
    # 我們準備兩個經度值來嘗試匹配
    lon_360 = lon if lon >= 0 else lon + 360
    
    # 提取最接近 HKO 總部的點
    try:
        ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    except Exception:
        logging.info(f"使用 0-360 經度格式 ({lon_360}) 重新選取...")
        ds_point = ds.sel(latitude=lat, longitude=lon_360, method="nearest")
        
    # 儲存為 NetCDF
    ds_point.to_netcdf(str(nc_path))
    logging.info(f"成功提取單一網格點並儲存至: {nc_path}")
    
    # 印出基本資訊
    logging.info(f"提取的實際座標: Lat={ds_point['latitude'].values:.3f}, Lon={ds_point['longitude'].values:.3f}")
    logging.info(f"資料維度: {dict(ds_point.dims)}")

def main():
    config = load_config()
    
    # 1. 下載 GRIB 檔案
    raw_grib_path = download_ecmwf_ens(config)
    
    # 2. 提取單點並存為 NetCDF
    extract_single_point(raw_grib_path, config)
    
    logging.info("所有步驟完成！")

if __name__ == "__main__":
    main()