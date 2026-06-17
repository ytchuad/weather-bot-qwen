# features/audit_no_leakage.py
"""
資料洩漏審計腳本
檢查訓練集與即時特徵中是否含有目標變數、未來觀測或結算資訊。
同時驗證早期小時 (00:00-02:00) 的覆蓋完整性。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路徑設定
ML_TRAIN_PATH = Path('data/intraday_ml_train.parquet')
LONG_TRAIN_MAX_PATH = Path('data/training_set_max.parquet')
LONG_TRAIN_MIN_PATH = Path('data/training_set_min.parquet')
REPORT_PATH = Path('reports/leakage_audit_report.json')
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 明確的目標變數名稱（這些絕不能出現在特徵欄位中）
TARGET_NAMES = [
    'tmax', 'tmin', 'actual_tmax', 'actual_tmin',
    'final_tmax', 'final_tmin', 'remaining_upside', 'remaining_downside',
    'is_upside_zero', 'is_downside_zero'
]

# 可能含有未來資訊的欄位模式（應人工審查）
SUSPICIOUS_PATTERNS = [
    'forecast_tmax_d0', 'forecast_tmin_d0',  # 當天預報可能用於11:30前？需確認時序
    'actual', 'final', 'settled', 'resolved'
]

def check_leakage(df, dataset_name, feature_cols, target_names, suspicious_patterns):
    """檢查 DataFrame 的特徵欄位是否有洩漏"""
    report = {
        'dataset': dataset_name,
        'num_rows': len(df),
        'num_features': len(feature_cols),
        'leakage_errors': [],
        'suspicious_warnings': [],
        'early_hour_coverage': None
    }

    # 1. 檢查是否包含目標變數
    for col in feature_cols:
        col_lower = col.lower()
        for target in target_names:
            if target in col_lower:
                report['leakage_errors'].append(
                    f"特徵 '{col}' 可能包含目標變數 '{target}'！"
                )

    # 2. 檢查可疑模式
    for col in feature_cols:
        col_lower = col.lower()
        for pat in suspicious_patterns:
            if pat in col_lower:
                report['suspicious_warnings'].append(
                    f"特徵 '{col}' 符合可疑模式 '{pat}'，請人工確認無未來資訊。"
                )

    # 3. 早期小時覆蓋檢查 (僅對 ML 訓練集)
    if 'minutes_since_midnight' in df.columns:
        early_mask = df['minutes_since_midnight'] < 120
        early_count = early_mask.sum()
        report['early_hour_coverage'] = {
            'rows_before_02:00': int(early_count),
            'has_hour_0': int((df['hour'] == 0).any()),
            'has_hour_1': int((df['hour'] == 1).any()),
            'min_minutes_since_midnight': int(df['minutes_since_midnight'].min())
        }
        if early_count == 0:
            report['leakage_errors'].append(
                "早期小時完全缺失！訓練集中沒有任何 minutes_since_midnight < 120 的資料。"
            )
        logger.info(f"早期小時 (<02:00) 樣本數: {early_count}")

    return report

def main():
    all_reports = {}

    # --- 檢查日內 ML 訓練集 ---
    if ML_TRAIN_PATH.exists():
        logger.info(f"載入 {ML_TRAIN_PATH}")
        df_ml = pd.read_parquet(ML_TRAIN_PATH)
        # 特徵欄位：排除 datetime, date 及已知目標
        target_cols_in_file = ['remaining_upside', 'remaining_downside', 'is_upside_zero', 'is_downside_zero']
        feature_cols_ml = [c for c in df_ml.columns if c not in target_cols_in_file and c not in ['datetime', 'date']]
        report_ml = check_leakage(df_ml, 'intraday_ml_train', feature_cols_ml, TARGET_NAMES, SUSPICIOUS_PATTERNS)
        all_reports['intraday_ml_train'] = report_ml
    else:
        logger.warning(f"{ML_TRAIN_PATH} 不存在，跳過檢查。")

    # --- 檢查長期模型訓練集 (最高溫) ---
    if LONG_TRAIN_MAX_PATH.exists():
        logger.info(f"載入 {LONG_TRAIN_MAX_PATH}")
        df_long = pd.read_parquet(LONG_TRAIN_MAX_PATH)
        # 目標為 tmax，排除它
        feature_cols_long = [c for c in df_long.columns if c not in ['tmax', 'date', 'target_date']]
        report_long = check_leakage(df_long, 'training_set_max', feature_cols_long, TARGET_NAMES, SUSPICIOUS_PATTERNS)
        all_reports['training_set_max'] = report_long

    if LONG_TRAIN_MIN_PATH.exists():
        logger.info(f"載入 {LONG_TRAIN_MIN_PATH}")
        df_long_min = pd.read_parquet(LONG_TRAIN_MIN_PATH)
        feature_cols_long_min = [c for c in df_long_min.columns if c not in ['tmin', 'date', 'target_date']]
        report_long_min = check_leakage(df_long_min, 'training_set_min', feature_cols_long_min, TARGET_NAMES, SUSPICIOUS_PATTERNS)
        all_reports['training_set_min'] = report_long_min

    # --- 寫出報告 ---
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    # 摘要
    total_errors = sum(len(r['leakage_errors']) for r in all_reports.values())
    if total_errors > 0:
        logger.error(f"❌ 發現 {total_errors} 個洩漏錯誤！詳見 {REPORT_PATH}")
    else:
        logger.info(f"✅ 未發現明確洩漏。報告已存至 {REPORT_PATH}")

    # 顯示早期小時摘要
    if 'intraday_ml_train' in all_reports:
        early = all_reports['intraday_ml_train']['early_hour_coverage']
        if early:
            logger.info(f"早期小時覆蓋: rows={early['rows_before_02:00']}, "
                        f"has_h0={early['has_hour_0']}, has_h1={early['has_hour_1']}, "
                        f"min_minutes={early['min_minutes_since_midnight']}")

if __name__ == '__main__':
    main()