"""
Train Model G — 結合 Model F 的官方預報特徵 (forecast_gap) 
與 Model C 的物理錨定特徵 (max_so_far, drop_from_max, time_since_max)。

此腳本會自動偵測 feature_store_enhanced.parquet 是否包含錨定特徵，
若無則即時從 hk_weather_raw 的溫度數據計算補上。

輸出模型至: models/intraday_minute_ml_model_g/
包含 5 個分位數 (q10~q90) + 二元分類器 (upside_zero)
"""

import json
import logging
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

# ========== 設定 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 資料路徑
FEATURE_STORE_PATH = Path("data/feature_store_enhanced.parquet")
RAW_TEMP_DIR = Path("data/hk_weather_raw")  # 存放 *_temperature.parquet 的資料夾
MODEL_DIR = Path("models/intraday_minute_ml_model_g")

# 時間分割 (與 Model C 一致)
TRAIN_END = "2024-06-11"
VALID_END = "2025-06-11"

# 分位數
ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

# LightGBM 超參數 (與 Model C 完全一致)
LGB_PARAMS = dict(
    max_depth=6,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_data_in_leaf=300,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

# ========== 定義 Model G 特徵清單 ==========
# 來自 Model F 的 12 個特徵
MODEL_F_FEATURES = [
    "value",                # 當前溫度 (T-8)
    "humidity",             # 當前濕度
    "pressure",             # 當前氣壓
    "temp_slope_30min",     # 過去 30 分鐘升溫斜率
    "temp_volatility_30min",# 過去 30 分鐘溫度波動
    "humid_delta_30min",    # 過去 30 分鐘濕度變化
    "pressure_delta_30min", # 過去 30 分鐘氣壓變化
    "forecast_gap",         # 官方預測最高溫 - max_so_far
    "hour",                 # 當前小時
    "minute",               # 當前分鐘
    "day_of_week",          # 星期幾
    "month",                # 月份
]

# 來自 Model C 的 3 個錨定特徵
MODEL_C_ANCHORS = [
    "max_so_far",           # 截至該時刻的當日最高溫
    "drop_from_max",        # 當前溫度離今日最高溫掉了幾度 (max_so_far - value)
    "time_since_max",       # 距離達到今日最高溫過了幾分鐘
]

# Model G 最終特徵清單 (15 個: 12 Model F + 3 錨定)
FEATURE_COLS = MODEL_F_FEATURES + MODEL_C_ANCHORS

# ========== 輔助函數 ==========
def enforce_monotonicity(preds_dict):
    """確保分位數預測單調遞增 (q10 <= q25 <= q50 <= q75 <= q90)"""
    preds_matrix = np.column_stack([preds_dict[f"q{int(a*100)}"] for a in ALPHAS])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(ALPHAS):
        preds_dict[f"q{int(a*100)}"] = preds_matrix[:, i]
    return preds_dict

# ========== 核心：載入資料並補上錨定特徵 ==========
def load_and_prepare_data():
    logger.info("📂 載入 Feature Store (Model F 基礎特徵)...")
    df = pd.read_parquet(FEATURE_STORE_PATH)
    logger.info(f"✅ 載入 {len(df):,} 筆決策點")

    # 檢查錨定特徵是否已存在
    if all(col in df.columns for col in MODEL_C_ANCHORS):
        logger.info("✅ 錨定特徵已存在，直接使用")
        return df

    logger.warning("⚠️ 錨定特徵不存在，將從原始分鐘數據即時計算...")

    # --- 步驟 1: 讀取所有溫度原始檔 ---
    temp_files = glob.glob(str(RAW_TEMP_DIR / "*_temperature.parquet"))
    if not temp_files:
        raise FileNotFoundError(f"找不到溫度原始檔: {RAW_TEMP_DIR}")

    logger.info(f"📂 讀取 {len(temp_files)} 個溫度原始檔...")
    df_raw_list = []
    for f in temp_files:
        df_raw_list.append(pd.read_parquet(f))
    df_raw = pd.concat(df_raw_list, ignore_index=True)
    df_raw = df_raw.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
    df_raw['date'] = df_raw['timestamp'].dt.date
    logger.info(f"✅ 原始溫度數據筆數: {len(df_raw):,}")

    # --- 步驟 2: 計算每日累積最高溫 (max_so_far) ---
    logger.info("📊 計算每日累積最高溫...")
    df_raw['max_so_far'] = df_raw.groupby('date')['value'].cummax()

    # --- 步驟 3: 計算每日最高溫發生的時間 (第一次達到) ---
    logger.info("📊 計算每日最高溫發生時間...")
    df_raw['is_daily_max'] = df_raw['value'] == df_raw['max_so_far']
    daily_max_time = df_raw[df_raw['is_daily_max']].groupby('date')['timestamp'].min().reset_index()
    daily_max_time.columns = ['date', 'max_time']

    # --- 步驟 4: 將錨定特徵 merge 回主 DataFrame ---
    # df 中有 'timestamp' (即 lookback_time) 和 'decision_time'
    # 我們需要將 df_raw 的 'timestamp', 'max_so_far' 對齊到 df 的 'timestamp'
    df_anchor = df_raw[['timestamp', 'max_so_far']].copy()
    
    # 因為 df['timestamp'] 就是觀測時間點 (已延遲 8 分鐘)，直接 merge
    df = df.merge(df_anchor, on='timestamp', how='left')

    # 補上每日最高溫時間 (以日期為 key)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.merge(daily_max_time, on='date', how='left')

    # --- 步驟 5: 計算衍生錨定特徵 ---
    # drop_from_max
    df['drop_from_max'] = df['max_so_far'] - df['value']

    # time_since_max (分鐘)
    df['time_since_max'] = (df['timestamp'] - pd.to_datetime(df['max_time'])).dt.total_seconds() / 60
    # 若尚未達到最高溫 (time_since_max 為負)，設為 0 (代表還在上升階段)
    df['time_since_max'] = df['time_since_max'].clip(lower=0)

    # 清理暫存欄位
    df = df.drop(columns=['date', 'max_time'])

    # 填補可能的 NaN (極少數邊界情況)
    for col in MODEL_C_ANCHORS:
        if col in df.columns:
            df[col] = df[col].fillna(df['value'])  # 若無 max_so_far，暫時用 value 代替

    logger.info(f"✅ 錨定特徵計算完成！")
    logger.info(f"   max_so_far 範圍: {df['max_so_far'].min():.1f} ~ {df['max_so_far'].max():.1f}")
    logger.info(f"   drop_from_max 平均值: {df['drop_from_max'].mean():.2f}")
    logger.info(f"   time_since_max 中位數: {df['time_since_max'].median():.1f} 分鐘")

    return df

# ========== 訓練與評估函數 (移植自 Model C) ==========
def time_split(df):
    df['target_date'] = pd.to_datetime(df['decision_time']).dt.date.astype(str)
    train = df[df['target_date'] < TRAIN_END].copy()
    valid = df[(df['target_date'] >= TRAIN_END) & (df['target_date'] < VALID_END)].copy()
    oot = df[df['target_date'] >= VALID_END].copy()
    logger.info(f"Split: train={len(train):,}  valid={len(valid):,}  oot={len(oot):,}")
    return train, valid, oot

def train_quantile_models(X_train, y_train, X_valid, y_valid):
    models = {}
    for alpha in ALPHAS:
        logger.info(f"Training upside quantile alpha={alpha}")
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        key = f"upside_q{int(alpha*100)}"
        model.booster_.save_model(str(MODEL_DIR / f"{key}.txt"))
        logger.info(f"  {key}: best_iter={model.best_iteration_}")
        models[key] = model
    return models

def train_classifier(X_train, y_train, X_valid, y_valid):
    logger.info("Training upside_zero classifier")
    model = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.booster_.save_model(str(MODEL_DIR / "upside_zero.txt"))
    logger.info(f"  upside_zero: best_iter={model.best_iteration_}")
    return model

def tune_threshold(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_valid, proba)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    logger.info(f"Best classifier threshold={best_thr:.4f}  F1={f1_scores[best_idx]:.4f}")
    return float(best_thr)

def oot_predict(quantile_models, clf, df_oot, best_thr):
    X = df_oot[FEATURE_COLS]
    df_out = df_oot[["target_date", "decision_time", "value", "max_so_far", "remaining_upside", "hour"]].copy()

    preds = {}
    for a in ALPHAS:
        key = f"upside_q{int(a*100)}"
        preds[f"q{int(a*100)}"] = quantile_models[key].predict(X)
    preds = enforce_monotonicity(preds)

    for a in ALPHAS:
        df_out[f"upside_q{int(a*100)}"] = preds[f"q{int(a*100)}"]
        df_out[f"pred_tmax_q{int(a*100)}"] = df_out["max_so_far"] + preds[f"q{int(a*100)}"]

    zero_proba = clf.predict_proba(X)[:, 1]
    df_out["zero_proba"] = zero_proba
    df_out["zero_pred"] = (zero_proba >= best_thr).astype(int)

    return df_out

def evaluate(df, label):
    n = len(df)
    if n == 0:
        logger.warning(f"Empty set: {label}")
        return
    actual = df["remaining_upside"].values
    q50 = df["upside_q50"].values
    mae_up = np.nanmean(np.abs(actual - q50))
    logger.info(f"[{label:>6s}] n={n:>7,}  MAE_up={mae_up:.3f}")

# ========== 主程式 ==========
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 載入資料 (自動補上錨定特徵)
    df = load_and_prepare_data()

    # 2. 移除 Label 極端值 (避免訓練不穩定)
    df = df[df['remaining_upside'].between(-2, 15)]
    logger.info(f"Filtered dataset: {len(df):,} rows")

    # 3. 時間分割
    train, valid, oot = time_split(df)

    # 4. 準備訓練資料
    X_train = train[FEATURE_COLS]
    y_train = train["remaining_upside"]
    if "is_upside_zero" in train.columns:
        y_train_zero = train["is_upside_zero"]
    else:
        y_train_zero = (train["remaining_upside"] <= 0.05).astype(int)

    X_valid = valid[FEATURE_COLS]
    y_valid = valid["remaining_upside"]
    if "is_upside_zero" in valid.columns:
        y_valid_zero = valid["is_upside_zero"]
    else:
        y_valid_zero = (valid["remaining_upside"] <= 0.05).astype(int)

    # 5. 訓練分位數模型 (5 個)
    logger.info("=" * 50)
    logger.info("TRAINING QUANTILE MODELS (Model G)")
    logger.info("=" * 50)
    quantile_models = train_quantile_models(X_train, y_train, X_valid, y_valid)

    # 6. 訓練分類器
    logger.info("=" * 50)
    logger.info("TRAINING CLASSIFIER (upside_zero)")
    logger.info("=" * 50)
    clf = train_classifier(X_train, y_train_zero, X_valid, y_valid_zero)
    best_thr = tune_threshold(clf, X_valid, y_valid_zero)

    # 7. OOT 預測與評估
    logger.info("=" * 50)
    logger.info("OOT PREDICTIONS")
    logger.info("=" * 50)
    df_oot = oot_predict(quantile_models, clf, oot, best_thr)
    evaluate(df_oot, "OOT Overall")

    # 8. 儲存特徵清單與設定
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(MODEL_DIR / "best_threshold.json", "w") as f:
        json.dump({"upside_zero_threshold": best_thr}, f)

    oot_out = MODEL_DIR / "oot_predictions.parquet"
    df_oot.to_parquet(oot_out, index=False)
    logger.info(f"✅ OOT predictions saved to {oot_out}")

    logger.info(f"🎉 Model G 訓練完成！模型儲存於: {MODEL_DIR}")
    logger.info(f"📊 特徵數量: {len(FEATURE_COLS)}")

if __name__ == "__main__":
    main()