# models/train_probabilistic_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
from pathlib import Path
from scipy.stats import norm
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_DATA_PATH = Path('data/training_set_max.parquet')
MIN_DATA_PATH = Path('data/training_set_min.parquet')
MODEL_DIR = Path('models')
CONFIG_PATH = MODEL_DIR / 'feature_config.json'

def train_probabilistic_model(df, target_col, prefix):
    logging.info(f"--- 開始訓練 {target_col} 機率回歸模型 ---")
    
    # 1. 準備特徵與目標
    exclude_cols = ['target_date', 'tmax', 'tmin']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. 時間序列分割 (嚴禁 Shuffle，避免未來資訊洩漏)
    train_mask = df['target_date'].dt.year <= 2022
    val_mask = df['target_date'].dt.year == 2023
    test_mask = df['target_date'].dt.year >= 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logging.info(f"資料分割: 訓練集 {len(X_train)}, 驗證集 {len(X_val)}, 測試集 {len(X_test)}")
    
    # 3. 訓練 Mean 模型
    logging.info("訓練 Mean 模型 (XGBoost Regressor)...")
    mean_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        verbosity=0
    )
    mean_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 4. 訓練 Std 模型
    logging.info("計算殘差並訓練 Std 模型...")
    y_train_pred_mean = mean_model.predict(X_train)
    abs_residuals = np.abs(y_train - y_train_pred_mean)
    
    # 驗證集的絕對殘差 (用於 early stopping)
    y_val_pred_mean = mean_model.predict(X_val)
    abs_residuals_val = np.abs(y_val - y_val_pred_mean)
    
    std_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        verbosity=0
    )
    std_model.fit(
        X_train, abs_residuals,
        eval_set=[(X_val, abs_residuals_val)],
        verbose=False
    )
    
    # 5. 測試集評估與校準 (Calibration)
    logging.info("在測試集上評估模型校準...")
    y_test_pred_mean = mean_model.predict(X_test)
    y_test_pred_abs_res = std_model.predict(X_test)
    
    # 將絕對殘差轉換為標準差: sigma = E[|X-mu|] * sqrt(pi/2)
    conversion_factor = np.sqrt(np.pi / 2)
    y_test_pred_std = y_test_pred_abs_res * conversion_factor
    y_test_pred_std = np.clip(y_test_pred_std, 0.5, None) # 強制最小標準差 0.5°C，防止極端自信
    
    # 計算 Z-score
    z_scores = (y_test - y_test_pred_mean) / y_test_pred_std
    
    # 經驗覆蓋率 (Empirical Coverage)
    within_1_std = np.mean(np.abs(z_scores) <= 1.0)
    within_2_std = np.mean(np.abs(z_scores) <= 2.0)
    
    logging.info(f"測試集覆蓋率: ±1σ 實際 {within_1_std:.1%} (理論 68.3%), ±2σ 實際 {within_2_std:.1%} (理論 95.4%)")
    logging.info(f"測試集 Mean Absolute Error (MAE): {np.mean(np.abs(y_test - y_test_pred_mean)):.2f} °C")
    
    # 6. 繪製 PIT 直方圖 (Probability Integral Transform)
    # 若模型校準完美，CDF 值應呈現 [0, 1] 的均勻分佈
    cdf_vals = norm.cdf(y_test, loc=y_test_pred_mean, scale=y_test_pred_std)
    
    fig = go.Figure(data=[go.Histogram(x=cdf_vals, nbinsx=20, histnorm='probability')])
    fig.add_hline(y=1/20, line_dash="dash", line_color="red", annotation_text="Uniform Expectation (5%)")
    fig.update_layout(
        title=f"PIT Histogram for {target_col} (Test Set)",
        xaxis_title="CDF Value (Probability Integral Transform)",
        yaxis_title="Frequency",
        height=400
    )
    pit_path = MODEL_DIR / f'pit_histogram_{target_col}.html'
    fig.write_html(pit_path)
    logging.info(f"PIT 校準直方圖已儲存至 {pit_path}")
    
    # 7. 儲存模型
    mean_model_path = MODEL_DIR / f'xgb_{prefix}_mean.json'
    std_model_path = MODEL_DIR / f'xgb_{prefix}_std.json'
    
    mean_model.save_model(str(mean_model_path))
    std_model.save_model(str(std_model_path))
    logging.info(f"模型已儲存: {mean_model_path}, {std_model_path}")
    
    return feature_cols

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 訓練 Tmax 模型
    df_max = pd.read_parquet(MAX_DATA_PATH)
    df_max['target_date'] = pd.to_datetime(df_max['target_date'])
    features_max = train_probabilistic_model(df_max, 'tmax', 'tmax')
    
    # 訓練 Tmin 模型
    df_min = pd.read_parquet(MIN_DATA_PATH)
    df_min['target_date'] = pd.to_datetime(df_min['target_date'])
    features_min = train_probabilistic_model(df_min, 'tmin', 'tmin')
    
    # 儲存特徵配置
    config = {
        'features': features_max,
        'target_cols': ['tmax', 'tmin'],
        'std_conversion_factor': float(np.sqrt(np.pi / 2)),
        'min_std': 0.5
    }
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    logging.info(f"特徵配置已儲存至 {CONFIG_PATH}")
    logging.info("✅ 所有機率回歸模型訓練完成！")

if __name__ == "__main__":
    main()