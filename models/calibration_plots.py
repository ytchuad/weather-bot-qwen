# models/calibration_plots.py
"""繪製桶機率可靠性圖，驗證模型校準"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # 載入驗證集（與訓練分割相同）
    df = pd.read_parquet('data/intraday_ml_train.parquet')
    df = df[df['datetime'] >= '2025-01-01']  # 驗證區間
    if len(df) < 1000:
        split_idx = int(len(pd.read_parquet('data/intraday_ml_train.parquet')) * 0.85)
        df = pd.read_parquet('data/intraday_ml_train.parquet').iloc[split_idx:]

    # 載入模型
    import lightgbm as lgb
    with open('models/intraday_ml/feature_list.json') as f:
        feature_cols = json.load(f)
    model = lgb.Booster(model_file='models/intraday_ml/upside_q50.txt')
    X = df[feature_cols].fillna(0)
    pred_upside = model.predict(X)

    # 預測最終最高溫
    pred_tmax = df['max_so_far'] + pred_upside
    actual_tmax = df['tmax'] if 'tmax' in df.columns else df['remaining_upside'] + df['max_so_far']

    # 建立桶（以 0.5°C 為區間）
    bins = np.arange(25.0, 36.5, 0.5)
    labels = [f"{b:.1f}-{b+0.5:.1f}" for b in bins[:-1]]
    df['bucket'] = pd.cut(actual_tmax, bins=bins, labels=labels, right=False)
    df['pred_bucket'] = pd.cut(pred_tmax, bins=bins, labels=labels, right=False)

    # 計算每個桶的實際頻率
    bucket_probs = df.groupby('pred_bucket').size() / len(df)
    bucket_actual = df.groupby('pred_bucket')['bucket'].apply(lambda x: (x == x.name).mean())

    # 繪製可靠性圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bucket_probs.values, y=bucket_actual.values, mode='markers+lines', name='校準曲線'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='完美校準'))
    fig.update_layout(title='桶機率可靠性圖', xaxis_title='預測機率', yaxis_title='實際頻率')
    fig.write_html('reports/calibration_reliability.html')
    logging.info("可靠性圖已存至 reports/calibration_reliability.html")

if __name__ == '__main__':
    import json
    main()