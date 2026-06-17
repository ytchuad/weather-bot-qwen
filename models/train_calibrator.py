# models/train_calibrator.py
import pandas as pd
import numpy as np
import logging
import yaml
import json
from pathlib import Path
from sklearn.metrics import log_loss
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_features_and_target(df, buckets):
    exclude_cols = ['forecast_date', 'lead_day', 'observed_bucket']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    bucket_to_class = {b: i for i, b in enumerate(buckets)}
    y = df['observed_bucket'].map(bucket_to_class).astype(int)
    return X, y, feature_cols

def time_series_cv_by_year(df, n_splits=5):
    df = df.copy()
    df['year'] = pd.to_datetime(df['forecast_date']).dt.year
    years = sorted(df['year'].unique())
    if len(years) < n_splits + 1:
        logging.warning(f"年份數量不足，調整 n_splits 為 {max(1, len(years)-1)}")
        n_splits = max(1, len(years)-1)
    
    split_indices = []
    for i in range(n_splits):
        train_years = years[:i+2]
        val_years = [years[i+2]] if i+2 < len(years) else []
        if not val_years: continue
        train_idx = df[df['year'].isin(train_years)].index.tolist()
        val_idx = df[df['year'].isin(val_years)].index.tolist()
        if train_idx and val_idx:
            split_indices.append((train_idx, val_idx))
    return split_indices

def compute_reliability_data(y_true, y_pred_proba, n_classes, n_bins=10):
    reliability_data = []
    for class_idx in range(n_classes):
        probs = y_pred_proba[:, class_idx]
        labels = (y_true == class_idx).astype(int)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if i == n_bins - 1: mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
            if mask.sum() > 0:
                reliability_data.append({
                    'class': class_idx, 'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                    'mean_predicted_prob': probs[mask].mean(), 'mean_actual_freq': labels[mask].mean(),
                    'count': int(mask.sum()), 'calibration_error': abs(probs[mask].mean() - labels[mask].mean())
                })
    return reliability_data

def train_calibrator(config):
    train_path = Path(config['paths']['training_data'])
    model_path = Path(config['paths']['model_path'])
    buckets = config['market']['buckets']
    n_classes = len(buckets)
    
    if not train_path.exists():
        logging.error(f"找不到訓練資料: {train_path}"); return

    logging.info(f"正在載入訓練資料: {train_path}")
    df = pd.read_parquet(train_path)
    X, y, feature_cols = prepare_features_and_target(df, buckets)
    logging.info(f"特徵數量: {len(feature_cols)}, 樣本數量: {len(X)}")
    
    cv_splits = time_series_cv_by_year(df, n_splits=5)
    all_y_true, all_y_pred_proba, cv_log_losses = [], [], []
    
    xgb_params = {
        'objective': 'multi:softprob', 'num_class': n_classes, 'eval_metric': 'mlogloss',
        'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'seed': 42, 'tree_method': 'hist', 'verbosity': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        logging.info(f"Fold {fold}: 訓練集 {len(train_idx)} 筆, 驗證集 {len(val_idx)} 筆")
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        model = xgb.XGBClassifier(**xgb_params)
        # 移除 eval_set 避免 XGBoost 對驗證集標籤做嚴格檢查
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_raw = model.predict_proba(X_val)
        if y_pred_raw.ndim == 1: y_pred_raw = y_pred_raw.reshape(-1, 1)
            
        # 對齊機率矩陣至 n_classes 欄位
        y_pred_aligned = np.zeros((len(y_val), n_classes))
        for i, cls in enumerate(model.classes_):
            if 0 <= int(cls) < n_classes:
                y_pred_aligned[:, int(cls)] = y_pred_raw[:, i]
                
        # 僅保留模型實際學過的類別樣本進行評估
        valid_mask = np.isin(y_val, model.classes_)
        y_val_f, y_pred_f = y_val[valid_mask], y_pred_aligned[valid_mask]
        
        if len(y_val_f) == 0:
            logging.warning(f"Fold {fold}: 驗證集無重疊類別，跳過。"); continue
            
        # 歸一化機率 (修正 sklearn 警告)
        row_sums = y_pred_f.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        y_pred_f = y_pred_f / row_sums
            
        fold_logloss = log_loss(y_val_f, y_pred_f, labels=list(range(n_classes)))
        cv_log_losses.append(fold_logloss)
        logging.info(f"  Fold {fold} Log Loss: {fold_logloss:.4f} (有效樣本: {len(y_val_f)})")
        
        all_y_true.extend(y_val_f.tolist())
        all_y_pred_proba.append(y_pred_f)
    
    if not all_y_true:
        logging.error("交叉驗證無有效資料。"); return

    all_y_pred_proba = np.vstack(all_y_pred_proba)
    overall_logloss = log_loss(all_y_true, all_y_pred_proba, labels=list(range(n_classes)))
    logging.info(f"\n整體驗證集 Log Loss: {overall_logloss:.4f}")
    
    logging.info("正在使用全部資料訓練最終模型...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X.values, y.values, verbose=False)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    logging.info(f"模型已儲存至: {model_path}")
    
    logging.info("正在生成校準報告...")
    reliability_data = compute_reliability_data(np.array(all_y_true), all_y_pred_proba, n_classes)
    
    report = {
        'overall_log_loss': float(overall_logloss),
        'cv_log_losses': [float(x) for x in cv_log_losses],
        'feature_importance': dict(zip(feature_cols, final_model.feature_importances_.tolist())),
        'reliability_data': reliability_data, 'buckets': buckets, 'training_samples': len(df)
    }
    report_path = model_path.parent / 'calibration_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"校準報告已儲存至: {report_path}")
    
    try: plot_reliability_diagram(reliability_data, buckets, model_path.parent)
    except Exception as e: logging.warning(f"繪圖失敗: {e}")
    
    logging.info("\n--- 特徵重要性 (Top 10) ---")
    for feat, imp in sorted(report['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  {feat}: {imp:.4f}")
    return report

def plot_reliability_diagram(reliability_data, buckets, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, bucket in enumerate(buckets):
        ax = axes[i]; data = [d for d in reliability_data if d['class'] == i]
        if not data: continue
        ax.scatter([d['mean_predicted_prob'] for d in data], [d['mean_actual_freq'] for d in data],
                   s=[d['count']*0.1 for d in data], alpha=0.6)
        ax.plot([0, 1], [0, 1], 'r--', lw=1)
        ax.set_title(f'Bucket {bucket}°C'); ax.grid(True, alpha=0.3); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'reliability_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"可靠性圖已儲存至: {output_dir / 'reliability_diagram.png'}")

def main():
    config = load_config()
    train_calibrator(config)

if __name__ == "__main__":
    main()