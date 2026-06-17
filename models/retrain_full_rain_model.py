# models/retrain_full_rain_model.py
"""重新訓練完整的降雨感知模型（含分位數迴歸 + 二元分類器）。

產出時間戳記的 candidate 目錄，不覆寫 production 模型。
驗證指標寫入 validation_metrics.json，完整報告寫入 candidate_model_report.json。
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path('data/intraday_ml_train.parquet')

# 時間戳記 candidate 目錄，避免覆寫 production 模型
_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_DIR = Path(f'models/intraday_ml_rain_candidate/{_TIMESTAMP}')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = MODEL_DIR / 'candidate_model_report.json'
METRICS_PATH = MODEL_DIR / 'validation_metrics.json'
CONFIG_PATH = MODEL_DIR / 'training_config.json'
FEATURE_IMPORTANCE_PATH = MODEL_DIR / 'feature_importance.png'

# 完整特徵清單（從中央 schema 載入）
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.feature_schema import get_feature_list, get_target_columns

FEATURES = get_feature_list("rain_aware")
TARGET_COLUMNS = get_target_columns()

# 訓練超參數
LGB_PARAMS = {
    'max_depth': 6,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 30,
    'random_state': 42,
    'verbose': -1,
}

QUANTILE_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

# 時間序列分割邊界
TRAIN_BEFORE = '2025-01-01'
VALID_BEFORE = '2026-01-01'

# 最小樣本數限制
MIN_TRAIN = 1000
MIN_VALID = 100


def validate_features(df):
    """驗證所有必要特徵欄位存在。"""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    logger.info(f"All {len(FEATURES)} features validated.")


def time_split(df):
    """時間序列分割（fail-fast，絕不使用測試集作為驗證）。"""
    df = df.sort_values('datetime').copy()
    train = df[df['datetime'] < TRAIN_BEFORE]
    valid = df[(df['datetime'] >= TRAIN_BEFORE) & (df['datetime'] < VALID_BEFORE)]
    test = df[df['datetime'] >= VALID_BEFORE]

    if len(train) < MIN_TRAIN:
        raise ValueError(f"Training set too small: {len(train)} < {MIN_TRAIN}")
    if len(valid) < MIN_VALID:
        raise ValueError(f"Validation set too small: {len(valid)} < {MIN_VALID}")
    if len(test) < 100:
        logger.warning(f"Test set small: {len(test)} rows — evaluation may be unstable.")

    logger.info(f"Split: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")
    return train, valid, test


def train_quantile_models(X_train, y_train, X_valid, y_valid, alphas, prefix, output_dir):
    """訓練多個分位數迴歸模型並儲存。"""
    models = {}
    for alpha in alphas:
        q = int(alpha * 100)
        logger.info(f"Training {prefix} quantile={alpha}")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            **LGB_PARAMS,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        path = output_dir / f'{prefix}_q{q}.txt'
        model.booster_.save_model(str(path))
        models[f'{prefix}_q{q}'] = model
        logger.info(f"  Saved {path.name}")
    return models


def train_classifier(X_train, y_train, X_valid, y_valid, name, output_dir):
    """訓練二元分類器並儲存。"""
    logger.info(f"Training classifier: {name}")
    model = lgb.LGBMClassifier(
        objective='binary',
        **LGB_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    path = output_dir / f'{name}.txt'
    model.booster_.save_model(str(path))
    logger.info(f"  Saved {path.name}")
    return model


def enforce_monotonicity(preds_dict, alpha_order):
    """強制分位數預測單調性 (q10 <= q25 <= q50 <= q75 <= q90)。"""
    preds_matrix = np.column_stack([preds_dict[f'q{int(a*100)}'] for a in alpha_order])
    preds_matrix.sort(axis=1)
    for i, a in enumerate(alpha_order):
        preds_dict[f'q{int(a*100)}'] = preds_matrix[:, i]
    return preds_dict


def evaluate_quantile_models(models, X_test, y_test, alphas, prefix):
    """評估分位數模型：MAE、覆蓋率、單調性。"""
    preds = {}
    for a in alphas:
        q = int(a * 100)
        preds[f'q{q}'] = models[f'{prefix}_q{q}'].predict(X_test)
    preds = enforce_monotonicity(preds, alphas)

    actual = y_test.values if hasattr(y_test, 'values') else y_test
    q50 = preds['q50']
    mae = float(np.mean(np.abs(actual - q50)))

    coverage_80 = float(np.mean((actual >= preds['q10']) & (actual <= preds['q90'])))
    coverage_50 = float(np.mean((actual >= preds['q25']) & (actual <= preds['q75'])))

    # Interval width
    interval_80_width = float(np.mean(preds['q90'] - preds['q10']))
    interval_50_width = float(np.mean(preds['q75'] - preds['q25']))

    # CRPS approximation (pinball loss at each quantile)
    crps_scores = []
    for a in alphas:
        q = int(a * 100)
        errors = actual - preds[f'q{q}']
        pinball = np.mean(np.where(errors >= 0, a * errors, (a - 1) * errors))
        crps_scores.append(pinball)
    crps = float(np.mean(crps_scores))

    return {
        'mae': round(mae, 4),
        'coverage_80': round(coverage_80, 4),
        'coverage_50': round(coverage_50, 4),
        'interval_80_width': round(interval_80_width, 4),
        'interval_50_width': round(interval_50_width, 4),
        'crps': round(crps, 4),
        'test_size': int(len(actual)),
    }


def evaluate_classifier(model, X_test, y_test, name):
    """評估二元分類器：accuracy, precision, recall, AUC。"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else preds
    actual = y_test.values if hasattr(y_test, 'values') else y_test

    return {
        'accuracy': round(float(accuracy_score(actual, preds)), 4),
        'precision': round(float(precision_score(actual, preds, zero_division=0)), 4),
        'recall': round(float(recall_score(actual, preds, zero_division=0)), 4),
        'auc': round(float(roc_auc_score(actual, proba)), 4),
        'test_size': int(len(actual)),
        'positive_rate': round(float(actual.mean()), 4),
    }


def plot_feature_importance(model, feature_names, output_path):
    """繪製並儲存特徵重要性圖。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        importance = model.feature_importances_
        indices = np.argsort(importance)[-30:]  # top 30

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('LightGBM Feature Importance (Top 30)')
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=100)
        plt.close(fig)
        logger.info(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save feature importance plot: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Full Rain Model Retraining — Starting")
    logger.info("=" * 60)

    # 1. 載入資料
    logger.info(f"Loading training data: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")

    # 2. 驗證特徵
    validate_features(df)

    # 3. 時間序列分割
    train, valid, test = time_split(df)

    X_train = train[FEATURES].fillna(0)
    X_valid = valid[FEATURES].fillna(0)
    X_test = test[FEATURES].fillna(0)

    y_train_u = train['remaining_upside']
    y_train_d = train['remaining_downside']
    y_train_uz = train['is_upside_zero']
    y_train_dz = train['is_downside_zero']

    y_valid_u = valid['remaining_upside']
    y_valid_d = valid['remaining_downside']
    y_valid_uz = valid['is_upside_zero']
    y_valid_dz = valid['is_downside_zero']

    y_test_u = test['remaining_upside']
    y_test_d = test['remaining_downside']
    y_test_uz = test['is_upside_zero']
    y_test_dz = test['is_downside_zero']

    # 4. 訓練分位數迴歸模型
    logger.info("")
    logger.info("=== Training Upside Quantile Models ===")
    upside_models = train_quantile_models(
        X_train, y_train_u, X_valid, y_valid_u, QUANTILE_ALPHAS, 'upside', MODEL_DIR
    )

    logger.info("")
    logger.info("=== Training Downside Quantile Models ===")
    downside_models = train_quantile_models(
        X_train, y_train_d, X_valid, y_valid_d, QUANTILE_ALPHAS, 'downside', MODEL_DIR
    )

    # 5. 訓練二元分類器
    logger.info("")
    logger.info("=== Training Upside Zero Classifier ===")
    upside_clf = train_classifier(
        X_train, y_train_uz, X_valid, y_valid_uz, 'upside_zero', MODEL_DIR
    )

    logger.info("")
    logger.info("=== Training Downside Zero Classifier ===")
    downside_clf = train_classifier(
        X_train, y_train_dz, X_valid, y_valid_dz, 'downside_zero', MODEL_DIR
    )

    # 6. 儲存特徵清單
    with open(MODEL_DIR / 'feature_list.json', 'w') as f:
        json.dump(FEATURES, f, indent=2)
    logger.info(f"Feature list saved ({len(FEATURES)} features)")

    # 7. 驗證指標
    logger.info("")
    logger.info("=== Validation Metrics ===")

    metrics = {}
    metrics['upside'] = evaluate_quantile_models(upside_models, X_valid, y_valid_u, QUANTILE_ALPHAS, 'upside')
    metrics['downside'] = evaluate_quantile_models(downside_models, X_valid, y_valid_d, QUANTILE_ALPHAS, 'downside')
    metrics['upside_zero_classifier'] = evaluate_classifier(upside_clf, X_valid, y_valid_uz, 'upside_zero')
    metrics['downside_zero_classifier'] = evaluate_classifier(downside_clf, X_valid, y_valid_dz, 'downside_zero')

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Validation metrics saved to {METRICS_PATH}")

    # Print summary
    for key in ['upside', 'downside']:
        m = metrics[key]
        logger.info(f"  {key}: MAE={m['mae']}, coverage_80={m['coverage_80']}, CRPS={m['crps']}")
    for key in ['upside_zero_classifier', 'downside_zero_classifier']:
        m = metrics[key]
        logger.info(f"  {key}: AUC={m['auc']}, precision={m['precision']}, recall={m['recall']}")

    # 8. 特徵重要性圖
    logger.info("")
    logger.info("=== Feature Importance ===")
    plot_feature_importance(upside_clf, FEATURES, FEATURE_IMPORTANCE_PATH)

    # 9. 訓練配置
    config = {
        'timestamp': _TIMESTAMP,
        'data_path': str(DATA_PATH),
        'model_dir': str(MODEL_DIR),
        'features': FEATURES,
        'num_features': len(FEATURES),
        'train_size': len(train),
        'valid_size': len(valid),
        'test_size': len(test),
        'train_before': TRAIN_BEFORE,
        'valid_before': VALID_BEFORE,
        'quantile_alphas': QUANTILE_ALPHAS,
        'lgbm_params': LGB_PARAMS,
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Training config saved to {CONFIG_PATH}")

    # 10. Candidate model report
    report = {
        'candidate_model_report': {
            'timestamp': _TIMESTAMP,
            'model_dir': str(MODEL_DIR),
            'data_shape': list(df.shape),
            'train_valid_test': {
                'train': len(train),
                'valid': len(valid),
                'test': len(test),
            },
            'features': {
                'count': len(FEATURES),
                'list': FEATURES,
            },
            'validation_metrics': metrics,
            'artifacts': {
                'quantile_models': [f'{prefix}_q{int(a*100)}.txt'
                                     for prefix in ['upside', 'downside']
                                     for a in QUANTILE_ALPHAS],
                'classifiers': ['upside_zero.txt', 'downside_zero.txt'],
                'feature_list': 'feature_list.json',
                'validation_metrics': 'validation_metrics.json',
                'training_config': 'training_config.json',
                'feature_importance': 'feature_importance.png',
            },
            'status': 'candidate_ready',
        }
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Candidate model report saved to {REPORT_PATH}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training complete. Candidate artifacts in: {MODEL_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
