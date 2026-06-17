# models/train_rain_nowcast_candidate.py
"""
Train rain_aware_nowcast candidate models for comparison.

Candidates:
  A. baseline
  B. rain_observed  (baseline + rain_observed + rain_interaction + metadata)
  C. rain_aware_nowcast (B + rain_nowcast features)

Outputs to: models/intraday_ml_rain_nowcast_candidate/<timestamp>/
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.feature_schema import get_feature_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/intraday_ml_train_with_rain_nowcast.parquet")
CANDIDATE_DIR = Path("models/intraday_ml_rain_nowcast_candidate")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = CANDIDATE_DIR / TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["remaining_upside", "remaining_downside", "is_upside_zero", "is_downside_zero"]
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
TRAIN_END = "2025-01-01"
VALID_END = "2026-01-01"


def time_split(df):
    train = df[df["datetime"] < TRAIN_END].copy()
    valid = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] < VALID_END)].copy()
    test = df[df["datetime"] >= VALID_END].copy()
    logger.info(f"Split: train={len(train)}, valid={len(valid)}, test={len(test)}")
    if len(train) < 1000 or len(valid) < 1000:
        raise ValueError("Train/valid too small.")
    return train, valid, test


def train_quantile(X_tr, y_tr, X_va, y_va, alpha, name, out_dir):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        max_depth=6, num_leaves=31, learning_rate=0.05,
        n_estimators=500, early_stopping_rounds=30,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    model.booster_.save_model(str(out_dir / f"{name}.txt"))
    return model


def train_classifier(X_tr, y_tr, X_va, y_va, name, out_dir):
    model = lgb.LGBMClassifier(
        objective="binary",
        max_depth=6, num_leaves=31, learning_rate=0.05,
        n_estimators=500, early_stopping_rounds=30,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    model.booster_.save_model(str(out_dir / f"{name}.txt"))
    return model


def enforce_monotonicity(preds_dict):
    keys = sorted(preds_dict.keys())
    mat = np.column_stack([preds_dict[k] for k in keys])
    mat.sort(axis=1)
    for i, k in enumerate(keys):
        preds_dict[k] = mat[:, i]
    return preds_dict


def evaluate_quantile(y_true, preds_dict, target_name):
    """preds_dict keys: e.g. remaining_upside_q10, remaining_upside_q50, ..."""
    def qkey(a):
        return f"{target_name}_q{int(a*100)}"
    q50 = preds_dict[qkey(0.5)]
    mae = float(np.mean(np.abs(y_true - q50)))
    coverage_80 = float(np.mean((y_true >= preds_dict[qkey(0.1)]) & (y_true <= preds_dict[qkey(0.9)])))
    coverage_50 = float(np.mean((y_true >= preds_dict[qkey(0.25)]) & (y_true <= preds_dict[qkey(0.75)])))
    return {
        "target": target_name,
        "mae": round(mae, 4),
        "coverage_80": round(coverage_80, 4),
        "coverage_50": round(coverage_50, 4),
    }


def evaluate_classifier(y_true, y_pred_proba, target_name):
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = float(np.mean(y_true == y_pred))
    fp_rate = float(np.mean((y_pred == 1) & (y_true == 0))) if len(y_true) > 0 else 0
    fn_rate = float(np.mean((y_pred == 0) & (y_true == 1))) if len(y_true) > 0 else 0
    return {
        "target": target_name,
        "accuracy": round(accuracy, 4),
        "fp_rate": round(fp_rate, 4),
        "fn_rate": round(fn_rate, 4),
    }


def train_candidate(df, feature_cols, label, out_dir):
    """Train all models for one candidate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = time_split(df)

    X_tr = train[feature_cols].fillna(0)
    X_va = valid[feature_cols].fillna(0)
    X_te = test[feature_cols].fillna(0)

    results = {}

    for target in ["remaining_upside", "remaining_downside"]:
        y_tr = train[target]
        y_va = valid[target]
        y_te = test[target]

        q_models = {}
        preds_train = {}
        preds_valid = {}
        preds_test = {}
        for a in ALPHAS:
            q_name = f"{target}_q{int(a*100)}"
            q_models[q_name] = train_quantile(X_tr, y_tr, X_va, y_va, a, q_name, out_dir)
            preds_train[q_name] = q_models[q_name].predict(X_tr)
            preds_valid[q_name] = q_models[q_name].predict(X_va)
            preds_test[q_name] = q_models[q_name].predict(X_te)

        # Enforce monotonicity on test
        preds_test_mono = enforce_monotonicity(preds_test.copy())
        results[f"{target}_quantile"] = evaluate_quantile(y_te, preds_test_mono, target)

    for target in ["is_upside_zero", "is_downside_zero"]:
        y_tr = train[target]
        y_va = valid[target]
        y_te = test[target]

        clf = train_classifier(X_tr, y_tr, X_va, y_va, target, out_dir)
        proba = clf.predict_proba(X_te)[:, 1]
        results[f"{target}_clf"] = evaluate_classifier(y_te.values, proba, target)

    # Save feature list
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    return results


def main():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    logger.info(f"Data shape: {df.shape}, range: {df['datetime'].min()} -> {df['datetime'].max()}")

    candidates = {
        "A_baseline": get_feature_list("baseline"),
        "B_rain_observed": get_feature_list("rain_aware"),
        "C_rain_aware_nowcast": get_feature_list("rain_aware_nowcast"),
    }

    all_results = {}
    for label, feature_cols in candidates.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training candidate: {label} ({len(feature_cols)} features)")
        logger.info(f"{'='*60}")
        candidate_dir = RUN_DIR / label
        results = train_candidate(df, feature_cols, label, candidate_dir)
        all_results[label] = results
        logger.info(f"Results for {label}:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")

    # Save comparison report
    report_path = RUN_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nComparison report saved to {report_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for label, results in all_results.items():
        logger.info(f"\n{label}:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
