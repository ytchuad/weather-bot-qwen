# models/evaluate_candidates.py
"""
Step 13 – Model comparison and validation slices.

Compares A (baseline) vs B (rain_observed) vs C (rain_aware_nowcast)
on the test set with full slice analysis.

Outputs: reports/candidate_evaluation_report.json
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import logging
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="Only one class is present in y_true*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = Path("reports/candidate_evaluation_report.json")

# ── Candidate model directories ──────────────────────────────────
_CANDIDATE_PRIMARY = Path("models/intraday_ai_rain_nowcast_candidate")
_CANDIDATE_FALLBACK = Path("models/intraday_ml_rain_nowcast_candidate")
CANDIDATE_BASE = _CANDIDATE_PRIMARY if _CANDIDATE_PRIMARY.exists() else _CANDIDATE_FALLBACK

# Use the latest run under the candidate base
_runs = sorted(CANDIDATE_BASE.iterdir(), key=lambda x: x.name, reverse=True) if CANDIDATE_BASE.exists() else []
CANDIDATE_DIR = _runs[0] if _runs else CANDIDATE_BASE  # fallback to base dir (will fail gracefully later)

CANDIDATES = {
    "A_baseline": {
        "dir": CANDIDATE_DIR / "A_baseline",
        "features": "baseline",
    },
    "B_rain_observed": {
        "dir": CANDIDATE_DIR / "B_rain_observed",
        "features": "rain_aware",
    },
    "C_rain_aware_nowcast": {
        "dir": CANDIDATE_DIR / "C_rain_aware_nowcast",
        "features": None,  # use trained feature list from disk
    },
}

# ── Load feature lists from schema ───────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.feature_schema import get_feature_list

for c in CANDIDATES.values():
    if c["features"] is not None:
        c["feature_list"] = get_feature_list(c["features"])
    else:
        # Load trained feature list from disk (for models trained before schema update)
        fl_path = c["dir"] / "feature_list.json"
        with open(fl_path, "r") as f:
            c["feature_list"] = json.load(f)


def load_models(model_dir):
    """Load all quantile and classifier models from a directory."""
    models = {}
    for q in [10, 25, 50, 75, 90]:
        models[f"upside_q{q}"] = lgb.Booster(model_file=str(model_dir / f"remaining_upside_q{q}.txt"))
        models[f"downside_q{q}"] = lgb.Booster(model_file=str(model_dir / f"remaining_downside_q{q}.txt"))
    for clf_name, file_name in [("upside_zero", "is_upside_zero.txt"), ("downside_zero", "is_downside_zero.txt")]:
        clf_path = model_dir / file_name
        if clf_path.exists():
            models[clf_name] = lgb.Booster(model_file=str(clf_path))
        else:
            models[clf_name] = None
    return models


def predict_quantile(models, X, prefix):
    """Predict quantiles and enforce monotonicity."""
    raw = {}
    for q in [10, 25, 50, 75, 90]:
        raw[f"q{q}"] = models[f"{prefix}_q{q}"].predict(X)
    stacked = np.column_stack([raw[f"q{q}"] for q in [10, 25, 50, 75, 90]])
    stacked.sort(axis=1)
    result = {}
    for i, q in enumerate([10, 25, 50, 75, 90]):
        result[f"q{q}"] = stacked[:, i]
    return result


def predict_classifier(models, X, clf_name):
    """Predict classifier probability."""
    if models.get(clf_name) is None:
        return None
    feat_names = models[clf_name].feature_name()
    X_c = X[feat_names]
    return models[clf_name].predict(X_c)


def compute_metrics(y_true, q_preds, y_binary=None, clf_preds=None):
    """Compute core metrics for a slice."""
    q50 = q_preds["q50"]
    mae = float(np.mean(np.abs(y_true - q50)))
    coverage_80 = float(np.mean((y_true >= q_preds["q10"]) & (y_true <= q_preds["q90"])))
    coverage_50 = float(np.mean((y_true >= q_preds["q25"]) & (y_true <= q_preds["q75"])))

    result = {
        "count": int(len(y_true)),
        "mae": round(mae, 4),
        "coverage_80": round(coverage_80, 4),
        "coverage_50": round(coverage_50, 4),
    }

    if y_binary is not None and clf_preds is not None:
        from sklearn.metrics import roc_auc_score
        y_pred = (clf_preds > 0.5).astype(int)
        accuracy = float(np.mean(y_binary == y_pred))
        fp_rate = float(np.mean((y_pred == 1) & (y_binary == 0))) if len(y_binary) > 0 else 0
        fn_rate = float(np.mean((y_pred == 0) & (y_binary == 1))) if len(y_binary) > 0 else 0
        try:
            auc = float(roc_auc_score(y_binary, clf_preds))
        except Exception:
            auc = None

        result.update({
            "accuracy": round(accuracy, 4),
            "fp_rate": round(fp_rate, 4),
            "fn_rate": round(fn_rate, 4),
            "auc": round(auc, 4) if auc is not None else None,
        })

    return result


def build_slice_masks(test):
    """Build boolean masks for all required slices."""
    hour = test["datetime"].dt.hour
    masks = {
        "all_data": np.ones(len(test), dtype=bool),
        "no_rain_nowcast": test["rain_nowcast_missing_flag"] == 1,
        "any_rain_nowcast_0_120m": test["rain_nc_any_0_120m"] > 0,
        "heavy_rain_nowcast_0_120m": test["rain_nc_heavy_0_120m"] > 0,
        "front_loaded_rain_nowcast": test["rain_nc_front_loaded_ratio"] > 0.5,
        "rain_present": test["rainfall_60m_filled"] > 0,
        "rain_absent": test["rainfall_60m_filled"] == 0,
        "post_peak_rain": test["post_peak_rain_flag"] == 1,
        "morning_peak_then_rain": test["morning_peak_then_rain_flag"] == 1,
        "afternoon_heavy_rain": (hour >= 12) & (hour < 18) & (test["rainfall_60m_filled"] >= 10),
        "rain_after_min_so_far": (test["rainfall_60m_filled"] > 0) & (test["drop_from_max"] > 0.5),
    }
    return masks


def evaluate_candidate(name, candidate, test, slice_masks):
    """Evaluate one candidate across all slices."""
    model_dir = candidate["dir"]
    feature_list = candidate["feature_list"]
    models = load_models(model_dir)

    # Prepare feature matrix
    X = test[feature_list].fillna(0)

    # Predict upside quantiles
    upside_q = predict_quantile(models, X, "upside")
    # Predict downside quantiles
    downside_q = predict_quantile(models, X, "downside")
    # Predict classifiers
    clf_up = predict_classifier(models, X, "upside_zero")
    clf_dn = predict_classifier(models, X, "downside_zero")

    y_up = test["remaining_upside"].values
    y_dn = test["remaining_downside"].values
    y_up_zero = test["is_upside_zero"].values
    y_dn_zero = test["is_downside_zero"].values

    results = {}
    for slice_name, mask in slice_masks.items():
        if mask.sum() < 10:
            results[slice_name] = {"count": int(mask.sum()), "note": "too few samples"}
            continue

        m_up = compute_metrics(y_up[mask], {k: v[mask] for k, v in upside_q.items()},
                               y_up_zero[mask], clf_up[mask] if clf_up is not None else None)
        m_dn = compute_metrics(y_dn[mask], {k: v[mask] for k, v in downside_q.items()},
                               y_dn_zero[mask], clf_dn[mask] if clf_dn is not None else None)

        results[slice_name] = {
            "remaining_upside": m_up,
            "remaining_downside": m_dn,
        }

    return results


def main():
    logger.info("Loading test data...")
    df = pd.read_parquet("data/intraday_ml_train_with_rain_nowcast.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    test = df[df["datetime"] >= "2026-01-01"].copy()
    logger.info(f"Test set: {len(test)} rows, {test['datetime'].min()} -> {test['datetime'].max()}")

    slice_masks = build_slice_masks(test)

    # Print slice sizes
    logger.info("Slice sizes:")
    for name, mask in slice_masks.items():
        logger.info(f"  {name}: {mask.sum()}")

    all_results = {}
    for cand_name, candidate in CANDIDATES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {cand_name} ({len(candidate['feature_list'])} features)")
        logger.info(f"{'='*60}")
        results = evaluate_candidate(cand_name, candidate, test, slice_masks)
        all_results[cand_name] = results

    # ── Summary table ────────────────────────────────────────────
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY — all_data")
    logger.info("=" * 100)
    header = f"{'Candidate':<25} {'up_MAE':>8} {'up_cov80':>9} {'dn_MAE':>8} {'dn_cov80':>9} {'up_acc':>8} {'dn_acc':>8} {'up_AUC':>8} {'dn_AUC':>8}"
    logger.info(header)
    for cand_name, results in all_results.items():
        sl = results.get("all_data", {})
        if "note" in sl:
            continue
        up = sl.get("remaining_upside", {})
        dn = sl.get("remaining_downside", {})
        logger.info(
            f"{cand_name:<25} {up.get('mae','N/A'):>8} {up.get('coverage_80','N/A'):>9} "
            f"{dn.get('mae','N/A'):>8} {dn.get('coverage_80','N/A'):>9} "
            f"{up.get('accuracy','N/A'):>8} {dn.get('accuracy','N/A'):>8} "
            f"{up.get('auc','N/A'):>8} {dn.get('auc','N/A'):>8}"
        )

    # Print key rain slices
    for slice_key in ["any_rain_nowcast_0_120m", "heavy_rain_nowcast_0_120m",
                      "front_loaded_rain_nowcast", "post_peak_rain",
                      "afternoon_heavy_rain", "rain_after_min_so_far"]:
        logger.info(f"\n--- {slice_key} ---")
        for cand_name, results in all_results.items():
            sl = results.get(slice_key, {})
            if "note" in sl:
                logger.info(f"  {cand_name:<25} {sl['note']}")
                continue
            up = sl.get("remaining_upside", {})
            dn = sl.get("remaining_downside", {})
            logger.info(
                f"  {cand_name:<25} up_MAE={up.get('mae','N/A'):>6} up_cov80={up.get('coverage_80','N/A'):>6} "
                f"dn_MAE={dn.get('mae','N/A'):>6} dn_cov80={dn.get('coverage_80','N/A'):>6} "
                f"n={up.get('count','N/A')}"
            )

    # ── Write report ─────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_range": f"{test['datetime'].min()} -> {test['datetime'].max()}",
        "test_rows": len(test),
        "candidates": {k: {"features": len(v["feature_list"])} for k, v in CANDIDATES.items()},
        "results": all_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
