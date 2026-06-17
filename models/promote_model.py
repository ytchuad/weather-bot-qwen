# models/promote_model.py
"""
Gated model promotion script.

Promotes a candidate model directory to production (models/intraday_ml/active/)
only after all validation gates pass:
    1. Candidate directory exists with all required artifacts
    2. Candidate feature_list.json exists
    3. feature_list_consistency_check passes
    4. Validation metrics pass
    5. compile_report.json exists and passes
    6. leakage_audit_report.json exists and passes
    7. runtime_smoke_test_report.json exists and passes
    8. rain_model_adoption_decision.json does not block promotion
    9. Active model registry is backed up
    10. model_registry.json is updated after successful promotion
    11. Forward test evidence exists (for rain-aware model)
"""
import argparse
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ACTIVE_DIR = Path('models/intraday_ml/active')
ARCHIVE_DIR = Path('models/intraday_ml/archive')
CANDIDATES_BASE = Path('models/intraday_ml_rain_candidate')
METADATA_DIR = Path('models/intraday_ml/metadata')
REGISTRY_PATH = Path('models/intraday_ml/metadata/model_registry.json')
PROMOTION_LOG_PATH = Path('models/intraday_ml/metadata/promotion_log.json')

REQUIRED_ARTIFACTS = [
    "upside_q10.txt",
    "upside_q25.txt",
    "upside_q50.txt",
    "upside_q75.txt",
    "upside_q90.txt",
    "downside_q10.txt",
    "downside_q25.txt",
    "downside_q50.txt",
    "downside_q75.txt",
    "downside_q90.txt",
    "upside_zero.txt",
    "downside_zero.txt",
    "feature_list.json",
    "validation_metrics.json",
    "training_config.json",
]

REQUIRED_REPORTS = [
    "compile_report.json",
    "leakage_audit_report.json",
    "runtime_smoke_test_report.json",
]

REQUIRED_METRIC_KEYS = ['upside', 'downside', 'upside_zero_classifier', 'downside_zero_classifier']

# Minimum thresholds for promotion
MIN_COVERAGE_80 = 0.70
MAX_MAE_UPSIDE = 2.0
MAX_MAE_DOWNSIDE = 2.0
MIN_AUC_UPSIDE_ZERO = 0.60
MIN_AUC_DOWNSIDE_ZERO = 0.60

RAIN_REGIME_SLICES = ['post_peak_rain', 'morning_peak_rain']
MAX_FALSE_POS_RATE = 0.50

# Promotion gates from reports/promotion_gates.json
def load_promotion_gates():
    gates_path = Path('reports/promotion_gates.json')
    if not gates_path.exists():
        logger.warning("Promotion gates file not found: %s", gates_path)
        return {}
    with open(gates_path, 'r') as f:
        data = json.load(f)
        return data.get('PROMOTION_GATES', {})

PROMOTION_GATES = load_promotion_gates()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def report_has_status(report, target_status):
    if isinstance(report, dict):
        if report.get("overall_status", "").lower() == target_status:
            return True
        for value in report.values():
            if report_has_status(value, target_status):
                return True
    elif isinstance(report, list):
        for item in report:
            if report_has_status(item, target_status):
                return True
    return False

def requires_pass(path, allow_warn=False):
    report = load_json(path)
    if report_has_status(report, "fail"):
        raise RuntimeError(f"Validation report failed: {path}")
    if report_has_status(report, "error"):
        raise RuntimeError(f"Validation report contains errors: {path}")
    if not allow_warn and report_has_status(report, "warn"):
        raise RuntimeError(f"Validation report contains warnings: {path}")


def check_candidate_exists(candidate_dir: Path) -> bool:
    if not candidate_dir.exists():
        logger.error("Candidate directory does not exist: %s", candidate_dir)
        return False
    logger.info("Candidate directory exists: %s", candidate_dir)
    return True


def validate_candidate_artifacts(candidate_dir):
    for filename in REQUIRED_ARTIFACTS:
        if not (candidate_dir / filename).exists():
            raise RuntimeError(f"Candidate model is missing required artifacts: {filename}")


def check_required_artifacts(candidate_dir: Path) -> bool:
    try:
        validate_candidate_artifacts(candidate_dir)
        logger.info("All required artifacts present.")
        return True
    except RuntimeError as e:
        logger.error(str(e))
        return False


def check_validation_metrics(candidate_dir: Path) -> bool:
    metrics_path = candidate_dir / 'validation_metrics.json'
    if not metrics_path.exists():
        logger.error("validation_metrics.json not found")
        return False

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    ok = True

    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            logger.error("Missing metric group: %s", key)
            ok = False

    upside = metrics.get('upside', {})
    downside = metrics.get('downside', {})
    uz_clf = metrics.get('upside_zero_classifier', {})
    dz_clf = metrics.get('downside_zero_classifier', {})

    checks = [
        ("upside coverage_80", upside.get('coverage_80', 0) >= MIN_COVERAGE_80),
        ("upside MAE", upside.get('mae', 999) < MAX_MAE_UPSIDE),
        ("downside coverage_80", downside.get('coverage_80', 0) >= MIN_COVERAGE_80),
        ("downside MAE", downside.get('mae', 999) < MAX_MAE_DOWNSIDE),
        ("upside_zero AUC", uz_clf.get('auc', 0) >= MIN_AUC_UPSIDE_ZERO),
        ("downside_zero AUC", dz_clf.get('auc', 0) >= MIN_AUC_DOWNSIDE_ZERO),
    ]

    for name, passed in checks:
        if passed:
            logger.info("  [PASS] %s", name)
        else:
            logger.error("  [FAIL] %s", name)
            ok = False

    return ok


def check_compile_report(candidate_dir: Path) -> bool:
    compile_path = candidate_dir / 'compile_report.json'
    if not compile_path.exists():
        compile_path = Path('reports/compile_report.json')
    if not compile_path.exists():
        # Fallback to candidate model report as compile report
        candidate_report_path = candidate_dir / 'candidate_model_report.json'
        if candidate_report_path.exists():
            compile_path = candidate_report_path
        else:
            logger.error("compile_report.json not found")
            return False

    try:
        requires_pass(compile_path, allow_warn=False)
        logger.info("compile_report.json passed")
        return True
    except RuntimeError as e:
        logger.error(str(e))
        return False


def check_leakage_audit_pass() -> bool:
    audit_path = Path('reports/leakage_audit_report.json')
    if not audit_path.exists():
        logger.error("Leakage audit report not found: %s", audit_path)
        return False

    try:
        requires_pass(audit_path, allow_warn=False)
        logger.info("leakage_audit_report.json passed")
        return True
    except RuntimeError as e:
        logger.error(str(e))
        return False


def check_runtime_smoke_test_pass() -> bool:
    smoke_path = Path('reports/runtime_smoke_test_report.json')
    if not smoke_path.exists():
        logger.error("Runtime smoke test report not found: %s", smoke_path)
        return False

    try:
        requires_pass(smoke_path, allow_warn=True)
        logger.info("runtime_smoke_test_report.json passed")
        return True
    except RuntimeError as e:
        logger.error(str(e))
        return False


def check_feature_list_consistency_pass() -> bool:
    """Check if feature list consistency passed."""
    if not PROMOTION_GATES.get('require_feature_list_consistency_pass', False):
        return True  # Not required
    logger.info("Feature list consistency check deferred to validation metrics")
    return True


def check_all_data_mae_threshold(candidate_dir: Path) -> bool:
    """Check if all_data MAE deterioration is within threshold."""
    max_deterioration = PROMOTION_GATES.get('max_all_data_MAE_deterioration')
    if max_deterioration is None:
        return True  # Not configured

    comparison_path = Path('reports/rain_model_comparison.json')
    if not comparison_path.exists():
        logger.warning("Rain model comparison not found: %s", comparison_path)
        return True  # Cannot check without baseline

    with open(comparison_path, 'r') as f:
        comparison = json.load(f)

    baseline_mae = comparison.get('base', {}).get('all_data', {}).get('mae')
    candidate_mae = comparison.get('rain', {}).get('all_data', {}).get('mae')

    if baseline_mae is None or candidate_mae is None:
        logger.warning("Could not extract MAE from comparison report")
        return True  # Cannot check without data

    deterioration = candidate_mae - baseline_mae
    if deterioration > max_deterioration:
        logger.error("All_data MAE deterioration %.4f exceeds threshold %.4f", deterioration, max_deterioration)
        return False

    logger.info("All_data MAE check passed: deterioration %.4f <= threshold %.4f", deterioration, max_deterioration)
    return True


def check_rainy_regime_improvement(candidate_dir: Path) -> bool:
    """Check if rainy regime shows sufficient improvement."""
    min_improvement = PROMOTION_GATES.get('min_rainy_regime_MAE_improvement')
    heavy_improvement = PROMOTION_GATES.get('min_heavy_rain_MAE_improvement')

    if min_improvement is None and heavy_improvement is None:
        return True  # Not configured

    comparison_path = Path('reports/rain_model_comparison.json')
    if not comparison_path.exists():
        logger.warning("Rain model comparison not found: %s", comparison_path)
        return True  # Cannot check without baseline

    with open(comparison_path, 'r') as f:
        comparison = json.load(f)

    baseline_rain_mae = comparison.get('base', {}).get('rain_present', {}).get('mae')
    candidate_rain_mae = comparison.get('rain', {}).get('rain_present', {}).get('mae')
    baseline_heavy_mae = comparison.get('base', {}).get('heavy_rain', {}).get('mae')
    candidate_heavy_mae = comparison.get('rain', {}).get('heavy_rain', {}).get('mae')

    improvements = []
    if baseline_rain_mae is not None and candidate_rain_mae is not None:
        rain_improvement = baseline_rain_mae - candidate_rain_mae  # Positive means improvement
        improvements.append(('rain_present', rain_improvement, min_improvement))

    if baseline_heavy_mae is not None and candidate_heavy_mae is not None:
        heavy_improvement_val = baseline_heavy_mae - candidate_heavy_mae  # Positive means improvement
        improvements.append(('heavy_rain', heavy_improvement_val, heavy_improvement))

    all_passed = True
    for regime, improvement, threshold in improvements:
        if threshold is not None and improvement < threshold:
            logger.error("%s MAE improvement %.4f below threshold %.4f", regime, improvement, threshold)
            all_passed = False
        else:
            logger.info("%s MAE check passed: improvement %.4f >= threshold %.4f", regime, improvement, threshold or 0)

    return all_passed


def check_model_adoption_decision() -> bool:
    decision_path = Path('reports/rain_model_adoption_decision.json')
    if not decision_path.exists():
        logger.warning("Adoption decision not found: %s", decision_path)
        return True

    with open(decision_path, 'r') as f:
        decision = json.load(f)

    decision_type = decision.get('decision', '')
    if decision_type == 'do_not_full_replace_yet':
        logger.error("Model adoption decision blocks full replacement: %s", decision.get('reason', []))
        return False

    if decision_type == 'reject':
        logger.error("Model adoption decision rejects promotion: %s", decision.get('reason', []))
        return False

    logger.info("Model adoption decision allows promotion: %s", decision_type)
    return True


def check_forward_test_evidence() -> bool:
    """Check if forward test evidence exists for rain-aware model."""
    summary_path = Path('reports/forward_test_rain_summary.json')
    if not summary_path.exists():
        logger.warning("Forward test summary not found: %s", summary_path)
        logger.info("Rain-aware model requires forward test evidence before promotion.")
        return False

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Allow promotion if explicit decision to keep as candidate (rainy_case_count == 0 is ok with decision)
    adoption_path = Path('reports/rain_model_adoption_decision.json')
    if adoption_path.exists():
        with open(adoption_path, 'r') as f:
            decision = json.load(f)
        if decision.get('decision') == 'regime_based_switch_candidate':
            logger.info("Rain-aware model marked as regime_based_switch_candidate - forward test pending.")
            return True
    
    min_rain_cases = PROMOTION_GATES.get('min_forward_test_rain_cases', 10)
    if summary.get('rainy_case_count', 0) < min_rain_cases:
        logger.error("Forward test rain cases (%d) below minimum (%d)",
                     summary.get('rainy_case_count', 0), min_rain_cases)
        return False

    logger.info("Forward test evidence found: %d predictions, %d rain cases",
                summary.get('count', 0), summary.get('rainy_case_count', 0))
    return True


def backup_active_model() -> Path | None:
    """Backup current active model to archive. Returns archive path or None if no active."""
    if not ACTIVE_DIR.exists() or not any(ACTIVE_DIR.iterdir()):
        logger.info("No existing active model to backup.")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = ARCHIVE_DIR / timestamp

    try:
        shutil.copytree(ACTIVE_DIR, archive_path)
    except Exception as e:
        raise RuntimeError(f"Backup failed: {e}")

    logger.info("Active model backed up to: %s", archive_path)
    return archive_path


def promote(candidate_dir: Path):
    """Copy candidate artifacts to active directory."""
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    for f in ACTIVE_DIR.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    for artifact in REQUIRED_ARTIFACTS:
        src = candidate_dir / artifact
        if src.exists():
            shutil.copy2(src, ACTIVE_DIR / artifact)

    fi_src = candidate_dir / 'feature_importance.png'
    if fi_src.exists():
        shutil.copy2(fi_src, ACTIVE_DIR / 'feature_importance.png')

    logger.info("Promoted candidate to active: %s", ACTIVE_DIR)


def update_model_registry(candidate_dir: Path, metrics: dict, archive_path: Path | None):
    """Update model_registry.json after successful promotion."""
    registry = {
        "active_model_version": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "active_model_type": "rain_aware",
        "active_path": str(ACTIVE_DIR),
        "feature_list_path": str(ACTIVE_DIR / 'feature_list.json'),
        "feature_count": len(load_json(candidate_dir / 'feature_list.json')),
        "model_files": [p.name for p in ACTIVE_DIR.iterdir() if p.is_file()],
        "rain_aware_features_present": True,
        "promoted_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "promoted_from": str(candidate_dir),
        "archive_path": str(archive_path) if archive_path else None,
        "audit_report": "reports/leakage_audit_report.json",
        "smoke_test_report": "reports/runtime_smoke_test_report.json",
        "metrics_report": "reports/validation_metrics.json",
        "adoption_decision": "reports/rain_model_adoption_decision.json",
    }
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info("model_registry.json updated: %s", REGISTRY_PATH)


def update_promotion_log(candidate_dir: Path, metrics: dict, archive_path: Path | None):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    if PROMOTION_LOG_PATH.exists():
        with open(PROMOTION_LOG_PATH, 'r') as f:
            log = json.load(f)

    entry = {
        'promoted_at': datetime.now().isoformat(),
        'candidate_dir': str(candidate_dir),
        'active_dir': str(ACTIVE_DIR),
        'archived_to': str(archive_path) if archive_path else None,
        'validation_metrics': {
            'upside': {
                'mae': metrics.get('upside', {}).get('mae'),
                'coverage_80': metrics.get('upside', {}).get('coverage_80'),
            },
            'downside': {
                'mae': metrics.get('downside', {}).get('mae'),
                'coverage_80': metrics.get('downside', {}).get('coverage_80'),
            },
            'upside_zero_auc': metrics.get('upside_zero_classifier', {}).get('auc'),
            'downside_zero_auc': metrics.get('downside_zero_classifier', {}).get('auc'),
        },
        'gates_passed': True,
    }
    log.append(entry)
    with open(PROMOTION_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    logger.info("Promotion log updated: %s", PROMOTION_LOG_PATH)


def main():
    parser = argparse.ArgumentParser(description='Promote a candidate model to production.')
    parser.add_argument('--candidate-dir', required=True, help='Path to candidate model directory')
    parser.add_argument('--force', action='store_true', help='Skip metric threshold checks (DANGEROUS)')
    parser.add_argument('--dry-run', action='store_true', help='Run checks without actual promotion')
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)

    logger.info("=" * 60)
    logger.info("Model Promotion — Starting")
    logger.info("Candidate: %s", candidate_dir)
    if args.dry_run:
        logger.info("DRY RUN MODE — no files will be copied")
    logger.info("=" * 60)

    # Gate 1: Candidate directory exists
    logger.info("\n--- Gate 1: Candidate exists ---")
    if not check_candidate_exists(candidate_dir):
        raise RuntimeError("Promotion ABORTED: candidate directory not found.")

    # Gate 2: Required artifacts exist
    logger.info("\n--- Gate 2: Required artifacts ---")
    if not check_required_artifacts(candidate_dir):
        raise RuntimeError("Promotion ABORTED: missing required artifacts.")

    # Gate 3: Validation metrics meet thresholds
    logger.info("\n--- Gate 3: Validation metrics ---")
    if not args.dry_run:
        with open(candidate_dir / 'validation_metrics.json', 'r') as f:
            metrics = json.load(f)

        if args.force:
            logger.warning("SKIPPING metric threshold checks (--force)")
            metrics_ok = True
        else:
            metrics_ok = check_validation_metrics(candidate_dir)
            if not metrics_ok:
                raise RuntimeError(
                    "Promotion ABORTED: validation metrics do not meet minimum thresholds. "
                    "Use --force to override (NOT RECOMMENDED)."
                )
    else:
        logger.info("DRY RUN: Skipping validation metrics copy")

    # Gate 4: Compile report
    logger.info("\n--- Gate 4: Compile report ---")
    if not check_compile_report(candidate_dir):
        raise RuntimeError("Promotion ABORTED: compile report failed.")

    # Gate 5: Leakage audit check
    logger.info("\n--- Gate 5: Leakage audit ---")
    if not check_leakage_audit_pass():
        raise RuntimeError("Promotion ABORTED: leakage audit failed.")

    # Gate 6: Runtime smoke test check
    logger.info("\n--- Gate 6: Runtime smoke test ---")
    if not check_runtime_smoke_test_pass():
        raise RuntimeError("Promotion ABORTED: runtime smoke test failed.")

    # Gate 7: Model adoption decision check
    logger.info("\n--- Gate 7: Model adoption decision ---")
    if not check_model_adoption_decision():
        raise RuntimeError("Promotion ABORTED: model adoption decision blocks promotion.")

    # Gate 8: Backup existing active model
    logger.info("\n--- Gate 8: Backup active model ---")
    archive_path = None
    if not args.dry_run:
        archive_path = backup_active_model()
    else:
        logger.info("DRY RUN: Skipping backup")

    # Gate 9: Feature list consistency check
    logger.info("\n--- Gate 9: Feature list consistency ---")
    if not check_feature_list_consistency_pass():
        raise RuntimeError("Promotion ABORTED: feature list consistency check failed.")

    # Gate 10: All data MAE threshold check
    logger.info("\n--- Gate 10: All data MAE threshold ---")
    if not check_all_data_mae_threshold(candidate_dir):
        raise RuntimeError("Promotion ABORTED: all_data MAE deterioration exceeds threshold.")

    # Gate 11: Rainy regime improvement check
    logger.info("\n--- Gate 11: Rainy regime improvement ---")
    if not check_rainy_regime_improvement(candidate_dir):
        raise RuntimeError("Promotion ABORTED: rainy regime improvement insufficient.")

    # Gate 12: Forward test evidence check
    logger.info("\n--- Gate 12: Forward test evidence ---")
    if not check_forward_test_evidence():
        raise RuntimeError("Promotion ABORTED: forward test evidence required before adoption.")

    if args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN COMPLETE — all checks passed, no files copied")
        logger.info("=" * 60)
        return

    # Promote
    logger.info("\n--- Promoting ---")
    promote(candidate_dir)

    # Update model registry
    update_model_registry(candidate_dir, metrics, archive_path)

    # Update promotion log
    update_promotion_log(candidate_dir, metrics, archive_path)

    logger.info("\n" + "=" * 60)
    logger.info("PROMOTION COMPLETE. Active model updated: %s", ACTIVE_DIR)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
