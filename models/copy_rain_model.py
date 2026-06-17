# models/copy_rain_model.py
# -------------------------
# DEPRECATED AND DISABLED.
#
# This script was intentionally disabled because direct model copying can bypass:
# - validation metrics
# - leakage audit
# - smoke tests
# - feature list consistency checks
# - artifact verification
# - model registry update
#
# Use models/promote_model.py instead:
#   python models/promote_model.py --candidate-dir models/intraday_ml_rain_candidate/<timestamp>

raise RuntimeError(
    "models/copy_rain_model.py is deprecated and disabled. "
    "Use models/promote_model.py --candidate-dir <candidate_dir> after validation gates pass."
)
