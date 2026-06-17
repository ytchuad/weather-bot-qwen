# execution/auto_rebalance.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import logging

from features.live_feature_builder import update_forecast_database
from execution.strategy_runner import run_enabled_strategies_once

logging.basicConfig(level=logging.INFO)

def run_auto_rebalance():
    logging.info("=== 啟動背景自動再平衡 ===")
    update_forecast_database()
    results = run_enabled_strategies_once()
    for r in results:
        logging.info(
            "  [%s] %s / %s  %s",
            r.get("status"), r.get("account_id"), r.get("strategy"),
            r.get("error", "")
        )
    return results

def run_auto_rebalance_dry(interval_sec=300):
    """Dry-run variant: does NOT call update_forecast_database.
    Useful for testing or dashboard-driven cycles where data is already fresh.
    """
    return run_enabled_strategies_once(interval_sec=interval_sec)

if __name__ == "__main__":
    run_auto_rebalance()