"""Real-CLOB PnL backtest driver.

Runs ``backtest_compare.run_variant`` over the days that actually CARRY
real Polymarket order-book depth (``market_depth_no``), comparing:
  * gamma_mid=False  -> walk the REAL CLOB book (spread + slippage)
  * gamma_mid=True   -> Gamma mid (no order-book cost, reference only)

Why only 07-11 / 07-12 / 07-13?  Those are the only dates whose export
CSVs contain ``market_depth_no`` (the real CLOB L2 book).  Earlier dates
have only Gamma-mid depth, so a "real orderbook" PnL is impossible there.

NOTE on the zero-inflated fix: the historical snapshots store ``model_probs``
computed at export time (OLD gaussian, no point mass) and never persisted
``prob_max_reached``.  This driver therefore measures the CURRENT (OLD)
strategy's PnL under real CLOB costs.  Measuring the NEW layer requires
re-exporting snapshots through the patched pipeline (see report).

Usage:
    python execution/ensemble/backtest_clob_pnl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.backtest_compare import run_variant, print_comparison
from execution.ensemble.params import EnsembleParams

REAL_CLOB_DATES = ["2026-07-13", "2026-07-14"]
EXCLUDE = ["2026-07-04"]


def main():
    print("=== REAL-CLOB PnL BACKTEST ===")
    print(f"  dates (real orderbook): {REAL_CLOB_DATES}")
    print(f"  capital $1000, model weights a/b/c = 1/3\n")

    results = []

    # (A) gamma_mid=False  -> REAL CLOB walk
    results.append(run_variant(
        "REAL-CLOB (gamma_mid=F)", exclude=EXCLUDE, dates=REAL_CLOB_DATES,
        params=EnsembleParams(capital=1000.0, gamma_mid=False),
        output_dir="output/clob_pnl_real",
    ))

    # (B) gamma_mid=True -> Gamma mid reference
    results.append(run_variant(
        "GAMMA-MID (gamma_mid=T)", exclude=EXCLUDE, dates=REAL_CLOB_DATES,
        params=EnsembleParams(capital=1000.0, gamma_mid=True),
        output_dir="output/clob_pnl_gamma",
    ))

    print_comparison(results)
    print("\n  Per-day PnL (real CLOB):")
    r = results[0]
    for d, v in sorted(r["day_pnl"].items()):
        print(f"    {d}: ${v:+.2f}   fees=${r['total_fees']:.2f}  slip=${r['total_slippage']:.2f}")


if __name__ == "__main__":
    main()
