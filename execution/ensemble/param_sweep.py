"""CLOB parameter sweep: find best edge_threshold + hold_behavior + exit_behavior."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.backtest_compare import run_variant, dedup_1sec
from execution.ensemble.params import EnsembleParams
from execution.ensemble.strategy import EnsembleStrategy
from execution.ensemble.strategy_candidate import CandidateEnsembleStrategy

DATES = ['2026-07-05', '2026-07-06', '2026-07-07', '2026-07-08', '2026-07-09']

SWEEP = []

# edge_threshold x hold_behavior x exit_behavior
for edge in [0.05, 0.08, 0.12, 0.16]:
    for hold in ["close_on_no_edge", "never_close"]:
        for exit_b in ["normal", "settlement_only"]:
            label = f"edge={edge} hold={hold} exit={exit_b}"
            cls = EnsembleStrategy
            params = EnsembleParams(
                capital=1000.0,
                edge_threshold=edge,
                hold_behavior=hold,
                exit_behavior=exit_b,
            )
            SWEEP.append((label, cls, params))

# Also add CandidateEnsembleStrategy (baseline C) variants
for edge in [0.05, 0.12]:
    for exit_b in ["normal", "settlement_only"]:
        label = f"C_edge={edge} exit={exit_b}"
        cls = CandidateEnsembleStrategy
        params = EnsembleParams(
            capital=1000.0,
            edge_threshold=edge,
            exit_behavior=exit_b,
        )
        SWEEP.append((label, cls, params))

results = []
print("=== CLOB Parameter Sweep ===")
print()
print("%-36s %8s %8s %6s %6s  %s" % ("Variant", "EndCap", "Return%", "Trds", "Fees", "Daily PnLs"))
print("-" * 100)

for label, cls, params in SWEEP:
    try:
        r = run_variant(label, exclude=[], strategy_cls=cls, params=params,
                        dedup_fn=dedup_1sec, dates=DATES)
        pnl_str = " ".join("%+5.0f" % r["day_pnl"].get(d, 0) for d in DATES)
        print("%-36s %8.2f %8.2f%% %6d %6.2f  %s" % (
            label, r["capital_end"], r["total_return"] * 100,
            r["total_trades"], r["total_fees"], pnl_str))
        results.append((label, r))
    except Exception as e:
        print("%-36s ERROR: %s" % (label, e))

print()
print("Sorted by total return:")
results.sort(key=lambda x: x[1]["total_return"], reverse=True)
for rank, (label, r) in enumerate(results[:10], 1):
    print("  #%d  %-36s %+8.2f%%  end=$%.0f  trades=%d" % (
        rank, label, r["total_return"] * 100, r["capital_end"], r["total_trades"]))
