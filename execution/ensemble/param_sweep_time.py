"""Walk-forward sensitivity analysis for the strategy's ENTRY / EXIT time gates.

Goal (user Q3): is the hard-flat-at-15:00 necessary, or should
entry/exit timing be tuned?  We sweep the time gates + edge and check
out-of-sample behaviour so we do NOT overfit to 5-7 days.

Design (anti-overfit):
  * Walk-forward: in-sample = 07-07..07-10 (4 days),
    out-of-sample = 07-11..07-12 (2 days). Best params must hold
    on OUT, not just IN.
  * Report Sharpe + MaxDD + per-day PnL (not just total PnL),
    because total PnL over 4-6 days is too noisy to rank on.
  * We sweep the TIME gates of EnsembleParams (morning_start / risk_reduction_start /
    hard_flat_start) and the top-level edge_threshold, NOT the per-model
    regime_thresholds (those are config-layer, not consumed by EnsembleStrategy).
  * Uses gamma_mid=True (Gamma mid execution) — the pre-CLOB-fix view.
    Re-run with gamma_mid=False once real CLOB depth is available.

Usage:
    python execution/ensemble/param_sweep_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from execution.ensemble.backtest_compare import run_variant
from execution.ensemble.params import EnsembleParams

IN_SAMPLE = ["2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"]
OUT_SAMPLE = ["2026-07-11", "2026-07-12"]

# Time-gate grid.  Keep it small & meaningful (48 combos) to avoid overfit.
RISK_REDUCTION = [13.0, 14.0, 15.0, 16.0]
HARD_FLAT = [14.0, 15.0, 16.0, 17.0]
EDGE = [0.03, 0.05, 0.07]


def _stats(r: dict) -> dict:
    """Pull PnL / Sharpe / MaxDD from a run_variant result."""
    eq = r.get("equity_curve", [])
    peak = 0.0
    max_dd = 0.0
    for pt in eq:
        v = pt.get("total_equity", 0.0)
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return {
        "capital_end": r.get("capital_end", 0.0),
        "total_return": r.get("total_return", 0.0),
        "sharpe": r.get("sharpe", 0.0),
        "max_dd": r.get("max_drawdown", 0.0),
        "trades": r.get("total_trades", 0),
        "day_pnl": r.get("day_pnl", {}),
    }


def _run(label: str, rr: float, hf: float, edge: float, dates: list[str]) -> dict:
    params = EnsembleParams(
        capital=1000.0,
        model_weights={"model_a": 1/3, "model_b": 1/3, "model_c": 1/3},
        gamma_mid=True,
        risk_reduction_start=rr,
        hard_flat_start=hf,
        edge_threshold=edge,
    )
    r = run_variant(label, exclude=[], dates=dates, params=params)
    return _stats(r)


def main():
    rows = []
    print("=== Walk-forward TIME-GATE sweep (gamma_mid) ===")
    print(f"  IN  = {IN_SAMPLE}")
    print(f"  OUT = {OUT_SAMPLE}")
    print()

    total = len(RISK_REDUCTION) * len(HARD_FLAT) * len(EDGE)
    done = 0
    for rr in RISK_REDUCTION:
        for hf in HARD_FLAT:
            for edge in EDGE:
                done += 1
                label = f"rr={rr:02.0f} hf={hf:02.0f} e={edge}"
                try:
                    sin = _run(label, rr, hf, edge, IN_SAMPLE)
                    sout = _run(label, rr, hf, edge, OUT_SAMPLE)
                except Exception as e:
                    print(f"  [{done}/{total}] {label}  ERROR: {e}")
                    continue
                rows.append({
                    "label": label, "rr": rr, "hf": hf, "edge": edge,
                    "in_ret": sin["total_return"], "in_sharpe": sin["sharpe"],
                    "in_dd": sin["max_dd"], "in_pnl": sin["day_pnl"],
                    "out_ret": sout["total_return"], "out_sharpe": sout["sharpe"],
                    "out_dd": sout["max_dd"], "out_pnl": sout["day_pnl"],
                    "in_trades": sin["trades"], "out_trades": sout["trades"],
                })
                if done % 8 == 0 or done == total:
                    print(f"  [{done}/{total}] {label}  in_ret={sin['total_return']*100:+.1f}%  out_ret={sout['total_return']*100:+.1f}%")

    print()
    print("=" * 132)
    print("  RANKED BY OUT-OF-SAMPLE RETURN (walk-forward = honest)")
    print("=" * 132)
    hdr = f"{'Config':<22}{'OUT ret%':>9}{'OUT Shrp':>10}{'OUT DD%':>9}{'IN ret%':>9}{'IN Shrp':>9}  flag"
    print(hdr)
    print("-" * len(hdr))

    rows.sort(key=lambda x: x["out_ret"], reverse=True)
    for r in rows:
        # Overfit flag: IN much better than OUT, or OUT is a loss while IN is a gain.
        flag = ""
        if r["in_ret"] > 0.10 and r["out_ret"] < 0:
            flag = "  <-- OVERFIT? (in +, out -)"
        elif r["out_ret"] - r["in_ret"] < -0.15:
            flag = "  <-- degrades OOS"
        print(f"{r['label']:<22}{r['out_ret']*100:>+8.1f}{r['out_sharpe']:>+10.2f}{r['out_dd']*100:>+8.1f}{r['in_ret']*100:>+8.1f}{r['in_sharpe']:>+9.2f}{flag}")

    print()
    print("  Best OOS configs (top 5) with per-day breakdown:")
    for r in rows[:5]:
        print(f"\n  {r['label']}  (OUT ret={r['out_ret']*100:+.1f}% Sharpe={r['out_sharpe']:.2f} DD={r['out_dd']*100:.1f}%)")
        op = r["out_pnl"]
        ip = r["in_pnl"]
        ins = "  ".join(f"{d[5:]}:{ip.get(d,0):+.0f}" for d in IN_SAMPLE)
        outs = "  ".join(f"{d[5:]}:{op.get(d,0):+.0f}" for d in OUT_SAMPLE)
        print(f"      IN  [{ins}]")
        print(f"      OUT [{outs}]")

    print("\n  NOTE: 07-12 is an in-progress day; OUT window is only 2 days — treat OOS as indicative, not conclusive.")
    print("  Re-run with real CLOB depth (gamma_mid=False) before locking any change.")


if __name__ == "__main__":
    main()
