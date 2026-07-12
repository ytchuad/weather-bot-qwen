"""Walk-forward comparison of STRATEGY VARIANTS (simulate before changing code).

Goal: before committing any change to ``strategy.py``, simulate different
strategy behaviours on the SAME fixed observations (model_probs + market_prices
+ max_so_far already stored in every snapshot) and compare PnL honestly.

Variants
--------
  V0  BASELINE (current behaviour): partial_reduction=False.  RISK_REDUCTION
      == full flat at rr.  Sweep rr in {13,14,15,16}, edge in {0.03,0.05,0.07}.
      This is the control group and MUST match param_sweep_time.py.
  V1  LINEAR TAPER: partial_reduction=True.  RISK_REDUCTION tapers exposure
      linearly from rr to hf (positions reduced, not force-flatted); HARD_FLAT
      still fully closes.  Sweep rr in {13,14,15}, hf in {16,17}, edge=0.05.
  V2  HOLD-TO-CLOSE: push risk_reduction_start to 23.9 so the strategy never
      intraday-flattens — it holds to settlement.  edge in {0.03,0.05}.
      Tests the upside of NOT cutting at 14:00.

Walk-forward:  IN = 07-07..07-10 (4d),  OUT = 07-11..07-12 (2d).
All variants use gamma_mid=True (pre-CLOB-fix view).  Re-run winners with
gamma_mid=False once real CLOB depth (07-13) is available.

Usage:
    python execution/ensemble/sim_strategy_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.backtest_compare import run_variant
from execution.ensemble.params import EnsembleParams

IN_SAMPLE = ["2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"]
OUT_SAMPLE = ["2026-07-11", "2026-07-12"]

WEIGHTS = {"model_a": 1/3, "model_b": 1/3, "model_c": 1/3}


def _stats(r: dict) -> dict:
    return {
        "ret": r.get("total_return", 0.0),
        "sharpe": r.get("sharpe", 0.0),
        "dd": r.get("max_drawdown", 0.0),
        "trades": r.get("total_trades", 0),
        "day_pnl": r.get("day_pnl", {}),
    }


def _run(label: str, params: EnsembleParams, dates: list[str]) -> dict:
    return _stats(run_variant(label, exclude=[], dates=dates, params=params))


def _mk(rr: float, hf: float, edge: float, partial: bool) -> EnsembleParams:
    return EnsembleParams(
        capital=1000.0,
        model_weights=WEIGHTS,
        gamma_mid=True,
        risk_reduction_start=rr,
        hard_flat_start=hf,
        edge_threshold=edge,
        partial_reduction=partial,
    )


def _flag(sin: dict, sout: dict) -> str:
    """Overfit / single-day-spike warnings."""
    if sin["ret"] > 0.10 and sout["ret"] < 0:
        return "  <-- OVERFIT (in +, out -)"
    if sout["ret"] - sin["ret"] < -0.15:
        return "  <-- degrades OOS"
    # single-day spike: OUT total dominated by one day (07-12 is in-progress)
    op = sout["day_pnl"]
    if op:
        tot = sum(op.values())
        mx = max(op.values(), key=abs) if op else 0
        if tot != 0 and abs(mx) > 0.85 * abs(tot) and tot > 0:
            return "  <-- OUT is ~1-day spike (fragile)"
    return ""


def _emit(rows: list[dict]):
    hdr = (f"{'Variant':<28}{'OUTret%':>8}{'OUTShrp':>9}{'OUTDD%':>8}"
           f"{'INret%':>8}{'INShrp':>8}{'Trd':>5}  flag")
    print(hdr)
    print("-" * len(hdr))
    rows.sort(key=lambda x: x["sout"]["ret"], reverse=True)
    for r in rows:
        sin, sout = r["sin"], r["sout"]
        print(f"{r['label']:<28}{sout['ret']*100:>+7.1f}{sout['sharpe']:>+9.2f}"
              f"{sout['dd']*100:>+7.1f}{sin['ret']*100:>+7.1f}{sin['sharpe']:>+8.2f}"
              f"{sout['trades']:>5}{_flag(sin, sout)}")


def _per_day(rows: list[dict], top: int = 3):
    print("\n  Top OOS per-day PnL breakdown:")
    rows_sorted = sorted(rows, key=lambda x: x["sout"]["ret"], reverse=True)
    for r in rows_sorted[:top]:
        ip, op = r["sin"]["day_pnl"], r["sout"]["day_pnl"]
        ins = "  ".join(f"{d[5:]}:{ip.get(d, 0):+.0f}" for d in IN_SAMPLE)
        outs = "  ".join(f"{d[5:]}:{op.get(d, 0):+.0f}" for d in OUT_SAMPLE)
        print(f"\n  {r['label']}  (OUT {r['sout']['ret']*100:+.1f}% "
              f"Sharpe {r['sout']['sharpe']:.2f} DD {r['sout']['dd']*100:.1f}%)")
        print(f"      IN  [{ins}]")
        print(f"      OUT [{outs}]")


def block(title: str, specs: list[tuple[str, EnsembleParams]]):
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)
    rows = []
    for label, params in specs:
        sin = _run(label, params, IN_SAMPLE)
        sout = _run(label, params, OUT_SAMPLE)
        rows.append({"label": label, "sin": sin, "sout": sout})
    _emit(rows)
    _per_day(rows)
    return rows


def main():
    print("=== STRATEGY VARIANT walk-forward simulation (gamma_mid) ===")
    print(f"  IN  = {IN_SAMPLE}")
    print(f"  OUT = {OUT_SAMPLE}")

    # V0 baseline (control) — must match param_sweep_time.py
    v0 = [
        (f"V0 rr={rr:02.0f} e={edge}", _mk(rr, 17.0, edge, partial=False))
        for rr in (13.0, 14.0, 15.0, 16.0)
        for edge in (0.03, 0.05, 0.07)
    ]
    block("V0 BASELINE (full-flat at rr) — control, should match param_sweep_time", v0)

    # V1 linear taper
    v1 = [
        (f"V1 rr={rr:02.0f} hf={hf:02.0f} taper", _mk(rr, hf, 0.05, partial=True))
        for rr in (13.0, 14.0, 15.0)
        for hf in (16.0, 17.0)
    ]
    block("V1 LINEAR TAPER (partial_reduction, rr->hf), edge=0.05", v1)

    # V2 hold-to-close
    v2 = [
        (f"V2 hold-to-close e={edge}", _mk(23.9, 23.95, edge, partial=False))
        for edge in (0.03, 0.05)
    ]
    block("V2 HOLD-TO-CLOSE (no intraday flatten)", v2)

    print("\n  NOTE: 07-12 is in-progress; OUT window is 2 days — indicative, not conclusive.")
    print("  Re-run winners with gamma_mid=False after real CLOB depth (07-13) lands.")


if __name__ == "__main__":
    main()
