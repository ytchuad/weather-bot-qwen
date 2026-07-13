"""End-to-end validation: model_2b vs model_2a_v2 on a real snapshot.

Loads one snapshot from data/export, reconstructs the intraday feature kwargs
that model_service would pass to predict_intraday_tmax_all, then compares the
2b and 2a_v2 bucket-probability outputs.

Usage:
    python scripts/validate_model_2b.py [--date 2026-07-07] [--ts-idx 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXE = str(ROOT)


def load_one_snapshot(date: str):
    fp = ROOT / "data" / "export" / f"{date}.csv"
    with open(fp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ctx = json.loads(row["context_json"])
            if ctx.get("model_probs"):
                return row, ctx
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-07-10")
    ap.add_argument("--no-rain", action="store_true", help="force rain features to 0 (degenerate 2b)")
    args = ap.parse_args()

    row, ctx = load_one_snapshot(args.date)
    if row is None:
        print(f"No snapshot with model_probs found for {args.date}")
        return
    print(f"Loaded snapshot {row['timestamp']} from {args.date}")
    print(f"  max_so_far={row.get('max_so_far')} actual_temp={row.get('actual_temp')}")

    # Pull the 9 rainfall features that model_service would have computed live.
    feature_metadata = ctx.get("feature_metadata", {})
    print(f"  feature_metadata keys: {list(feature_metadata.keys())}")

    # We cannot perfectly reconstruct the minute buffer from the CSV, so we
    # build a synthetic but realistic kwargs set and call the public entry
    # points directly. Rain features default to the live values recorded in
    # context_json (fall back to 0).
    rain_keys = ["rainfall_60m", "rainfall_120m", "has_recent_rainfall_obs",
                 "rain_intensity_max_120m", "rain_cooling_60m",
                 "rain_after_max_flag", "post_peak_rain_flag",
                 "rain_data_gap_flag", "rainfall_data_age_minutes"]
    rkw = {k: (ctx.get(k, 0) if k in ctx else 0) for k in rain_keys}
    if args.no_rain:
        rkw = {k: 0 for k in rain_keys}

    max_so_far = float(row["max_so_far"]) if row.get("max_so_far") not in (None, "") else 30.0
    temp_now = max_so_far - 0.5
    hour = int(row["timestamp"][11:13])

    from models.intraday_inference import (
        set_active_model,
        predict_intraday_tmax_model_2a_v2,
        predict_intraday_tmax_model_2b,
    )

    common = dict(
        current_datetime=None,
        max_so_far=max_so_far,
        temp_now=temp_now,
        humidity=70.0,
        min_so_far=max_so_far - 3.0,
        time_since_max=30.0,
        temp_change_30m_pre=0.1,
        temp_change_60m_pre=0.2,
        temp_volatility_60m_pre=0.3,
        temp_acceleration_60m_pre=0.0,
        rh_change_60m_pre=-1.0,
        dew_point_change_60m_pre=0.0,
        dew_point_spread_change_60m_pre=0.0,
        temp_buffer=[temp_now - 0.4, temp_now - 0.2, temp_now],
        rh_buffer=[70.0, 69.0, 70.0],
        forecast_tmax=max_so_far + 1.5,
        forecast_tmin=max_so_far - 5.0,
        hour=hour,
        minute=int(row["timestamp"][14:16]),
    )

    set_active_model("model_2a_v2")
    r2a = predict_intraday_tmax_model_2a_v2(**common)
    set_active_model("model_2b")
    r2b = predict_intraday_tmax_model_2b(**common, **rkw)

    def finite_report(r):
        quants = {k: r.get(k) for k in ("remaining_upside_p10", "remaining_upside_p25",
                                        "remaining_upside_p50", "remaining_upside_p75",
                                        "remaining_upside_p90", "prob_max_reached")}
        return quants

    print("\n[model_2a_v2] remaining_upside quantiles:")
    for k, v in finite_report(r2a).items():
        print(f"    {k:<24} = {v}")
    print("[model_2b] remaining_upside quantiles (rain={}):".format(
        "ZERO" if args.no_rain else "live/default"))
    for k, v in finite_report(r2b).items():
        print(f"    {k:<24} = {v}")

    # Check degeneracy
    allq = [finite_report(r2b)[k] for k in ("remaining_upside_p10", "remaining_upside_p50", "remaining_upside_p90")]
    ok = all(isinstance(v, (int, float)) and np.isfinite(v) for v in allq)
    spread = float(np.max(allq) - np.min(allq))
    print(f"\nDegeneracy check: finite={ok}  upside_spread={spread:.3f}  prob_max_reached={finite_report(r2b)['prob_max_reached']}")

    # Diff vs 2a_v2 median upside
    d = finite_report(r2b)["remaining_upside_p50"] - finite_report(r2a)["remaining_upside_p50"]
    print(f"Median upside delta (2b - 2a_v2): {d:+.3f} degC")

    if not ok or spread < 0.05:
        print("\n[FAIL] 2b produces degenerate / non-finite bucket probs")
        sys.exit(1)
    print("\n[OK] 2b produces finite, non-degenerate upside estimates.")


if __name__ == "__main__":
    main()
