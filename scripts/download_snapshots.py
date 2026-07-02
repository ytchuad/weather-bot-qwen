"""Download snapshot exports from HF Space and merge into local CSV files.

Usage:
    python scripts/download_snapshots.py https://shea-hilton-weather-prediction.hf.space

Tries the new ``/api/data/export-snapshots`` endpoint first. If not deployed
(404), falls back to ``/api/charts/models-comparison`` to extract whatever
time-series data is available.

Appends only NEW rows (by timestamp+strategy_key) to local ``data/export/`` CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from urllib.request import urlopen

EXPORT_DIR = Path("data/export")
CSV_FIELDS = [
    "timestamp", "snapshot_date", "slug", "strategy_key", "model_key",
    "pm_weighted_temp", "model_predicted_temp", "actual_temp",
    "max_so_far", "predicted_upside", "model_std",
    "position_size", "position_value",
    "all_model_predictions", "context_json",
]
JSON_FIELDS = {"all_model_predictions", "context_json"}


def _load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("timestamp", ""), row.get("strategy_key", "")))
    return keys


def _append_rows(csv_path: Path, rows: list[dict]) -> int:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            r = dict(row)
            for jf in JSON_FIELDS:
                if jf in r and isinstance(r[jf], (dict, list)):
                    r[jf] = json.dumps(r[jf], ensure_ascii=False)
            writer.writerow(r)
    return len(rows)


def _try_export_endpoint(base_url: str) -> list[dict] | None:
    """Try the full export endpoint. Returns None if not available."""
    url = f"{base_url}/api/data/export-snapshots"
    print(f"  Trying {url} ... ", end="", flush=True)
    try:
        resp = urlopen(url, timeout=30)
        payload = json.loads(resp.read().decode())
        print(f"{payload.get('snapshot_count', 0)} snapshots")
        return payload.get("snapshots", [])
    except Exception as e:
        print(f"unavailable ({e})")
        return None


def _try_models_comparison(base_url: str, date: str) -> list[dict]:
    """Fallback: extract per-strategy snapshot rows from models-comparison.

    Returns a list of partial snapshot dicts (only timestamp/pm/actual/models).
    """
    url = f"{base_url}/api/charts/models-comparison?date={date}"
    print(f"    {url} ... ", end="", flush=True)
    try:
        resp = urlopen(url, timeout=30)
        data = json.loads(resp.read().decode())
        timestamps = data.get("timestamps", [])
        if not timestamps:
            print("0 timestamps")
            return []
        print(f"{len(timestamps)} timestamps")
    except Exception as e:
        print(f"failed ({e})")
        return []

    models = data.get("models", {})
    market_temps = data.get("market_temps", [])
    actual_temps = data.get("actual_temps", [])

    # The models-comparison endpoint merges data across strategies.
    # For each timestamp, create one row per strategy (v1 and v2)
    # so the CSV import can reconstruct the time series.
    strategies = ["enhanced_v1_paper", "enhanced_v2_paper"]
    rows: list[dict] = []
    for i, ts in enumerate(timestamps):
        for sk in strategies:
            all_preds = {}
            for mk, vals in models.items():
                if i < len(vals) and vals[i] is not None:
                    all_preds[mk] = vals[i]
            rows.append({
                "timestamp": ts,
                "snapshot_date": date,
                "slug": "",
                "strategy_key": sk,
                "model_key": "",
                "pm_weighted_temp": market_temps[i] if i < len(market_temps) else None,
                "model_predicted_temp": None,
                "actual_temp": actual_temps[i] if i < len(actual_temps) else None,
                "max_so_far": None,
                "predicted_upside": None,
                "model_std": None,
                "position_size": 0,
                "position_value": 0,
                "all_model_predictions": all_preds,
                "context_json": {},
            })
    return rows


def _merge_into_csv(
    new_rows: list[dict], date: str,
) -> int:
    """Append new rows for *date* to the local CSV, deduplicating by
    (timestamp, strategy_key). Returns count of new rows added."""
    csv_path = EXPORT_DIR / f"{date}.csv"
    existing_keys = _load_existing_keys(csv_path)
    fresh = [
        r for r in new_rows
        if (r.get("timestamp", ""), r.get("strategy_key", "")) not in existing_keys
    ]
    if not fresh:
        print(f"    -> 0 new (all {len(existing_keys)} already exist)")
        return 0
    n = _append_rows(csv_path, fresh)
    print(f"    -> +{n} new (total now {len(existing_keys) + n})")
    return n


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_snapshots.py <HF_SPACE_URL>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    print(f"Downloading from {base_url}\n")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    total_new = 0

    # ---- Try export endpoint first ----
    snapshots = _try_export_endpoint(base_url)

    if snapshots is not None:
        # Full export available — group by date and merge
        by_date: dict[str, list[dict]] = {}
        for snap in snapshots:
            d = snap.get("snapshot_date", "unknown")
            by_date.setdefault(d, []).append(snap)
        for date in sorted(by_date.keys()):
            print(f"  {date}: ", end="")
            n = _merge_into_csv(by_date[date], date)
            total_new += n
    else:
        # ---- Fallback: use models-comparison for each date ----
        # HF Spaces only keeps recent data, but try a wide window
        print("  Falling back to models-comparison endpoint per date ...\n")
        for date in ("2026-07-02", "2026-07-01", "2026-06-30", "2026-06-29"):
            print(f"  {date}:")
            rows = _try_models_comparison(base_url, date)
            if rows:
                n = _merge_into_csv(rows, date)
                total_new += n

    print(f"\nDone. {total_new} new snapshot(s) added to {EXPORT_DIR}/")
    if total_new:
        print("Run:  git add data/export/ && git commit -m 'sync snapshot exports'")


if __name__ == "__main__":
    main()