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


def _csv_has_real_data(csv_path: Path) -> bool:
    """True if the CSV already has any row with non-empty context_json."""
    if not csv_path.exists():
        return False
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ctx = row.get("context_json", "")
            if ctx and ctx.strip() not in ("", "{}"):
                return True
    return False


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


def _try_export_endpoint(base_url: str, date: str | None = None) -> list[dict] | None:
    """Try the per-date export endpoint. Returns None if not available.

    IMPORTANT: the unfiltered endpoint applies ``ORDER BY timestamp ASC
    LIMIT 10000`` server-side, so once the total snapshot count exceeds
    10000 it silently drops the NEWEST rows (everything past row 10000).
    Querying with an explicit ``date`` keeps each response under the limit
    and avoids that truncation.
    """
    url = f"{base_url}/api/data/export-snapshots"
    if date:
        url += f"?date={date}"
    print(f"  Trying {url} ... ", end="", flush=True)
    try:
        resp = urlopen(url, timeout=180)
        payload = json.loads(resp.read().decode())
        snaps = payload.get("snapshots", [])
        print(f"{len(snaps)} snapshots")
        return snaps
    except Exception as e:
        print(f"unavailable ({e})")
        return None


def _recent_dates(days_back: int = 21) -> list[str]:
    """Dates to poll, oldest→newest, covering the recent window.

    A fixed window (not derived from local latest) is intentional: the
    unfiltered endpoint truncates, so the local "latest" can be stale and
    must not be trusted as the lower bound.
    """
    from datetime import date, timedelta
    today = date.today()
    return [(today - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days_back, -1, -1)]


def _try_models_comparison(base_url: str, date: str) -> list[dict]:
    """Fallback: extract per-strategy snapshot rows from models-comparison.

    Returns a list of partial snapshot dicts (only timestamp/pm/actual/models).
    """
    url = f"{base_url}/api/charts/models-comparison?date={date}"
    print(f"    {url} ... ", end="", flush=True)
    try:
        resp = urlopen(url, timeout=180)
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
    new_rows: list[dict], date: str, *, from_fallback: bool = False,
) -> int:
    """Append new rows for *date* to the local CSV, deduplicating by
    (timestamp, strategy_key). Returns count of new rows added.

    When *from_fallback* is True and the existing CSV already contains
    rows with non-empty *context_json*, the fallback rows are skipped
    (they lack market_prices and would dilute real data).
    """
    csv_path = EXPORT_DIR / f"{date}.csv"
    if from_fallback and _csv_has_real_data(csv_path):
        print(f"    -> skipped (CSV already has real context_json)")
        return 0
    existing_keys = _load_existing_keys(csv_path)
    fresh = [
        r for r in new_rows
        if (r.get("timestamp", ""), r.get("strategy_key", "")) not in existing_keys
        and "T" in (r.get("timestamp", ""))  # skip malformed date-only rows
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

    # ---- Primary path: per-date export endpoint ----
    # The unfiltered endpoint silently truncates at 10000 rows (oldest kept),
    # so we poll each recent date explicitly. Each day's volume is well under
    # the limit, guaranteeing no rows are dropped.
    dates = _recent_dates()
    print(f"Polling per-date export endpoint for {len(dates)} dates ...\n")
    any_available = False
    for date in dates:
        snaps = _try_export_endpoint(base_url, date)
        if snaps is None:
            continue
        any_available = True
        if snaps:
            n = _merge_into_csv(snaps, date)
            total_new += n

    if not any_available:
        # ---- Fallback: models-comparison per date ----
        print("\nExport endpoint unavailable — falling back to models-comparison ...\n")
        for date in _recent_dates(8):
            print(f"  {date}:")
            rows = _try_models_comparison(base_url, date)
            if rows:
                n = _merge_into_csv(rows, date, from_fallback=True)
                total_new += n

    print(f"\nDone. {total_new} new snapshot(s) added to {EXPORT_DIR}/")
    if total_new:
        print("Run:  git add data/export/ && git commit -m 'sync snapshot exports'")


if __name__ == "__main__":
    main()