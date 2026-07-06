"""Strategy Snapshot Logger — SQLite-backed time-series persistence.

Records per-cycle snapshots (Polymarket weighted temp, model predicted temp,
actual temp) so the frontend can render strategy performance charts without
re-running models or fetching live APIs.

Usage:
    from features.strategy_snapshot_logger import write_snapshot, read_snapshots

    # write after each strategy cycle
    write_snapshot({
        "timestamp": hkt_now(),
        "snapshot_date": "2026-06-29",
        "slug": "highest-temperature-in-hong-kong-on-june-29-2026",
        "strategy_key": "enhanced_v2_paper",
        "model_key": "model_c",
        "pm_weighted_temp": 30.2,
        "model_predicted_temp": 30.8,
        "actual_temp": 29.5,
        "max_so_far": 29.5,
        "predicted_upside": 1.3,
        "model_std": 1.2,
        "position_size": 42.5,
        "position_value": 21.25,
    })

    # read for chart
    df = read_snapshots("enhanced_v2_paper", date="2026-06-29")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path("data/strategy_snapshots.db")
EXPORT_DIR = Path("data/export")
_LOCAL = threading.local()

CSV_FIELDS = [
    "timestamp", "snapshot_date", "slug", "strategy_key", "model_key",
    "pm_weighted_temp", "model_predicted_temp", "actual_temp",
    "max_so_far", "predicted_upside", "model_std",
    "position_size", "position_value",
    "all_model_predictions", "context_json",
]
JSON_FIELDS = {"all_model_predictions", "context_json"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    snapshot_date   TEXT    NOT NULL,
    slug            TEXT    NOT NULL,
    strategy_key    TEXT    NOT NULL,
    model_key       TEXT    NOT NULL,

    pm_weighted_temp         REAL,
    model_predicted_temp     REAL,
    actual_temp              REAL,
    max_so_far               REAL,
    predicted_upside         REAL,
    model_std                REAL,
    position_size            REAL DEFAULT 0,
    position_value           REAL DEFAULT 0,
    all_model_predictions    TEXT,
    context_json             TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
    ON snapshots(strategy_key, snapshot_date, slug);

CREATE INDEX IF NOT EXISTS idx_snapshots_date
    ON snapshots(snapshot_date);
"""


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_LOCAL, "conn") or _LOCAL.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        import_from_csv(conn)
        _LOCAL.conn = conn
    return _LOCAL.conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing in databases created by older code."""
    _add_column_if_missing(conn, "snapshots", "all_model_predictions", "TEXT")
    _add_column_if_missing(conn, "snapshots", "context_json", "TEXT")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_ts_strat
        ON snapshots(timestamp, strategy_key)
    """)


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.info("Migration: added column %s to %s", column, table)


def write_snapshot(record: dict) -> None:
    conn = _get_conn()
    _insert_row(conn, record)
    conn.commit()
    _append_to_csv(record)


def read_snapshots(
    strategy_key: str | None = None,
    slug: str | None = None,
    date: str | None = None,
    model_key: str | None = None,
    limit: int = 10000,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if strategy_key:
        clauses.append("strategy_key = ?")
        params.append(strategy_key)
    if slug:
        clauses.append("slug = ?")
        params.append(slug)
    if date:
        clauses.append("snapshot_date = ?")
        params.append(date)
    if model_key:
        clauses.append("model_key = ?")
        params.append(model_key)
    where = " AND ".join(clauses) if clauses else "1"
    cursor = conn.execute(
        f"SELECT * FROM snapshots WHERE {where} ORDER BY timestamp ASC LIMIT ?",
        (*params, limit),
    )
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    results = []
    for row in rows:
        r = dict(zip(columns, row))
        if isinstance(r.get("all_model_predictions"), str):
            try:
                r["all_model_predictions"] = json.loads(r["all_model_predictions"])
            except (json.JSONDecodeError, TypeError):
                r["all_model_predictions"] = {}
        if isinstance(r.get("context_json"), str):
            try:
                r["context_json"] = json.loads(r["context_json"])
            except (json.JSONDecodeError, TypeError):
                r["context_json"] = {}
        results.append(r)
    return results


def delete_snapshots(strategy_key: str | None = None, date: str | None = None) -> int:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if strategy_key:
        clauses.append("strategy_key = ?")
        params.append(strategy_key)
    if date:
        clauses.append("snapshot_date = ?")
        params.append(date)
    where = " AND ".join(clauses) if clauses else "1"
    conn.execute(f"DELETE FROM snapshots WHERE {where}", params)
    conn.commit()
    return conn.total_changes


def calc_pm_weighted_temp(
    markets: list[dict],
    prices_dict: dict[str, float] | None = None,
) -> float:
    """Compute Polymarket weighted-average temperature from bucket markets.

    Each market dict must have 'lower', 'upper' (float bounds), and
    'yes_price' (or the price is looked up in *prices_dict* by bucket name).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for m in markets:
        lo = m.get("lower", -np.inf)
        hi = m.get("upper", np.inf)
        bucket = m.get("bucket", "")
        price = (prices_dict or {}).get(bucket, m.get("yes_price", 0.5))
        if price is None or price <= 0:
            continue
        if np.isfinite(lo) and np.isfinite(hi):
            midpoint = (lo + hi) / 2.0
        elif np.isfinite(hi):
            midpoint = hi - 0.5
        elif np.isfinite(lo):
            midpoint = lo + 0.5
        else:
            continue
        weighted_sum += midpoint * price
        total_weight += price
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def calc_model_predicted_temp(
    max_so_far: float | None,
    post_mean: float | None,
) -> float | None:
    return post_mean


def read_models_comparison(
    date: str,
    slug: str | None = None,
    limit: int = 10000,
) -> list[dict]:
    """Read all snapshots for a date and merge model predictions.

    Returns rows with timestamps, market/actual temps, and merged model predictions
    from all strategies.
    """
    conn = _get_conn()
    clauses = ["snapshot_date = ?"]
    params: list[Any] = [date]
    if slug:
        clauses.append("slug = ?")
        params.append(slug)
    where = " AND ".join(clauses)
    cursor = conn.execute(
        f"SELECT timestamp, pm_weighted_temp, actual_temp, all_model_predictions FROM snapshots WHERE {where} ORDER BY timestamp ASC LIMIT ?",
        (*params, limit),
    )
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    merged: dict[str, dict] = {}
    for row in rows:
        r = dict(zip(columns, row))
        ts = r["timestamp"]
        if ts not in merged:
            merged[ts] = {
                "timestamp": ts,
                "pm_weighted_temp": r.get("pm_weighted_temp"),
                "actual_temp": r.get("actual_temp"),
                "model_predictions": {},
            }
        existing = merged[ts]
        if r.get("pm_weighted_temp") is not None:
            existing["pm_weighted_temp"] = r["pm_weighted_temp"]
        if r.get("actual_temp") is not None:
            existing["actual_temp"] = r["actual_temp"]

        preds = r.get("all_model_predictions")
        if isinstance(preds, str):
            try:
                preds = json.loads(preds)
            except (json.JSONDecodeError, TypeError):
                preds = {}
        if isinstance(preds, dict):
            for mk, val in preds.items():
                if mk != "_intraday_error" and val is not None:
                    existing["model_predictions"][mk] = val

    return list(merged.values())


def _insert_row(conn: sqlite3.Connection, record: dict) -> None:
    """Low-level INSERT without CSV/export side-effects."""
    all_preds = record.get("all_model_predictions")
    all_preds_json = json.dumps(all_preds, default=str) if isinstance(all_preds, dict) else (all_preds or "{}")
    ctx = record.get("context_json")
    ctx_json = json.dumps(ctx, default=str) if isinstance(ctx, dict) else (ctx or None)
    conn.execute(
        """INSERT OR IGNORE INTO snapshots
           (timestamp, snapshot_date, slug, strategy_key, model_key,
            pm_weighted_temp, model_predicted_temp, actual_temp,
            max_so_far, predicted_upside, model_std,
            position_size, position_value, all_model_predictions,
            context_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record["timestamp"],
            record["snapshot_date"],
            record["slug"],
            record["strategy_key"],
            record["model_key"],
            _as_float(record.get("pm_weighted_temp")),
            _as_float(record.get("model_predicted_temp")),
            _as_float(record.get("actual_temp")),
            _as_float(record.get("max_so_far")),
            _as_float(record.get("predicted_upside")),
            _as_float(record.get("model_std")),
            _as_float(record.get("position_size", 0)),
            _as_float(record.get("position_value", 0)),
            all_preds_json,
            ctx_json,
        ),
    )


def _append_to_csv(record: dict) -> None:
    """Append one snapshot row to the daily CSV export file."""
    import csv
    date = record.get("snapshot_date", "unknown")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{date}.csv"
    is_new = not path.exists()
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            row = dict(record)
            for jf in JSON_FIELDS:
                if jf in row and isinstance(row[jf], (dict, list)):
                    row[jf] = json.dumps(row[jf], ensure_ascii=False)
            writer.writerow(row)
    except Exception as e:
        logger.warning("Failed to append to CSV %s: %s", path, e)


def import_from_csv(conn: sqlite3.Connection) -> int:
    """Import snapshot rows from CSV exports into SQLite.

    Called once at startup to restore data from git-tracked CSV files.
    Skips rows where (timestamp, strategy_key) already exist.
    """
    import csv
    if not EXPORT_DIR.exists():
        return 0
    count = 0
    for csv_path in sorted(EXPORT_DIR.glob("*.csv")):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for jf in JSON_FIELDS:
                        if jf in row and isinstance(row[jf], str):
                            try:
                                row[jf] = json.loads(row[jf])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    try:
                        _insert_row(conn, row)
                        count += 1
                    except sqlite3.IntegrityError:
                        pass
            conn.commit()
        except Exception as e:
            logger.warning("Failed to import CSV %s: %s", csv_path, e)
    if count:
        logger.info("Imported %d snapshot(s) from CSV exports", count)
    return count


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
