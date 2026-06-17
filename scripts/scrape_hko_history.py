"""Scrape minute-level temperature & RH from i-lens.hk (2016-12-08 to 2026-06-10).

Resumes from last checkpoint. Uses ThreadPoolExecutor(10) for concurrent fetches.
Output: data/hko_history.parquet with columns: datetime, date, time, temp, rh.
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "hko_chunks"
OUTPUT = ROOT / "data" / "hko_history.parquet"
CHECKPOINT = ROOT / "data" / "hko_history_checkpoint.txt"
ERROR_LOG = ROOT / "data" / "hko_history_errors.txt"

HKT = ZoneInfo("Asia/Hong_Kong")
BASE_URL = "https://i-lens.hk/hkweather/history_chart.php"
DELAY = 0.3  # seconds between requests per worker
CHUNK_SIZE = 35
MAX_WORKERS = 10
START = date(2016, 12, 8)
END = date(2026, 6, 10)

ENTRY_PATTERN = re.compile(
    r"\[Date\.UTC\((\d+),(\d+),(\d+),(\d+),(\d+)\),([\d.]+)\]"
)
DATA_ARRAY_PATTERN = re.compile(r"data:\s*\[")


def _match_to_datetime_val(m) -> tuple:
    """Convert a regex match of [Date.UTC(y,m,d,H,M),val] → (HKT_datetime, value)."""
    year = int(m.group(1))
    month = int(m.group(2)) + 1
    day = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    value = float(m.group(6))
    dt = datetime(year, month, day, hour, minute, tzinfo=HKT)
    return dt, value


def fetch_and_parse(single_date: date) -> pd.DataFrame | None:
    """Fetch one day's page and return DataFrame with temp & RH."""
    date_str = single_date.isoformat()
    params = {"date": date_str, "chart_type": ""}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("HTTP error for %s: %s", date_str, e)
        return None

    html = resp.text

    # Locate data: [ positions to split temp vs RH series
    data_starts = [m.end() for m in DATA_ARRAY_PATTERN.finditer(html)]
    if len(data_starts) < 2:
        logger.warning("Could not find two data arrays for %s", date_str)
        return None

    all_entries = list(ENTRY_PATTERN.finditer(html))
    first_end = data_starts[1]
    temp_entries = [e for e in all_entries if e.start() < first_end]
    rh_entries  = [e for e in all_entries if e.start() >= first_end]

    if not temp_entries and not rh_entries:
        logger.warning("No Date.UTC entries found for %s", date_str)
        return None

    temp_dts, temp_vals = [], []
    for e in temp_entries:
        dt, val = _match_to_datetime_val(e)
        temp_dts.append(dt)
        temp_vals.append(val)

    rh_dts, rh_vals = [], []
    for e in rh_entries:
        dt, val = _match_to_datetime_val(e)
        rh_dts.append(dt)
        rh_vals.append(val)

    parts = []
    if temp_vals:
        parts.append(pd.DataFrame({"datetime": temp_dts, "temp": temp_vals}))
    if rh_vals:
        parts.append(pd.DataFrame({"datetime": rh_dts, "rh": rh_vals}))

    df = parts[0] if len(parts) == 1 else pd.merge(parts[0], parts[1], on="datetime", how="outer")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = date_str
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    return df


def process_chunk(chunk_dates: list[date], chunk_id: int) -> list[str]:
    """Process a chunk of dates, return list of warnings/errors."""
    errors = []
    chunk_dfs = []
    for d in chunk_dates:
        date_str = d.isoformat()
        logger.info("  [chunk %d] %s", chunk_id, date_str)
        try:
            df = fetch_and_parse(d)
            if df is not None and not df.empty:
                chunk_dfs.append(df)
        except Exception as e:
            msg = f"{date_str}: {e}"
            logger.warning(msg)
            errors.append(msg)
        time.sleep(DELAY)

    if chunk_dfs:
        combined = pd.concat(chunk_dfs, ignore_index=True)
        out_path = CHUNKS_DIR / f"chunk_{chunk_id:04d}.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("  [chunk %d] wrote %d rows → %s", chunk_id, len(combined), out_path.name)
    else:
        logger.info("  [chunk %d] no data", chunk_id)

    return errors


def merge_chunks():
    """Merge all chunk parquet files into final output."""
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.parquet"))
    if not chunk_files:
        logger.info("No chunk files to merge")
        return

    dfs = []
    for f in chunk_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning("Failed to read %s: %s", f.name, e)

    if not dfs:
        return

    final = pd.concat(dfs, ignore_index=True)
    final = final.sort_values("datetime").reset_index(drop=True)

    # Ensure proper column order
    final = final[["datetime", "date", "time", "temp", "rh"]]

    # Append to existing if present
    if OUTPUT.exists():
        try:
            existing = pd.read_parquet(OUTPUT)
            final = pd.concat([existing, final], ignore_index=True)
            final = final.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        except Exception:
            pass

    final.to_parquet(OUTPUT, index=False)
    logger.info("Merged %d total rows → %s", len(final), OUTPUT.name)


def read_checkpoint() -> date:
    """Read last successfully scraped date from checkpoint file."""
    if CHECKPOINT.exists():
        raw = CHECKPOINT.read_text().strip()
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return START


def write_checkpoint(d: date):
    CHECKPOINT.write_text(d.isoformat())


def load_errors() -> set[str]:
    if ERROR_LOG.exists():
        return set(ERROR_LOG.read_text().strip().splitlines())
    return set()


def save_errors(errors: list[str], previous: set[str]):
    all_errors = previous | set(errors)
    ERROR_LOG.write_text("\n".join(sorted(all_errors)))


def main():
    parser = argparse.ArgumentParser(description="Scrape HKO temperature & RH history")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, scrape from start")
    parser.add_argument("--start", type=str, default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Override end date YYYY-MM-DD")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else (START if args.force else read_checkpoint())
    end_date = date.fromisoformat(args.end) if args.end else END

    # If checkpoint is already at or past end_date, skip
    if not args.force and start_date >= end_date:
        logger.info("Checkpoint %s >= end %s, nothing to do", start_date, end_date)
        return

    # Build date list
    all_dates: list[date] = []
    d = start_date
    while d <= end_date:
        all_dates.append(d)
        d += timedelta(days=1)

    if not all_dates:
        logger.info("No dates to scrape")
        return

    logger.info("Scraping %d dates from %s to %s", len(all_dates), all_dates[0], all_dates[-1])
    logger.info("Workers=%d  ChunkSize=%d  Delay=%.1fs", MAX_WORKERS, CHUNK_SIZE, DELAY)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Split into chunks
    chunks = [all_dates[i:i + CHUNK_SIZE] for i in range(0, len(all_dates), CHUNK_SIZE)]
    logger.info("Split into %d chunks", len(chunks))

    previous_errors = load_errors()
    all_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_chunk, chunk, i): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                errs = future.result()
                all_errors.extend(errs)
            except Exception as e:
                msg = f"chunk_{chunk_id}: {e}"
                logger.error(msg)
                all_errors.append(msg)

    save_errors(all_errors, previous_errors)

    # Merge chunks
    merge_chunks()

    # Update checkpoint to the last date
    write_checkpoint(all_dates[-1])
    logger.info("Checkpoint updated to %s", all_dates[-1])

    # Clean up chunks
    for f in CHUNKS_DIR.glob("chunk_*.parquet"):
        f.unlink()
    try:
        CHUNKS_DIR.rmdir()
    except OSError:
        pass

    n_err = len(all_errors)
    if n_err:
        logger.warning("Completed with %d errors (see %s)", n_err, ERROR_LOG)
    else:
        logger.info("All done — no errors")


if __name__ == "__main__":
    main()
