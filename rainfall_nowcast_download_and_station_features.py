from __future__ import annotations

import argparse
import io
import json
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

# ############################################################################
# Config
# ############################################################################

BASE_GET_FILE = "https://app.data.gov.hk/v1/historical-archive/get-file"
ORIGINAL_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"

# HKO station official coordinate
# 22°18'07"N, 114°10'27"E
DEFAULT_HKO_LAT = 22 + 18 / 60 + 7 / 3600
DEFAULT_HKO_LON = 114 + 10 / 60 + 27 / 3600

REQUIRED_COLS = ["issue_time", "valid_time", "lat", "lon", "rain_nowcast_mm"]

# Defensive mapping:
# Supports both official column names and observed renamed/trimmed variants.
COL_MAP = {
    "Updated Date and Time (in Hong Kong Time)": "issue_time",
    "Updated Date and Time (in Hong Kong Time) ": "issue_time",
    "Ending Date and Time (in Hong Kong Time)": "valid_time",
    "Ending Date and Time (in Hong Kong Time) ": "valid_time",
    "Valid time end": "valid_time",
    "Longitude (degree)": "lon",
    "Longitude (degree) ": "lon",
    "lon": "lon",
    "Latitude (degree)": "lat",
    "Latitude (degree) ": "lat",
    "lat": "lat",
    "Half-hourly Nowcast Accumulated Rainfall (mm)": "rain_nowcast_mm",
    "Half-hourly Nowcast Accumulated Rainfall (mm) ": "rain_nowcast_mm",
}


# ############################################################################
# Logging / utility
# ############################################################################

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8")
        ]
    )


def str_to_bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["true", "1", "yes", "y"]:
        return True
    if x in ["false", "0", "no", "n"]:
        return False
    raise ValueError(f"Invalid boolean value: {x}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def resolve_time_key(month: str | None, time_key: str | None) -> str:
    if time_key:
        x = str(time_key).strip()
    elif month:
        x = str(month).strip()
    else:
        raise ValueError("Either --month or --time-key is required.")
    if len(x) == 6 and x.isdigit():
        return x + "01"
    if len(x) == 8 and x.isdigit():
        return x
    raise ValueError(f"Invalid input: {x}. Use YYYYMM or YYYYMMDD.")


def build_archive_url(time_key: str) -> str:
    return BASE_GET_FILE + "?" + urlencode({
        "url": ORIGINAL_URL,
        "time": time_key,
    })


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def parse_hko_time(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(x, format="%Y%m%d%H%M", errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * R * asin(sqrt(a))


# ############################################################################
# Download ZIP
# ############################################################################

def download_zip(
    url: str,
    out_path: Path,
    timeout: int,
    verify_ssl: bool,
    keep_existing: bool = True
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {
            "status": "skip_exists",
            "zip_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "url": url
        }
    logging.info("Downloading: %s", url)
    r = requests.get(url, timeout=timeout, verify=verify_ssl)
    if r.status_code != 200:
        return {
            "status": "failed_http",
            "http_status": r.status_code,
            "url": url,
            "content_length": len(r.content or b""),
            "response_text_head": r.text[:500],
        }
    content = r.content or b""
    import zipfile
    if not zipfile.is_zipfile(io.BytesIO(content)):
        diag_path = out_path.with_suffix(".not_zip_response.bin")
        diag_path.write_bytes(content)
        return {
            "status": "failed_not_zip",
            "http_status": r.status_code,
            "url": url,
            "content_length": len(content),
            "diagnostic_path": str(diag_path),
            "response_text_head": content[:500].decode("utf-8", errors="replace"),
        }
    out_path.write_bytes(content)
    return {
        "status": "downloaded",
        "http_status": r.status_code,
        "zip_path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "url": url
    }


# ############################################################################
# CSV normalization
# ############################################################################

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COL_MAP)
    return df


def normalize_nowcast_df(df: pd.DataFrame, source_member: str) -> pd.DataFrame:
    df = normalize_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_member} missing columns: {missing}; actual: {list(df.columns)}")
    out = df[REQUIRED_COLS].copy()
    out["issue_time"] = parse_hko_time(out["issue_time"])
    out["valid_time"] = parse_hko_time(out["valid_time"])
    out["lat"] = safe_numeric(out["lat"])
    out["lon"] = safe_numeric(out["lon"])
    out["rain_nowcast_mm"] = safe_numeric(out["rain_nowcast_mm"])
    out = out.dropna(subset=REQUIRED_COLS).copy()
    out["lead_minutes"] = (
        out["valid_time"] - out["issue_time"]
    ).dt.total_seconds() / 60
    out = out[out["lead_minutes"].between(0, 180)].copy()
    out["source_member"] = source_member
    return out


# ############################################################################
# Station feature aggregation
# ############################################################################

def find_nearest_grid_points(
    df: pd.DataFrame,
    station_lat: float,
    station_lon: float,
    top_n: int = 20,
) -> pd.DataFrame:
    grid = df[["lat", "lon"]].drop_duplicates().copy()
    grid["dist_km"] = grid.apply(
        lambda r: haversine_km(station_lat, station_lon, r["lat"], r["lon"]),
        axis=1
    )
    grid["station_lat"] = station_lat
    grid["station_lon"] = station_lon
    return grid.sort_values("dist_km").head(top_n).reset_index(drop=True)


def aggregate_station_features_long(
    df: pd.DataFrame,
    station_lat: float,
    station_lon: float,
    radius_km: float,
    heavy_rain_mm: float,
) -> pd.DataFrame:
    box = df[
        df["lat"].between(station_lat - 0.2, station_lat + 0.2) &
        df["lon"].between(station_lon - 0.2, station_lon + 0.2)
    ].copy()
    if box.empty:
        return pd.DataFrame()
    box["dist_km"] = box.apply(
        lambda r: haversine_km(station_lat, station_lon, r["lat"], r["lon"]),
        axis=1
    )
    near = box[box["dist_km"] <= radius_km].copy()
    if near.empty:
        return pd.DataFrame()
    keys = ["issue_time", "valid_time", "lead_minutes"]
    nearest_idx = near.groupby(keys)["dist_km"].idxmin()
    nearest = near.loc[nearest_idx].copy()
    nearest = nearest.rename(
        columns={
            "lat": "nearest_lat",
            "lon": "nearest_lon",
            "dist_km": "nearest_dist_km",
            "rain_nowcast_mm": "rain_nc_nearest_mm",
        }
    )
    agg = near.groupby(keys).agg(
        rain_nc_mean_r5km=("rain_nowcast_mm", "mean"),
        rain_nc_max_r5km=("rain_nowcast_mm", "max"),
        rain_nc_min_r5km=("rain_nowcast_mm", "min"),
        rain_nc_p90_r5km=(
            "rain_nowcast_mm",
            lambda x: float(np.nanpercentile(x, 90)),
        ),
        rain_nc_area_gt0_r5km=(
            "rain_nowcast_mm",
            lambda x: float((x > 0).mean()),
        ),
        rain_nc_area_gt5_r5km=(
            "rain_nowcast_mm",
            lambda x: float((x > heavy_rain_mm).mean()),
        ),
        grid_count_r5km=("rain_nowcast_mm", "size"),
    ).reset_index()
    out = agg.merge(nearest, on=keys, how="left")
    out["station_lat"] = station_lat
    out["station_lon"] = station_lon
    out["radius_km"] = radius_km
    return out


def extract_lead_from_col(col: str) -> int:
    return int(col.split("_")[-1].replace("m", ""))


def pivot_station_features(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    df = long_df.copy()
    df["lead_bucket"] = (df["lead_minutes"] / 30).round().astype(int)
    value_cols = [
        "rain_nc_nearest_mm",
        "rain_nc_mean_r5km",
        "rain_nc_max_r5km",
        "rain_nc_min_r5km",
        "rain_nc_p90_r5km",
        "rain_nc_area_gt0_r5km",
        "rain_nc_area_gt5_r5km",
        "grid_count_r5km"
    ]
    parts = []
    for col in value_cols:
        p = df.pivot_table(
            index="issue_time",
            columns="lead_bucket",
            values=col,
            aggfunc="mean"
        )
        p.columns = [f"{col}_sum_{int(c)*30}m" for c in p.columns]
        parts.append(p)
    wide = pd.concat(parts, axis=1).reset_index()
    mean_cols = [c for c in wide.columns if c.startswith("rain_nc_mean_r5km_sum_")]
    mean_60 = [c for c in mean_cols if extract_lead_from_col(c) <= 60]
    mean_120 = [c for c in mean_cols if extract_lead_from_col(c) <= 120]
    wide["rain_nc_sum_0_60m"] = (
        wide[mean_60].sum(axis=1, skipna=True) if mean_60 else np.nan
    )
    wide["rain_nc_sum_0_120m"] = (
        wide[mean_120].sum(axis=1, skipna=True) if mean_120 else np.nan
    )
    wide["rain_nc_front_loaded_ratio"] = (
        wide["rain_nc_sum_0_60m"] / wide["rain_nc_sum_0_120m"].replace(0, np.nan)
    )
    wide["rain_nc_any_0_120m"] = (
        (wide["rain_nc_sum_0_120m"].fillna(0) > 0).astype(int)
    )
    wide["rain_nc_heavy_0_120m"] = (
        (wide["rain_nc_sum_0_120m"].fillna(0) > 5).astype(int)
    )
    wide["rain_nc_valid_horizon_count"] = (
        wide[mean_cols].notna().sum(axis=1) if mean_cols else 0
    )
    wide["rain_nc_missing_flag"] = (
        (wide["rain_nc_valid_horizon_count"] == 0).astype(int)
    )
    return wide


# ############################################################################
# ZIP member processing
# ############################################################################

def process_zip_members(
    zip_path: Path,
    station_lat: float,
    station_lon: float,
    radius_km: float,
    heavy_rain_mm: float,
    member_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import zipfile
    manifest_rows = []
    long_parts = []
    nearest_rows = []
    with zipfile.ZipFile(zip_path) as z:
        members = sorted([
            m for m in z.namelist()
            if m.lower().endswith("gridded_rainfall_nowcast.csv")
        ])
        if member_limit is not None:
            members = members[:member_limit]
        logging.info("ZIP members selected: %s", len(members))
        for i, member in enumerate(members, start=1):
            row = {
                "zip_path": str(zip_path),
                "member_name": member,
                "member_index": i,
                "status": None,
                "raw_rows": None,
                "normalized_rows": None,
                "feature_rows": None,
                "error": None
            }
            try:
                with z.open(member) as f:
                    raw = pd.read_csv(f, dtype=str)
                norm = normalize_nowcast_df(raw, source_member=member)
                feat = aggregate_station_features_long(
                    norm,
                    station_lat=station_lat,
                    station_lon=station_lon,
                    radius_km=radius_km,
                    heavy_rain_mm=heavy_rain_mm,
                )
                if i == 1:
                    nearest = find_nearest_grid_points(
                        norm, station_lat=station_lat, station_lon=station_lon, top_n=20,
                    )
                    nearest["sample_member"] = member
                    nearest_rows.append(nearest)
                if not feat.empty:
                    feat["source_member"] = member
                    long_parts.append(feat)
                row.update({
                    "status": "processed",
                    "raw_rows": len(raw),
                    "normalized_rows": len(norm),
                    "feature_rows": len(feat),
                })
            except Exception as e:
                row.update({"status": "failed", "error": repr(e)})
            manifest_rows.append(row)
            if i % 100 == 0:
                logging.info("Processed members: %s/%s", i, len(members))
    manifest = pd.DataFrame(manifest_rows)
    long_all = (
        pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    )
    wide_all = (
        pivot_station_features(long_all) if not long_all.empty else pd.DataFrame()
    )
    nearest_all = (
        pd.concat(nearest_rows, ignore_index=True) if nearest_rows else pd.DataFrame()
    )
    return manifest, long_all, wide_all, nearest_all


# ############################################################################
# Main
# ############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download monthly ZIP and build HKO station rainfall nowcast features."
    )
    parser.add_argument("--month", default=None, help="YYYYMM or YYYYMMDD.")
    parser.add_argument("--time-key", default=None, help="Exact DATA.GOV.HK archive key.")
    parser.add_argument("--raw-zip-dir", default="data/raw_zip/rainfall_nowcast")
    parser.add_argument("--out-dir", default="data/features/rainfall_nowcast")
    parser.add_argument("--reports-dir", default="reports/rainfall_nowcast")
    parser.add_argument("--station-lat", type=float, default=DEFAULT_HKO_LAT)
    parser.add_argument("--station-lon", type=float, default=DEFAULT_HKO_LON)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--heavy-rain-mm", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded ZIP.")
    parser.add_argument("--verify-ssl", default="true", help="True/False.")
    parser.add_argument("--member-limit", type=int, default=None, help="Debug: process first N members.")
    args = parser.parse_args()
    verify_ssl = str_to_bool(args.verify_ssl)
    time_key = resolve_time_key(args.month, args.time_key)
    archive_id = time_key[:8]
    raw_zip_dir = Path(args.raw_zip_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    raw_zip_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(reports_dir / f"pipeline_{archive_id}.log")
    url = build_archive_url(time_key)
    zip_path = raw_zip_dir / f"gridded_rainfall_nowcast_{archive_id}.zip"
    logging.info("Resolved time_key: %s", time_key)
    logging.info("Archive URL: %s", url)
    logging.info("Verify SSL: %s", verify_ssl)
    download_result = download_zip(
        url=url, out_path=zip_path, timeout=args.timeout,
        verify_ssl=verify_ssl, keep_existing=True
    )
    logging.info("Download result: %s", download_result)
    if download_result["status"] not in ["downloaded", "skip_exists"]:
        write_json(reports_dir / f"summary_failed_{archive_id}.json", {
            "status": "failed_download", "download_result": download_result,
            "resolved_time_key": time_key,
        })
        raise SystemExit(1)
    manifest, long_all, wide_all, nearest_all = process_zip_members(
        zip_path=zip_path, station_lat=args.station_lat, station_lon=args.station_lon,
        radius_km=args.radius_km, heavy_rain_mm=args.heavy_rain_mm,
        member_limit=args.member_limit,
    )
    manifest_path = reports_dir / f"member_manifest_{archive_id}.parquet"
    manifest.to_parquet(manifest_path, index=False)
    nearest_path = reports_dir / f"nearest_hko_grid_points_{archive_id}.csv"
    nearest_all.to_csv(nearest_path, index=False)
    long_path = None
    if not long_all.empty:
        long_path = out_dir / f"rainfall_nowcast_station_features_long_{archive_id}.parquet"
        long_all.to_parquet(long_path, index=False, compression="zstd")
    wide_path = None
    if not wide_all.empty:
        wide_path = out_dir / f"rainfall_nowcast_station_features_wide_{archive_id}.parquet"
        wide_all.to_parquet(wide_path, index=False, compression="zstd")
    summary = {
        "status": "completed", "archive_id": archive_id,
        "resolved_time_key": time_key, "download_result": download_result,
        "station_lat": args.station_lat, "station_lon": args.station_lon,
        "radius_km": args.radius_km,
        "member_count": int(len(manifest)),
        "member_status_counts": (
            manifest["status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
        ),
        "long_rows": int(len(long_all)), "wide_rows": int(len(wide_all)),
        "manifest_path": str(manifest_path),
        "nearest_grid_path": str(nearest_path) if not nearest_all.empty else None,
        "long_features_path": str(long_path) if long_path else None,
        "wide_features_path": str(wide_path) if wide_path else None,
    }
    write_json(reports_dir / f"summary_{archive_id}.json", summary)
    logging.info("Summary: %s", summary)
    if not args.keep_zip:
        try:
            zip_path.unlink()
            logging.info("Deleted ZIP: %s", zip_path)
        except Exception as e:
            logging.warning("Could not delete ZIP: %s; repr: %s", zip_path, repr(e))


if __name__ == "__main__":
    main()
