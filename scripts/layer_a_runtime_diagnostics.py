"""Deployment-safe Layer A runtime checklist.

This command reads only local health/history endpoints and filesystem metadata.
It never prints HF credentials or environment values containing secrets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

HKT = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_a.storage import _timestamp  # noqa: E402

try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize)


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _latest_legacy_timestamp(csv_path: Path) -> str | None:
    latest: datetime | None = None
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp = _timestamp(row.get("timestamp"))
                if timestamp is not None and (latest is None or timestamp > latest):
                    latest = timestamp
    except (OSError, csv.Error):
        return None
    return latest.astimezone(HKT).isoformat() if latest else None


def _frontend_bundle() -> dict[str, Any]:
    assets = sorted((ROOT / "app" / "frontend" / "dist" / "assets").glob("*.js"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not assets:
        return {"present": False, "latest_bundle": None, "sha256": None, "build_timestamp": None, "minute_panel_present": False}
    latest = assets[0]
    payload = latest.read_bytes()
    text = payload.decode("utf-8", errors="ignore")
    return {
        "present": True,
        "latest_bundle": str(latest.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "build_timestamp": datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc).isoformat(),
        "minute_panel_present": "Minute history" in text or "layer-a-minute-history" in text,
    }


def collect(base_url: str, date_value: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actual_server_entrypoint": "uvicorn app.api.server:app --host 0.0.0.0 --port 7860",
        "local_development_command": "uvicorn app.api.server:app --host 0.0.0.0 --port 7860",
        "hf_docker_entrypoint": "uvicorn app.api.server:app --host 0.0.0.0 --port 7860",
        "frontend_bundle": _frontend_bundle(),
        "date": date_value,
    }
    try:
        health = _get_json(base_url, "/api/health")
        layer_a = health.get("layer_a", {})
        result["health_http"] = "ok"
        result["collectors"] = {
            "market_running": layer_a.get("market_collector_running"),
            "weather_running": layer_a.get("weather_collector_running"),
            "market_last_tick": layer_a.get("market_last_tick"),
            "weather_last_tick": layer_a.get("weather_last_tick"),
            "market_last_success": layer_a.get("market_last_success"),
            "weather_last_success": layer_a.get("weather_last_success"),
            "market_last_error": layer_a.get("market_last_error"),
            "weather_last_error": layer_a.get("weather_last_error"),
        }
        result["latest_local_timestamps"] = {
            "market": layer_a.get("last_market_snapshot"),
            "weather": layer_a.get("last_weather_snapshot"),
            "model": layer_a.get("last_model_cycle"),
        }
        result["remote_history"] = {
            "repo_configured": layer_a.get("remote_history", {}).get("repo_configured"),
            "status": layer_a.get("remote_history_status"),
            "last_refresh": layer_a.get("remote_history_last_refresh"),
            "latest_timestamp": layer_a.get("remote_history_latest_timestamp"),
            "files_cached": layer_a.get("remote_history_files_cached"),
            "files_found": layer_a.get("remote_history", {}).get("files_found"),
            "files_downloaded": layer_a.get("remote_history", {}).get("files_downloaded"),
        }
        result["api_schema"] = {
            "health_collector_fields": all(
                key in layer_a
                for key in (
                    "market_collector_running",
                    "weather_collector_running",
                    "market_last_tick",
                    "weather_last_tick",
                )
            ),
        }
        result["chunks"] = {
            "open": layer_a.get("local_minute_chunks_open"),
            "closed": layer_a.get("local_minute_chunks_closed"),
            "oldest_unuploaded": layer_a.get("oldest_unuploaded_chunk"),
        }
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        result["health_http"] = f"unavailable:{type(exc).__name__}"

    try:
        minute = _get_json(base_url, f"/api/history/minute?date={date_value}&limit=10000")
        result["history_api"] = {
            "status": minute.get("status"),
            "row_count_today": minute.get("count", 0),
            "sources": minute.get("sources", []),
            "retrieved_at": minute.get("retrieved_at"),
        }
        result.setdefault("api_schema", {})["minute_response_fields"] = all(
            key in minute for key in ("retrieved_at", "sources", "minutes", "count")
        )
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        result["history_api"] = {"status": f"unavailable:{type(exc).__name__}", "row_count_today": 0}

    legacy_path = ROOT / "data" / "export" / f"{date_value}.csv"
    result["legacy_csv"] = {
        "path": str(legacy_path.relative_to(ROOT)).replace("\\", "/"),
        "exists": legacy_path.exists(),
        "last_timestamp": _latest_legacy_timestamp(legacy_path) if legacy_path.exists() else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LAYER_A_DIAGNOSTIC_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--date", default=datetime.now(timezone.utc).astimezone(HKT).date().isoformat())
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = collect(args.base_url, args.date)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for key, value in result.items():
        print(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
