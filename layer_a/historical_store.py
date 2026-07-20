"""Read-only remote/local Layer A history with rebuild-safe caching."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .market_schema import validate_market_snapshot
from .market_storage import MarketSnapshotStore
from .minute_view import build_minute_view
from .schema import jsonable, validate_layer_a_record
from .storage import LayerAStore, _timestamp
from .weather_schema import validate_weather_snapshot
from .weather_storage import WeatherSnapshotStore

logger = logging.getLogger(__name__)
HKT = timezone(timedelta(hours=8))
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize)


def _legacy_csv_root() -> Path:
    configured = os.getenv("LAYER_A_LEGACY_CSV_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        from app.config import LAYER_A_LEGACY_CSV_DIR

        return Path(LAYER_A_LEGACY_CSV_DIR)
    except ImportError:
        return Path(__file__).resolve().parents[1] / "data" / "export"


def _csv_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _csv_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _date_strings(
    *,
    date_value: str | None = None,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 7,
) -> set[str]:
    if date_value:
        return {date_value}
    parsed_start = _timestamp(start) if start else None
    parsed_end = _timestamp(end) if end else None
    if parsed_start is not None or parsed_end is not None:
        first = (parsed_start or parsed_end).astimezone(HKT).date()
        last = (parsed_end or parsed_start).astimezone(HKT).date()
        if last < first:
            first, last = last, first
        return {
            (first + timedelta(days=offset)).isoformat()
            for offset in range((last - first).days + 1)
        }
    today = datetime.now(timezone.utc).astimezone(HKT).date()
    return {(today - timedelta(days=offset)).isoformat() for offset in range(max(0, lookback_days) + 1)}


class HistoricalStore:
    """Merge local writable capture with a separate remote read-only cache."""

    _PREFIXES = {
        "model": "layer_a",
        "market": "layer_a_market",
        "weather": "layer_a_weather",
    }

    def __init__(
        self,
        *,
        local_store: LayerAStore | None = None,
        local_market_store: MarketSnapshotStore | None = None,
        local_weather_store: WeatherSnapshotStore | None = None,
        cache_dir: Path | str | None = None,
        repo_id: str | None = None,
        token: str | None = None,
        lookback_days: int | None = None,
        auto_refresh: bool | None = None,
        refresh_interval_minutes: float | None = None,
        legacy_csv_dir: Path | str | None = None,
        api: Any = None,
    ) -> None:
        self.local_store = local_store or LayerAStore(auto_upload=False)
        self.local_market_store = local_market_store or MarketSnapshotStore(auto_upload=False)
        self.local_weather_store = local_weather_store or WeatherSnapshotStore(auto_upload=False)
        self.cache_dir = Path(
            cache_dir
            or os.getenv("LAYER_A_REMOTE_CACHE_DIR", "/tmp/layer_a_remote_cache")
        ).expanduser()
        local_roots = (
            self.local_store.root,
            self.local_market_store.root,
            self.local_weather_store.root,
        )
        if any(_inside(self.cache_dir, root) or _inside(root, self.cache_dir) for root in local_roots):
            raise ValueError("LAYER_A_REMOTE_CACHE_DIR must be physically separate from local capture roots")
        self.repo_id = repo_id if repo_id is not None else os.getenv("HF_LAYER_A_REPO_ID", "").strip()
        self.token = token if token is not None else os.getenv("HF_LAYER_A_TOKEN", "")
        self.lookback_days = (
            int(os.getenv("LAYER_A_HISTORY_LOOKBACK_DAYS", "7"))
            if lookback_days is None
            else max(0, int(lookback_days))
        )
        self.auto_refresh = (
            _env_bool("LAYER_A_HISTORY_AUTO_REFRESH", True)
            if auto_refresh is None
            else bool(auto_refresh)
        )
        self.refresh_interval_minutes = (
            float(os.getenv("LAYER_A_HISTORY_REFRESH_INTERVAL_MINUTES", "10"))
            if refresh_interval_minutes is None
            else max(0.1, float(refresh_interval_minutes))
        )
        self.api = api
        self.legacy_csv_dir = Path(legacy_csv_dir or _legacy_csv_root()).expanduser()
        self._api_instance: Any = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self._generation = 0
        self._status: dict[str, Any] = {
            "status": "not_started" if self.repo_id and self.token else "disabled",
            "last_refresh": None,
            "latest_timestamp": None,
            "files_cached": 0,
            "files_found": 0,
            "files_downloaded": 0,
            "refresh_failures": 0,
            "last_error": None,
            "repo_configured": bool(self.repo_id and self.token),
        }

    @property
    def index_path(self) -> Path:
        return self.cache_dir / ".remote_index.json"

    def _read_index(self) -> dict[str, Any]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"files": {}}

    def _write_index(self, index: Mapping[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_dir / f".remote_index.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(jsonable(index), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def _get_api(self) -> Any:
        if self.api is not None:
            return self.api
        if self._api_instance is not None:
            return self._api_instance
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for Layer A history") from exc
        self._api_instance = HfApi(token=self.token)
        return self._api_instance

    def _list_remote_files(self) -> list[dict[str, Any]]:
        api = self._get_api()
        if hasattr(api, "list_repo_files"):
            raw = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        elif hasattr(api, "list_repo_tree"):
            raw = api.list_repo_tree(repo_id=self.repo_id, repo_type="dataset", recursive=True)
        else:
            raise RuntimeError("HF history client does not expose a repository listing method")
        result: list[dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, str):
                result.append({"path": entry})
                continue
            path = getattr(entry, "rfilename", None) or getattr(entry, "path", None)
            if path is None and isinstance(entry, Mapping):
                path = entry.get("rfilename") or entry.get("path")
            if path:
                metadata = dict(entry) if isinstance(entry, Mapping) else {}
                metadata["path"] = str(path)
                result.append(metadata)
        return result

    def _remote_entries_for_dates(self, dates: set[str]) -> list[dict[str, Any]]:
        entries = []
        for entry in self._list_remote_files():
            path = str(entry.get("path", "")).replace("\\", "/")
            if not any(path.startswith(f"{prefix}/date={date_value}/") for prefix in self._PREFIXES.values() for date_value in dates):
                continue
            if not path.endswith((".parquet", ".jsonl.zst", ".jsonl", ".json")):
                continue
            entries.append({**entry, "path": path})
        return sorted(entries, key=lambda item: str(item["path"]))

    @staticmethod
    def _metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: entry.get(key)
            for key in ("size", "oid", "sha256", "lfs", "last_commit_id")
            if entry.get(key) is not None
        }

    def _download_one(self, remote_path: str, destination: Path) -> None:
        api = self._get_api()
        downloaded: Any = None
        if hasattr(api, "download_file"):
            for kwargs in (
                {
                    "repo_id": self.repo_id,
                    "filename": remote_path,
                    "repo_type": "dataset",
                    "token": self.token,
                },
                {"repo_id": self.repo_id, "filename": remote_path, "repo_type": "dataset"},
            ):
                try:
                    downloaded = api.download_file(**kwargs)
                    break
                except TypeError:
                    continue
        else:
            try:
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=str(self.cache_dir),
                    local_dir_use_symlinks=False,
                )
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required for Layer A history download") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        if isinstance(downloaded, bytes):
            temporary.write_bytes(downloaded)
        elif isinstance(downloaded, (str, os.PathLike)) and Path(downloaded).exists():
            source = Path(downloaded)
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, temporary)
            else:
                return
        elif downloaded is not None and hasattr(downloaded, "read"):
            temporary.write_bytes(downloaded.read())
        else:
            raise RuntimeError(f"remote history download returned no file for {remote_path}")
        os.replace(temporary, destination)

    def refresh(
        self,
        *,
        date_value: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Download closed remote date partitions into the read-only cache."""
        with self._lock:
            if not self.repo_id or not self.token:
                self._status.update({"status": "disabled", "last_error": None, "files_found": 0, "files_downloaded": 0})
                logger.info("Layer A remote history refresh: repo configured: false; status=disabled")
                return self.health_summary()
            self._status.update({"status": "loading", "last_error": None})
        logger.info("Layer A remote history refresh: repo configured: true; status=loading")
        dates = _date_strings(
            date_value=date_value,
            start=start,
            end=end,
            lookback_days=self.lookback_days,
        )
        entries: list[dict[str, Any]] = []
        downloaded_count = 0
        try:
            entries = self._remote_entries_for_dates(dates)
            index = self._read_index()
            files_index = index.setdefault("files", {})
            for entry in entries:
                remote_path = str(entry["path"])
                destination = self.cache_dir / Path(remote_path)
                metadata = self._metadata(entry)
                previous = files_index.get(remote_path)
                if destination.exists() and previous is not None and previous.get("metadata", {}) == metadata:
                    continue
                if destination.exists() and previous is None:
                    # Hub paths are content-addressed/immutable for this use;
                    # retain an existing cache file instead of redownloading it.
                    files_index[remote_path] = {"metadata": metadata, "cached_at": datetime.now(timezone.utc).isoformat()}
                    continue
                self._download_one(remote_path, destination)
                files_index[remote_path] = {"metadata": metadata, "cached_at": datetime.now(timezone.utc).isoformat()}
                downloaded_count += 1
            self._write_index({"schema_version": "layer_a.remote_index.v1", "files": files_index})
            with self._lock:
                self._generation += 1
                self._records_cache.clear()
                cached_files = len([path for path in files_index if Path(path).suffix in {".json", ".parquet"} or path.endswith((".jsonl", ".jsonl.zst"))])
                self._status.update(
                    {
                        "status": "available",
                        "last_refresh": datetime.now(timezone.utc).isoformat(),
                        "files_cached": cached_files,
                        "files_found": len(entries),
                        "files_downloaded": downloaded_count,
                        "last_error": None,
                    }
                )
            logger.info(
                "Layer A remote history refresh: status=available files_found=%d files_downloaded=%d latest_cached_timestamp=%s",
                len(entries),
                downloaded_count,
                self.health_summary().get("latest_timestamp"),
            )
            return self.health_summary()
        except Exception as exc:
            logger.warning("Layer A remote history refresh failed: %s", type(exc).__name__)
            with self._lock:
                self._status.update(
                    {
                        "status": "degraded" if self._cache_has_files() else "unavailable",
                        "last_refresh": self._status.get("last_refresh"),
                        "files_found": len(entries),
                        "files_downloaded": downloaded_count,
                        "refresh_failures": int(self._status.get("refresh_failures", 0)) + 1,
                        "last_error": type(exc).__name__,
                    }
                )
            return self.health_summary()

    def manual_refresh(self, **kwargs: Any) -> dict[str, Any]:
        return self.refresh(**kwargs)

    def _cache_has_files(self) -> bool:
        return any(
            path.is_file() and not path.name.startswith(".")
            for prefix in self._PREFIXES.values()
            for path in (self.cache_dir / prefix).rglob("*")
            if (self.cache_dir / prefix).exists()
        )

    def start_background_refresh(self) -> None:
        if not self.auto_refresh or self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            self.refresh()
            while not self._stop.wait(self.refresh_interval_minutes * 60.0):
                self.refresh()

        self._thread = threading.Thread(target=loop, daemon=True, name="layer-a-history-refresh")
        self._thread.start()

    def stop_background_refresh(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def _remote_model_store(self) -> LayerAStore:
        return LayerAStore(self.cache_dir / self._PREFIXES["model"], auto_upload=False)

    def _remote_market_store(self) -> MarketSnapshotStore:
        return MarketSnapshotStore(self.cache_dir / self._PREFIXES["market"], auto_upload=False)

    def _remote_weather_store(self) -> WeatherSnapshotStore:
        return WeatherSnapshotStore(self.cache_dir / self._PREFIXES["weather"], auto_upload=False)

    @staticmethod
    def _valid(kind: str, record: Mapping[str, Any]) -> bool:
        try:
            if kind == "model":
                validate_layer_a_record(record)
            elif kind == "market":
                validate_market_snapshot(record)
            else:
                validate_weather_snapshot(record)
            return True
        except (TypeError, ValueError):
            return False

    def _local_records(self, kind: str) -> list[dict[str, Any]]:
        if kind == "model":
            return [record for _info, record in self.local_store.iter_records()]
        if kind == "market":
            return [
                record
                for info in self.local_market_store.scan()
                if info.status in {"complete", "incomplete"}
                for record in self.local_market_store.read_partition_snapshots(info)
            ]
        return [
            record
            for info in self.local_weather_store.scan()
            if info.status in {"complete", "incomplete"}
            for record in self.local_weather_store.read_partition_snapshots(info)
        ]

    def _legacy_minute_rows(self, dates: set[str]) -> list[dict[str, Any]]:
        """Read the repository-synced legacy CSV only as a display fallback.

        These rows intentionally do not become model/market/weather records:
        the CSV has no Layer A snapshot identity and no CLOB book identity.
        Timestamps without an offset are historical HKT wall-clock values.
        """
        grouped: dict[datetime, list[tuple[datetime, Path, dict[str, Any]]]] = {}
        for date_value in sorted(dates):
            path = self.legacy_csv_dir / f"{date_value}.csv"
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for raw in csv.DictReader(handle):
                        parsed = _timestamp(raw.get("timestamp"))
                        if parsed is None:
                            continue
                        local_time = parsed.astimezone(HKT)
                        row_date = str(raw.get("snapshot_date") or local_time.date().isoformat())
                        if row_date not in dates:
                            continue
                        minute = local_time.replace(second=0, microsecond=0)
                        grouped.setdefault(minute, []).append((parsed, path, dict(raw)))
            except (OSError, csv.Error) as exc:
                logger.warning("Legacy CSV history read failed for %s: %s", path.name, type(exc).__name__)

        rows: list[dict[str, Any]] = []
        for minute, entries in sorted(grouped.items()):
            entries.sort(key=lambda item: item[0])
            latest_timestamp, latest_path, latest = entries[-1]
            context = _csv_json(latest.get("context_json"))
            if not isinstance(context, Mapping):
                context = {}
            predictions: dict[str, Any] = {}
            probabilities: dict[str, Any] = {}
            actual_temperature = None
            max_so_far = None
            min_so_far = None
            for _timestamp_value, _path, raw in entries:
                actual_temperature = _csv_float(raw.get("actual_temp")) if actual_temperature is None else actual_temperature
                max_so_far = _csv_float(raw.get("max_so_far")) if max_so_far is None else max_so_far
                raw_context = _csv_json(raw.get("context_json"))
                if isinstance(raw_context, Mapping):
                    min_so_far = (
                        _csv_float(raw_context.get("min_so_far"))
                        if min_so_far is None
                        else min_so_far
                    )
                    raw_probs = raw_context.get("model_probs")
                    if isinstance(raw_probs, Mapping):
                        probabilities.update({str(key): value for key, value in raw_probs.items()})
                raw_predictions = _csv_json(raw.get("all_model_predictions"))
                if isinstance(raw_predictions, Mapping):
                    for key, value in raw_predictions.items():
                        parsed_value = _csv_float(value)
                        if parsed_value is not None:
                            predictions[str(key)] = {"point_prediction": parsed_value}
                model_key = str(raw.get("model_key") or "")
                model_value = _csv_float(raw.get("model_predicted_temp"))
                if model_key and model_value is not None:
                    predictions.setdefault(model_key, {"point_prediction": model_value})
            if min_so_far is None:
                min_so_far = _csv_float(context.get("min_so_far"))
            if not predictions:
                model_key = str(latest.get("model_key") or "")
                model_value = _csv_float(latest.get("model_predicted_temp"))
                if model_key and model_value is not None:
                    predictions[model_key] = {"point_prediction": model_value}
            timestamp = latest_timestamp.astimezone(HKT).isoformat()
            rows.append(
                {
                    "timestamp": timestamp,
                    "source": "legacy_csv",
                    "source_path": latest_path.name,
                    "legacy_csv_timestamp": latest.get("timestamp"),
                    "legacy_csv_strategy_rows": len(entries),
                    "actual_temperature": actual_temperature,
                    "max_so_far": max_so_far,
                    "min_so_far": min_so_far,
                    "weather_source_timestamp": timestamp if actual_temperature is not None else None,
                    "weather_age_seconds": 0 if actual_temperature is not None else None,
                    "weather_quality_status": "observed" if actual_temperature is not None else "missing",
                    "weather_snapshot_id": None,
                    "weather_observations": {},
                    "market_prices": {},
                    "best_bid": {},
                    "best_ask": {},
                    "spread": {},
                    "market_book_timestamp": {},
                    "market_book_age_seconds": {},
                    "market": {},
                    "markets_by_kind": {},
                    "market_snapshot_id": None,
                    "market_validation_status": "legacy_csv_no_clob_identity",
                    "latest_model_cycle_timestamp": timestamp if predictions else None,
                    "model_cycle_timestamp": timestamp if predictions else None,
                    "model_cycle_id": None,
                    "model_age_seconds": 0 if predictions else None,
                    "model_predictions": predictions,
                    "model_probabilities": probabilities,
                    "models": predictions,
                }
            )
        return rows

    def _remote_records(self, kind: str) -> list[dict[str, Any]]:
        if kind == "model":
            return [record for _info, record in self._remote_model_store().iter_records()]
        if kind == "market":
            return [record for _info, record in self._remote_market_store().read_snapshot_records()]
        return [record for _info, record in self._remote_weather_store().read_snapshot_records()]

    @staticmethod
    def _key(kind: str, record: Mapping[str, Any]) -> str | None:
        field = {
            "model": "decision_cycle_id",
            "market": "market_snapshot_id",
            "weather": "weather_snapshot_id",
        }[kind]
        value = record.get(field)
        return str(value) if value not in (None, "") else None

    def _merged_records(self, kind: str) -> list[dict[str, Any]]:
        local = self._local_records(kind)
        remote = self._remote_records(kind)
        merged: dict[str, tuple[int, dict[str, Any]]] = {}
        for rank, records in ((1, remote), (2, local)):
            for record in records:
                key = self._key(kind, record)
                if key is None:
                    continue
                valid_rank = rank + (2 if self._valid(kind, record) else 0)
                existing = merged.get(key)
                if existing is None or valid_rank > existing[0]:
                    merged[key] = (valid_rank, jsonable(record))
        result = [record for _rank, record in merged.values()]
        timestamp_field = "decision_timestamp" if kind != "weather" else "snapshot_timestamp"
        result.sort(key=lambda record: str(record.get(timestamp_field) or ""))
        return result

    def records(
        self,
        kind: str,
        *,
        date_value: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if kind not in self._PREFIXES:
            raise ValueError(f"unsupported history kind: {kind}")
        # Local market/weather partitions remain open for up to ten minutes.
        # Their mtime changes on every append, so a generation counter that is
        # advanced only by remote refresh would make the UI stale indefinitely.
        # Re-scan the small local/remote read set on every API query instead.
        with self._lock:
            data = self._merged_records(kind)
        start_dt = _timestamp(start) if start else None
        end_dt = _timestamp(end) if end else None
        timestamp_field = "decision_timestamp" if kind != "weather" else "snapshot_timestamp"
        result = []
        for record in data:
            if date_value and str(record.get("event_date")) != date_value:
                continue
            timestamp = _timestamp(record.get(timestamp_field))
            if start_dt and (timestamp is None or timestamp < start_dt):
                continue
            if end_dt and (timestamp is None or timestamp > end_dt):
                continue
            result.append(jsonable(record))
        if limit is not None:
            return result[: max(0, int(limit))]
        return result

    def minute_history(
        self,
        *,
        date_value: str | None = None,
        start: str | None = None,
        end: str | None = None,
        bucket_filters: Iterable[str] | None = None,
        model_filters: Iterable[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        # Keep records before ``start`` available for backward as-of model and
        # prior-valid-weather joins.  The projection applies the requested
        # range after joining, so a 09:01 row can still see a 09:00 cycle.
        cycles = self.records("model", date_value=date_value, end=end)
        markets = self.records("market", date_value=date_value, end=end)
        weather = self.records("weather", date_value=date_value, end=end)
        has_layer_a_records = bool(cycles or markets or weather)
        if not has_layer_a_records:
            dates = _date_strings(date_value=date_value, start=start, end=end, lookback_days=self.lookback_days)
            legacy_rows = self._legacy_minute_rows(dates)
            start_dt = _timestamp(start) if start else None
            end_dt = _timestamp(end) if end else None
            filtered = [
                row
                for row in legacy_rows
                if (start_dt is None or (_timestamp(row["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc)) >= start_dt)
                and (end_dt is None or (_timestamp(row["timestamp"]) or datetime.max.replace(tzinfo=timezone.utc)) <= end_dt)
            ]
            return filtered[: max(0, int(limit))]
        rows = build_minute_view(
            cycles,
            markets,
            weather,
            start=start,
            end=end,
            bucket_filters=list(bucket_filters or []),
            model_filters=list(model_filters or []),
            limit=limit,
        )
        for row in rows:
            row.setdefault("source", "layer_a")
        return rows

    # Explicit read aliases keep callers from depending on the generic kind
    # string and make the read-only boundary obvious in service code/tests.
    def read_model_cycles(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.records("model", **kwargs)

    def read_market_snapshots(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.records("market", **kwargs)

    def read_weather_snapshots(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.records("weather", **kwargs)

    def read_minute_view(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.minute_history(**kwargs)

    def health_summary(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
        timestamps: list[datetime] = []
        for kind in self._PREFIXES:
            try:
                # This field describes the durable remote cache only; local
                # current capture is reported by the individual stores.
                for record in self._remote_records(kind):
                    timestamp = _timestamp(
                        record.get("snapshot_timestamp" if kind == "weather" else "decision_timestamp")
                    )
                    if timestamp is not None:
                        timestamps.append(timestamp)
            except Exception:
                continue
        result["latest_timestamp"] = max(timestamps).isoformat() if timestamps else None
        result["files_cached"] = max(int(result.get("files_cached", 0)), self._count_cache_files())
        return result

    def _count_cache_files(self) -> int:
        count = 0
        for prefix in self._PREFIXES.values():
            root = self.cache_dir / prefix
            if root.exists():
                count += sum(1 for path in root.rglob("*") if path.is_file())
        return count


_DEFAULT_HISTORICAL_STORE: HistoricalStore | None = None
_DEFAULT_HISTORICAL_STORE_LOCK = threading.Lock()


def get_default_historical_store() -> HistoricalStore:
    global _DEFAULT_HISTORICAL_STORE
    if _DEFAULT_HISTORICAL_STORE is None:
        with _DEFAULT_HISTORICAL_STORE_LOCK:
            if _DEFAULT_HISTORICAL_STORE is None:
                _DEFAULT_HISTORICAL_STORE = HistoricalStore()
    return _DEFAULT_HISTORICAL_STORE


__all__ = ["HistoricalStore", "get_default_historical_store"]
