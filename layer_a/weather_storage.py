"""Append-only ten-minute storage for one-minute weather snapshots."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .minute_partition import (
    get_minute_partition_minutes,
    minute_partition_start,
    partition_is_due,
    partition_start_from_directory,
)
from .schema import jsonable
from .storage import _date_scan_root, _default_root, _json_dump, _sha256, _timestamp, read_books
from .weather_schema import WEATHER_SCHEMA_VERSION, validate_weather_snapshot

logger = logging.getLogger(__name__)
HKT = timezone(timedelta(hours=8))
_PART_FILE_RE = re.compile(
    r"^(?P<kind>snapshots|manifest)-(?P<partition>[A-Za-z0-9_-]+)"
    r"(?P<suffix>\.jsonl\.zst|\.jsonl|\.json)(?P<temporary>\.tmp)?$"
)


def _weather_root() -> Path:
    configured = os.getenv("LAYER_A_WEATHER_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        from app.config import LAYER_A_WEATHER_DIR

        return Path(LAYER_A_WEATHER_DIR)
    except ImportError:
        return _default_root().parent / "layer_a_weather"


def _write_compressed(path: Path, payload: bytes) -> str:
    try:
        import zstandard as zstd  # type: ignore

        path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
        return "zstd"
    except ImportError:
        path.write_bytes(payload)
        return "plain_fallback"


def _read_raw_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
    return records


def _is_buffer_backfill(snapshot: Mapping[str, Any]) -> bool:
    source_status = snapshot.get("source_status")
    return isinstance(source_status, Mapping) and bool(source_status.get("buffer_backfill"))


@dataclass
class WeatherPartitionInfo:
    partition_id: str
    directory: Path
    files: dict[str, Path] = field(default_factory=dict)
    temporary_files: list[Path] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    status: str = "incomplete"
    reasons: list[str] = field(default_factory=list)
    checksum_valid: bool | None = None
    uploaded: bool = False

    @property
    def closed(self) -> bool:
        return self.status == "complete"

    def as_dict(self, root: Path | None = None) -> dict[str, Any]:
        base = root or self.directory.parent
        try:
            directory = str(self.directory.relative_to(base)).replace("\\", "/")
        except ValueError:
            directory = str(self.directory).replace("\\", "/")
        return {
            "partition_id": self.partition_id,
            "directory": directory,
            "files": {key: str(value.name) for key, value in self.files.items()},
            "temporary_files": [str(value.name) for value in self.temporary_files],
            "status": self.status,
            "closed": self.closed,
            "uploaded": self.uploaded,
            "checksum_valid": self.checksum_valid,
            "reasons": list(self.reasons),
            "manifest": self.manifest,
        }


@dataclass
class WeatherCaptureResult:
    status: str
    weather_snapshot_id: str
    partition_id: str | None = None
    snapshot: dict[str, Any] | None = None


class WeatherSnapshotStore:
    """Store weather snapshots in immutable, configurable minute chunks."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        uploader: Any = None,
        auto_upload: bool | None = None,
        upload_interval_minutes: float | None = None,
        partition_minutes: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _weather_root()
        self.partition_minutes = get_minute_partition_minutes(partition_minutes)
        self._lock = threading.RLock()
        self._uploader = uploader
        self._auto_upload = self._env_bool("HF_LAYER_A_AUTO_UPLOAD") if auto_upload is None else bool(auto_upload)
        self._upload_interval_minutes = (
            float(os.getenv("HF_LAYER_A_UPLOAD_INTERVAL_MINUTES", "30"))
            if upload_interval_minutes is None
            else float(upload_interval_minutes)
        )
        self._last_upload_attempt: datetime | None = None

    @staticmethod
    def _env_bool(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def receipts_dir(self) -> Path:
        return self.root / ".upload_receipts"

    @property
    def failures_dir(self) -> Path:
        return self.root / ".upload_failures"

    def _has_receipt(self, partition_id: str) -> bool:
        return (self.receipts_dir / f"{partition_id}.json").exists()

    def _parse_part_file(self, path: Path) -> tuple[str, str, bool] | None:
        match = _PART_FILE_RE.match(path.name)
        if not match:
            return None
        kind = match.group("kind")
        suffix = match.group("suffix")
        if kind == "snapshots" and suffix not in {".jsonl.zst", ".jsonl"}:
            return None
        if kind == "manifest" and suffix != ".json":
            return None
        return kind, match.group("partition"), bool(match.group("temporary"))

    def scan(self, date_value: str | None = None) -> list[WeatherPartitionInfo]:
        scan_root = _date_scan_root(self.root, date_value)
        if scan_root is None or not scan_root.exists():
            return []
        groups: dict[tuple[Path, str], WeatherPartitionInfo] = {}
        for path in scan_root.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(scan_root).parts):
                continue
            parsed = self._parse_part_file(path)
            if parsed is None:
                continue
            kind, partition_id, temporary = parsed
            info = groups.setdefault((path.parent, partition_id), WeatherPartitionInfo(partition_id, path.parent))
            if temporary:
                info.temporary_files.append(path)
            else:
                info.files[kind] = path
        result: list[WeatherPartitionInfo] = []
        for info in groups.values():
            missing = sorted({"snapshots", "manifest"} - set(info.files))
            if missing:
                info.reasons.append(f"missing_closed_files:{','.join(missing)}")
            manifest_path = info.files.get("manifest")
            if manifest_path is not None:
                try:
                    info.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    info.reasons.append(f"manifest_unreadable:{type(exc).__name__}")
            if info.manifest is not None:
                if info.manifest.get("partition_id") not in (None, info.partition_id):
                    info.reasons.append("manifest_partition_id_mismatch")
                info.checksum_valid, checksum_reasons = self._verify_manifest(info)
                info.reasons.extend(checksum_reasons)
            if (
                not missing
                and info.manifest is not None
                and not any(
                    reason.startswith("manifest_unreadable") or reason.startswith("manifest_partition_id")
                    for reason in info.reasons
                )
            ):
                info.status = "complete"
            else:
                info.status = "incomplete"
            info.uploaded = self._has_receipt(info.partition_id)
            result.append(info)
        return sorted(result, key=lambda item: (str(item.directory), item.partition_id))

    def _verify_manifest(self, info: WeatherPartitionInfo) -> tuple[bool, list[str]]:
        checksums = (info.manifest or {}).get("file_checksums", {})
        path = info.files.get("snapshots")
        if not isinstance(checksums, Mapping):
            return False, ["manifest_checksums_missing"]
        if path is None:
            return False, ["snapshots_file_missing"]
        expected = checksums.get(path.name)
        if not expected:
            return False, [f"checksum_missing:{path.name}"]
        try:
            actual = _sha256(path)
        except OSError as exc:
            return False, [f"checksum_read_failed:{path.name}:{type(exc).__name__}"]
        return (True, []) if actual == expected else (False, [f"checksum_mismatch:{path.name}"])

    def _read_info_snapshots(self, info: WeatherPartitionInfo) -> list[dict[str, Any]]:
        path = info.files.get("snapshots")
        if path is not None:
            return read_books(path, (info.manifest or {}).get("compression"))
        temporary = next(
            (item for item in info.temporary_files if item.name.startswith("snapshots-")),
            None,
        )
        return _read_raw_jsonl(temporary)

    def read_partition_snapshots(self, info: WeatherPartitionInfo) -> list[dict[str, Any]]:
        return self._read_info_snapshots(info) if info.status in {"complete", "incomplete"} else []

    def _known_snapshot_ids(self, partitions: Iterable[WeatherPartitionInfo] | None = None) -> set[str]:
        known: set[str] = set()
        for info in partitions if partitions is not None else self.scan():
            ids = (info.manifest or {}).get("weather_snapshot_ids", [])
            if isinstance(ids, list):
                known.update(str(item) for item in ids)
            known.update(
                str(snapshot["weather_snapshot_id"])
                for snapshot in self._read_info_snapshots(info)
                if snapshot.get("weather_snapshot_id")
            )
        return known

    def _known_snapshot_records(
        self,
        partitions: Iterable[WeatherPartitionInfo] | None = None,
    ) -> dict[str, dict[str, Any]]:
        known: dict[str, dict[str, Any]] = {}
        for info in partitions if partitions is not None else self.scan():
            for snapshot in self._read_info_snapshots(info):
                snapshot_id = snapshot.get("weather_snapshot_id")
                if snapshot_id:
                    known[str(snapshot_id)] = snapshot
        return known

    def _find_snapshot_partition(
        self,
        snapshot_id: str,
        partitions: Iterable[WeatherPartitionInfo],
    ) -> WeatherPartitionInfo | None:
        for info in partitions:
            for snapshot in self._read_info_snapshots(info):
                if str(snapshot.get("weather_snapshot_id") or "") == str(snapshot_id):
                    return info
        return None

    def _replace_open_snapshot(
        self,
        info: WeatherPartitionInfo,
        snapshot: Mapping[str, Any],
    ) -> bool:
        """Replace a same-ID record in an open append buffer atomically."""
        raw_path = next(
            (item for item in info.temporary_files if item.name.startswith("snapshots-") and item.suffix == ".tmp"),
            None,
        )
        if raw_path is None or not raw_path.exists():
            return False
        records = _read_raw_jsonl(raw_path)
        snapshot_id = str(snapshot.get("weather_snapshot_id") or "")
        replaced = False
        for index, current in enumerate(records):
            if str(current.get("weather_snapshot_id") or "") == snapshot_id:
                records[index] = jsonable(snapshot)
                replaced = True
        if not replaced:
            return False
        replacement = Path(f"{raw_path}.replace.tmp")
        replacement.write_bytes(
            "".join(f"{_json_dump(record)}\n" for record in records).encode("utf-8")
        )
        os.replace(replacement, raw_path)
        return True

    def _replace_closed_snapshot(
        self,
        info: WeatherPartitionInfo,
        snapshot: Mapping[str, Any],
    ) -> WeatherPartitionInfo | None:
        """Rewrite one closed chunk so a deterministic ID remains unique.

        A CSV buffer backfill is an authoritative correction for a bad capture
        of the same minute.  The old line is replaced in the closed chunk and
        the checksum/manifest are regenerated atomically; a second line with
        the same ID is never written.
        """
        final_path = info.files.get("snapshots")
        if final_path is None or not final_path.exists():
            return None
        records = self._read_info_snapshots(info)
        snapshot_id = str(snapshot.get("weather_snapshot_id") or "")
        replaced = False
        for index, current in enumerate(records):
            if str(current.get("weather_snapshot_id") or "") == snapshot_id:
                records[index] = jsonable(snapshot)
                replaced = True
        if not replaced:
            return None

        raw_payload = "".join(f"{_json_dump(record)}\n" for record in records).encode("utf-8")
        compressed_tmp = Path(f"{final_path}.tmp")
        manifest_path = info.files.get("manifest") or info.directory / f"manifest-{info.partition_id}.json"
        manifest_tmp = Path(f"{manifest_path}.tmp")
        decision_times = [
            _timestamp(record.get("snapshot_timestamp"))
            for record in records
            if _timestamp(record.get("snapshot_timestamp")) is not None
        ]
        partition_start = partition_start_from_directory(info.directory)
        if partition_start is None and decision_times:
            partition_start = minute_partition_start(min(decision_times), self.partition_minutes)
        compression = _write_compressed(compressed_tmp, raw_payload)
        manifest = {
            "schema_version": WEATHER_SCHEMA_VERSION,
            "partition_id": info.partition_id,
            "partition_relative_path": str(info.directory.relative_to(self.root)).replace("\\", "/"),
            "files": {"snapshots": final_path.name, "manifest": manifest_path.name},
            "file_checksums": {final_path.name: _sha256(compressed_tmp)},
            "file_sizes": {final_path.name: compressed_tmp.stat().st_size},
            "first_timestamp": min(decision_times).isoformat() if decision_times else None,
            "last_timestamp": max(decision_times).isoformat() if decision_times else None,
            "snapshot_count": len(records),
            "weather_snapshot_ids": [str(record["weather_snapshot_id"]) for record in records],
            "partition_minutes": self.partition_minutes,
            "partition_start": partition_start.isoformat() if partition_start else None,
            "partition_end": (
                partition_start + timedelta(minutes=self.partition_minutes)
            ).isoformat() if partition_start else None,
            "compression": compression,
            "created_at": (info.manifest or {}).get("created_at") or datetime.now(timezone.utc).isoformat(),
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "checksum_scope": "snapshots; manifest hash is calculated by export checksums",
        }
        manifest_tmp.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
        os.replace(compressed_tmp, final_path)
        os.replace(manifest_tmp, manifest_path)
        receipt = self.receipts_dir / f"{info.partition_id}.json"
        receipt.unlink(missing_ok=True)
        updated = WeatherPartitionInfo(
            partition_id=info.partition_id,
            directory=info.directory,
            files={"snapshots": final_path, "manifest": manifest_path},
            manifest=manifest,
            status="complete",
            checksum_valid=True,
            uploaded=False,
        )
        self._maybe_auto_upload(updated)
        return updated

    def _minute_directory(self, snapshot: Mapping[str, Any]) -> Path:
        decision = _timestamp(snapshot["snapshot_timestamp"])
        if decision is None:
            raise ValueError("weather snapshot snapshot_timestamp is invalid")
        slot = minute_partition_start(decision, self.partition_minutes)
        event_date = str(snapshot["event_date"])
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
            raise ValueError("weather snapshot event_date is not YYYY-MM-DD")
        return (
            self.root
            / f"date={event_date}"
            / f"hour={slot.hour:02d}"
            / f"minute={slot.minute:02d}"
        )

    def _open_info(self, directory: Path) -> WeatherPartitionInfo:
        directory.mkdir(parents=True, exist_ok=True)
        candidates = [
            info
            for info in self.scan()
            if info.directory == directory
            and info.status != "complete"
            and any(item.name.startswith("snapshots-") for item in info.temporary_files)
        ]
        if candidates:
            return sorted(candidates, key=lambda item: item.partition_id)[0]
        partition_id = uuid.uuid4().hex
        return WeatherPartitionInfo(
            partition_id=partition_id,
            directory=directory,
            temporary_files=[directory / f"snapshots-{partition_id}.jsonl.tmp"],
        )

    def _close_info(self, info: WeatherPartitionInfo) -> WeatherPartitionInfo | None:
        raw_path = next(
            (item for item in info.temporary_files if item.name.startswith("snapshots-") and item.suffix == ".tmp"),
            None,
        )
        if raw_path is None or not raw_path.exists():
            return None
        raw_payload = raw_path.read_bytes()
        try:
            records = [json.loads(line) for line in raw_payload.decode("utf-8").splitlines() if line.strip()]
            for record in records:
                validate_weather_snapshot(record)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            info.reasons.append(f"open_jsonl_unreadable:{type(exc).__name__}")
            return None
        if not records:
            return None
        final_path = info.directory / f"snapshots-{info.partition_id}.jsonl.zst"
        compressed_tmp = Path(f"{final_path}.tmp")
        manifest_path = info.directory / f"manifest-{info.partition_id}.json"
        manifest_tmp = Path(f"{manifest_path}.tmp")
        decision_times = [
            _timestamp(record.get("snapshot_timestamp"))
            for record in records
            if _timestamp(record.get("snapshot_timestamp")) is not None
        ]
        partition_start = partition_start_from_directory(info.directory)
        if partition_start is None and decision_times:
            partition_start = minute_partition_start(min(decision_times), self.partition_minutes)
        try:
            compression = _write_compressed(compressed_tmp, raw_payload)
            manifest = {
                "schema_version": WEATHER_SCHEMA_VERSION,
                "partition_id": info.partition_id,
                "partition_relative_path": str(info.directory.relative_to(self.root)).replace("\\", "/"),
                "files": {"snapshots": final_path.name, "manifest": manifest_path.name},
                "file_checksums": {final_path.name: _sha256(compressed_tmp)},
                "file_sizes": {final_path.name: compressed_tmp.stat().st_size},
                "first_timestamp": min(decision_times).isoformat() if decision_times else None,
                "last_timestamp": max(decision_times).isoformat() if decision_times else None,
                "snapshot_count": len(records),
                "weather_snapshot_ids": [str(record["weather_snapshot_id"]) for record in records],
                "partition_minutes": self.partition_minutes,
                "partition_start": partition_start.isoformat() if partition_start else None,
                "partition_end": (
                    partition_start
                    + timedelta(minutes=self.partition_minutes)
                ).isoformat()
                if partition_start
                else None,
                "compression": compression,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "checksum_scope": "snapshots; manifest hash is calculated by export checksums",
            }
            manifest_tmp.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
            os.replace(compressed_tmp, final_path)
            os.replace(manifest_tmp, manifest_path)
            raw_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Weather partition close failed for %s", info.partition_id)
            raise
        return WeatherPartitionInfo(
            partition_id=info.partition_id,
            directory=info.directory,
            files={"snapshots": final_path, "manifest": manifest_path},
            manifest=manifest,
            status="complete",
            checksum_valid=True,
        )

    def close_due(self, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=HKT)
        now_local = current.astimezone(HKT)
        closed = failed = 0
        with self._lock:
            for info in self.scan():
                if info.status == "complete":
                    continue
                start = partition_start_from_directory(info.directory)
                if start is None or not partition_is_due(start, now=now_local, minutes=self.partition_minutes):
                    continue
                try:
                    closed_info = self._close_info(info)
                    if closed_info is not None:
                        closed += 1
                        self._maybe_auto_upload(closed_info)
                except Exception:
                    failed += 1
        return {"closed": closed, "failed": failed}

    def capture(self, snapshot: Mapping[str, Any]) -> WeatherCaptureResult:
        return self.capture_many([snapshot])[0]

    def capture_many(self, snapshots: Iterable[Mapping[str, Any]]) -> list[WeatherCaptureResult]:
        clean = [jsonable(snapshot) for snapshot in snapshots]
        for snapshot in clean:
            validate_weather_snapshot(snapshot)
        results: list[WeatherCaptureResult] = []
        with self._lock:
            self.close_due()
            partitions = self.scan()
            known = self._known_snapshot_ids(partitions)
            known_records = self._known_snapshot_records(partitions)
            grouped: dict[Path, list[dict[str, Any]]] = {}
            for snapshot in clean:
                snapshot_id = str(snapshot["weather_snapshot_id"])
                existing = known_records.get(snapshot_id)
                if snapshot_id in known:
                    if _is_buffer_backfill(snapshot) and existing is not None and not _is_buffer_backfill(existing):
                        existing_info = self._find_snapshot_partition(snapshot_id, partitions)
                        replaced = False
                        if existing_info is not None and existing_info.status != "complete":
                            replaced = self._replace_open_snapshot(existing_info, snapshot)
                        elif existing_info is not None:
                            replaced = self._replace_closed_snapshot(existing_info, snapshot) is not None
                        if replaced:
                            known_records[snapshot_id] = snapshot
                            results.append(
                                WeatherCaptureResult(
                                    "captured",
                                    snapshot_id,
                                    partition_id=existing_info.partition_id if existing_info else None,
                                    snapshot=snapshot,
                                )
                            )
                            continue
                    results.append(WeatherCaptureResult("duplicate", snapshot_id, snapshot=snapshot))
                    continue
                grouped.setdefault(self._minute_directory(snapshot), []).append(snapshot)
                known.add(snapshot_id)
                known_records[snapshot_id] = snapshot
            for directory, records in grouped.items():
                info = self._open_info(directory)
                raw_path = next(
                    (item for item in info.temporary_files if item.name.startswith("snapshots-")),
                    info.directory / f"snapshots-{info.partition_id}.jsonl.tmp",
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("ab") as handle:
                    for snapshot in records:
                        handle.write((_json_dump(snapshot) + "\n").encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                results.extend(
                    WeatherCaptureResult(
                        "captured",
                        str(snapshot["weather_snapshot_id"]),
                        partition_id=info.partition_id,
                        snapshot=snapshot,
                    )
                    for snapshot in records
                )
        return results

    def read_snapshot_records(
        self,
        *,
        date_value: str | None = None,
        start: str | None = None,
        end: str | None = None,
        only_unuploaded: bool = False,
        verify_checksums: bool = False,
    ) -> list[tuple[WeatherPartitionInfo, dict[str, Any]]]:
        start_dt = _timestamp(start) if start else None
        end_dt = _timestamp(end) if end else None
        result: list[tuple[WeatherPartitionInfo, dict[str, Any]]] = []
        for info in self.scan(date_value=date_value):
            if info.status != "complete" or (only_unuploaded and info.uploaded):
                continue
            if verify_checksums and info.checksum_valid is not True:
                raise ValueError(f"weather checksum verification failed for partition {info.partition_id}")
            for snapshot in self.read_partition_snapshots(info):
                if date_value and snapshot.get("event_date") != date_value:
                    continue
                timestamp = _timestamp(snapshot.get("snapshot_timestamp"))
                if start_dt and (timestamp is None or timestamp < start_dt):
                    continue
                if end_dt and (timestamp is None or timestamp > end_dt):
                    continue
                result.append((info, snapshot))
        return result

    def _configured_uploader(self) -> Any:
        if self._uploader is not None:
            return self._uploader
        if not os.getenv("HF_LAYER_A_REPO_ID") or not os.getenv("HF_LAYER_A_TOKEN"):
            return None
        from .upload import DatasetUploader

        self._uploader = DatasetUploader.from_environment()
        return self._uploader

    def _record_upload_failure(self, info: WeatherPartitionInfo, error: Exception) -> None:
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        message = str(error).replace(os.getenv("HF_LAYER_A_TOKEN", "\0"), "[REDACTED]")
        (self.failures_dir / f"{info.partition_id}-{uuid.uuid4().hex}.json").write_text(
            _json_dump(
                {
                    "partition_id": info.partition_id,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": message,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _maybe_auto_upload(self, info: WeatherPartitionInfo) -> None:
        if not self._auto_upload:
            return
        now = datetime.now(timezone.utc)
        if self._last_upload_attempt is not None and (
            now - self._last_upload_attempt
        ).total_seconds() / 60.0 < max(0.0, self._upload_interval_minutes):
            return
        self._last_upload_attempt = now
        uploader = self._configured_uploader()
        if uploader is None:
            return
        try:
            uploader.upload_partition(info, self.root, remote_prefix="layer_a_weather")
        except Exception as exc:
            self._record_upload_failure(info, exc)
            logger.warning("Weather Layer A remote upload failed for partition %s", info.partition_id)

    def close_all(self) -> dict[str, int]:
        closed = failed = 0
        with self._lock:
            for info in self.scan():
                if info.status == "complete":
                    continue
                try:
                    closed_info = self._close_info(info)
                    if closed_info is not None:
                        closed += 1
                        self._maybe_auto_upload(closed_info)
                except Exception:
                    failed += 1
        return {"closed": closed, "failed": failed}

    def retry_pending_uploads(self) -> dict[str, Any]:
        uploader = self._configured_uploader()
        if uploader is None:
            return {"configured": False, "attempted": 0, "uploaded": 0, "failed": 0}
        attempted = uploaded = failed = 0
        for info in self.scan():
            if info.status != "complete" or info.uploaded:
                continue
            attempted += 1
            try:
                uploader.upload_partition(info, self.root, remote_prefix="layer_a_weather")
                uploaded += 1
            except Exception as exc:
                failed += 1
                self._record_upload_failure(info, exc)
        return {"configured": True, "attempted": attempted, "uploaded": uploaded, "failed": failed}

    def startup_scan(self) -> dict[str, Any]:
        closed = self.close_due()
        partitions = self.scan()
        retry = self.retry_pending_uploads() if self._auto_upload else {
            "configured": False,
            "attempted": 0,
            "uploaded": 0,
            "failed": 0,
        }
        return {
            "partitions": len(partitions),
            "complete": sum(info.status == "complete" for info in partitions),
            "incomplete": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(len(info.temporary_files) for info in partitions),
            "closed_due": closed,
            "upload_retry": retry,
        }

    def health_summary(self) -> dict[str, Any]:
        partitions = self.scan()
        records = [
            snapshot
            for info in partitions
            if info.status == "complete"
            for snapshot in self.read_partition_snapshots(info)
        ]
        today = datetime.now(timezone.utc).astimezone(HKT).date().isoformat()
        today_records = [record for record in records if record.get("event_date") == today]
        timestamps = [
            _timestamp(record.get("snapshot_timestamp"))
            for record in records
            if _timestamp(record.get("snapshot_timestamp")) is not None
        ]
        pending = [info for info in partitions if info.status == "complete" and not info.uploaded]
        oldest = min(
            (
                _timestamp((info.manifest or {}).get("first_timestamp"))
                for info in pending
                if _timestamp((info.manifest or {}).get("first_timestamp")) is not None
            ),
            default=None,
        )
        open_chunks = [
            info
            for info in partitions
            if info.status != "complete"
            and any(item.name.startswith("snapshots-") for item in info.temporary_files)
        ]
        return {
            "last_weather_snapshot": max(timestamps).isoformat() if timestamps else None,
            "weather_snapshots_today": len(today_records),
            "weather_capture_failures": len(list(self.failures_dir.glob("*.json")))
            if self.failures_dir.exists()
            else 0,
            "pending_local_partitions": len(pending),
            "local_minute_chunks_open": len(open_chunks),
            "local_minute_chunks_closed": sum(info.status == "complete" for info in partitions),
            "oldest_unuploaded_chunk": oldest.isoformat() if oldest else None,
            "partition_minutes": self.partition_minutes,
            "incomplete_partitions": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(len(info.temporary_files) for info in partitions),
        }


_DEFAULT_WEATHER_STORE: WeatherSnapshotStore | None = None
_DEFAULT_WEATHER_STORE_LOCK = threading.Lock()


def get_default_weather_store() -> WeatherSnapshotStore:
    global _DEFAULT_WEATHER_STORE
    if _DEFAULT_WEATHER_STORE is None:
        with _DEFAULT_WEATHER_STORE_LOCK:
            if _DEFAULT_WEATHER_STORE is None:
                _DEFAULT_WEATHER_STORE = WeatherSnapshotStore()
    return _DEFAULT_WEATHER_STORE


__all__ = [
    "WeatherCaptureResult",
    "WeatherPartitionInfo",
    "WeatherSnapshotStore",
    "get_default_weather_store",
]
