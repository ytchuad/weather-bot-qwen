"""Append-and-close storage for one-minute market snapshots."""

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

from .market_schema import MARKET_SCHEMA_VERSION, validate_market_snapshot
from .minute_partition import (
    get_minute_partition_minutes,
    minute_partition_start,
    partition_is_due,
    partition_start_from_directory,
)
from .schema import jsonable
from .storage import (
    _default_root,
    _json_dump,
    _sha256,
    _timestamp,
    read_books,
)

logger = logging.getLogger(__name__)

HKT = timezone(timedelta(hours=8))
_PART_FILE_RE = re.compile(
    r"^(?P<kind>snapshots|manifest)-(?P<partition>[A-Za-z0-9_-]+)"
    r"(?P<suffix>\.jsonl\.zst|\.jsonl|\.json)(?P<temporary>\.tmp)?$"
)


def _market_root() -> Path:
    configured = os.getenv("LAYER_A_MARKET_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        from app.config import LAYER_A_MARKET_DIR

        return Path(LAYER_A_MARKET_DIR)
    except ImportError:
        return _default_root().parent / "layer_a_market"


def _write_compressed(path: Path, payload: bytes) -> str:
    try:
        import zstandard as zstd  # type: ignore

        path.write_bytes(zstd.ZstdCompressor(level=3).compress(payload))
        return "zstd"
    except ImportError:
        # Keep a truthful fallback for minimal test environments.  Production
        # requirements install zstandard, so closed Space partitions are zstd.
        path.write_bytes(payload)
        return "plain_fallback"


def _read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A collector can be interrupted while appending the final
                # line.  Keep prior complete records visible and ignore only
                # that incomplete tail; a later scan will read it again.
                if index == len(lines) - 1:
                    continue
                raise
    return records


@dataclass
class MarketPartitionInfo:
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
class MarketCaptureResult:
    status: str
    market_snapshot_id: str
    partition_id: str | None = None
    snapshot: dict[str, Any] | None = None


class MarketSnapshotStore:
    """Append minute snapshots to one compressed partition per local hour."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        uploader: Any = None,
        auto_upload: bool | None = None,
        upload_interval_minutes: float | None = None,
        partition_minutes: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _market_root()
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

    def _receipt_path(self, partition_id: str) -> Path:
        return self.receipts_dir / f"{partition_id}.json"

    def _has_receipt(self, partition_id: str) -> bool:
        return self._receipt_path(partition_id).exists()

    def _parse_part_file(self, path: Path) -> tuple[str, str, bool] | None:
        match = _PART_FILE_RE.match(path.name)
        if not match:
            return None
        kind = match.group("kind")
        suffix = match.group("suffix")
        temporary = bool(match.group("temporary"))
        if kind == "snapshots" and suffix not in {".jsonl.zst", ".jsonl"}:
            return None
        if kind == "manifest" and suffix != ".json":
            return None
        return kind, match.group("partition"), temporary

    def scan(self) -> list[MarketPartitionInfo]:
        """Scan closed partitions and recoverable open hourly files."""
        if not self.root.exists():
            return []
        groups: dict[tuple[Path, str], MarketPartitionInfo] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            parsed = self._parse_part_file(path)
            if parsed is None:
                continue
            kind, partition_id, temporary = parsed
            info = groups.setdefault((path.parent, partition_id), MarketPartitionInfo(partition_id, path.parent))
            if temporary:
                info.temporary_files.append(path)
            else:
                info.files[kind] = path

        result: list[MarketPartitionInfo] = []
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
                checksum_ok, checksum_reasons = self._verify_manifest(info)
                info.checksum_valid = checksum_ok
                info.reasons.extend(checksum_reasons)
            if (
                not missing
                and info.manifest is not None
                and not any(
                    reason.startswith("manifest_unreadable")
                    or reason.startswith("manifest_partition_id")
                    for reason in info.reasons
                )
            ):
                # A raw open temp left after a crash immediately after the
                # manifest rename does not invalidate the immutable closed
                # payload; it remains visible as a recoverable diagnostic.
                info.status = "complete"
            else:
                info.status = "incomplete"
            info.uploaded = self._has_receipt(info.partition_id)
            result.append(info)
        return sorted(result, key=lambda item: (str(item.directory), item.partition_id))

    def _verify_manifest(self, info: MarketPartitionInfo) -> tuple[bool, list[str]]:
        manifest = info.manifest or {}
        checksums = manifest.get("file_checksums", {})
        if not isinstance(checksums, Mapping):
            return False, ["manifest_checksums_missing"]
        path = info.files.get("snapshots")
        if path is None:
            return False, ["snapshots_file_missing"]
        expected = checksums.get(path.name)
        if not expected:
            return False, [f"checksum_missing:{path.name}"]
        try:
            actual = _sha256(path)
        except OSError as exc:
            return False, [f"checksum_read_failed:{path.name}:{type(exc).__name__}"]
        if actual != expected:
            return False, [f"checksum_mismatch:{path.name}"]
        return True, []

    def _read_info_snapshots(self, info: MarketPartitionInfo) -> list[dict[str, Any]]:
        path = info.files.get("snapshots")
        if path is not None:
            compression = (info.manifest or {}).get("books_compression")
            return read_books(path, compression)
        temporary = next(
            (item for item in info.temporary_files if item.name.startswith("snapshots-")),
            None,
        )
        return _read_raw_jsonl(temporary) if temporary is not None else []

    def read_partition_snapshots(self, info: MarketPartitionInfo) -> list[dict[str, Any]]:
        if info.status not in {"complete", "incomplete"}:
            return []
        return self._read_info_snapshots(info)

    def _known_snapshot_ids(self, partitions: Iterable[MarketPartitionInfo] | None = None) -> set[str]:
        known: set[str] = set()
        for info in partitions if partitions is not None else self.scan():
            ids = (info.manifest or {}).get("market_snapshot_ids", [])
            if isinstance(ids, list):
                known.update(str(item) for item in ids)
            for snapshot in self._read_info_snapshots(info):
                snapshot_id = snapshot.get("market_snapshot_id")
                if snapshot_id:
                    known.add(str(snapshot_id))
        return known

    @staticmethod
    def _minute_directory(
        root: Path,
        snapshot: Mapping[str, Any],
        partition_minutes: int = 10,
    ) -> Path:
        event_date = str(snapshot["event_date"])
        decision = _timestamp(snapshot["decision_timestamp"])
        if decision is None:
            raise ValueError("market snapshot decision_timestamp is invalid")
        local = decision.astimezone(HKT)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
            raise ValueError("market snapshot event_date is not YYYY-MM-DD")
        slot = minute_partition_start(local, partition_minutes)
        return (
            root
            / f"date={event_date}"
            / f"hour={local.hour:02d}"
            / f"minute={slot.minute:02d}"
        )

    def _open_info(self, directory: Path) -> MarketPartitionInfo:
        directory.mkdir(parents=True, exist_ok=True)
        candidates: list[MarketPartitionInfo] = []
        for info in self.scan():
            if info.directory != directory or info.status == "complete":
                continue
            if any(item.name.startswith("snapshots-") for item in info.temporary_files):
                candidates.append(info)
        if candidates:
            return sorted(candidates, key=lambda item: item.partition_id)[0]
        partition_id = uuid.uuid4().hex
        raw_path = directory / f"snapshots-{partition_id}.jsonl.tmp"
        return MarketPartitionInfo(
            partition_id=partition_id,
            directory=directory,
            temporary_files=[raw_path],
            status="incomplete",
        )

    def _close_info(self, info: MarketPartitionInfo) -> MarketPartitionInfo | None:
        raw_path = next(
            (item for item in info.temporary_files if item.name.startswith("snapshots-") and item.suffix == ".tmp"),
            None,
        )
        if raw_path is None or not raw_path.exists():
            return None
        raw_payload = raw_path.read_bytes()
        try:
            records = [json.loads(line) for line in raw_payload.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            info.reasons.append(f"open_jsonl_unreadable:{type(exc).__name__}")
            return None
        if not records:
            return None
        final_path = info.directory / f"snapshots-{info.partition_id}.jsonl.zst"
        compressed_tmp = Path(f"{final_path}.tmp")
        manifest_path = info.directory / f"manifest-{info.partition_id}.json"
        manifest_tmp = Path(f"{manifest_path}.tmp")
        try:
            compression = _write_compressed(compressed_tmp, raw_payload)
            snapshots_hash = _sha256(compressed_tmp)
            decision_times = [
                _timestamp(record.get("decision_timestamp"))
                for record in records
                if _timestamp(record.get("decision_timestamp")) is not None
            ]
            partition_start = partition_start_from_directory(info.directory)
            if partition_start is None and decision_times:
                partition_start = minute_partition_start(min(decision_times), self.partition_minutes)
            manifest = {
                "schema_version": MARKET_SCHEMA_VERSION,
                "partition_id": info.partition_id,
                "partition_relative_path": str(info.directory.relative_to(self.root)).replace("\\", "/"),
                "files": {
                    "snapshots": final_path.name,
                    "manifest": manifest_path.name,
                },
                "file_checksums": {final_path.name: snapshots_hash},
                "file_sizes": {final_path.name: compressed_tmp.stat().st_size},
                "first_timestamp": min(decision_times).isoformat() if decision_times else None,
                "last_timestamp": max(decision_times).isoformat() if decision_times else None,
                "snapshot_count": len(records),
                "market_snapshot_ids": [str(record["market_snapshot_id"]) for record in records],
                "partition_minutes": self.partition_minutes,
                "partition_start": partition_start.isoformat() if partition_start else None,
                "partition_end": (
                    partition_start
                    + timedelta(minutes=self.partition_minutes)
                ).isoformat()
                if partition_start
                else None,
                "books_compression": compression,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "checksum_scope": "snapshots; manifest hash is calculated by export checksums",
            }
            manifest_tmp.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
            os.replace(compressed_tmp, final_path)
            os.replace(manifest_tmp, manifest_path)
            # The compressed immutable payload and manifest are authoritative.
            # Remove only the temporary append file created by this store.
            raw_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Market partition close failed for %s", info.partition_id)
            raise
        return MarketPartitionInfo(
            partition_id=info.partition_id,
            directory=info.directory,
            files={"snapshots": final_path, "manifest": manifest_path},
            manifest=manifest,
            status="complete",
            checksum_valid=True,
            uploaded=False,
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
                hour_match = re.search(r"hour=(\d{2})", str(info.directory))
                date_match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(info.directory))
                if not hour_match or not date_match:
                    continue
                partition_start = partition_start_from_directory(info.directory)
                if partition_start is None:
                    try:
                        partition_start = datetime.strptime(
                            f"{date_match.group(1)} {hour_match.group(1)}", "%Y-%m-%d %H"
                        ).replace(tzinfo=HKT)
                    except ValueError:
                        continue
                    # Preserve recovery semantics for pre-minute-partition
                    # hourly temporary files.
                    due = partition_start < now_local.replace(minute=0, second=0, microsecond=0)
                else:
                    due = partition_is_due(
                        partition_start,
                        now=now_local,
                        minutes=self.partition_minutes,
                    )
                if not due:
                    continue
                try:
                    closed_info = self._close_info(info)
                    if closed_info is not None:
                        closed += 1
                        self._maybe_auto_upload(closed_info)
                except Exception:
                    failed += 1
        return {"closed": closed, "failed": failed}

    def capture(self, snapshot: Mapping[str, Any]) -> MarketCaptureResult:
        results = self.capture_many([snapshot])
        return results[0]

    def capture_many(self, snapshots: Iterable[Mapping[str, Any]]) -> list[MarketCaptureResult]:
        clean_snapshots = [jsonable(snapshot) for snapshot in snapshots]
        for snapshot in clean_snapshots:
            validate_market_snapshot(snapshot)
        results: list[MarketCaptureResult] = []
        with self._lock:
            self.close_due()
            known = self._known_snapshot_ids()
            grouped: dict[Path, list[dict[str, Any]]] = {}
            for snapshot in clean_snapshots:
                snapshot_id = str(snapshot["market_snapshot_id"])
                if snapshot_id in known:
                    results.append(MarketCaptureResult("duplicate", snapshot_id, snapshot=snapshot))
                    continue
                directory = self._minute_directory(
                    self.root,
                    snapshot,
                    self.partition_minutes,
                )
                grouped.setdefault(directory, []).append(snapshot)
                known.add(snapshot_id)
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
                for snapshot in records:
                    results.append(
                        MarketCaptureResult(
                            "captured",
                            str(snapshot["market_snapshot_id"]),
                            partition_id=info.partition_id,
                            snapshot=snapshot,
                        )
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
    ) -> list[tuple[MarketPartitionInfo, dict[str, Any]]]:
        start_dt = _timestamp(start) if start else None
        end_dt = _timestamp(end) if end else None
        result: list[tuple[MarketPartitionInfo, dict[str, Any]]] = []
        for info in self.scan():
            if info.status != "complete":
                continue
            if only_unuploaded and info.uploaded:
                continue
            if verify_checksums and info.checksum_valid is not True:
                raise ValueError(f"market checksum verification failed for partition {info.partition_id}")
            for snapshot in self.read_partition_snapshots(info):
                if date_value and snapshot.get("event_date") != date_value:
                    continue
                decision = _timestamp(snapshot.get("decision_timestamp"))
                if start_dt and (decision is None or decision < start_dt):
                    continue
                if end_dt and (decision is None or decision > end_dt):
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

    def _record_upload_failure(self, info: MarketPartitionInfo, error: Exception) -> None:
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        message = str(error)
        token = os.getenv("HF_LAYER_A_TOKEN")
        if token:
            message = message.replace(token, "[REDACTED]")
        path = self.failures_dir / f"{info.partition_id}-{uuid.uuid4().hex}.json"
        temporary = Path(f"{path}.tmp")
        temporary.write_text(
            _json_dump({
                "partition_id": info.partition_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": message,
            })
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _maybe_auto_upload(self, info: MarketPartitionInfo) -> None:
        if not self._auto_upload:
            return
        now = datetime.now(timezone.utc)
        if self._last_upload_attempt is not None:
            elapsed = (now - self._last_upload_attempt).total_seconds() / 60.0
            if elapsed < max(0.0, self._upload_interval_minutes):
                return
        self._last_upload_attempt = now
        uploader = self._configured_uploader()
        if uploader is None:
            return
        try:
            uploader.upload_partition(info, self.root, remote_prefix="layer_a_market")
        except Exception as exc:
            self._record_upload_failure(info, exc)
            logger.warning("Market Layer A remote upload failed for partition %s", info.partition_id)

    def close_all(self) -> dict[str, int]:
        """Close all open hourly partitions; intended for tests/operator tools."""
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
                uploader.upload_partition(info, self.root, remote_prefix="layer_a_market")
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
        # Previous-hour append files are closed on startup; current-hour files
        # remain open for the next minute and can continue appending.
        return {
            "partitions": len(partitions),
            "complete": sum(info.status == "complete" for info in partitions),
            "incomplete": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(len(info.temporary_files) for info in partitions),
            "closed_previous_hours": closed,
            "upload_retry": retry,
        }

    def health_summary(self) -> dict[str, Any]:
        partitions = self.scan()
        records: list[dict[str, Any]] = []
        for info in partitions:
            if info.status == "complete":
                records.extend(self.read_partition_snapshots(info))
        now_local = datetime.now(timezone.utc).astimezone(HKT)
        today = now_local.date().isoformat()
        today_records = [record for record in records if record.get("event_date") == today]
        eligible = [
            record
            for record in today_records
            if (record.get("completeness") or {}).get("replay_eligible_for_market_replay") is True
        ]
        pending = [info for info in partitions if info.status == "complete" and not info.uploaded]
        timestamps = [
            _timestamp(record.get("decision_timestamp"))
            for record in records
            if _timestamp(record.get("decision_timestamp")) is not None
        ]
        receipts: list[dict[str, Any]] = []
        if self.receipts_dir.exists():
            for path in self.receipts_dir.glob("*.json"):
                try:
                    receipts.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        last_upload = max(
            (item.get("uploaded_at") for item in receipts if item.get("uploaded_at")),
            default=None,
        )
        failures = len(list(self.failures_dir.glob("*.json"))) if self.failures_dir.exists() else 0
        disk_usage = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) if self.root.exists() else 0
        open_chunks = [
            info
            for info in partitions
            if info.status != "complete"
            and any(item.name.startswith("snapshots-") for item in info.temporary_files)
        ]
        closed_chunks = [info for info in partitions if info.status == "complete"]
        oldest_unuploaded = min(
            (
                _timestamp((info.manifest or {}).get("first_timestamp"))
                for info in closed_chunks
                if not info.uploaded
                and _timestamp((info.manifest or {}).get("first_timestamp")) is not None
            ),
            default=None,
        )
        return {
            "last_successful_snapshot": max(timestamps).isoformat() if timestamps else None,
            "market_snapshots_captured_today": len(today_records),
            "market_replay_eligible_percentage": round(100.0 * len(eligible) / len(today_records), 2) if today_records else 0.0,
            "pending_local_partitions": len(pending),
            "last_successful_remote_upload": last_upload,
            "upload_failures": failures,
            "local_disk_usage_bytes": disk_usage,
            "incomplete_partitions": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(len(info.temporary_files) for info in partitions),
            "partition_minutes": self.partition_minutes,
            "local_minute_chunks_open": len(open_chunks),
            "local_minute_chunks_closed": len(closed_chunks),
            "oldest_unuploaded_chunk": oldest_unuploaded.isoformat() if oldest_unuploaded else None,
            "last_market_snapshot": max(timestamps).isoformat() if timestamps else None,
            "market_snapshots_today": len(today_records),
        }


_DEFAULT_MARKET_STORE: MarketSnapshotStore | None = None
_DEFAULT_MARKET_STORE_LOCK = threading.Lock()


def get_default_market_store() -> MarketSnapshotStore:
    global _DEFAULT_MARKET_STORE
    if _DEFAULT_MARKET_STORE is None:
        with _DEFAULT_MARKET_STORE_LOCK:
            if _DEFAULT_MARKET_STORE is None:
                _DEFAULT_MARKET_STORE = MarketSnapshotStore()
    return _DEFAULT_MARKET_STORE


__all__ = [
    "MarketCaptureResult",
    "MarketPartitionInfo",
    "MarketSnapshotStore",
    "get_default_market_store",
]
