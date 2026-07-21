"""Append-only local persistence for immutable Layer A cycles."""

from __future__ import annotations

import hashlib
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

from .schema import SCHEMA_VERSION, build_layer_a_record, jsonable, validate_layer_a_record

logger = logging.getLogger(__name__)

HKT = timezone(timedelta(hours=8))
_PART_FILE_RE = re.compile(
    r"^(?P<kind>cycles|books|manifest)-(?P<partition>[A-Za-z0-9_-]+)"
    r"(?P<suffix>\.parquet|\.jsonl\.zst|\.json)(?P<temporary>\.tmp)?$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _date_scan_root(root: Path, date_value: str | None) -> Path | None:
    """Restrict a partition scan to one event date when it is provided."""
    if not date_value:
        return root
    date_text = str(date_value)
    if not _DATE_RE.fullmatch(date_text):
        return None
    return root / f"date={date_text}"


def _default_root() -> Path:
    configured = os.getenv("LAYER_A_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        from app.config import LAYER_A_DIR

        return Path(LAYER_A_DIR)
    except ImportError:
        pass
    return Path(__file__).resolve().parents[1] / "data" / "layer_a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=HKT)
    return parsed.astimezone(timezone.utc)


def _json_dump(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_books(path: Path, books: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(f"{_json_dump(book)}\n" for book in books).encode("utf-8")
    try:
        import zstandard as zstd  # type: ignore

        compressor = zstd.ZstdCompressor(level=3)
        path.write_bytes(compressor.compress(payload))
        return "zstd"
    except ImportError:
        # ``zstandard`` is an optional runtime dependency so capture still
        # works in a minimal local test/diagnostic environment.  The manifest
        # records the fallback truthfully; production requirements install
        # zstandard and produce a real .zst stream.
        path.write_bytes(payload)
        return "plain_fallback"


def read_books_bytes(payload: bytes, compression: str | None = None) -> list[dict[str, Any]]:
    """Decode a books payload without creating a second mutable artifact."""
    mode = compression
    if mode is None:
        try:
            import zstandard as zstd  # type: ignore

            payload = zstd.ZstdDecompressor().decompress(payload)
            mode = "zstd"
        except Exception:
            mode = "plain_fallback"
    elif mode == "zstd":
        import zstandard as zstd  # type: ignore

        payload = zstd.ZstdDecompressor().decompress(payload)
    if mode not in {"zstd", "plain_fallback"}:
        raise ValueError(f"unsupported Layer A books compression: {mode}")
    records: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_books(path: Path, compression: str | None = None) -> list[dict[str, Any]]:
    """Read a books file, accepting the documented optional local fallback."""
    return read_books_bytes(path.read_bytes(), compression)


@dataclass
class PartitionInfo:
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

    @property
    def relative_directory(self) -> str:
        return str(self.directory).replace("\\", "/")

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
class CaptureResult:
    status: str
    decision_cycle_id: str
    partition: PartitionInfo | None = None
    record: dict[str, Any] | None = None

    def as_dict(self, root: Path | None = None) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_cycle_id": self.decision_cycle_id,
            "partition": self.partition.as_dict(root) if self.partition else None,
        }


class LayerAStore:
    """Store one immutable cycle per unique closed partition.

    A partition is written through sibling ``.tmp`` files and the manifest is
    renamed last.  If a process crashes between renames, the next startup
    sees the partial files and reports them; it never removes or rewrites
    them.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        uploader: Any = None,
        auto_upload: bool | None = None,
        upload_interval_minutes: float | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
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
        suffix = match.group("suffix")
        expected = {
            "cycles": ".parquet",
            "books": ".jsonl.zst",
            "manifest": ".json",
        }
        if expected[match.group("kind")] != suffix:
            return None
        return match.group("kind"), match.group("partition"), bool(match.group("temporary"))

    def scan(self, date_value: str | None = None) -> list[PartitionInfo]:
        """Scan complete, incomplete and temporary partition state."""
        scan_root = _date_scan_root(self.root, date_value)
        if scan_root is None or not scan_root.exists():
            return []
        groups: dict[tuple[Path, str], PartitionInfo] = {}
        for path in scan_root.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(scan_root).parts):
                continue
            parsed = self._parse_part_file(path)
            if parsed is None:
                continue
            kind, partition_id, temporary = parsed
            key = (path.parent, partition_id)
            info = groups.setdefault(key, PartitionInfo(partition_id, path.parent))
            if temporary:
                info.temporary_files.append(path)
            else:
                info.files[kind] = path

        result: list[PartitionInfo] = []
        for info in groups.values():
            required = {"cycles", "books", "manifest"}
            missing = sorted(required - set(info.files))
            if missing:
                info.reasons.append(f"missing_closed_files:{','.join(missing)}")
            if info.temporary_files:
                info.reasons.append("temporary_files_present")
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
            if not missing and not info.temporary_files and info.manifest is not None and not any(
                reason.startswith("manifest_unreadable") or reason.startswith("manifest_partition_id")
                for reason in info.reasons
            ):
                info.status = "complete"
            else:
                info.status = "incomplete"
            info.uploaded = self._has_receipt(info.partition_id)
            result.append(info)
        return sorted(result, key=lambda item: (str(item.directory), item.partition_id))

    def _verify_manifest(self, info: PartitionInfo) -> tuple[bool, list[str]]:
        manifest = info.manifest or {}
        checksums = manifest.get("file_checksums", manifest.get("checksums", {}))
        if not isinstance(checksums, Mapping):
            return False, ["manifest_checksums_missing"]
        reasons: list[str] = []
        valid = True
        for role in ("cycles", "books"):
            path = info.files.get(role)
            if path is None:
                valid = False
                continue
            expected = checksums.get(path.name)
            if not expected:
                valid = False
                reasons.append(f"checksum_missing:{path.name}")
                continue
            try:
                actual = _sha256(path)
            except OSError as exc:
                valid = False
                reasons.append(f"checksum_read_failed:{path.name}:{type(exc).__name__}")
                continue
            if actual != expected:
                valid = False
                reasons.append(f"checksum_mismatch:{path.name}")
        return valid, reasons

    def _known_cycle_ids(self, partitions: Iterable[PartitionInfo] | None = None) -> set[str]:
        known: set[str] = set()
        for info in partitions if partitions is not None else self.scan():
            manifest = info.manifest or {}
            ids = manifest.get("decision_cycle_ids", [])
            if isinstance(ids, list):
                known.update(str(item) for item in ids)
            path = info.files.get("cycles")
            candidates = [path] if path is not None else []
            candidates.extend(
                item for item in info.temporary_files if item.name.startswith("cycles-")
            )
            for candidate in candidates:
                try:
                    import pandas as pd

                    frame = pd.read_parquet(candidate)
                    if "decision_cycle_id" in frame.columns:
                        known.update(str(item) for item in frame["decision_cycle_id"].dropna().tolist())
                except Exception:
                    # An incomplete file remains visible in scan(); a corrupt
                    # temporary parquet is not silently repaired or removed.
                    continue
        return known

    def capture(self, record: Mapping[str, Any]) -> CaptureResult:
        """Atomically close one new partition, or return ``duplicate``."""
        validate_layer_a_record(record)
        clean = jsonable(record)
        decision_cycle_id = str(clean["decision_cycle_id"])
        with self._lock:
            partitions = self.scan()
            if decision_cycle_id in self._known_cycle_ids(partitions):
                return CaptureResult("duplicate", decision_cycle_id, record=clean)

            decision = _timestamp(clean.get("decision_timestamp")) or datetime.now(timezone.utc)
            local = decision.astimezone(HKT)
            directory = self.root / f"date={clean['event_date']}" / f"hour={local.hour:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            partition_id = uuid.uuid4().hex
            cycles_path = directory / f"cycles-{partition_id}.parquet"
            books_path = directory / f"books-{partition_id}.jsonl.zst"
            manifest_path = directory / f"manifest-{partition_id}.json"
            cycles_tmp = Path(f"{cycles_path}.tmp")
            books_tmp = Path(f"{books_path}.tmp")
            manifest_tmp = Path(f"{manifest_path}.tmp")

            row = {
                "decision_cycle_id": clean["decision_cycle_id"],
                "schema_version": clean["schema_version"],
                "decision_timestamp": clean["decision_timestamp"],
                "capture_timestamp": clean["capture_timestamp"],
                "event_date": clean["event_date"],
                "location": clean["location"],
                "market_kind": clean["market_kind"],
                "event_slug": clean["event_slug"],
                "weather_state_json": _json_dump(clean["weather_state"]),
                "models_json": _json_dump(clean["models"]),
                "market_identity_json": _json_dump(clean["market_identity"]),
                "completeness_json": _json_dump(clean["completeness"]),
                "source_status_json": _json_dump(clean.get("source_status", {})),
                "gamma_reference_prices_json": _json_dump(clean.get("gamma_reference_prices", {})),
                # Keeping the complete record in one column makes exported
                # cycles self-describing; the book file is the exact token
                # level stream used for book-oriented replay.
                "record_json": _json_dump(clean),
            }
            try:
                import pandas as pd

                pd.DataFrame([row]).to_parquet(cycles_tmp, index=False, engine="pyarrow")
                books_payload = [
                    {**book, "decision_cycle_id": decision_cycle_id}
                    for book in clean.get("clob_books", [])
                    if isinstance(book, Mapping)
                ]
                compression = _write_books(books_tmp, books_payload)
                cycles_hash = _sha256(cycles_tmp)
                books_hash = _sha256(books_tmp)
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "partition_id": partition_id,
                    "partition_relative_path": str(directory.relative_to(self.root)).replace("\\", "/"),
                    "files": {
                        "cycles": cycles_path.name,
                        "books": books_path.name,
                        "manifest": manifest_path.name,
                    },
                    "file_checksums": {
                        cycles_path.name: cycles_hash,
                        books_path.name: books_hash,
                    },
                    "file_sizes": {
                        cycles_path.name: cycles_tmp.stat().st_size,
                        books_path.name: books_tmp.stat().st_size,
                    },
                    "first_timestamp": clean["decision_timestamp"],
                    "last_timestamp": clean["decision_timestamp"],
                    "cycle_count": 1,
                    "decision_cycle_ids": [decision_cycle_id],
                    "books_compression": compression,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "closed_at": None,
                    "checksum_scope": "cycles_and_books; manifest hash is calculated by export checksums",
                }
                manifest["closed_at"] = datetime.now(timezone.utc).isoformat()
                manifest_tmp.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
                # Manifest is renamed last.  Any crash before this line leaves
                # a detectable temporary/incomplete partition.
                os.replace(cycles_tmp, cycles_path)
                os.replace(books_tmp, books_path)
                os.replace(manifest_tmp, manifest_path)
            except Exception:
                logger.exception("Layer A partition close failed for cycle %s", decision_cycle_id)
                # Do not clean up .tmp/final partial files: startup scan must
                # expose them for operator recovery.
                raise

            info = PartitionInfo(
                partition_id=partition_id,
                directory=directory,
                files={"cycles": cycles_path, "books": books_path, "manifest": manifest_path},
                manifest=manifest,
                status="complete",
                checksum_valid=True,
            )
            self._maybe_auto_upload(info)
            return CaptureResult("captured", decision_cycle_id, partition=info, record=clean)

    def capture_context(self, context: Mapping[str, Any], **kwargs: Any) -> CaptureResult:
        return self.capture(build_layer_a_record(context, **kwargs))

    def read_partition_records(self, info: PartitionInfo) -> list[dict[str, Any]]:
        if info.status != "complete":
            return []
        path = info.files.get("cycles")
        if path is None:
            return []
        import pandas as pd

        frame = pd.read_parquet(path)
        records: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            raw = row.get("record_json")
            if isinstance(raw, str):
                records.append(json.loads(raw))
        return records

    def iter_records(
        self,
        *,
        date_value: str | None = None,
        start: str | None = None,
        end: str | None = None,
        only_unuploaded: bool = False,
        verify_checksums: bool = False,
    ) -> list[tuple[PartitionInfo, dict[str, Any]]]:
        result: list[tuple[PartitionInfo, dict[str, Any]]] = []
        start_dt = _timestamp(start) if start else None
        end_dt = _timestamp(end) if end else None
        for info in self.scan(date_value=date_value):
            if info.status != "complete":
                continue
            if only_unuploaded and info.uploaded:
                continue
            if verify_checksums and info.checksum_valid is not True:
                raise ValueError(f"checksum verification failed for partition {info.partition_id}")
            records = self.read_partition_records(info)
            for record in records:
                if date_value and record.get("event_date") != date_value:
                    continue
                decision = _timestamp(record.get("decision_timestamp"))
                if start_dt and (decision is None or decision < start_dt):
                    continue
                if end_dt and (decision is None or decision > end_dt):
                    continue
                result.append((info, record))
        return result

    def latest_completed_model_record(
        self,
        *,
        event_date: str,
        event_slug: str,
        market_kind: str,
        before_timestamp: Any = None,
    ) -> dict[str, Any] | None:
        """Return the latest persisted model cycle at or before a market time."""
        before = _timestamp(before_timestamp) if before_timestamp is not None else None
        best_record: dict[str, Any] | None = None
        best_timestamp: datetime | None = None
        for _info, record in self.iter_records(date_value=event_date):
            if record.get("event_slug") != event_slug or record.get("market_kind") != market_kind:
                continue
            decision = _timestamp(record.get("decision_timestamp"))
            if decision is None or (before is not None and decision > before):
                continue
            if best_timestamp is None or decision > best_timestamp:
                best_timestamp = decision
                best_record = record
        return best_record

    def _configured_uploader(self) -> Any:
        if self._uploader is not None:
            return self._uploader
        if not os.getenv("HF_LAYER_A_REPO_ID") or not os.getenv("HF_LAYER_A_TOKEN"):
            return None
        from .upload import DatasetUploader

        self._uploader = DatasetUploader.from_environment()
        return self._uploader

    def _record_upload_failure(self, info: PartitionInfo, error: Exception) -> None:
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        path = self.failures_dir / f"{info.partition_id}-{uuid.uuid4().hex}.json"
        message = str(error)
        token = os.getenv("HF_LAYER_A_TOKEN")
        if token:
            message = message.replace(token, "[REDACTED]")
        _write_text_atomic(
            path,
            _json_dump(
                {
                    "partition_id": info.partition_id,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": message,
                }
            )
            + "\n",
        )

    def _maybe_auto_upload(self, info: PartitionInfo) -> None:
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
            uploader.upload_partition(info, self.root)
        except Exception as exc:
            self._record_upload_failure(info, exc)
            logger.warning("Layer A remote upload failed for partition %s", info.partition_id)

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
                uploader.upload_partition(info, self.root)
                uploaded += 1
            except Exception as exc:
                failed += 1
                self._record_upload_failure(info, exc)
                logger.warning("Layer A pending upload failed for partition %s", info.partition_id)
        return {"configured": True, "attempted": attempted, "uploaded": uploaded, "failed": failed}

    def startup_scan(self) -> dict[str, Any]:
        partitions = self.scan()
        retry = self.retry_pending_uploads() if self._auto_upload else {"configured": False, "attempted": 0, "uploaded": 0, "failed": 0}
        return {
            "partitions": len(partitions),
            "complete": sum(info.status == "complete" for info in partitions),
            "incomplete": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(bool(info.temporary_files) for info in partitions),
            "upload_retry": retry,
        }

    def health_summary(self) -> dict[str, Any]:
        partitions = self.scan()
        now_local = datetime.now(timezone.utc).astimezone(HKT)
        records: list[dict[str, Any]] = []
        for info in partitions:
            if info.status == "complete":
                records.extend(self.read_partition_records(info))
        today = now_local.date().isoformat()
        today_records = [record for record in records if record.get("event_date") == today]
        eligible = [
            record for record in today_records
            if (record.get("completeness") or {}).get("replay_eligible_for_clob_strategy") is True
        ]
        pending = [info for info in partitions if info.status == "complete" and not info.uploaded]
        timestamps = [
            _timestamp(record.get("decision_timestamp"))
            for record in records
            if _timestamp(record.get("decision_timestamp")) is not None
        ]
        latest = max(timestamps).isoformat() if timestamps else None
        oldest_pending = None
        pending_times: list[datetime] = []
        for info in pending:
            value = (info.manifest or {}).get("first_timestamp")
            parsed = _timestamp(value)
            if parsed is not None:
                pending_times.append(parsed)
        if pending_times:
            oldest_pending = min(pending_times).isoformat()
        receipts = []
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
        disk_usage = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) if self.root.exists() else 0
        failure_count = len(list(self.failures_dir.glob("*.json"))) if self.failures_dir.exists() else 0
        return {
            "last_successful_cycle": latest,
            "last_model_cycle": latest,
            "cycles_captured_today": len(today_records),
            "model_cycles_today": len(today_records),
            "clob_replay_eligible_percentage": round(100.0 * len(eligible) / len(today_records), 2) if today_records else 0.0,
            "pending_local_partitions": len(pending),
            "last_successful_remote_upload": last_upload,
            "upload_failures": failure_count,
            "local_disk_usage_bytes": disk_usage,
            "oldest_unuploaded_partition": oldest_pending,
            "incomplete_partitions": sum(info.status != "complete" for info in partitions),
            "temporary_files": sum(len(info.temporary_files) for info in partitions),
        }


_DEFAULT_STORE: LayerAStore | None = None
_DEFAULT_STORE_LOCK = threading.Lock()


def get_default_store() -> LayerAStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = LayerAStore()
    return _DEFAULT_STORE


__all__ = [
    "CaptureResult",
    "LayerAStore",
    "PartitionInfo",
    "get_default_store",
    "read_books",
    "read_books_bytes",
]
