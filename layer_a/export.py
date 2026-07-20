"""Downloadable Layer A export bundle creation."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION
from .market_schema import MARKET_SCHEMA_VERSION
from .market_storage import MarketPartitionInfo, MarketSnapshotStore, get_default_market_store
from .storage import LayerAStore, PartitionInfo


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalise_boundary(value: str | None, *, end: bool = False) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return f"{raw}T23:59:59.999999+08:00" if end else f"{raw}T00:00:00+08:00"
    return raw


def _partition_member(root: Path, path: Path, prefix: str = "layer_a") -> str:
    return f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"


def export_layer_a(
    *,
    store: LayerAStore | None = None,
    market_store: MarketSnapshotStore | None = None,
    output: Path | str | None = None,
    date_value: str | None = None,
    start: str | None = None,
    end: str | None = None,
    only_unuploaded: bool = False,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    """Create one archive without mutating any closed Layer A partition."""
    store = store or LayerAStore()
    market_store = market_store or get_default_market_store()
    start_boundary = _normalise_boundary(start)
    end_boundary = _normalise_boundary(end, end=True)
    selected: dict[str, PartitionInfo] = {}
    records_count = 0
    market_selected: dict[str, MarketPartitionInfo] = {}
    market_records_count = 0
    # Iterating records applies date/time filters while selecting a partition
    # only once even if a future multi-row partition contains many cycles.
    for info, _record in store.iter_records(
        date_value=date_value,
        start=start_boundary,
        end=end_boundary,
        only_unuploaded=only_unuploaded,
        verify_checksums=verify_checksums,
    ):
        selected[info.partition_id] = info
        records_count += 1

    for info, _snapshot in market_store.read_snapshot_records(
        date_value=date_value,
        start=start_boundary,
        end=end_boundary,
        only_unuploaded=only_unuploaded,
        verify_checksums=verify_checksums,
    ):
        market_selected[info.partition_id] = info
        market_records_count += 1

    all_partitions = store.scan()
    pending = [
        info.partition_id
        for info in all_partitions
        if info.status == "complete" and not info.uploaded
    ]
    incomplete = [info.as_dict(store.root) for info in all_partitions if info.status != "complete"]
    all_market_partitions = market_store.scan()
    market_pending = [
        info.partition_id
        for info in all_market_partitions
        if info.status == "complete" and not info.uploaded
    ]
    market_incomplete = [
        info.as_dict(market_store.root)
        for info in all_market_partitions
        if info.status != "complete"
    ]
    members: dict[str, bytes] = {}
    selected_manifest_data: list[dict[str, Any]] = []
    for info in selected.values():
        selected_manifest_data.append(
            {
                "partition_id": info.partition_id,
                "directory": str(info.directory.relative_to(store.root)).replace("\\", "/"),
                "uploaded": info.uploaded,
                "files": {},
                "first_timestamp": (info.manifest or {}).get("first_timestamp"),
                "last_timestamp": (info.manifest or {}).get("last_timestamp"),
                "cycle_count": (info.manifest or {}).get("cycle_count", 0),
            }
        )
        for path in info.files.values():
            member = _partition_member(store.root, path)
            members[member] = path.read_bytes()
            selected_manifest_data[-1]["files"][path.name] = {
                "sha256": _sha256_bytes(members[member]),
                "bytes": len(members[member]),
            }

    selected_market_manifest_data: list[dict[str, Any]] = []
    for info in market_selected.values():
        selected_market_manifest_data.append(
            {
                "partition_id": info.partition_id,
                "directory": str(info.directory.relative_to(market_store.root)).replace("\\", "/"),
                "uploaded": info.uploaded,
                "files": {},
                "first_timestamp": (info.manifest or {}).get("first_timestamp"),
                "last_timestamp": (info.manifest or {}).get("last_timestamp"),
                "snapshot_count": (info.manifest or {}).get("snapshot_count", 0),
            }
        )
        for path in info.files.values():
            member = _partition_member(market_store.root, path, "layer_a_market")
            members[member] = path.read_bytes()
            selected_market_manifest_data[-1]["files"][path.name] = {
                "sha256": _sha256_bytes(members[member]),
                "bytes": len(members[member]),
            }

    now = datetime.now(timezone.utc).isoformat()
    export_manifest = {
        "schema_version": "layer_a.export_manifest.v1",
        "layer_a_schema_version": SCHEMA_VERSION,
        "layer_a_market_schema_version": MARKET_SCHEMA_VERSION,
        "created_at": now,
        "filters": {
            "date": date_value,
            "start": start,
            "end": end,
            "only_unuploaded": only_unuploaded,
            "verify_checksums": verify_checksums,
        },
        "cycle_count": records_count,
        "partitions": selected_manifest_data,
        "market_snapshot_count": market_records_count,
        "market_partitions": selected_market_manifest_data,
        "unuploaded_partition_list": pending,
        "unuploaded_market_partition_list": market_pending,
        "incomplete_partition_list": incomplete,
        "incomplete_market_partition_list": market_incomplete,
        "notes": [
            "Only closed immutable partitions are included as payload files.",
            "Incomplete temporary partitions remain on local storage and are listed for recovery.",
            "No paper-account, position, order, fill or PnL state is exported by Layer A.",
        ],
    }
    members["export_manifest.json"] = json.dumps(
        export_manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    checksum_lines = [
        f"{digest}  {name}"
        for name, payload in sorted(members.items())
        for digest in [_sha256_bytes(payload)]
    ]
    members["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    members["docs/layer_a_schema.md"] = _schema_doc_bytes()
    # Add the schema doc to the checksums after its bytes are known.
    checksum_lines = [
        f"{_sha256_bytes(payload)}  {name}"
        for name, payload in sorted(members.items())
        if name != "checksums.sha256"
    ]
    members["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    output_path = Path(output) if output is not None else store.root.parent / f"layer_a_export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(name, payload)
        os.replace(temporary, output_path)
    except Exception:
        # Preserve a failed temp archive for operator diagnosis, just as local
        # partition temp files are preserved after an interrupted close.
        raise
    return {
        "output": str(output_path),
        "cycle_count": records_count,
        "market_snapshot_count": market_records_count,
        "partition_count": len(selected),
        "market_partition_count": len(market_selected),
        "unuploaded_partition_count": len(pending),
        "unuploaded_market_partition_count": len(market_pending),
        "incomplete_partition_count": len(incomplete),
        "incomplete_market_partition_count": len(market_incomplete),
        "members": sorted(members),
    }


def _schema_doc_bytes() -> bytes:
    path = Path(__file__).resolve().parents[1] / "docs" / "layer_a_schema.md"
    if path.exists():
        return path.read_bytes()
    return b"# layer_a.v1\n"


__all__ = ["export_layer_a"]
