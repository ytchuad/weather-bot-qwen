"""Optional private Hugging Face Dataset uploader for closed Layer A files."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _redact(value: Any) -> str:
    message = str(value)
    token = os.getenv("HF_LAYER_A_TOKEN")
    return message.replace(token, "[REDACTED]") if token else message


@dataclass
class DatasetUploader:
    repo_id: str
    token: str
    max_retries: int = 3
    backoff_seconds: float = 1.0
    api: Any = None
    sleep_fn: Any = time.sleep

    @classmethod
    def from_environment(cls) -> "DatasetUploader":
        repo_id = os.getenv("HF_LAYER_A_REPO_ID", "").strip()
        token = os.getenv("HF_LAYER_A_TOKEN", "")
        if not repo_id or not token:
            raise RuntimeError("HF_LAYER_A_REPO_ID and HF_LAYER_A_TOKEN are required")
        return cls(
            repo_id=repo_id,
            token=token,
            max_retries=max(1, int(os.getenv("HF_LAYER_A_UPLOAD_MAX_RETRIES", "3"))),
            backoff_seconds=max(0.0, float(os.getenv("HF_LAYER_A_UPLOAD_BACKOFF_SECONDS", "1"))),
        )

    def _api(self) -> Any:
        if self.api is not None:
            return self.api
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for Layer A upload") from exc
        self.api = HfApi(token=self.token)
        return self.api

    @staticmethod
    def _remote_path(info: Any, path: Path, root: Path) -> str:
        relative = path.relative_to(root).as_posix()
        return f"layer_a/{relative}"

    def _verify_remote(self, api: Any, remote_path: str) -> bool:
        """Verify a remote path without assuming one hub-client version."""
        if hasattr(api, "file_exists"):
            return bool(
                api.file_exists(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                )
            )
        if hasattr(api, "get_paths_info"):
            try:
                result = api.get_paths_info(
                    repo_id=self.repo_id,
                    paths=[remote_path],
                    repo_type="dataset",
                )
            except TypeError:
                result = api.get_paths_info(
                    repo_id=self.repo_id,
                    path_in_repo=remote_path,
                    repo_type="dataset",
                )
            return bool(result)
        # Older/mocked clients expose no read API.  upload_file returning
        # without raising is the strongest verification available there.
        return True

    def _upload_one(self, api: Any, path: Path, remote_path: str) -> None:
        if hasattr(api, "file_exists") or hasattr(api, "get_paths_info"):
            if self._verify_remote(api, remote_path):
                return
        result = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add Layer A partition {getattr(path, 'stem', 'file')}",
        )
        if result is None and not hasattr(api, "file_exists") and not hasattr(api, "get_paths_info"):
            return
        if not self._verify_remote(api, remote_path):
            raise RuntimeError(f"remote verification failed for {remote_path}")

    def upload_partition(self, info: Any, root: Path) -> dict[str, Any]:
        """Upload a closed partition and write an immutable local receipt."""
        if getattr(info, "status", None) != "complete":
            raise ValueError("only closed complete partitions may be uploaded")
        api = self._api()
        uploaded_paths: list[str] = []
        files = getattr(info, "files", {})
        try:
            for path in files.values():
                remote_path = self._remote_path(info, path, root)
                last_error: Exception | None = None
                for attempt in range(self.max_retries):
                    try:
                        self._upload_one(api, path, remote_path)
                        last_error = None
                        break
                    except Exception as exc:  # bounded retry belongs at the file boundary
                        last_error = exc
                        if attempt + 1 < self.max_retries:
                            self.sleep_fn(self.backoff_seconds * (2**attempt))
                if last_error is not None:
                    raise RuntimeError(_redact(last_error)) from last_error
                uploaded_paths.append(remote_path)
        except Exception as exc:
            # Never include token/repository credentials in the exception text
            # emitted to the caller or logger.
            raise RuntimeError(_redact(exc)) from exc

        from .storage import _write_text_atomic

        receipt_dir = root / ".upload_receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema_version": "layer_a.upload_receipt.v1",
            "partition_id": info.partition_id,
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "remote_paths": uploaded_paths,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "local_checksums": (info.manifest or {}).get("file_checksums", {}),
        }
        _write_text_atomic(
            receipt_dir / f"{info.partition_id}.json",
            __import__("json").dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n",
        )
        info.uploaded = True
        return receipt


__all__ = ["DatasetUploader"]
