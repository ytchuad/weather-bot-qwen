"""Background worker for closed immutable Layer A Dataset uploads."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any


class LayerAUploadWorker:
    def __init__(self, *, interval_minutes: float | None = None, stores: tuple[Any, ...] | None = None) -> None:
        self.interval_minutes = max(
            0.1,
            float(interval_minutes or os.getenv("HF_LAYER_A_UPLOAD_INTERVAL_MINUTES", "30")),
        )
        self.stores = stores
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run: str | None = None
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(
            os.getenv("HF_LAYER_A_AUTO_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}
            and os.getenv("HF_LAYER_A_REPO_ID", "").strip()
            and os.getenv("HF_LAYER_A_TOKEN", "")
        )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _resolve_stores(self) -> tuple[Any, ...]:
        if self.stores is not None:
            return self.stores
        from .market_storage import get_default_market_store
        from .storage import get_default_store
        from .weather_storage import get_default_weather_store

        return (get_default_store(), get_default_market_store(), get_default_weather_store())

    def run_once(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"closed": 0, "uploaded": 0, "failed": 0}
        try:
            for store in self._resolve_stores():
                close_due = getattr(store, "close_due", None)
                if callable(close_due):
                    result = close_due()
                    summary["closed"] += int(result.get("closed", 0))
                retry = getattr(store, "retry_pending_uploads", None)
                if callable(retry):
                    result = retry()
                    summary["uploaded"] += int(result.get("uploaded", 0))
                    summary["failed"] += int(result.get("failed", 0))
            try:
                from .quality import build_and_write_daily_quality_report

                quality = build_and_write_daily_quality_report()
                summary["quality_report"] = {
                    "report_date": quality.get("report_date"),
                    "gate_passed": quality.get("gate_passed", False),
                    "report_path": quality.get("report_path"),
                }
            except Exception as quality_error:
                summary["quality_report_error"] = type(quality_error).__name__
            self._last_run = datetime.now(timezone.utc).isoformat()
            self._last_error = None
        except Exception as exc:
            self._last_error = type(exc).__name__
        return summary

    def _loop(self) -> None:
        self.run_once()
        while not self._stop.wait(self.interval_minutes * 60.0):
            self.run_once()

    def start(self) -> None:
        if not self.enabled or self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="layer-a-upload-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def health_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "interval_minutes": self.interval_minutes,
            "last_run": self._last_run,
            "last_error": self._last_error,
        }


__all__ = ["LayerAUploadWorker"]
