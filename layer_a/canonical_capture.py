"""Independent five-minute canonical model-cycle capture scheduler."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class CanonicalCycleCollector:
    """Warm/persist the shared canonical cycles without account execution."""

    def __init__(self, *, interval_seconds: float = 300.0) -> None:
        self.interval_seconds = max(60.0, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._runs = 0
        self._failed_runs = 0
        self._last_success: str | None = None
        self._last_failure: str | None = None
        self._last_run: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict[str, Any]:
        report: dict[str, Any] = {"cycles": [], "errors": []}
        try:
            from app.services.canonical_cycle import get_canonical_cycle

            for is_min_temp in (False, True):
                try:
                    cycle = get_canonical_cycle(is_min_temp=is_min_temp)
                    report["cycles"].append(
                        {
                            "decision_cycle_id": cycle.decision_cycle_id,
                            "decision_timestamp": cycle.decision_timestamp.isoformat(),
                            "market_kind": "lowest_temperature" if is_min_temp else "highest_temperature",
                        }
                    )
                except Exception as exc:
                    report["errors"].append(
                        {
                            "market_kind": "lowest_temperature" if is_min_temp else "highest_temperature",
                            "error": type(exc).__name__,
                        }
                    )
        except Exception as exc:
            report["errors"].append({"stage": "collector", "error": type(exc).__name__})
        with self._lock:
            self._runs += 1
            self._last_run = report
            if report["cycles"]:
                self._last_success = datetime.now(timezone.utc).isoformat()
            if report["errors"] and not report["cycles"]:
                self._failed_runs += 1
                self._last_failure = datetime.now(timezone.utc).isoformat()
        return report

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self.run_once()
            self._stop.wait(max(0.1, self.interval_seconds - (time.monotonic() - started)))

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="layer-a-canonical-cycle")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def health_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "interval_seconds": self.interval_seconds,
                "runs": self._runs,
                "failed_runs": self._failed_runs,
                "last_success": self._last_success,
                "last_failure": self._last_failure,
                "last_run": self._last_run,
            }


_DEFAULT_CANONICAL_COLLECTOR: CanonicalCycleCollector | None = None
_DEFAULT_CANONICAL_COLLECTOR_LOCK = threading.Lock()


def get_default_canonical_collector() -> CanonicalCycleCollector:
    global _DEFAULT_CANONICAL_COLLECTOR
    if _DEFAULT_CANONICAL_COLLECTOR is None:
        with _DEFAULT_CANONICAL_COLLECTOR_LOCK:
            if _DEFAULT_CANONICAL_COLLECTOR is None:
                _DEFAULT_CANONICAL_COLLECTOR = CanonicalCycleCollector()
    return _DEFAULT_CANONICAL_COLLECTOR


__all__ = ["CanonicalCycleCollector", "get_default_canonical_collector"]
