from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter

from app.api.schemas import BacktestResult, BacktestStatus, BacktestTaskCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])

_backtest_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _run_backtest_worker(task_id: str, initial_capital: float, kelly_fraction: float):
    try:
        from app.services.backtest_service import evaluate_forward_test

        with _lock:
            _backtest_tasks[task_id]["progress"] = 10.0

        result = evaluate_forward_test(
            initial_capital=initial_capital,
            kelly_fraction=kelly_fraction,
        )

        summary = {}
        pnl_curve = []
        brier_scores = []

        for engine, data in result.items():
            summary[engine] = data.get("summary", {})
            pnl_df = data.get("pnl_df")
            if pnl_df is not None and not pnl_df.empty:
                for _, row in pnl_df.iterrows():
                    pnl_curve.append({
                        "engine": engine,
                        "date": str(row.get("date", "")),
                        "pnl": float(row.get("pnl", 0)),
                        "bankroll": float(row.get("bankroll", 0)),
                    })
            brier_df = data.get("brier_df")
            if brier_df is not None and not brier_df.empty:
                for _, row in brier_df.iterrows():
                    brier_scores.append({
                        "engine": engine,
                        "date": str(row.get("date", "")),
                        "brier_score": float(row.get("brier_score", 0)),
                    })

        with _lock:
            _backtest_tasks[task_id] = {
                "status": "done",
                "progress": 100.0,
                "result": BacktestResult(
                    summary=summary,
                    pnl_curve=pnl_curve,
                    brier_scores=brier_scores,
                ),
            }
    except Exception as e:
        logger.exception("Backtest worker failed")
        with _lock:
            _backtest_tasks[task_id] = {
                "status": "error",
                "progress": 0.0,
                "message": str(e),
            }


@router.post("/run", response_model=BacktestTaskCreate)
def run_backtest(initial_capital: float = 10000.0, kelly_fraction: float = 0.25):
    task_id = uuid.uuid4().hex
    with _lock:
        _backtest_tasks[task_id] = {"status": "running", "progress": 0.0}
    thread = threading.Thread(
        target=_run_backtest_worker,
        args=(task_id, initial_capital, kelly_fraction),
        daemon=True,
    )
    thread.start()
    return BacktestTaskCreate(task_id=task_id)


@router.get("/status/{task_id}", response_model=BacktestStatus)
def get_status(task_id: str):
    with _lock:
        task = _backtest_tasks.get(task_id)
    if task is None:
        return BacktestStatus(task_id=task_id, status="error", message="Task not found")
    return BacktestStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        message=task.get("message", ""),
    )


@router.get("/result/{task_id}", response_model=BacktestResult | None)
def get_result(task_id: str):
    with _lock:
        task = _backtest_tasks.get(task_id)
    if task is None:
        return None
    if task["status"] != "done":
        return None
    return task["result"]
