"""Read-only legacy-vs-CLOB shadow comparison records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from execution.clob_execution import CLOBExecutionSnapshot
from execution.paper_execution_config import get_shadow_log_path


def _decision_by_bucket(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for decision in summary.get("decisions", []) or []:
        bucket = decision.get("bucket")
        if bucket:
            result[str(bucket)] = dict(decision)
    return result


def _edge(model_probability: float, token_side: str, token_price: Any) -> float | None:
    try:
        price = float(token_price)
        probability = float(model_probability)
    except (TypeError, ValueError):
        return None
    if token_side == "YES":
        return probability - price
    return (1.0 - probability) - price


def build_shadow_records(
    *,
    strategy: str,
    model: str,
    target_probs: Mapping[str, float],
    gamma_reference_prices: Mapping[str, float],
    legacy_summary: Mapping[str, Any],
    clob_summary: Mapping[str, Any],
    execution_snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
    snapshot_error: str | None = None,
) -> list[dict[str, Any]]:
    """Build one comparison record per bucket without mutating positions."""
    legacy_positions = legacy_summary.get("target_positions", {}) or {}
    clob_positions = clob_summary.get("target_positions", {}) or {}
    legacy_decisions = _decision_by_bucket(legacy_summary)
    clob_decisions = _decision_by_bucket(clob_summary)
    buckets = sorted(set(target_probs) | set(legacy_positions) | set(clob_positions))
    records: list[dict[str, Any]] = []

    for bucket in buckets:
        legacy_position = legacy_positions.get(bucket) or {}
        clob_position = clob_positions.get(bucket) or {}
        legacy_decision = legacy_decisions.get(bucket, {})
        clob_decision = clob_decisions.get(bucket, {})
        side = str(
            clob_position.get("side")
            or legacy_position.get("side")
            or ("YES" if clob_decision.get("action") == "BUY_YES" else "NO")
        ).upper()
        token_snapshot = (execution_snapshots.get(bucket) or {}).get(side)
        best_bid = token_snapshot.best_bid.price if token_snapshot and token_snapshot.best_bid else None
        best_ask = token_snapshot.best_ask.price if token_snapshot and token_snapshot.best_ask else None
        legacy_price = legacy_position.get("target_price")
        clob_price = clob_position.get("target_price")
        if clob_price is None:
            clob_price = clob_decision.get("depth_adjusted_vwap")
        legacy_edge = _edge(target_probs.get(bucket, 0.5), side, legacy_price)
        clob_edge = clob_decision.get("executable_edge_at_final_size")
        if clob_edge is None:
            clob_edge = _edge(target_probs.get(bucket, 0.5), side, clob_price)
        records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "model": model,
            "bucket": bucket,
            "side": side,
            "model_probability": target_probs.get(bucket),
            "gamma_reference_price": gamma_reference_prices.get(bucket),
            "legacy_simulated_price": legacy_price,
            "clob_best_bid": best_bid,
            "clob_best_ask": best_ask,
            "requested_size": clob_decision.get(
                "requested_shares", clob_position.get("quantity")
            ),
            "depth_adjusted_vwap": clob_price,
            "fee": clob_decision.get("fee"),
            "fill_ratio": clob_decision.get("fill_ratio"),
            "legacy_executable_edge": legacy_edge,
            "clob_executable_edge": clob_edge,
            "legacy_would_trade": bool(legacy_position),
            "clob_would_trade": bool(clob_position),
            "legacy_action": legacy_decision.get("action"),
            "clob_action": clob_decision.get("action"),
            "snapshot_error": snapshot_error,
            "difference_reason": _difference_reason(
                bool(legacy_position), bool(clob_position), legacy_decision, clob_decision,
                snapshot_error,
            ),
        })
    return records


def _difference_reason(
    legacy_would_trade: bool,
    clob_would_trade: bool,
    legacy_decision: Mapping[str, Any],
    clob_decision: Mapping[str, Any],
    snapshot_error: str | None,
) -> str:
    if snapshot_error:
        return f"clob_snapshot_rejected:{snapshot_error}"
    if legacy_would_trade != clob_would_trade:
        return "would_trade_changed"
    if clob_decision.get("reason") in {"NO_EXECUTABLE_QUOTE", "PARTIAL_FILL_REJECTED"}:
        return str(clob_decision["reason"])
    if legacy_decision.get("action") != clob_decision.get("action"):
        return "action_changed"
    if clob_decision.get("depth_adjusted_vwap") is not None:
        return "depth_price_or_fee_changed"
    return "no_difference"


def write_shadow_records(
    records: list[Mapping[str, Any]], path: str | Path | None = None
) -> None:
    """Append JSONL records; this writer never touches positions or balances."""
    if not records:
        return
    target = Path(path) if path is not None else get_shadow_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")
