"""Strict replay eligibility audit for historical strategy exports.

The historical CSVs contain depth diagnostics, but a replay is executable only
when the exported row also contains the canonical market/token/book metadata.
This command never fills missing metadata from Gamma and never writes runtime
state.  It emits JSON to stdout so a report or CI job can consume the result.

Example::

    python scripts/replay_clob_execution.py \
        --start 2026-07-14 --end 2026-07-19
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from execution.clob_execution import (  # noqa: E402
    SnapshotValidationError,
    build_execution_snapshots,
)


DEFAULT_EXPORT_DIR = Path("data/export")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _model_probs(context: dict[str, Any], model_key: str) -> dict[str, float] | None:
    direct = context.get("target_probs")
    if isinstance(direct, dict) and direct:
        return direct
    all_probs = context.get("model_probs")
    if isinstance(all_probs, dict):
        selected = all_probs.get(model_key)
        if isinstance(selected, dict) and selected:
            return selected
    return None


def _markets_from_export(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct markets only from persisted canonical metadata.

    ``gamma_market_info`` is accepted because current snapshots persist the
    same market metadata there.  The function does not infer NO token IDs,
    outcomes, tick sizes, or minimum sizes.
    """
    raw = context.get("markets")
    if isinstance(raw, list) and raw:
        return [dict(m) for m in raw if isinstance(m, dict)]
    info = context.get("gamma_market_info")
    if not isinstance(info, dict):
        return []
    markets: list[dict[str, Any]] = []
    for bucket, metadata in info.items():
        if isinstance(metadata, dict):
            markets.append({"bucket": str(bucket), **metadata})
    return markets


def _metadata_missing(
    markets: list[dict[str, Any]],
    yes_depth: dict[str, Any],
    no_depth: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if not markets:
        missing.append("market_metadata")
    for market in markets:
        bucket = str(market.get("bucket", ""))
        if not market.get("conditionId") and not market.get("condition_id"):
            missing.append(f"{bucket}:condition_id")
        if not market.get("token_id"):
            missing.append(f"{bucket}:yes_token_id")
        if not market.get("no_token_id"):
            missing.append(f"{bucket}:no_token_id")
        outcomes = market.get("outcomes")
        if outcomes in (None, "", []):
            missing.append(f"{bucket}:outcomes")
        if not any(market.get(key) is not None for key in (
            "tick_size", "orderPriceMinTickSize"
        )):
            missing.append(f"{bucket}:tick_size")
        if not any(market.get(key) is not None for key in (
            "minimum_order_size", "orderMinSize", "minimumOrderSize"
        )):
            missing.append(f"{bucket}:minimum_order_size")
    if not context.get("depth_fetch_cycle_id"):
        missing.append("fetch_cycle_id")
    for side_name, depth in (("yes", yes_depth), ("no", no_depth)):
        for bucket, book in depth.items():
            if not isinstance(book, dict):
                missing.append(f"{side_name}:{bucket}:book")
                continue
            for key in ("asset_id", "timestamp", "fetch_cycle_id", "source_name"):
                if book.get(key) in (None, ""):
                    missing.append(f"{side_name}:{bucket}:{key}")
    return missing


def _book_stats(yes_depth: dict[str, Any], no_depth: dict[str, Any]) -> dict[str, int]:
    both_books = 0
    yes_asks = no_asks = yes_bids = no_bids = 0
    for bucket in set(yes_depth) & set(no_depth):
        yes = yes_depth.get(bucket) or {}
        no = no_depth.get(bucket) or {}
        if not isinstance(yes, dict) or not isinstance(no, dict):
            continue
        both_books += 1
        yes_asks += int(bool(yes.get("top_asks") or yes.get("asks")))
        no_asks += int(bool(no.get("top_asks") or no.get("asks")))
        yes_bids += int(bool(yes.get("top_bids") or yes.get("bids")))
        no_bids += int(bool(no.get("top_bids") or no.get("bids")))
    return {
        "bucket_pairs_with_both_books": both_books,
        "yes_buckets_with_asks": yes_asks,
        "no_buckets_with_asks": no_asks,
        "yes_buckets_with_bids": yes_bids,
        "no_buckets_with_bids": no_bids,
    }


def _book_age_status(
    row: dict[str, str],
    context: dict[str, Any],
    max_book_age_seconds: float = 60.0,
) -> str:
    yes_depth = context.get("market_depth")
    no_depth = context.get("market_depth_no")
    if not isinstance(yes_depth, dict) or not isinstance(no_depth, dict):
        return "missing_yes_or_no_depth"
    try:
        decision = _parse_datetime(
            context.get("decision_timestamp") or row.get("timestamp", "")
        )
    except (TypeError, ValueError):
        return "invalid_decision_timestamp"
    ages: list[float] = []
    for books in (yes_depth, no_depth):
        for book in books.values():
            if not isinstance(book, dict) or book.get("timestamp") in (None, ""):
                return "missing_book_timestamp"
            try:
                numeric = float(book["timestamp"])
                if numeric > 100_000_000_000:
                    numeric /= 1000.0
                book_time = datetime.fromtimestamp(numeric, tz=decision.tzinfo)
                ages.append((decision - book_time).total_seconds())
            except (TypeError, ValueError, OverflowError, OSError):
                return "invalid_book_timestamp"
    if any(age < 0.0 for age in ages):
        return "future_book"
    if any(age > max_book_age_seconds for age in ages):
        return "stale_book"
    return "within_max_age"


def _has_mismatched_books(context: dict[str, Any]) -> bool:
    yes_depth = context.get("market_depth")
    no_depth = context.get("market_depth_no")
    if not isinstance(yes_depth, dict) or not isinstance(no_depth, dict):
        return False
    if set(yes_depth) != set(no_depth):
        return True
    for bucket in yes_depth:
        yes = yes_depth.get(bucket) or {}
        no = no_depth.get(bucket) or {}
        if not isinstance(yes, dict) or not isinstance(no, dict):
            return True
        # Historical exports do not retain fetch_cycle_id, so timestamp
        # disagreement is the only available coherence check here.
        if yes.get("timestamp") != no.get("timestamp"):
            return True
    return False


def _missing_metadata_reason(missing: list[str]) -> str:
    market_fields = (
        "condition_id", "yes_token_id", "no_token_id", "outcomes",
        "tick_size", "minimum_order_size", "market_metadata",
    )
    book_fields = ("asset_id", "timestamp", "fetch_cycle_id", "source_name")
    has_market = any(
        item == "market_metadata" or item.rsplit(":", 1)[-1] in market_fields
        for item in missing
    )
    has_book = any(
        item == "fetch_cycle_id" or item.rsplit(":", 1)[-1] in book_fields
        for item in missing
    )
    if has_market and has_book:
        return "missing_canonical_market_and_book_metadata"
    if has_market:
        return "missing_canonical_market_metadata"
    if has_book:
        return "missing_canonical_book_metadata"
    return missing[0] if missing else "missing_metadata"


def _row_result(row: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    context_raw = row.get("context_json") or "{}"
    try:
        context = json.loads(context_raw)
    except json.JSONDecodeError:
        return {"status": "rejected", "reason": "invalid_context_json"}, [
            "invalid_context_json"
        ]
    if not isinstance(context, dict):
        return {"status": "rejected", "reason": "context_not_object"}, [
            "context_not_object"
        ]

    yes_depth = context.get("market_depth")
    no_depth = context.get("market_depth_no")
    age_status = _book_age_status(row, context)
    mismatched_books = _has_mismatched_books(context)
    if not isinstance(yes_depth, dict) or not isinstance(no_depth, dict):
        return {
            "status": "rejected",
            "reason": "missing_yes_or_no_depth",
            "book_age_status": age_status,
            "mismatched_books": mismatched_books,
        }, [
            "missing_yes_or_no_depth"
        ]

    markets = _markets_from_export(context)
    missing = _metadata_missing(markets, yes_depth, no_depth, context)
    target_probs = _model_probs(context, row.get("model_key", ""))
    if target_probs is None:
        missing.append("model_probabilities")
    if missing:
        return {
            "status": "rejected",
            "reason": _missing_metadata_reason(missing),
            "missing_metadata": missing,
            "book_age_status": age_status,
            "mismatched_books": mismatched_books,
            **_book_stats(yes_depth, no_depth),
        }, [_missing_metadata_reason(missing)]

    try:
        snapshots = build_execution_snapshots(
            markets=markets,
            target_probs=target_probs or {},
            market_depth=yes_depth,
            market_depth_no=no_depth,
            event_slug=row.get("slug", ""),
            decision_timestamp=_parse_datetime(
                context.get("decision_timestamp") or row.get("timestamp", "")
            ),
            expected_market_date=_parse_date(row["snapshot_date"]),
            fetch_cycle_id=context.get("depth_fetch_cycle_id"),
            is_min_temp=row.get("slug", "").startswith("lowest-temperature-"),
        )
    except (SnapshotValidationError, KeyError, TypeError, ValueError) as exc:
        reason = str(exc).split(":", 1)[0]
        return {
            "status": "rejected",
            "reason": reason,
            "book_age_status": age_status,
            "mismatched_books": mismatched_books,
            **_book_stats(yes_depth, no_depth),
        }, [reason]

    return {
        "status": "eligible",
        "reason": "",
        "book_age_status": age_status,
        "mismatched_books": mismatched_books,
        "snapshot_buckets": len(snapshots),
        **_book_stats(yes_depth, no_depth),
    }, []


def run_replay_audit(
    export_dir: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    csv.field_size_limit(10_000_000)
    totals = Counter()
    reasons = Counter()
    missing_fields = Counter()
    by_day: dict[str, Counter] = defaultdict(Counter)
    by_strategy_day: dict[str, Counter] = defaultdict(Counter)
    depth_stats = Counter()
    files: list[str] = []

    current = start
    while current <= end:
        path = export_dir / f"{current.isoformat()}.csv"
        if not path.exists():
            current = current.fromordinal(current.toordinal() + 1)
            continue
        files.append(str(path))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                result, _ = _row_result(row)
                day = row.get("snapshot_date") or current.isoformat()
                strategy = row.get("strategy_key") or "unknown"
                totals["rows"] += 1
                by_day[day]["rows"] += 1
                by_strategy_day[f"{day}|{strategy}"]["rows"] += 1
                if result.get("bucket_pairs_with_both_books", 0):
                    totals["book_rows"] += 1
                    by_day[day]["book_rows"] += 1
                    by_strategy_day[f"{day}|{strategy}"]["book_rows"] += 1
                age_status = result.get("book_age_status")
                if age_status:
                    totals[f"{age_status}_rows"] += 1
                    by_day[day][f"{age_status}_rows"] += 1
                    by_strategy_day[f"{day}|{strategy}"][f"{age_status}_rows"] += 1
                if result.get("mismatched_books"):
                    totals["mismatched_book_rows"] += 1
                    by_day[day]["mismatched_book_rows"] += 1
                    by_strategy_day[f"{day}|{strategy}"]["mismatched_book_rows"] += 1
                for key in (
                    "yes_buckets_with_asks", "no_buckets_with_asks",
                    "yes_buckets_with_bids", "no_buckets_with_bids",
                ):
                    depth_stats[key] += result.get(key, 0)
                status = result.get("status")
                totals[status] += 1
                by_day[day][status] += 1
                by_strategy_day[f"{day}|{strategy}"][status] += 1
                if status == "rejected":
                    reasons[result.get("reason", "unknown")] += 1
                    for field in result.get("missing_metadata", []):
                        missing_fields[field.rsplit(":", 1)[-1]] += 1
                    by_day[day]["rejected_rows"] += 1
                    by_strategy_day[f"{day}|{strategy}"]["rejected_rows"] += 1
                elif status == "eligible":
                    by_day[day]["eligible_rows"] += 1
                    by_strategy_day[f"{day}|{strategy}"]["eligible_rows"] += 1
        current = current.fromordinal(current.toordinal() + 1)

    # These exports are periodic context snapshots, not persisted order
    # decisions.  Never call a row a changed trade when target orders are not
    # available for both the legacy and CLOB variants.
    totals["comparable_trade_decisions"] = 0
    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "files": files,
        "totals": dict(totals),
        "rejection_reasons": reasons.most_common(),
        "missing_metadata_fields": missing_fields.most_common(),
        "book_diagnostics": dict(depth_stats),
        "by_day": {key: dict(value) for key, value in sorted(by_day.items())},
        "by_strategy_day": {
            key: dict(value) for key, value in sorted(by_strategy_day.items())
        },
        "decision_comparison": {
            "status": "not_computable",
            "trade_decisions_changed": None,
            "average_vwap_slippage": None,
            "fill_ratio": None,
            "fee_difference": None,
            "turnover_difference": None,
            "realized_pnl_difference": None,
            "unrealized_pnl_difference": None,
            "positions_unable_to_fully_exit": None,
            "reason": "exports contain context/depth snapshots but no persisted target orders",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must not precede --start")
    csv.field_size_limit(10_000_000)
    print(json.dumps(
        run_replay_audit(args.export_dir, args.start, args.end),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
