"""Read-only Layer A replay smoke capability."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from execution.clob_execution import CLOBExecutionSnapshot, DepthLevel, walk_depth

from .storage import LayerAStore, read_books, read_books_bytes


def _dt(value: Any) -> datetime:
    raw = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _snapshot(book: dict[str, Any]) -> CLOBExecutionSnapshot:
    bids = tuple(
        DepthLevel(float(level["price"]), float(level["available_shares"]))
        for level in book.get("bids", [])
        if level.get("price") is not None and level.get("available_shares") is not None
    )
    asks = tuple(
        DepthLevel(float(level["price"]), float(level["available_shares"]))
        for level in book.get("asks", [])
        if level.get("price") is not None and level.get("available_shares") is not None
    )
    return CLOBExecutionSnapshot(
        market_id=str(book["market_id"]),
        condition_id=str(book["condition_id"]),
        bucket=str(book["bucket"]),
        token_side=str(book["token_side"]).upper(),
        token_id=str(book["token_id"]),
        decision_timestamp=_dt(book["decision_timestamp"]),
        book_timestamp=_dt(book["book_timestamp"]),
        book_age_seconds=float(book["book_age_seconds"]),
        tick_size=float(book["tick_size"]),
        minimum_order_size=float(book["minimum_order_size"]),
        bids=bids,
        asks=asks,
        fetch_cycle_id=str(book["fetch_cycle_id"]),
        source_name=str(book.get("source_name") or "polymarket_clob"),
    )


def _merge_books(records: list[dict[str, Any]], books: Iterable[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for book in books:
        cycle_id = book.get("decision_cycle_id")
        if cycle_id:
            grouped.setdefault(str(cycle_id), []).append(book)
    for record in records:
        cycle_id = str(record.get("decision_cycle_id"))
        if grouped.get(cycle_id):
            record["clob_books"] = grouped[cycle_id]


def _records_from_directory(path: Path) -> list[dict[str, Any]]:
    root = path / "layer_a" if (path / "layer_a").exists() else path
    store = LayerAStore(root)
    records: list[dict[str, Any]] = []
    for info in store.scan():
        if info.status != "complete":
            continue
        records.extend(store.read_partition_records(info))
        book_path = info.files.get("books")
        if book_path is not None:
            _merge_books(records, read_books(book_path, (info.manifest or {}).get("books_compression")))
    return records


def _records_from_zip(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    books: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            if not name.endswith(".parquet") or "/cycles-" not in name:
                continue
            import pandas as pd

            frame = pd.read_parquet(io.BytesIO(archive.read(name)))
            records.extend(
                json.loads(raw)
                for raw in frame.get("record_json", []).tolist()
                if isinstance(raw, str)
            )
        for name in sorted(names):
            if not name.endswith(".jsonl.zst") or "/books-" not in name:
                continue
            manifest_name = name.replace("books-", "manifest-").replace(".jsonl.zst", ".json")
            compression = None
            if manifest_name in names:
                try:
                    compression = json.loads(archive.read(manifest_name)).get("books_compression")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    compression = None
            books.extend(read_books_bytes(archive.read(name), compression))
    _merge_books(records, books)
    return records


def load_export_records(path: Path | str) -> list[dict[str, Any]]:
    """Load records from a Layer A directory or export zip without writes."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        return _records_from_zip(path)
    return _records_from_directory(path)


def _strategy_candidates(record: dict[str, Any], threshold: float, shares: float) -> list[dict[str, Any]]:
    books = {
        (str(book.get("bucket")), str(book.get("token_side")).upper()): book
        for book in record.get("clob_books", [])
        if isinstance(book, dict)
    }
    snapshots: dict[tuple[str, str], CLOBExecutionSnapshot] = {}
    for key, book in books.items():
        try:
            snapshots[key] = _snapshot(book)
        except (KeyError, TypeError, ValueError):
            continue
    prices = record.get("gamma_reference_prices") or {}
    candidates: list[dict[str, Any]] = []
    for model in record.get("models", []):
        if not isinstance(model, dict):
            continue
        probabilities = model.get("full_bucket_probabilities") or {}
        for bucket, probability in probabilities.items():
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                continue
            snapshot = snapshots.get((str(bucket), "YES"))
            if snapshot is None or snapshot.best_ask is None:
                continue
            fill = walk_depth(snapshot, "BUY", shares)
            executable = fill.execution_vwap
            if executable is None:
                continue
            edge = probability - executable
            if edge >= threshold:
                gamma = prices.get(bucket)
                candidates.append(
                    {
                        "model_name": model.get("model_name"),
                        "bucket": bucket,
                        "probability": probability,
                        "execution_price": executable,
                        "gamma_reference_price": gamma,
                        "edge": edge,
                        "price_source": "YES_CLOB_buy_walk",
                    }
                )
    return candidates


def replay_layer_a(
    path: Path | str,
    *,
    strategy_a_threshold: float = 0.03,
    kelly_fraction: float = 0.25,
    requested_shares: float = 1.0,
) -> dict[str, Any]:
    """Reconstruct probabilities/books and run a read-only replay probe."""
    if requested_shares <= 0:
        raise ValueError("requested_shares must be positive")
    if kelly_fraction < 0:
        raise ValueError("kelly_fraction must be non-negative")
    records = load_export_records(path)
    model_snapshots = sum(len(record.get("models", [])) for record in records)
    reconstructed_books = 0
    eligible_cycles = 0
    depth_walk: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for record in records:
        books = record.get("clob_books", [])
        reconstructed_books += len(books)
        if (record.get("completeness") or {}).get("replay_eligible_for_clob_strategy") is True:
            eligible_cycles += 1
        if depth_walk is None:
            by_key = {
                (str(book.get("bucket")), str(book.get("token_side")).upper()): book
                for book in books
                if isinstance(book, dict)
            }
            for bucket in sorted({key[0] for key in by_key}):
                yes = by_key.get((bucket, "YES"))
                no = by_key.get((bucket, "NO"))
                if yes is None or no is None:
                    continue
                try:
                    yes_snapshot = _snapshot(yes)
                    no_snapshot = _snapshot(no)
                    buy = walk_depth(yes_snapshot, "BUY", requested_shares)
                    sell = walk_depth(yes_snapshot, "SELL", requested_shares)
                    no_buy = walk_depth(no_snapshot, "BUY", requested_shares)
                    depth_walk = {
                        "decision_cycle_id": record.get("decision_cycle_id"),
                        "bucket": bucket,
                        "yes_buy": buy.to_dict(),
                        "yes_sell": sell.to_dict(),
                        "no_buy": no_buy.to_dict(),
                        "fee_and_vwap_recomputed": True,
                    }
                    break
                except (KeyError, TypeError, ValueError):
                    continue
        candidates.extend(_strategy_candidates(record, strategy_a_threshold, requested_shares))

    for candidate in candidates:
        price = float(candidate["execution_price"])
        probability = float(candidate["probability"])
        candidate["kelly_fraction"] = kelly_fraction
        candidate["kelly_fractional_bankroll"] = max(
            0.0,
            (probability - price) / max(1.0 - price, 1e-12),
        ) * kelly_fraction

    return {
        "schema_version": "layer_a.replay_smoke.v1",
        "input": str(path),
        "records_loaded": len(records),
        "model_probability_snapshots_reconstructed": model_snapshots,
        "clob_books_reconstructed": reconstructed_books,
        "clob_replay_eligible_cycles": eligible_cycles,
        "depth_walk": depth_walk,
        "strategy_a_probe": {
            "threshold": strategy_a_threshold,
            "requested_shares": requested_shares,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "kelly_probe": {
            "fraction": kelly_fraction,
            "stored_data_modified": False,
        },
    }


__all__ = ["load_export_records", "replay_layer_a"]
