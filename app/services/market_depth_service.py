"""Polymarket CLOB market depth (order book) data fetching.

Provides full order book, bid/ask spread, depth summary, order-book walking
for trade simulation, and a background cache that auto-refreshes every 10 s.
All CLOB endpoints are public — no authentication required.
"""
from __future__ import annotations

import logging
import threading
import time

import requests

from ..config import PM_CLOB_API

logger = logging.getLogger(__name__)

_DEPTH_TOP_LEVELS = 20  # number of bid/ask levels stored in summaries


# ── raw API fetchers ──────────────────────────────────────────────────


def fetch_order_book(token_id: str) -> dict | None:
    """Full order book for one token via ``GET /book``.

    Returns the raw response dict (bids, asks, timestamp, …) or *None*.
    """
    url = f"{PM_CLOB_API}/book?token_id={token_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning("CLOB book fetch failed for token %s: %s", token_id, e)
        return None


def fetch_order_books_batch(token_ids: list[str]) -> list[dict]:
    """Batch order books for up to 500 tokens via ``POST /books``.

    The body is an array of ``{"token_id": "<id>"}`` objects.
    Returns a list of book dicts in the same order as *token_ids*.
    Any failed token gets *None* in its slot.
    """
    if not token_ids:
        return []
    url = f"{PM_CLOB_API}/books"
    body = [{"token_id": tid} for tid in token_ids]
    try:
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        logger.warning("Unexpected /books response type: %s", type(data))
        return []
    except requests.RequestException as e:
        logger.warning("CLOB batch books fetch failed: %s", e)
        return []


# ── depth summary with correct sort order ─────────────────────────────


def compute_depth_summary(
    book: dict | None,
    fetch_cycle_id: str | None = None,
) -> dict | None:
    """Compact depth summary from a raw order book dict.

    CLOB returns asks *descending* (highest price first) and bids *ascending*
    (lowest price first).  This function **re-sorts** so that:

    * ``top_asks`` — lowest (best) price first
    * ``top_bids`` — highest (best) price first

    Only the top ``_DEPTH_TOP_LEVELS`` (10) levels are included in ``top_*``.
    Returns *None* if the book is *None* or empty.
    """
    if not book or not isinstance(book, dict):
        return None

    raw_bids = book.get("bids", [])
    raw_asks = book.get("asks", [])
    ts_raw = book.get("timestamp")

    validation_errors: list[str] = []

    def _extract(levels: list, side_name: str) -> list[dict]:
        """Parse all price/size levels, filtering out zero-size entries."""
        out: list[dict] = []
        if not isinstance(levels, list):
            validation_errors.append(f"{side_name}_levels_not_list")
            return out
        for index, level in enumerate(levels):
            try:
                p = float(level["price"])
                s = float(level["size"])
                if not (0.0 < p < 1.0):
                    validation_errors.append(f"{side_name}[{index}]_invalid_price")
                    continue
                if s <= 0:
                    validation_errors.append(f"{side_name}[{index}]_non_positive_size")
                    continue
                out.append({"price": round(p, 8), "size": round(s, 8)})
            except (KeyError, ValueError, TypeError):
                validation_errors.append(f"{side_name}[{index}]_non_numeric")
                continue
        return out

    bids_parsed = _extract(raw_bids, "bids")
    asks_parsed = _extract(raw_asks, "asks")

    # bids: highest price first (best for seller)
    # asks: lowest price first (best for buyer)
    bids_sorted = sorted(bids_parsed, key=lambda x: x["price"], reverse=True)
    asks_sorted = sorted(asks_parsed, key=lambda x: x["price"])

    top_bids = bids_sorted[:_DEPTH_TOP_LEVELS]
    top_asks = asks_sorted[:_DEPTH_TOP_LEVELS]

    best_bid = top_bids[0] if top_bids else None
    best_ask = top_asks[0] if top_asks else None

    total_bid_size = round(sum(level["size"] for level in bids_sorted), 2)
    total_ask_size = round(sum(level["size"] for level in asks_sorted), 2)

    spread = round(best_ask["price"] - best_bid["price"], 6) if (best_bid and best_ask) else None
    midpoint = round((best_bid["price"] + best_ask["price"]) / 2, 6) if (best_bid and best_ask) else None

    # size-weighted mid (VWAP across top levels)
    weighted_mid = None
    if top_bids and top_asks:
        bid_vwap = sum(level["price"] * level["size"] for level in top_bids) / sum(level["size"] for level in top_bids)
        ask_vwap = sum(level["price"] * level["size"] for level in top_asks) / sum(level["size"] for level in top_asks)
        weighted_mid = round((bid_vwap + ask_vwap) / 2, 6)

    # depth imbalance: +1 = all bid pressure, -1 = all ask pressure
    depth_imbalance = None
    denom = total_bid_size + total_ask_size
    if denom > 0:
        depth_imbalance = round((total_bid_size - total_ask_size) / denom, 4)

    return {
        "asset_id": book.get("asset_id") or book.get("market"),
        "timestamp": ts_raw,
        "tick_size": book.get("tick_size"),
        "minimum_order_size": book.get("min_order_size", book.get("minimum_order_size")),
        "fetch_cycle_id": fetch_cycle_id,
        "source_name": "polymarket_clob",
        "validation_errors": validation_errors,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "midpoint": midpoint,
        "weighted_mid": weighted_mid,
        "bid_count": len(bids_sorted),
        "ask_count": len(asks_sorted),
        "total_bid_size": total_bid_size,
        "total_ask_size": total_ask_size,
        "depth_imbalance": depth_imbalance,
        "top_bids": top_bids,
        "top_asks": top_asks,
        # Keep the complete normalized depth for execution.  The top_* fields
        # remain compact diagnostics for the dashboard.
        "bids": bids_sorted,
        "asks": asks_sorted,
    }


# ── order-book walking for trade simulation ───────────────────────────


def walk_book(
    side: str,
    order_value: float,
    asks_sorted: list[dict],
    bids_sorted: list[dict],
) -> dict:
    """Simulate a market order walking the order book.

    Parameters
    ----------
    side : ``"BUY"`` or ``"SELL"``
        BUY  → walk asks (lowest price first)
        SELL → walk bids (highest price first)
    order_value : float
        USD amount to spend (BUY) / receive (SELL).
    asks_sorted : list[dict]
        Ask levels **already sorted ascending** (lowest price first).
    bids_sorted : list[dict]
        Bid levels **already sorted descending** (highest price first).

    Returns
    -------
    dict with keys:
        avg_price     — weighted average fill price
        shares_filled — total shares filled
        value_filled  — total USD value filled
        levels_used   — number of price levels consumed
        fill_frac     — fraction of the order that was filled (0..1)
    """
    if order_value <= 0:
        return {"avg_price": 0, "shares_filled": 0, "value_filled": 0,
                "levels_used": 0, "fill_frac": 0.0}

    levels = asks_sorted if side == "BUY" else bids_sorted

    remaining = order_value
    total_shares = 0.0
    total_cost = 0.0
    levels_used = 0

    for lvl in levels:
        p = lvl["price"]
        s = lvl["size"]
        cost_at_level = p * s

        if cost_at_level <= remaining:
            total_shares += s
            total_cost += cost_at_level
            remaining -= cost_at_level
            levels_used += 1
        else:
            partial_shares = remaining / p
            total_shares += partial_shares
            total_cost += remaining
            remaining = 0
            levels_used += 1
            break

        if remaining <= 0:
            break

    value_filled = total_cost
    fill_frac = value_filled / order_value
    avg_price = total_cost / total_shares if total_shares > 0 else 0.0

    return {
        "avg_price": round(avg_price, 6),
        "shares_filled": round(total_shares, 2),
        "value_filled": round(value_filled, 2),
        "levels_used": levels_used,
        "fill_frac": round(fill_frac, 4),
    }


def compute_execution_estimate(
    side: str,
    order_value: float,
    depth_summary: dict | None,
    gamma_ask: float | None = None,
    gamma_bid: float | None = None,
) -> dict:
    """CLOB-only execution estimate; missing depth fails closed.

    For small orders the Gamma best-ask/best-bid (which includes AMM
    liquidity) is diagnostic only and is intentionally ignored here.  This
    function never substitutes Gamma for a missing CLOB quote.

    Returns the same keys as :func:`walk_book`.
    """
    # Gamma arguments are retained for API compatibility only.  They cannot
    # supply an executable quote when the CLOB book is absent or incomplete.
    del gamma_ask, gamma_bid
    if depth_summary:
        depth_summary = dict(depth_summary)
        if depth_summary.get("asks") is not None:
            depth_summary["top_asks"] = depth_summary["asks"]
        if depth_summary.get("bids") is not None:
            depth_summary["top_bids"] = depth_summary["bids"]
    if not depth_summary:
        return {"avg_price": 0, "shares_filled": 0, "value_filled": 0,
                "levels_used": 0, "fill_frac": 0.0}

    asks = sorted(depth_summary.get("top_asks", []), key=lambda x: x["price"])
    bids = sorted(depth_summary.get("top_bids", []), key=lambda x: x["price"], reverse=True)
    if side == "BUY":
        return walk_book("BUY", order_value, asks, bids)
    return walk_book("SELL", order_value, asks, bids)


# ── one-shot convenience wrappers ─────────────────────────────────────


def fetch_market_depth(token_id: str) -> dict | None:
    """One-shot: fetch order book + compute depth summary for one token."""
    book = fetch_order_book(token_id)
    return compute_depth_summary(book)


def fetch_market_depths_batch(
    bucket_token_map: dict[str, str],
    fetch_cycle_id: str | None = None,
) -> dict[str, dict | None]:
    """Batch depth fetch for many buckets at once.

    *bucket_token_map* maps bucket label → Yes token ID.
    Returns {bucket_label: depth_summary_or_None}.
    """
    if not bucket_token_map:
        return {}

    buckets = list(bucket_token_map.keys())
    token_ids = [bucket_token_map[b] for b in buckets]

    books = fetch_order_books_batch(token_ids)
    # The CLOB ``POST /books`` endpoint does NOT guarantee the response order
    # matches the request order, so align by ``asset_id`` instead of position.
    by_asset: dict[str, dict] = {}
    for raw in books:
        if isinstance(raw, dict) and raw.get("asset_id"):
            by_asset[raw["asset_id"]] = raw
    result: dict[str, dict | None] = {}
    for bucket in buckets:
        raw = by_asset.get(bucket_token_map[bucket])
        result[bucket] = compute_depth_summary(raw, fetch_cycle_id=fetch_cycle_id)
    return result


# ── background depth cache (refreshes every 10 s) ─────────────────────


class DepthCache:
    """Thread-safe cache that keeps CLOB order-book depth fresh.

    A background daemon thread calls ``POST /books`` every 10 seconds and
    stores the results.  Callers read via :meth:`get` — which is a
    non-blocking dict lookup.

    Usage
    -----
    .. code-block:: python

        cache = DepthCache()
        cache.update_token_ids(
            {"32-33": "0x...", "33-34": "0x..."},
            {"32-33": "0xNO...", "33-34": "0xNO..."},
        )
        cache.start()

        # … later, from any thread:
        depths = cache.get()        # dict[str, dict | None]  — YES token books
        no_depths = cache.get_no()  # dict[str, dict | None]  — NO  token books
        depth_33 = depths.get("33-34")
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict | None] = {}
        self._no_cache: dict[str, dict | None] = {}
        self._token_ids: dict[str, str] = {}
        self._no_token_ids: dict[str, str] = {}
        self._fetch_cycle_id: str | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── public API ────────────────────────────────────────────────

    def update_token_ids(self, yes_token_ids: dict[str, str],
                         no_token_ids: dict[str, str] | None = None) -> None:
        """Replace bucket→token-id maps for YES and optionally NO tokens (thread-safe)."""
        with self._lock:
            self._token_ids = dict(yes_token_ids)
            self._no_token_ids = dict(no_token_ids) if no_token_ids else {}

    def get(self) -> dict[str, dict | None]:
        """Return a snapshot of the current YES-token cache (thread-safe)."""
        with self._lock:
            return dict(self._cache)

    def get_no(self) -> dict[str, dict | None]:
        """Return a snapshot of the current NO-token cache (thread-safe)."""
        with self._lock:
            return dict(self._no_cache)

    def get_bundle(self) -> tuple[dict[str, dict | None], dict[str, dict | None], str | None]:
        """Return YES/NO depth and the coherent refresh cycle identifier."""
        with self._lock:
            return dict(self._cache), dict(self._no_cache), self._fetch_cycle_id

    def start(self) -> None:
        """Start the background refresh thread (daemon)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="depth-cache")
        self._thread.start()
        logger.info("DepthCache started (10 s interval)")

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False

    # ── internal ──────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                cycle_id = str(time.time_ns())
                with self._lock:
                    yes_tids = dict(self._token_ids)
                    no_tids = dict(self._no_token_ids)

                if yes_tids:
                    depths = fetch_market_depths_batch(yes_tids, fetch_cycle_id=cycle_id)
                    with self._lock:
                        self._cache = depths

                if no_tids:
                    no_depths = fetch_market_depths_batch(no_tids, fetch_cycle_id=cycle_id)
                    with self._lock:
                        self._no_cache = no_depths
                if yes_tids and no_tids:
                    with self._lock:
                        self._fetch_cycle_id = cycle_id
            except Exception as exc:
                logger.warning("DepthCache refresh failed: %s", exc)

            time.sleep(10)


# Convenience global instance
_depth_cache: DepthCache | None = None
_depth_cache_lock = threading.Lock()


def get_global_depth_cache() -> DepthCache:
    """Return (or create) the singleton :class:`DepthCache`."""
    global _depth_cache
    if _depth_cache is None:
        with _depth_cache_lock:
            if _depth_cache is None:
                _depth_cache = DepthCache()
    return _depth_cache
