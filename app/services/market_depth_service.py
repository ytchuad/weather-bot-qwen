"""Polymarket CLOB market depth (order book) data fetching.

Provides full order book, bid/ask spread, depth summary, order-book walking
for trade simulation, and a background cache that auto-refreshes every 10 s.
All CLOB endpoints are public — no authentication required.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

from ..config import PM_CLOB_API

logger = logging.getLogger(__name__)

_DEPTH_TOP_LEVELS = 10  # number of bid/ask levels stored in summaries


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


def compute_depth_summary(book: dict | None) -> dict | None:
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

    def _extract(levels: list) -> list[dict]:
        """Parse price/size, filter zero-size, cap at top N."""
        out: list[dict] = []
        for l in levels:
            try:
                p = float(l["price"])
                s = float(l["size"])
                if s > 0:
                    out.append({"price": round(p, 6), "size": round(s, 2)})
                    if len(out) >= _DEPTH_TOP_LEVELS * 4:
                        break
            except (ValueError, TypeError):
                continue
        return out

    bids_parsed = _extract(raw_bids)
    asks_parsed = _extract(raw_asks)

    # bids: highest price first (best for seller)
    # asks: lowest price first (best for buyer)
    bids_sorted = sorted(bids_parsed, key=lambda x: x["price"], reverse=True)
    asks_sorted = sorted(asks_parsed, key=lambda x: x["price"])

    top_bids = bids_sorted[:_DEPTH_TOP_LEVELS]
    top_asks = asks_sorted[:_DEPTH_TOP_LEVELS]

    best_bid = top_bids[0] if top_bids else None
    best_ask = top_asks[0] if top_asks else None

    total_bid_size = round(sum(l["size"] for l in bids_sorted), 2)
    total_ask_size = round(sum(l["size"] for l in asks_sorted), 2)

    spread = round(best_ask["price"] - best_bid["price"], 6) if (best_bid and best_ask) else None
    midpoint = round((best_bid["price"] + best_ask["price"]) / 2, 6) if (best_bid and best_ask) else None

    # size-weighted mid (VWAP across top levels)
    weighted_mid = None
    if top_bids and top_asks:
        bid_vwap = sum(l["price"] * l["size"] for l in top_bids) / sum(l["size"] for l in top_bids)
        ask_vwap = sum(l["price"] * l["size"] for l in top_asks) / sum(l["size"] for l in top_asks)
        weighted_mid = round((bid_vwap + ask_vwap) / 2, 6)

    # depth imbalance: +1 = all bid pressure, -1 = all ask pressure
    depth_imbalance = None
    denom = total_bid_size + total_ask_size
    if denom > 0:
        depth_imbalance = round((total_bid_size - total_ask_size) / denom, 4)

    return {
        "timestamp": ts_raw,
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
    """Best-effort execution estimate using CLOB depth + Gamma fallback.

    For small orders the Gamma best-ask/best-bid (which includes AMM
    liquidity) is usually tighter than the CLOB top-of-book.  This function
    picks the cheaper source automatically.

    Returns the same keys as :func:`walk_book`.
    """
    if not depth_summary:
        # no CLOB data — fall back to Gamma only
        price = gamma_ask if side == "BUY" else gamma_bid
        if price is None or price <= 0:
            return {"avg_price": 0, "shares_filled": 0, "value_filled": 0,
                    "levels_used": 0, "fill_frac": 0.0}
        shares = order_value / price
        return {"avg_price": round(price, 6), "shares_filled": round(shares, 2),
                "value_filled": round(order_value, 2), "levels_used": 1,
                "fill_frac": 1.0}

    asks = sorted(depth_summary.get("top_asks", []), key=lambda x: x["price"])
    bids = sorted(depth_summary.get("top_bids", []), key=lambda x: x["price"], reverse=True)
    all_asks = asks
    all_bids = bids

    if side == "BUY":
        # Prefer the better of CLOB top ask and Gamma bestAsk
        best = None
        if all_asks:
            best = all_asks[0]["price"]
        if gamma_ask is not None and (best is None or gamma_ask < best):
            best = gamma_ask
        if best is None:
            return {"avg_price": 0, "shares_filled": 0, "value_filled": 0,
                    "levels_used": 0, "fill_frac": 0.0}
        # If Gamma offers a better price, use Gamma for full fill
        if gamma_ask is not None and gamma_ask <= (all_asks[0]["price"] if all_asks else float("inf")):
            shares = order_value / gamma_ask
            return {"avg_price": round(gamma_ask, 6), "shares_filled": round(shares, 2),
                    "value_filled": round(order_value, 2), "levels_used": 1,
                    "fill_frac": 1.0}
        return walk_book("BUY", order_value, all_asks, all_bids)
    else:
        best = None
        if all_bids:
            best = all_bids[0]["price"]
        if gamma_bid is not None and (best is None or gamma_bid > best):
            best = gamma_bid
        if best is None:
            return {"avg_price": 0, "shares_filled": 0, "value_filled": 0,
                    "levels_used": 0, "fill_frac": 0.0}
        if gamma_bid is not None and gamma_bid >= (all_bids[0]["price"] if all_bids else 0):
            shares = order_value / gamma_bid
            return {"avg_price": round(gamma_bid, 6), "shares_filled": round(shares, 2),
                    "value_filled": round(order_value, 2), "levels_used": 1,
                    "fill_frac": 1.0}
        return walk_book("SELL", order_value, all_asks, all_bids)


# ── one-shot convenience wrappers ─────────────────────────────────────


def fetch_market_depth(token_id: str) -> dict | None:
    """One-shot: fetch order book + compute depth summary for one token."""
    book = fetch_order_book(token_id)
    return compute_depth_summary(book)


def fetch_market_depths_batch(
    bucket_token_map: dict[str, str],
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
    result: dict[str, dict | None] = {}
    for i, bucket in enumerate(buckets):
        book = books[i] if i < len(books) else None
        result[bucket] = compute_depth_summary(book)
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
                with self._lock:
                    yes_tids = dict(self._token_ids)
                    no_tids = dict(self._no_token_ids)

                if yes_tids:
                    depths = fetch_market_depths_batch(yes_tids)
                    with self._lock:
                        self._cache = depths

                if no_tids:
                    no_depths = fetch_market_depths_batch(no_tids)
                    with self._lock:
                        self._no_cache = no_depths
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
