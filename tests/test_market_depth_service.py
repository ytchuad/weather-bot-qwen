"""Tests for CLOB market-depth fetching / alignment.

The Polymarket ``POST /books`` batch endpoint does NOT guarantee the
response order matches the request order, so ``fetch_market_depths_batch``
must align results by ``asset_id`` rather than list position.
"""
from unittest.mock import patch

import pytest

from app.services import market_depth_service as mds


def _book(asset_id: str, ask_price: float):
    """Minimal order book with a single ask at ``ask_price``."""
    return {
        "asset_id": asset_id,
        "bids": [],
        "asks": [{"price": str(ask_price), "size": "1"}],
    }


def test_batch_aligns_by_asset_id_not_position():
    # Request buckets in this order...
    bucket_map = {
        "34-35": "TOKEN_A",
        "33-34": "TOKEN_B",
        "<27": "TOKEN_C",
    }
    # ...but the API returns them scrambled / reversed.
    scrambled = [
        _book("TOKEN_C", 0.01),  # <27  -> ask 0.01
        _book("TOKEN_A", 0.97),  # 34-35 -> ask 0.97
        _book("TOKEN_B", 0.50),  # 33-34 -> ask 0.50
    ]

    with patch.object(mds, "fetch_order_books_batch", return_value=scrambled):
        result = mds.fetch_market_depths_batch(bucket_map)

    assert result["34-35"]["best_ask"]["price"] == pytest.approx(0.97)
    assert result["33-34"]["best_ask"]["price"] == pytest.approx(0.50)
    assert result["<27"]["best_ask"]["price"] == pytest.approx(0.01)


def test_batch_handles_missing_books():
    bucket_map = {"34-35": "TOKEN_A", "33-34": "TOKEN_B"}
    # Only one of the two requested tokens is returned.
    partial = [_book("TOKEN_A", 0.97)]

    with patch.object(mds, "fetch_order_books_batch", return_value=partial):
        result = mds.fetch_market_depths_batch(bucket_map)

    assert result["34-35"]["best_ask"]["price"] == pytest.approx(0.97)
    assert result["33-34"] is None  # missing book -> None, not crash


def test_batch_empty_input():
    assert mds.fetch_market_depths_batch({}) == {}
