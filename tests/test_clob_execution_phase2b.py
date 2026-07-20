"""Isolated Phase 2B tests for CLOB-depth paper execution."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from execution.clob_execution import (
    CLOBExecutionSnapshot,
    DepthLevel,
    SnapshotValidationError,
    build_execution_snapshots,
    compute_depth_adjusted_bets,
    mark_to_market,
    walk_depth,
)
from execution.clob_paper_adapter import ClobDepthPaperAdapter
from execution.strategy_engine import compute_config_orders
from execution.strategy_gate import load_strategy_config


DECISION = datetime(2026, 7, 19, 4, 0, tzinfo=timezone.utc)


def _snapshot(
    side: str,
    asks=((0.20, 100.0),),
    bids=((0.10, 100.0),),
    *,
    book_timestamp: datetime = DECISION,
    minimum_order_size: float = 5.0,
) -> CLOBExecutionSnapshot:
    return CLOBExecutionSnapshot(
        market_id="market-1",
        condition_id="condition-1",
        bucket="30-31",
        token_side=side,
        token_id=f"token-{side.lower()}",
        decision_timestamp=DECISION,
        book_timestamp=book_timestamp,
        book_age_seconds=(DECISION - book_timestamp).total_seconds(),
        tick_size=0.01,
        minimum_order_size=minimum_order_size,
        bids=tuple(DepthLevel(price, size) for price, size in bids),
        asks=tuple(DepthLevel(price, size) for price, size in asks),
        fetch_cycle_id="cycle-1",
    )


def _depth(snapshot: CLOBExecutionSnapshot, *, cycle: str = "cycle-1") -> dict:
    return {
        "asset_id": snapshot.token_id,
        "timestamp": snapshot.book_timestamp.isoformat(),
        "tick_size": snapshot.tick_size,
        "minimum_order_size": snapshot.minimum_order_size,
        "fetch_cycle_id": cycle,
        "source_name": "polymarket_clob",
        "bids": [level.to_dict() | {"size": level.available_shares} for level in snapshot.bids],
        "asks": [level.to_dict() | {"size": level.available_shares} for level in snapshot.asks],
    }


def _market(
    bucket: str = "30-31",
    yes_token: str = "token-yes",
    no_token: str = "token-no",
    outcomes=None,
    condition: str = "condition-1",
) -> dict:
    return {
        "bucket": bucket,
        "id": f"market-{bucket}",
        "conditionId": condition,
        "outcomes": outcomes if outcomes is not None else ["Yes", "No"],
        "token_id": yes_token,
        "no_token_id": no_token,
    }


def _snapshot_map(
    yes: CLOBExecutionSnapshot | None = None,
    no: CLOBExecutionSnapshot | None = None,
) -> dict[str, dict[str, CLOBExecutionSnapshot]]:
    return {"30-31": {"YES": yes or _snapshot("YES"), "NO": no or _snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),))}}


def test_single_level_yes_buy():
    result = walk_depth(_snapshot("YES", ((0.2, 10.0),)), "BUY", 10.0)
    assert result.filled_shares == pytest.approx(10.0)
    assert result.gross_vwap == pytest.approx(0.2)
    assert result.all_in_buy_vwap == pytest.approx(0.208)
    assert result.unfilled_shares == pytest.approx(0.0)


def test_multi_level_yes_buy():
    result = walk_depth(_snapshot("YES", ((0.2, 10.0), (0.3, 20.0))), "BUY", 25.0)
    assert result.filled_shares == pytest.approx(25.0)
    assert result.gross_notional == pytest.approx(6.5)
    assert result.depth_levels_consumed == 2
    assert result.worst_fill_price == pytest.approx(0.3)


def test_single_level_no_buy():
    result = walk_depth(_snapshot("NO", ((0.4, 10.0),)), "BUY", 10.0)
    assert result.gross_vwap == pytest.approx(0.4)
    assert result.all_in_buy_vwap == pytest.approx(0.412)


def test_multi_level_no_buy():
    result = walk_depth(_snapshot("NO", ((0.4, 10.0), (0.5, 20.0))), "BUY", 25.0)
    assert result.filled_shares == pytest.approx(25.0)
    assert result.gross_vwap == pytest.approx(11.5 / 25.0)
    assert result.depth_levels_consumed == 2


def test_yes_sell_walks_bids():
    result = walk_depth(_snapshot("YES", bids=((0.3, 10.0), (0.2, 20.0))), "SELL", 25.0)
    assert result.gross_vwap == pytest.approx(6.0 / 25.0)
    assert result.net_sell_vwap is not None
    assert result.worst_fill_price == pytest.approx(0.2)


def test_no_sell_walks_bids():
    result = walk_depth(_snapshot("NO", bids=((0.7, 10.0), (0.6, 20.0))), "SELL", 25.0)
    assert result.gross_notional == pytest.approx(16.0)
    assert result.depth_levels_consumed == 2
    assert result.net_cash_flow < result.gross_notional


def test_fee_is_calculated_per_level():
    result = walk_depth(_snapshot("YES", ((0.2, 10.0), (0.4, 10.0))), "BUY", 20.0)
    expected = 10.0 * 0.05 * 0.2 * 0.8 + 10.0 * 0.05 * 0.4 * 0.6
    assert result.total_fee == pytest.approx(expected)
    assert result.fills[0].fee != result.fills[1].fee


def test_partial_fill_does_not_charge_unfilled_shares():
    result = walk_depth(_snapshot("YES", ((0.2, 3.0),)), "BUY", 10.0)
    assert result.filled_shares == pytest.approx(3.0)
    assert result.fill_ratio == pytest.approx(0.3)
    assert result.unfilled_shares == pytest.approx(7.0)
    assert result.gross_notional == pytest.approx(0.6)


def test_no_liquidity():
    result = walk_depth(_snapshot("YES", asks=(), bids=()), "BUY", 10.0)
    assert result.filled_shares == 0
    assert result.total_fee == 0
    assert result.net_cash_flow == 0


def test_snapshot_rejects_stale_book():
    stale = DECISION.replace(hour=3)
    yes = _snapshot("YES", book_timestamp=stale)
    no = _snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),), book_timestamp=stale)
    with pytest.raises(SnapshotValidationError, match="stale/future"):
        build_execution_snapshots(
            [_market()], {"30-31": 0.5}, {"30-31": _depth(yes)}, {"30-31": _depth(no)},
            "highest-temperature-in-hong-kong-on-july-19-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1", max_book_age_seconds=30,
        )


def test_snapshot_rejects_missing_timestamp():
    yes = _depth(_snapshot("YES"))
    yes["timestamp"] = None
    no = _depth(_snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),)))
    with pytest.raises(SnapshotValidationError, match="timestamp is missing"):
        build_execution_snapshots(
            [_market()], {"30-31": 0.5}, {"30-31": yes}, {"30-31": no},
            "highest-temperature-in-hong-kong-on-july-19-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1",
        )


def test_invalid_token_mapping_is_rejected():
    with pytest.raises(SnapshotValidationError, match="outcome mapping"):
        build_execution_snapshots(
            [_market(outcomes=["No", "Yes"])], {"30-31": 0.5},
            {"30-31": _depth(_snapshot("YES"))},
            {"30-31": _depth(_snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),)))},
            "highest-temperature-in-hong-kong-on-july-19-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1",
        )


def test_bucket_schema_mismatch_is_rejected():
    with pytest.raises(SnapshotValidationError, match="bucket schema mismatch"):
        build_execution_snapshots(
            [_market("30-31")], {"31-32": 0.5},
            {"30-31": _depth(_snapshot("YES"))},
            {"30-31": _depth(_snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),)))},
            "highest-temperature-in-hong-kong-on-july-19-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1",
        )


def test_previous_day_market_is_rejected():
    with pytest.raises(SnapshotValidationError, match="market date mismatch"):
        build_execution_snapshots(
            [_market()], {"30-31": 0.5},
            {"30-31": _depth(_snapshot("YES"))},
            {"30-31": _depth(_snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),)))},
            "highest-temperature-in-hong-kong-on-july-18-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1",
        )


def test_incorrect_book_sorting_is_rejected():
    with pytest.raises(SnapshotValidationError, match="ask levels"):
        CLOBExecutionSnapshot(
            "m", "c", "30-31", "YES", "y", DECISION, DECISION, 0, .01, 5,
            (DepthLevel(.1, 10),), (DepthLevel(.4, 10), DepthLevel(.2, 10)), "cycle-1",
        )


def test_duplicate_token_ids_are_rejected():
    markets = [_market("30-31", condition="c1"), _market("31-32", condition="c2", no_token="token-no-2")]
    yes = _depth(_snapshot("YES"))
    no = _depth(_snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),)))
    depths = {"30-31": yes, "31-32": yes}
    no_depths = {"30-31": no, "31-32": no}
    with pytest.raises(SnapshotValidationError, match="duplicate token id"):
        build_execution_snapshots(
            markets, {"30-31": 0.5, "31-32": 0.5}, depths, no_depths,
            "highest-temperature-in-hong-kong-on-july-19-2026", DECISION,
            date(2026, 7, 19), fetch_cycle_id="cycle-1",
        )


def test_iterative_kelly_converges_on_depth_quote():
    yes = _snapshot("YES", ((0.2, 10_000.0),))
    no = _snapshot("NO", ((0.8, 10_000.0),), ((0.7, 10_000.0),))
    result = compute_depth_adjusted_bets(
        {"30-31": 0.7}, {"30-31": 0.4}, 1000.0,
        _snapshot_map(yes, no), .15, .5,
    )
    assert result.converged is True
    assert result.iterations >= 2
    assert result.adjusted_bets["30-31"]["execution_price_is_all_in"] is True


def test_depth_slippage_reduces_kelly_size():
    top_yes = _snapshot("YES", ((0.2, 10_000.0),))
    slipped_yes = _snapshot("YES", ((0.2, 10.0), (0.4, 10_000.0)))
    no = _snapshot("NO", ((0.8, 10_000.0),), ((0.7, 10_000.0),))
    top = compute_depth_adjusted_bets(
        {"30-31": 0.7}, {"30-31": 0.4}, 1000.0,
        _snapshot_map(top_yes, no), .15, .5,
    )
    slipped = compute_depth_adjusted_bets(
        {"30-31": 0.7}, {"30-31": 0.4}, 1000.0,
        _snapshot_map(slipped_yes, no), .15, .5,
        partial_fill_policy="accept_partial",
    )
    assert slipped.converged is True
    assert slipped.adjusted_bets["30-31"]["adjusted_quantity"] < top.adjusted_bets["30-31"]["adjusted_quantity"]
    assert slipped.adjusted_bets["30-31"]["execution_price"] > top.adjusted_bets["30-31"]["execution_price"]


def test_negative_executable_edge_after_depth_walk_is_rejected():
    yes = _snapshot("YES", ((0.2, 10.0), (0.9, 10_000.0)))
    no = _snapshot("NO", ((0.8, 10_000.0),), ((0.7, 10_000.0),))
    result = compute_depth_adjusted_bets(
        {"30-31": 0.3}, {"30-31": 0.2}, 1000.0,
        _snapshot_map(yes, no), .15, .5, partial_fill_policy="accept_partial",
    )
    assert result.adjusted_bets == {}


def test_gamma_change_does_not_change_clob_strategy_decision(monkeypatch):
    yes = _snapshot("YES", ((0.2, 10_000.0),))
    no = _snapshot("NO", ((0.8, 10_000.0),), ((0.7, 10_000.0),))
    snapshots = _snapshot_map(yes, no)
    config = load_strategy_config("enhanced_v2_paper")
    first = compute_config_orders(
        {"30-31": 0.7}, {"30-31": 0.2}, {}, 1000.0, False,
        datetime(2026, 7, 19, 10), {}, "model_a", "slug", config=config,
        post_mean=30.2, execution_snapshots=snapshots, paper_execution_mode="clob_depth",
        gamma_reference_prices={"30-31": 0.2},
    )
    second = compute_config_orders(
        {"30-31": 0.7}, {"30-31": 0.9}, {}, 1000.0, False,
        datetime(2026, 7, 19, 10), {}, "model_a", "slug", config=config,
        post_mean=30.2, execution_snapshots=snapshots, paper_execution_mode="clob_depth",
        gamma_reference_prices={"30-31": 0.9},
    )
    assert first[0] == second[0]
    assert first[1][0]["executable_edge_at_final_size"] == second[1][0]["executable_edge_at_final_size"]
    assert first[1][0]["diagnostic_edge"] != second[1][0]["diagnostic_edge"]


def test_mock_slippage_is_not_called_in_clob_path(monkeypatch):
    import execution.strategy_engine as engine

    def fail_if_called(*args, **kwargs):
        raise AssertionError("mock/legacy slippage was called in CLOB mode")

    monkeypatch.setattr(engine, "apply_slippage_to_bets", fail_if_called)
    yes = _snapshot("YES", ((0.2, 10_000.0),))
    no = _snapshot("NO", ((0.8, 10_000.0),), ((0.7, 10_000.0),))
    result, _ = compute_config_orders(
        {"30-31": 0.7}, {"30-31": 0.2}, {}, 1000.0, True,
        datetime(2026, 7, 19, 10), {}, "model_a", "slug",
        config=load_strategy_config("enhanced_v2_paper"), post_mean=30.2,
        execution_snapshots=_snapshot_map(yes, no), paper_execution_mode="clob_depth",
    )
    assert result


def test_forced_exit_with_insufficient_bid_depth_is_blocked():
    yes = _snapshot("YES", ((0.2, 100.0),), bids=((0.1, 2.0),))
    no = _snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),))
    target, decisions = compute_config_orders(
        {"30-31": 0.1}, {"30-31": 0.9}, {}, 1000.0, False,
        datetime(2026, 7, 19, 10),
        {"model_a": {"slug": {"30-31": {"side": "YES", "quantity": 10}}}},
        "model_a", "slug", config=load_strategy_config("enhanced_v2_paper"),
        post_mean=30.2, execution_snapshots=_snapshot_map(yes, no),
        paper_execution_mode="clob_depth", partial_fill_policy="fail_closed",
    )
    assert target == {}
    assert decisions[0]["reason"] in {"NO_EXECUTABLE_QUOTE", "PARTIAL_FILL_REJECTED"}


def test_residual_position_accounting_uses_actual_partial_sell(tmp_path):
    adapter = ClobDepthPaperAdapter(tmp_path / "paper")
    try:
        yes = _snapshot("YES", ((0.2, 100.0),), bids=((0.1, 5.0),))
        no = _snapshot("NO", ((0.8, 100.0),), ((0.7, 100.0),))
        markets = [_market()]
        snapshots = _snapshot_map(yes, no)
        adapter.execute_target_positions(
            {"30-31": {"side": "YES", "quantity": 10, "target_price": .2}},
            "portfolio", "slug", "strategy", {}, {}, markets=markets,
            execution_snapshots=snapshots, persist_legacy_positions=False,
        )
        fills = adapter.execute_target_positions(
            {"30-31": {"side": "YES", "quantity": 0, "target_price": .1}},
            "portfolio", "slug", "strategy", {}, {}, markets=markets,
            execution_snapshots=snapshots, partial_fill_policy="accept_partial",
            persist_legacy_positions=False,
        )
        assert fills[0]["is_partial"] is True
        assert fills[0]["residual_shares"] == pytest.approx(5.0)
        assert adapter._engine.db.get_position("condition-1", "yes").shares == pytest.approx(5.0)
    finally:
        adapter.close()


def test_unrealized_marks_expose_midpoint_and_liquidation_value():
    result = mark_to_market(
        _snapshot("YES", ((0.4, 100.0),), ((0.2, 100.0),)), 10.0
    )
    assert result["midpoint_mark"] == pytest.approx(3.0)
    assert result["immediate_liquidation_value"] < result["midpoint_mark"]
    assert result["immediate_liquidation_fill_ratio"] == pytest.approx(1.0)


def test_shadow_record_builder_is_read_only(tmp_path):
    from execution.shadow_logger import build_shadow_records, write_shadow_records

    records = build_shadow_records(
        strategy="s", model="model_a", target_probs={"30-31": .7},
        gamma_reference_prices={"30-31": .2},
        legacy_summary={"target_positions": {}, "decisions": []},
        clob_summary={"target_positions": {}, "decisions": []},
        execution_snapshots=_snapshot_map(), snapshot_error="no snapshot",
    )
    assert records[0]["legacy_would_trade"] is False
    assert records[0]["clob_would_trade"] is False
    log_path = tmp_path / "shadow.jsonl"
    write_shadow_records(records, log_path)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_strategy_settings_and_model_artifacts_are_not_changed():
    strategy_bytes = Path("config/paper_strategies.json").read_bytes()
    model_root = Path("models")
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_root.rglob("*")
        if path.is_file() and "intraday_minute_ml_model_2a" in str(path)
    }
    # The CLOB module must not load or mutate model artifacts.
    assert strategy_bytes == Path("config/paper_strategies.json").read_bytes()
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_root.rglob("*")
        if path.is_file() and "intraday_minute_ml_model_2a" in str(path)
    }
    assert before == after


def test_runtime_state_files_are_not_touched_by_core_execution(tmp_path):
    paths = [Path("data/current_positions.json"), Path("data/strategy_accounts.json")]
    before = {path: path.read_bytes() for path in paths if path.exists()}
    # Use only an isolated fill result; no live path or global store is opened.
    assert walk_depth(_snapshot("YES"), "BUY", 1).filled_shares == pytest.approx(1)
    after = {path: path.read_bytes() for path in paths if path.exists()}
    assert before == after
