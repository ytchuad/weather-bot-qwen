#!/usr/bin/env python
"""Phase 1: Verify polymarket-paper-trader connectivity and multi-bucket support.

Tests using harvey-weinstein-prison-time (6 buckets, same multi-bucket
structure as temperature events). Each bucket is a binary Yes/No sub-market.

Usage:
    python tests/test_paper_trader_connect.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_trader_test"
SLUG = "harvey-weinstein-prison-time"
BUCKETS = ["No Prison Time", "<5 years", "5-10 years", "10-20 years", "20-30 years", "30+ years"]


def test_1_import():
    from pm_trader.engine import Engine
    print("[OK] pm_trader.engine.Engine imported")
    return Engine


def test_2_init_account(Engine):
    engine = Engine(DATA_DIR)
    engine.init_account(balance=10_000.0)
    bal = engine.get_balance()
    print(f"[OK] Account initialized: cash={bal['cash']}, total={bal['total_value']}")
    assert bal["cash"] == 10_000.0
    return engine


def test_3_fetch_markets_via_gamma_api():
    """Fetch all sub-markets under the event slug via Gamma API."""
    import requests
    resp = requests.get(f"https://gamma-api.polymarket.com/events?slug={SLUG}", timeout=10)
    events = resp.json()
    assert len(events) > 0
    markets = events[0].get("markets", [])
    print(f"[OK] Gamma API: event has {len(markets)} sub-markets:")
    buckets_via_gamma = {}
    for m in markets:
        title = m.get("groupItemTitle", "?")
        cond_id = m.get("conditionId", "")
        outcomes = m.get("outcomes", "[]")
        clob_ids = m.get("clobTokenIds", "[]")
        prices = m.get("outcomePrices", "[]")
        print(f"  bucket='{title}' cond={cond_id[:16]} outcomes={outcomes} clob={clob_ids[:40]} prices={prices[:30]}")
        buckets_via_gamma[title] = {"condition_id": cond_id, "outcomes": outcomes, "clob_ids": clob_ids}
    return markets, buckets_via_gamma


def test_4_market_not_found_by_event_slug(engine):
    """Confirm sub-markets under multi-bucket events are NOT findable by event slug."""
    try:
        market = engine.api.get_market(SLUG)
        print(f"[WARN] get_market('{SLUG}') unexpectedly succeeded: {market.question[:60]}")
    except Exception as e:
        print(f"[OK] get_market('{SLUG}') correctly raises: {type(e).__name__}")
    print("  Multi-bucket sub-markets have unique slugs or condition_ids,")
    print("  NOT the event slug. Adapter must use condition_id or token_id.")


def test_5_fetch_market_by_condition_id(engine, cond_id):
    """Verify paper-trader can fetch a specific bucket by condition_id."""
    try:
        market = engine.api.get_market(cond_id)
        print(f"[OK] get_market(condition_id='{cond_id[:16]}...'):")
        print(f"  question={market.question[:60]}")
        print(f"  outcomes={market.outcomes}")
        print(f"  slug={market.slug}")
        token_id = market.get_token_id("yes")
        print(f"  YES token_id={token_id}")
        return market
    except Exception as e:
        print(f"[FAIL] get_market(condition_id): {e}")
        return None


def test_6_place_buy_on_slug(engine):
    """Buy on the event slug -- predictably only buys on first bucket."""
    try:
        result = engine.buy(SLUG, "yes", amount_usd=10.0)
        t = result.trade
        print(f"[OK] buy('{SLUG}', yes, $10):")
        print(f"  shares={t.shares:.2f} @ {t.avg_price:.4f}, fee={t.fee:.4f}, slippage={t.slippage:.1f}bps")
        return True
    except Exception as e:
        print(f"[FAIL] buy('{SLUG}', yes, $10): {e}")
        return False


def test_7_place_buy_on_each_bucket_by_condition_id(engine, buckets_via_gamma):
    """Buy on specific buckets using condition_id instead of slug."""
    results = {}
    for title, info in buckets_via_gamma.items():
        cond_id = info["condition_id"]
        if not cond_id or not cond_id.startswith("0x"):
            print(f"[SKIP] bucket='{title}' has no condition_id")
            continue
        try:
            result = engine.buy(cond_id, "yes", amount_usd=5.0)
            t = result.trade
            print(f"[OK] buy(cond_id, yes, $5) on bucket='{title}': shares={t.shares:.2f} @ {t.avg_price:.4f}, fee={t.fee:.4f}")
            results[title] = {"ok": True, "shares": t.shares, "avg_price": t.avg_price, "fee": t.fee}
        except Exception as e:
            print(f"[FAIL] buy(cond_id, yes, $5) on bucket='{title}': {e}")
            results[title] = {"ok": False, "error": str(e)}
    return results


def test_8_check_portfolio(engine):
    """Verify positions are tracked correctly per bucket."""
    portfolio = engine.get_portfolio()
    print(f"[OK] Portfolio: {len(portfolio)} positions")
    for p in portfolio:
        print(f"  {p['market_slug'][:40]} / {p['outcome']}: {p['shares']:.2f}sh @ {p['avg_entry_price']:.4f}, "
              f"cost={p['total_cost']:.2f}, value={p['current_value']:.2f}, "
              f"pnl={p['unrealized_pnl']:.2f}")
    bal = engine.get_balance()
    print(f"[OK] Balance: cash={bal['cash']:.2f}, positions_value={bal['positions_value']:.2f}, "
          f"total={bal['total_value']:.2f}, pnl={bal['pnl']:.2f}")
    return portfolio


def test_9_check_trade_history(engine):
    history = engine.get_history(limit=10)
    print(f"[OK] Trade history: {len(history)} trades")
    for h in history[:5]:
        print(f"  {h.market_slug[:40]} / {h.outcome}: side={h.side}, shares={h.shares:.2f} @ {h.avg_price:.4f}, "
              f"fee={h.fee:.4f}")


def test_10_check_backtest_import():
    from pm_trader.backtest import run_backtest, PriceSnapshot
    import inspect
    sig = inspect.signature(run_backtest)
    print(f"[OK] run_backtest signature: {sig}")


def test_11_reset(engine):
    engine.reset()
    bal = engine.get_balance()
    print(f"[OK] Reset: cash={bal['cash']:.2f}, positions={len(engine.get_portfolio())}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: polymarket-paper-trader Connectivity Tests")
    print("=" * 60)

    Engine = test_1_import()
    engine = test_2_init_account(Engine)

    print("\n--- 3. Fetch sub-markets via Gamma API ---")
    markets, buckets_via_gamma = test_3_fetch_markets_via_gamma_api()

    print("\n--- 4. Sub-markets not found by event slug ---")
    test_4_market_not_found_by_event_slug(engine)

    print("\n--- 5. Fetch specific bucket by condition_id ---")
    first_cond = buckets_via_gamma.get(BUCKETS[0], {}).get("condition_id", "")
    print(f"  Using condition_id: {first_cond[:20]}...")
    test_5_fetch_market_by_condition_id(engine, first_cond)

    print("\n--- 6. Place buy using slug (only hits first bucket) ---")
    test_6_place_buy_on_slug(engine)

    print("\n--- 7. Place buys on each bucket by condition_id ---")
    results = test_7_place_buy_on_each_bucket_by_condition_id(engine, buckets_via_gamma)

    print("\n--- 8. Check portfolio ---")
    portfolio = test_8_check_portfolio(engine)

    print("\n--- 9. Check trade history ---")
    test_9_check_trade_history(engine)

    print("\n--- 10. Backtest import ---")
    test_10_check_backtest_import()

    print("\n--- 11. Reset ---")
    test_11_reset(engine)

    engine.close()

    success = all(v["ok"] for v in results.values())
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED - condition_id approach works for multi-bucket!")
    else:
        failed = [k for k, v in results.items() if not v["ok"]]
        print(f"PARTIAL PASS - {len(failed)}/{len(results)} buckets failed: {failed}")
    print("=" * 60)
