# execution/portfolio_manager.py
import sys
import json
import logging
import re
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from copy import deepcopy

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

_HKT_OFFSET = timedelta(hours=8)
def _hkt_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + _HKT_OFFSET

from execution.strategy_runner import (
    load_strategy_registry, load_strategy_state, save_strategy_state,
    run_single_strategy_cycle, check_entry_rules, _default_slug, _save_json, _load_json,
    DEFAULT_ACCOUNTS
)

logger = logging.getLogger(__name__)

PORTFOLIO_CONFIG_PATH = Path('config/portfolios.json')
PORTFOLIO_STATE_PATH = Path('data/portfolio_state.json')
POSITIONS_PATH = Path('data/current_positions.json')

AVAILABLE_MODEL_KEYS = ["9d", "aws", "baseline_paper", "rain_nowcast_paper", "intra"]

DEFAULT_PORTFOLIO_CONFIG = {
    "version": 1,
    "portfolios": {
        "weather_main": {
            "label": "我的天氣組合",
            "capital": 10000.0,
            "strategies": ["enhanced_v1_paper"],
            "strategy_models": {
                "enhanced_v1_paper": "baseline_paper"
            },
            "watched_slugs": []
        }
    }
}

DEFAULT_PORTFOLIO_STATE = {
    "version": 1,
    "portfolios": {}
}


def load_portfolio_config():
    cfg = _load_json(PORTFOLIO_CONFIG_PATH, None)
    if cfg is None:
        cfg = deepcopy(DEFAULT_PORTFOLIO_CONFIG)
    # Normalize: ensure every portfolio has strategy_models
    _known_model_keys = {'baseline_paper', 'rain_nowcast_paper', 'model_a_paper', 'model_b_paper', 'model_c_paper', 'intra', 'aws', '9d'}
    for pid, pf in cfg.get("portfolios", {}).items():
        if "strategy_models" not in pf:
            pf["strategy_models"] = {}
        for sk in pf.get("strategies", []):
            if sk not in pf["strategy_models"]:
                if sk in _known_model_keys:
                    pf["strategy_models"][sk] = sk
                else:
                    pf["strategy_models"][sk] = 'model_c_paper'
    return cfg


def save_portfolio_config(cfg):
    _save_json(PORTFOLIO_CONFIG_PATH, cfg)


def create_portfolio(portfolio_id, label, capital=10000.0, strategies=None, strategy_models=None):
    """Create a new portfolio in config."""
    cfg = load_portfolio_config()
    if portfolio_id in cfg["portfolios"]:
        raise KeyError(f"Portfolio '{portfolio_id}' already exists")
    if strategies is None:
        strategies = ["enhanced_v1_paper"]
    if strategy_models is None:
        strategy_models = {sk: sk for sk in strategies}
    cfg["portfolios"][portfolio_id] = {
        "label": label,
        "capital": capital,
        "strategies": list(strategies),
        "strategy_models": dict(strategy_models),
        "watched_slugs": [],
    }
    save_portfolio_config(cfg)
    return cfg["portfolios"][portfolio_id]


def delete_portfolio(portfolio_id):
    """Delete a portfolio from config."""
    cfg = load_portfolio_config()
    if portfolio_id not in cfg["portfolios"]:
        raise KeyError(f"Portfolio '{portfolio_id}' not found")
    del cfg["portfolios"][portfolio_id]
    save_portfolio_config(cfg)
    # Also clean up state
    state = load_portfolio_state()
    state["portfolios"].pop(portfolio_id, None)
    save_portfolio_state(state)


def load_portfolio_state():
    state = _load_json(PORTFOLIO_STATE_PATH, None)
    if state is None:
        return deepcopy(DEFAULT_PORTFOLIO_STATE)
    return state


def save_portfolio_state(state):
    _save_json(PORTFOLIO_STATE_PATH, state)


def get_portfolio(portfolio_id):
    cfg = load_portfolio_config()
    pf = cfg["portfolios"].get(portfolio_id)
    if pf is None:
        raise KeyError(f"Portfolio '{portfolio_id}' not found")
    return pf


def list_portfolios():
    cfg = load_portfolio_config()
    return {
        pid: {
            "label": pf.get("label", pid),
            "capital": pf.get("capital", 0),
            "strategy_count": len(pf.get("strategies", [])),
        }
        for pid, pf in cfg["portfolios"].items()
    }


def parse_date_from_slug(slug):
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    pattern = rf"({months})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?"
    m = re.search(pattern, slug.replace("-", " "), re.IGNORECASE)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3) or _hkt_now().year}", "%B %d %Y")
        except Exception:
            pass
    return None


def search_polymarket_events(query="hong-kong-temperature"):
    """Search Polymarket for weather events matching the query."""
    try:
        url = "https://gamma-api.polymarket.com/public-search"
        resp = requests.get(url, params={"q": query}, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        events = []
        for e in raw:
            title = e.get("title", "")
            slug = e.get("slug", "")
            if title and slug:
                events.append({"title": title, "slug": slug})
        return events
    except Exception as ex:
        logger.warning("Polymarket search failed: %s", ex)
        return []


def discover_events_for_portfolio(portfolio_id):
    """Discover active events for each strategy in a portfolio based on entry_rules.

    Returns {strategy_key: [matched_slug, ...]}
    """
    pf = get_portfolio(portfolio_id)
    registry = load_strategy_registry()
    strategies = pf.get("strategies", [])
    all_events = search_polymarket_events()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    results = {}
    for sk in strategies:
        sdef = registry["strategies"].get(sk)
        if sdef is None:
            continue
        matched = []
        for evt in all_events:
            slug = evt["slug"]
            if check_entry_rules(sdef, event_slug=slug, now=now):
                matched.append(slug)
        if not matched:
            today_slug = _default_slug(now)
            if all(check_entry_rules(sdef, event_slug=today_slug, now=now) for evt in all_events if evt["slug"] == today_slug) or True:
                try:
                    if check_entry_rules(sdef, event_slug=today_slug, now=now):
                        if today_slug not in matched:
                            matched.append(today_slug)
                except Exception:
                    pass
        results[sk] = matched
    return results


def run_portfolio(portfolio_id, context_overrides=None, strategy_contexts=None, force=False, prices_dict=None):
    """Run all enabled strategies for a portfolio across their active events.

    Parameters
    ----------
    portfolio_id : str
    context_overrides : dict, optional
        Flat context dict shared by ALL strategies (backward compat).
    strategy_contexts : dict of {strategy_key: dict}, optional
        Per-strategy context dicts. Keys are strategy IDs, values are
        the context overrides for that strategy. Takes precedence over
        context_overrides for matching strategies.
    force : bool
        If True, skip the scheduler_on check. Use for manual Run Now.
    prices_dict : dict, optional
        Current market prices for PnL snapshot. If None, tries to
        extract from context.

    Returns list of result dicts.
    """
    pf = get_portfolio(portfolio_id)
    state = load_portfolio_state()
    pf_state = state["portfolios"].get(portfolio_id, {})
    if not force and not pf_state.get("scheduler_on", False):
        return [{"status": "skipped_scheduler_off", "portfolio_id": portfolio_id}]

    registry = load_strategy_registry()
    events_per_strategy = discover_events_for_portfolio(portfolio_id)
    base_ctx = context_overrides or {}
    strategy_contexts = strategy_contexts or {}

    strategy_models = pf.get("strategy_models", {})
    results = []
    for sk in pf.get("strategies", []):
        sdef = registry["strategies"].get(sk)
        if sdef is None:
            continue
        sk_model_key = strategy_models.get(sk, sk)
        # Merge: base_ctx first, then per-strategy overrides
        sk_ctx = dict(base_ctx)
        sk_ctx.update(strategy_contexts.get(sk, {}))
        sk_ctx.setdefault("model_key", sk_model_key)
        for slug in events_per_strategy.get(sk, []):
            result = run_single_strategy_cycle(
                strategy_key=sk,
                strategy_config=sdef,
                portfolio_id=portfolio_id,
                event_slug=slug,
                **sk_ctx
            )
            result["portfolio_id"] = portfolio_id
            result["event_slug"] = slug
            results.append(result)

    # Save PnL snapshot after running
    if prices_dict is None:
        for sk_ctx in strategy_contexts.values():
            prices_dict = sk_ctx.get("prices_dict")
            if prices_dict:
                break
    if prices_dict is None and base_ctx:
        prices_dict = base_ctx.get("prices_dict")
    try:
        save_pnl_snapshot(portfolio_id, prices_dict)
    except Exception as e:
        logger.warning("Failed to save PnL snapshot: %s", e)

    return results


def check_portfolio_capital(portfolio_id, new_order_value=0):
    """Check if adding new_order_value would exceed portfolio capital limit."""
    pf = get_portfolio(portfolio_id)
    capital = pf.get("capital", 0)
    current_exposure = get_portfolio_exposure(portfolio_id)
    return (current_exposure + new_order_value) <= capital


def get_portfolio_exposure(portfolio_id):
    """Sum all position quantities * prices for a portfolio."""
    pos = _load_json(POSITIONS_PATH, {})
    pf_pos = pos.get(portfolio_id, {})
    total = 0.0
    for slug, slug_pos in pf_pos.items():
        for sk, buckets in slug_pos.items():
            for bucket, data in buckets.items():
                total += data.get("quantity", 0) * data.get("entry_price", 0)
    return total


def get_portfolio_positions(portfolio_id: str) -> dict:
    """Return all positions for a portfolio, keyed by slug -> strategy -> bucket.

    Returns empty dict if no positions found.
    """
    pos = _load_json(POSITIONS_PATH, {})
    return pos.get(portfolio_id, {})


def _get_adapter_fees() -> dict:
    """Fetch fees from paper-trader adapter, mapped by (event_slug, bucket).

    Returns dict: {(event_slug, bucket): fee}
    """
    try:
        from execution.paper_adapter import _get_adapter
        adapter = _get_adapter()
        raw = adapter.get_position_fees()
        mapped = {}
        for (sub_slug, outcome), fee in raw.items():
            try:
                # Map: adapter.cache_buckets stores {bucket: {sub_slug, ...}}
                # Find which bucket this sub_slug maps to
                for bucket, info in adapter._buckets.items():
                    if info.get("slug") == sub_slug:
                        # Returned slug was stored during execute_target_positions
                        mapped[(adapter._slug, bucket)] = fee
            except Exception:
                pass
        return mapped
    except Exception:
        return {}


def get_portfolio_pnl(portfolio_id, prices_dict=None, strategy_keys=None):
    """Calculate total PnL for a portfolio across all strategies and events.

    If prices_dict is None, returns cost basis only (no mark-to-market).
    Handles both YES and NO positions correctly (NO side uses 1 - price).
    If strategy_keys is provided, only include positions for those strategies.
    """
    pos = _load_json(POSITIONS_PATH, {})
    pf_pos = pos.get(portfolio_id, {})
    if strategy_keys is None:
        try:
            pf = get_portfolio(portfolio_id)
            strategy_keys = set(pf.get("strategies", []))
        except (KeyError, TypeError):
            strategy_keys = None
    total_cost = 0.0
    total_market = 0.0
    total_fees = 0.0
    details = []
    adapter_fees = _get_adapter_fees()
    for slug, slug_pos in pf_pos.items():
        slug_cost = 0.0
        slug_market = 0.0
        slug_fees = 0.0
        for sk, buckets in slug_pos.items():
            if strategy_keys and sk not in strategy_keys:
                continue
            for bucket, data in buckets.items():
                qty = data.get("quantity", 0)
                is_no = data.get("side", "YES") == "NO"
                entry_px_raw = data.get("entry_price", 0)
                entry_px = 1.0 - entry_px_raw if is_no else entry_px_raw
                cost = qty * entry_px
                if prices_dict:
                    yes_price = prices_dict.get(bucket)
                    if yes_price is None:
                        cur_px = entry_px
                    elif is_no:
                        cur_px = 1.0 - yes_price
                    else:
                        cur_px = yes_price
                else:
                    cur_px = entry_px
                market = qty * cur_px
                fee = adapter_fees.get((slug, bucket), 0.0)
                total_cost += cost
                total_market += market
                total_fees += fee
                slug_cost += cost
                slug_market += market
                slug_fees += fee
                details.append({
                    "slug": slug, "strategy": sk, "bucket": bucket,
                    "side": "YES" if not is_no else "NO",
                    "quantity": qty, "entry_price": entry_px,
                    "current_price": cur_px,
                    "cost_basis": cost, "market_value": market,
                    "fee": fee,
                })
        if slug_cost > 0:
            details.append({
                "slug": slug, "_summary": True,
                "cost_basis": slug_cost, "market_value": slug_market,
                "pnl": slug_market - slug_cost, "fee": slug_fees,
            })
    return {
        "portfolio_id": portfolio_id,
        "cost_basis": total_cost,
        "market_value": total_market,
        "unrealized_pnl": total_market - total_cost,
        "total_fees": total_fees,
        "details": details,
    }


PNL_HISTORY_DIR = Path('data/pnl_history')


def save_pnl_snapshot(portfolio_id, prices_dict=None):
    """Record current PnL snapshot to historical parquet file."""
    pnl = get_portfolio_pnl(portfolio_id, prices_dict)
    PNL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = PNL_HISTORY_DIR / f'{portfolio_id}.parquet'
    snapshot = pd.DataFrame([{
        'timestamp': _hkt_now(),
        'cost_basis': pnl['cost_basis'],
        'market_value': pnl['market_value'],
        'unrealized_pnl': pnl['unrealized_pnl'],
    }])
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, snapshot], ignore_index=True)
    else:
        df = snapshot
    df.to_parquet(path, index=False)
    return pnl


def reset_portfolio(portfolio_id):
    """Clear all positions, PnL history, and audit log for a portfolio."""
    # Reset paper-trader adapter
    try:
        from execution.paper_adapter import _get_adapter
        ap = _get_adapter()
        ap.close_all_positions()
        ap.reset()
    except Exception as e:
        logger.warning("Failed to reset paper-trader adapter: %s", e)

    # Clear positions
    try:
        from execution.portfolio_reconciler import load_positions, save_positions
        pos = load_positions()
        if portfolio_id in pos:
            del pos[portfolio_id]
            save_positions(pos)
            logger.info("Cleared positions for portfolio '%s'", portfolio_id)
    except Exception as e:
        logger.warning("Failed to clear positions for '%s': %s", portfolio_id, e)

    # Clear PnL history
    try:
        path = PNL_HISTORY_DIR / f'{portfolio_id}.parquet'
        if path.exists():
            path.unlink()
            logger.info("Cleared PnL history for '%s'", portfolio_id)
    except Exception as e:
        logger.warning("Failed to clear PnL history for '%s': %s", portfolio_id, e)

    # Clear audit log entries for this portfolio
    try:
        audit_path = Path('data/paper_trade_audit.parquet')
        if audit_path.exists():
            import pandas as _pd
            df = _pd.read_parquet(audit_path)
            if 'portfolio_id' in df.columns:
                df = df[df['portfolio_id'] != portfolio_id]
                df.to_parquet(audit_path, index=False)
                logger.info("Cleared audit log for '%s'", portfolio_id)
    except Exception as e:
        logger.warning("Failed to clear audit log for '%s': %s", portfolio_id, e)

    # Reset portfolio state
    try:
        state = load_portfolio_state()
        if portfolio_id in state.get("portfolios", {}):
            state["portfolios"][portfolio_id] = {"scheduler_on": False}
            save_portfolio_state(state)
            logger.info("Reset state for '%s'", portfolio_id)
    except Exception as e:
        logger.warning("Failed to reset state for '%s': %s", portfolio_id, e)


def load_pnl_history(portfolio_id):
    """Load PnL history parquet into a DataFrame."""
    path = PNL_HISTORY_DIR / f'{portfolio_id}.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Drawdown Tracker
# ═══════════════════════════════════════════════════════════════

class DrawdownTracker:
    """Track portfolio peak and compute drawdown for risk management.

    Parameters
    ----------
    initial_equity : float
        Starting equity value.
    config : dict, optional
        Config with drawdown thresholds. If None, loads from paper_strategies.json.
    """

    def __init__(self, initial_equity: float, config: dict = None):
        self._peak = initial_equity
        self._current = initial_equity
        self._high_water_mark = initial_equity
        self._drawdown_pct = 0.0
        self._history = []  # list of (timestamp, equity, drawdown_pct)
        if config is None:
            config = load_strategy_config()
        self._config = config

        from copy import deepcopy
        self._snapshot = {
            'peak': self._peak,
            'trough': initial_equity,
            'recovery_peak': initial_equity,
            'max_drawdown': 0.0,
        }

    @property
    def drawdown_pct(self) -> float:
        return self._drawdown_pct

    @property
    def max_drawdown_pct(self) -> float:
        return self._snapshot['max_drawdown']

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def current_equity(self) -> float:
        return self._current

    def update(self, equity: float, timestamp=None):
        """Update tracker with current equity value. Returns (drawdown_pct, action)."""
        self._current = equity
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

        if equity > self._peak:
            self._peak = equity
            self._snapshot['recovery_peak'] = equity

        dd = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
        self._drawdown_pct = dd

        if dd > self._snapshot['max_drawdown']:
            self._snapshot['max_drawdown'] = dd
            self._snapshot['trough'] = equity

        self._history.append((timestamp, equity, dd))

        # Determine action from config
        action, mult = self._get_action(dd)
        return dd, action, mult

    def _get_action(self, dd: float):
        """Map current drawdown to action + multiplier."""
        dd_cfg = self._config.get('exit', {}).get('drawdown', {})
        hard = dd_cfg.get('hard_flatten', -0.15)
        reduce = dd_cfg.get('reduce_risk', -0.075)
        stop = dd_cfg.get('stop_new_entries', -0.10)

        dd_signed = -dd  # config uses negative values

        if dd_signed <= hard:
            return 'HARD_FLATTEN', 0.0
        if dd_signed <= reduce:
            ratio = max(0.0, dd_signed / reduce)
            return 'REDUCE_RISK', ratio
        if dd_signed <= stop:
            return 'STOP_ENTRIES', 0.0
        return 'NORMAL', 1.0

    def get_action(self) -> str:
        """Return current risk action based on drawdown."""
        action, _ = self._get_action(self._drawdown_pct)
        return action

    def get_multiplier(self) -> float:
        """Return current sizing multiplier based on drawdown."""
        _, mult = self._get_action(self._drawdown_pct)
        return mult

    def get_history(self) -> list:
        """Return list of (timestamp, equity, drawdown_pct)."""
        return list(self._history)

    def get_max_drawdown_info(self) -> dict:
        """Return dict with peak, trough, max_drawdown_pct, recovery_peak."""
        return dict(self._snapshot)

    def reset(self, new_equity: float = None):
        """Reset tracker state, optionally with a new starting equity."""
        if new_equity is not None:
            self._peak = new_equity
            self._current = new_equity
        else:
            self._peak = self._current
        self._high_water_mark = self._peak
        self._drawdown_pct = 0.0
        self._history = []
        self._snapshot = {
            'peak': self._peak,
            'trough': self._current,
            'recovery_peak': self._peak,
            'max_drawdown': 0.0,
        }
