# execution/strategy_runner.py
import sys
import json
import yaml
import logging
import importlib
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

from execution.strategy_engine import load_config_for_strategy

CONFIG_PATH = Path('config/paper_strategies.json')
STATE_PATH = Path('data/paper_strategy_state.json')
ROOT_CONFIG_PATH = Path('config.yaml')

DEFAULT_ACCOUNTS = [
    'baseline_paper', 'rain_observed_paper',
    'rain_nowcast_paper', 'gated_ensemble_paper',
    'enhanced_v1_paper', 'enhanced_v2_paper',
    'enhanced_v2_aggressive_paper', 'enhanced_v2_conservative_paper'
]

DEFAULT_STATE = {
    "version": 1,
    "accounts": {
        acct: {"strategy": acct, "status": "idle", "last_run": None, "scheduler_on": False}
        for acct in DEFAULT_ACCOUNTS
    }
}

_STRATEGY_REGISTRY = None


def _read_config_yaml():
    with open(ROOT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_json(path, default):
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Cannot read %s: %s", path, e)
    return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_strategy_registry():
    global _STRATEGY_REGISTRY
    raw = _load_json(CONFIG_PATH, None)
    if raw is None:
        logger.warning("paper_strategies.json not found, using embedded defaults")
        return _embedded_registry()
    validate_strategy_config(raw)
    _STRATEGY_REGISTRY = raw
    return raw


def save_strategy_registry(registry):
    _save_json(CONFIG_PATH, registry)
    global _STRATEGY_REGISTRY
    _STRATEGY_REGISTRY = registry


def _embedded_registry():
    return {
        "version": 1,
        "default_strategy": "enhanced_v1_paper",
        "strategies": {
            acct: {
                "label": acct.replace("_", " ").title(),
                "module": "execution.rebalancer",
                "entry_point": "generate_orders_from_probs",
                "paper_only": True,
                "description": "",
                "exposure_limits": None
            }
            for acct in DEFAULT_ACCOUNTS
        }
    }


def validate_strategy_config(config):
    if not isinstance(config, dict):
        raise ValueError("Strategy config must be a dict")
    if "strategies" not in config:
        raise ValueError("Strategy config missing 'strategies' key")
    for sid, sdef in config["strategies"].items():
        if not isinstance(sdef, dict):
            raise ValueError(f"Strategy '{sid}' definition must be a dict")
        if sdef.get("paper_only") is not True:
            raise ValueError(
                f"Strategy '{sid}' has paper_only={sdef.get('paper_only')!r}. "
                f"paper_only must be true. Invalid config rejected."
            )
        if "module" not in sdef:
            raise ValueError(f"Strategy '{sid}' missing 'module'")
        if "entry_point" not in sdef:
            raise ValueError(f"Strategy '{sid}' missing 'entry_point'")


def load_strategy_state():
    state = _load_json(STATE_PATH, None)
    if state is None:
        state = DEFAULT_STATE.copy()
        state["accounts"] = {
            acct: dict(cfg) for acct, cfg in DEFAULT_STATE["accounts"].items()
        }
    return state


def save_strategy_state(state):
    _save_json(STATE_PATH, state)


def get_strategy_entry_point(strategy_id, registry=None):
    if registry is None:
        registry = _STRATEGY_REGISTRY or load_strategy_registry()
    sdef = registry["strategies"].get(strategy_id)
    if sdef is None:
        raise KeyError(f"Strategy '{strategy_id}' not found in registry")
    mod = importlib.import_module(sdef["module"])
    fn = getattr(mod, sdef["entry_point"], None)
    if fn is None:
        raise AttributeError(
            f"Module '{sdef['module']}' has no entry_point '{sdef['entry_point']}'"
        )
    return fn


def _paper_guard():
    root_cfg = _read_config_yaml()
    exec_cfg = root_cfg.get("execution", {})
    if exec_cfg.get("allow_live_orders", False):
        raise RuntimeError(
            "Paper guard triggered: allow_live_orders is True but strategy_runner "
            "requires paper-only mode. Set allow_live_orders=False in config.yaml."
        )
    return True


def _state_for(account_id, state=None):
    if state is None:
        state = load_strategy_state()
    acct = state["accounts"].get(account_id)
    if acct is None:
        raise KeyError(f"Account '{account_id}' not found in strategy state")
    return acct


def start_strategy(account_id, strategy_id=None, registry=None):
    if registry is None:
        registry = _STRATEGY_REGISTRY or load_strategy_registry()
    _paper_guard()
    if strategy_id is None:
        strategy_id = registry.get("default_strategy", DEFAULT_ACCOUNTS[0])
    if strategy_id not in registry["strategies"]:
        raise KeyError(
            f"Strategy '{strategy_id}' not in registry. "
            f"Available: {list(registry['strategies'].keys())}"
        )
    state = load_strategy_state()
    if account_id not in state["accounts"]:
        state["accounts"][account_id] = {
            "strategy": strategy_id,
            "status": "idle",
            "last_run": None,
            "scheduler_on": False
        }
    acct = state["accounts"][account_id]
    acct["strategy"] = strategy_id
    acct["status"] = "running"
    acct["scheduler_on"] = True
    save_strategy_state(state)
    return state


def pause_strategy(account_id):
    _paper_guard()
    state = load_strategy_state()
    acct = _state_for(account_id, state)
    if acct["status"] != "running":
        logger.warning("Account '%s' is not running (status=%s), pause skipped", account_id, acct["status"])
    acct["status"] = "paused"
    acct["scheduler_on"] = False
    save_strategy_state(state)
    return state


def stop_strategy(account_id):
    _paper_guard()
    state = load_strategy_state()
    acct = _state_for(account_id, state)
    acct["status"] = "stopped"
    acct["scheduler_on"] = False
    acct["last_run"] = None
    acct.pop("last_decisions", None)
    save_strategy_state(state)
    return state


def is_due_to_run(account_id, interval_sec=300):
    state = load_strategy_state()
    acct = _state_for(account_id, state)
    if acct["scheduler_on"] is not True:
        return False
    if acct["status"] != "running":
        return False
    if acct["last_run"] is None:
        return True
    try:
        last = datetime.fromisoformat(acct["last_run"])
    except (TypeError, ValueError):
        return True
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed = (now - last).total_seconds()
    return elapsed >= interval_sec


def preview_reconcile(portfolio_id, slug, strategy_key, target_positions, strategy_context=None):
    """Preview reconciliation without writing positions or audit log.

    Returns the ReconciliationResult with preview=True flag.
    Useful for dry-run / what-if analysis before committing.
    """
    from execution.portfolio_reconciler import reconcile_positions, load_positions
    all_pos = load_positions()
    return reconcile_positions(
        all_pos, target_positions, portfolio_id, slug, strategy_key,
        strategy_context=strategy_context, preview=True
    )


def list_strategies(registry=None):
    if registry is None:
        registry = _STRATEGY_REGISTRY or load_strategy_registry()
    return {
        sid: {
            "label": sdef.get("label", sid),
            "description": sdef.get("description", ""),
            "paper_only": sdef.get("paper_only", True),
        }
        for sid, sdef in registry["strategies"].items()
    }


def _result(status, account_id, strategy, error=None, **extra):
    d = {"status": status, "account_id": account_id, "strategy": strategy}
    if error:
        d["error"] = error
    d.update(extra)
    return d


def _default_slug(now=None):
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    from datetime import timedelta
    target_dt = now + timedelta(hours=8)
    month_str = target_dt.strftime("%B").lower()
    return f"highest-temperature-in-hong-kong-on-{month_str}-{target_dt.day}-{target_dt.year}"


def check_entry_rules(strategy_config, event_slug=None, now=None):
    entry_rules = strategy_config.get("entry_rules", {})
    if not entry_rules:
        return True
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    from datetime import timedelta
    hkt_now_dt = now + timedelta(hours=8)
    min_hour = entry_rules.get("min_hour", 0)
    if hkt_now_dt.hour < min_hour:
        return False
    only_on_event_date = entry_rules.get("only_on_event_date", False)
    if only_on_event_date and event_slug:
        try:
            from re import search as _re_search
            months = "January|February|March|April|May|June|July|August|September|October|November|December"
            pattern = rf"({months})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?"
            m = _re_search(pattern, event_slug.replace("-", " "))
            if m:
                from calendar import month_name
                slug_month = m.group(1)
                slug_day = int(m.group(2))
                slug_year = int(m.group(3)) if m.group(3) else hkt_now_dt.year
                if slug_month != hkt_now_dt.strftime("%B") or slug_day != hkt_now_dt.day or slug_year != hkt_now_dt.year:
                    return False
        except Exception:
            pass
    return True


def run_single_strategy_cycle(strategy_key, strategy_config, portfolio_id=None, event_slug=None, now=None, **context):
    """Run one full paper strategy cycle safely.

    Parameters
    ----------
    strategy_key : str
        Account / paper_model_key (e.g. 'baseline_paper').
    strategy_config : dict
        Strategy definition from registry (must include module, entry_point,
        paper_only=True).
    portfolio_id : str, optional
        Portfolio identifier (e.g. 'weather_main').
    event_slug : str, optional
        Specific event slug to trade. If None, uses context slug or _default_slug().
    now : datetime, optional
        Override for deterministic testing.
    **context : dict
        Runtime context. Expected keys:
        - target_probs, prices_dict, token_ids_dict
        - capital, mock_slippage
        - model_std, recent_price_volatility, temp_now, max_so_far, rain_regime

    Returns
    -------
    dict with keys: status, account_id, strategy, error (if any).
    """
    _paper_guard()

    if not isinstance(strategy_config, dict):
        return _result("blocked", strategy_key, strategy_key,
                       "strategy_config not a dict")
    if strategy_config.get("paper_only") is not True:
        return _result("blocked", strategy_key, strategy_key,
                       "paper_only must be true")

    slug = event_slug or context.get("slug") or _default_slug(now)
    if not check_entry_rules(strategy_config, event_slug=slug, now=now):
        return _result("skipped_entry_rules", strategy_key, strategy_key,
                       "entry_rules not satisfied (min_hour or only_on_event_date)")

    entry_point = strategy_config.get("entry_point")
    module_path = strategy_config.get("module")
    if not entry_point or not module_path:
        return _result("blocked", strategy_key, strategy_key,
                       "missing entry_point or module in strategy_config")

    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        return _result("dependency_missing", strategy_key, strategy_key,
                       f"cannot import {module_path}: {e}")

    fn = getattr(mod, entry_point, None)
    if fn is None:
        return _result("dependency_missing", strategy_key, strategy_key,
                       f"entry_point '{entry_point}' not found in {module_path}")

    pid = portfolio_id or context.get("portfolio_id", "default")
    model_key = context.get("model_key", strategy_key)
    capital = context.get("capital", 1000.0)
    mock_slippage = context.get("mock_slippage", True)

    try:
        if entry_point == "run_config_rebalance_cycle":
            target_probs = context.get("target_probs")
            prices_dict = context.get("prices_dict")
            token_ids_dict = context.get("token_ids_dict")
            if not target_probs or not prices_dict:
                return _result("dependency_missing", strategy_key, model_key,
                               "target_probs and prices_dict required in context")

            config = load_config_for_strategy(strategy_key)

            summary = fn(
                slug=slug, model_key=model_key, capital=capital,
                mock_slippage=mock_slippage,
                target_probs=target_probs, prices_dict=prices_dict,
                token_ids_dict=token_ids_dict,
                config=config, strategy_key=strategy_key,
                dt_now=None, current_positions=None,
                temp_now=context.get("temp_now"),
                max_so_far=context.get("max_so_far"),
                rain_regime=context.get("rain_regime"),
                model_std=context.get("model_std", 1.0),
                recent_price_volatility=context.get("recent_price_volatility", 0.0),
                hours_to_settlement=context.get("hours_to_settlement", 24.0),
                nowcast_stale=context.get("nowcast_stale", False),
                data_missing=context.get("data_missing", False),
                drawdown_pct=context.get("drawdown_pct", 0.0),
                probs_old=context.get("probs_old"),
                probs_new=context.get("probs_new"),
            )

            target_positions = summary.get("target_positions", {})
            if target_positions:
                from execution.paper_adapter import _get_adapter
                strategy_context = {
                    "strategy_key": strategy_key,
                    "strategy_version": strategy_config.get("label", ""),
                    "scheduler_source": context.get("scheduler_source", "manual"),
                    "selected_model": model_key,
                }
                _get_adapter().execute_target_positions(
                    target_positions, pid, slug, strategy_key, prices_dict, strategy_context
                )

            return _result("completed", strategy_key, model_key,
                           f"{len(target_positions)} buckets updated",
                           decisions=summary.get("decisions", []),
                           time_slot=summary.get("time_slot", ""),
                           rebalance_triggers=summary.get("rebalance_triggers", []))

        if entry_point == "run_rebalance_cycle":
            rain_regime = context.get("rain_regime")
            target_dt = context.get("target_dt")
            fn(
                slug=slug, model_key=model_key, capital=capital,
                mock_slippage=mock_slippage, rain_regime=rain_regime,
                target_dt=target_dt
            )
            return _result("completed", strategy_key, model_key,
                           "run_rebalance_cycle handled internally")

        if entry_point == "run_enhanced_rebalance_cycle":
            target_probs = context.get("target_probs")
            prices_dict = context.get("prices_dict")
            token_ids_dict = context.get("token_ids_dict")
            if not target_probs or not prices_dict:
                return _result("dependency_missing", strategy_key, model_key,
                               "target_probs and prices_dict required in context")

            summary = fn(
                slug=slug, model_key=model_key, capital=capital,
                mock_slippage=mock_slippage,
                target_probs=target_probs, prices_dict=prices_dict,
                token_ids_dict=token_ids_dict,
                dt_now=now, current_positions=None,
                temp_now=context.get("temp_now"),
                max_so_far=context.get("max_so_far"),
                rain_regime=context.get("rain_regime"),
                model_std=context.get("model_std", 1.0),
                recent_price_volatility=context.get("recent_price_volatility", 0.0),
            )

            target_positions = summary.get("target_positions", {})
            if target_positions:
                from execution.paper_adapter import _get_adapter
                strategy_context = {
                    "strategy_key": strategy_key,
                    "strategy_version": strategy_config.get("label", ""),
                    "scheduler_source": context.get("scheduler_source", "manual"),
                    "selected_model": model_key,
                }
                _get_adapter().execute_target_positions(
                    target_positions, pid, slug, strategy_key, prices_dict, strategy_context
                )

            return _result("completed", strategy_key, model_key,
                           f"{len(target_positions)} buckets updated",
                           decisions=summary.get("decisions", []),
                           time_slot=summary.get("time_slot", ""),
                           effective_limit=summary.get("effective_exposure_limit", 0))

        if entry_point == "generate_orders_from_probs":
            target_probs = context.get("target_probs")
            prices_dict = context.get("prices_dict")
            token_ids_dict = context.get("token_ids_dict")
            if not target_probs or not prices_dict:
                return _result("dependency_missing", strategy_key, model_key,
                               "target_probs and prices_dict required in context")

            orders = fn(
                target_probs, prices_dict,
                token_ids_dict or {}, capital, mock_slippage
            )
            target_positions = orders

            if target_positions:
                from execution.paper_adapter import _get_adapter
                strategy_context = {
                    "strategy_key": strategy_key,
                    "strategy_version": strategy_config.get("label", ""),
                    "scheduler_source": context.get("scheduler_source", "manual"),
                    "selected_model": model_key,
                }
                _get_adapter().execute_target_positions(
                    target_positions, pid, slug, strategy_key, prices_dict, strategy_context
                )

            return _result("completed", strategy_key, model_key,
                           f"{len(target_positions)} buckets updated")

        logger.warning("Unknown entry_point '%s' for strategy '%s', trying generic call",
                       entry_point, strategy_key)
        result = fn(**context)
        return _result("completed", strategy_key, model_key,
                       f"generic call returned {type(result).__name__}")

    except Exception as e:
        logger.exception("Strategy cycle failed for %s/%s", strategy_key, entry_point)
        return _result("error", strategy_key, model_key, str(e))


def run_enabled_strategies_once(registry=None, interval_sec=None, portfolio_id=None, event_slug=None):
    """Iterate all enabled, due strategies and run one cycle each.

    Steps:
    1. Load and validate registry (paper_only guardrail).
    2. Load state.
    3. For each account with scheduler_on=True, status=running, and due:
       call run_single_strategy_cycle, then update last_run.
    4. Return list of result dicts.
    """
    if registry is None:
        registry = _STRATEGY_REGISTRY or load_strategy_registry()
    validate_strategy_config(registry)
    _paper_guard()

    if interval_sec is None:
        interval_sec = registry.get("rebalance_interval_minutes", 5) * 60

    state = load_strategy_state()
    results = []

    for account_id, acct in list(state["accounts"].items()):
        if acct.get("scheduler_on") is not True:
            continue
        if acct.get("status") != "running":
            continue

        strategy_id = acct.get("strategy")
        if strategy_id not in registry["strategies"]:
            results.append({
                "status": "error",
                "account_id": account_id,
                "strategy": strategy_id,
                "error": f"Strategy '{strategy_id}' not in registry",
            })
            continue

        if not is_due_to_run(account_id, interval_sec=interval_sec):
            results.append({
                "status": "skipped_not_due",
                "account_id": account_id,
                "strategy": strategy_id,
            })
            continue

        strategy_config = registry["strategies"].get(strategy_id)
        result = run_single_strategy_cycle(
            strategy_key=account_id,
            strategy_config=strategy_config,
            portfolio_id=portfolio_id,
            event_slug=event_slug,
            **{}
        )
        results.append(result)

        if result.get("status") not in ("error", "blocked", "dependency_missing", "skipped_entry_rules"):
            state["accounts"][account_id]["last_run"] = (
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            )
            save_strategy_state(state)

    return results
