# execution/rebalancer.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import numpy as np
import pandas as pd
import requests
import re
import math
import yaml
from datetime import datetime, timedelta, timezone

_HKT_OFFSET = timedelta(hours=8)
def _hkt_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + _HKT_OFFSET

from features.live_feature_builder import build_features_for_date, update_forecast_database
from models.inference import predict_distribution, predict_bucket_probabilities
from execution.kelly_betting import compute_multi_kelly_bets
from execution.clob_slippage import apply_slippage_to_bets
from execution.strategy_gate import (
    evaluate_all_gates, validate_bucket_parsing, classify_rain_regime,
    select_prediction_engine, check_entry_gate_phase2, decide_entry_exit
)
from execution.strategy_engine import (
    compute_enhanced_orders, run_enhanced_rebalance_cycle,
    compute_config_orders, run_config_rebalance_cycle,
    get_time_slot, get_effective_exposure_limit,
    get_confidence_multiplier, get_volatility_multiplier,
    hkt_now, ENHANCED_VERSION, load_config_for_strategy
)
from execution.portfolio_reconciler import (
    load_positions, save_positions, reconcile_positions,
    build_audit_events, write_audit_log, PM_MIN_QTY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PM_QTY_STEP = 0.01
PM_MIN_PRICE = 0.001
PM_MAX_PRICE = 0.999
PM_PRICE_STEP = 0.001


def sanitize_polymarket_order(qty: float, price_usd: float):
    price_usd = max(PM_MIN_PRICE, min(PM_MAX_PRICE, price_usd))
    price_usd = round(round(price_usd / PM_PRICE_STEP) * PM_PRICE_STEP, 3)
    qty = math.floor(qty / PM_QTY_STEP) * PM_QTY_STEP
    return round(qty, 2), price_usd


def reset_portfolio(model_key: str):
    pos = load_positions()
    if model_key in pos:
        del pos[model_key]
        save_positions(pos)


def close_position_manually(model_key: str, slug: str, bucket: str):
    pos = load_positions()
    if model_key in pos and slug in pos[model_key] and bucket in pos[model_key][slug]:
        del pos[model_key][slug][bucket]
        if not pos[model_key][slug]:
            del pos[model_key][slug]
        if not pos[model_key]:
            del pos[model_key]
        save_positions(pos)


def calculate_pnl(model_positions: dict, prices_yes: dict) -> list:
    records = []
    total_unrealized_pnl = 0.0
    total_cost_basis = 0.0

    for slug, buckets in model_positions.items():
        for bucket, pos_data in buckets.items():
            side = pos_data['side']
            qty = pos_data['quantity']
            entry_price = pos_data.get('entry_price', 0.5)
            price_yes = prices_yes.get(bucket, 0.5)

            if side == 'YES':
                current_market_price = price_yes
            else:
                current_market_price = 1.0 - price_yes

            pnl = (current_market_price - entry_price) * qty
            cost_basis = entry_price * qty

            total_unrealized_pnl += pnl
            total_cost_basis += cost_basis

            records.append({
                'Market': slug.split('-in-hong-kong-on-')[0].upper() if '-in-hong-kong-on-' in slug else slug,
                'Bucket': bucket,
                'Side': side,
                'Quantity': qty,
                'Entry Price (¢)': round(entry_price * 100, 1),
                'Current Price (¢)': round(current_market_price * 100, 1),
                'Unrealized PnL ($)': round(pnl, 2)
            })

    return records, round(total_unrealized_pnl, 2), round(total_cost_basis, 2)


def fetch_market_state(slug: str) -> dict:
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return None
        markets = events[0].get('markets', [])
        buckets, prices_dict, token_ids_dict = [], {}, {}
        for m in markets:
            title = m.get('groupItemTitle', '')
            match = re.search(r'(\d+)', title)
            if not match:
                continue
            val = float(match.group(1))
            title_lower = title.lower()
            if any(kw in title_lower for kw in ['below', 'lower', 'under']):
                lower, upper = -np.inf, val + 1.0
            elif any(kw in title_lower for kw in ['higher', 'above', 'over']):
                lower, upper = val, np.inf
            else:
                lower, upper = val, val + 1.0
            try:
                outcomes = json.loads(m.get('outcomes', '[]'))
                prices = json.loads(m.get('outcomePrices', '[]'))
                yes_idx = next((i for i, out in enumerate(outcomes) if out.lower() == 'yes'), 0)
                price_yes = float(prices[yes_idx]) if yes_idx < len(prices) else 0.01
            except Exception:
                price_yes = 0.01
            clob_ids_str = m.get('clobTokenIds', '[]')
            try:
                clob_ids = json.loads(clob_ids_str)
                if len(clob_ids) >= 2:
                    token_ids_dict[title] = (clob_ids[0], clob_ids[1])
            except Exception:
                pass
            buckets.append({'name': title, 'lower': lower, 'upper': upper})
            prices_dict[title] = price_yes
        return {
            'slug': slug, 'buckets': buckets, 'prices_dict': prices_dict,
            'token_ids_dict': token_ids_dict, 'bucket_parse_ok': True, 'bucket_labels': [b['name'] for b in buckets]
        }
    except Exception as e:
        logger.error(f"獲取市場狀態失敗: {e}")
        return None


def generate_orders_from_probs(probs_dict: dict, prices_dict: dict, token_ids_dict: dict, capital: float, mock_slippage: bool) -> list:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if config.get('execution', {}).get('allow_live_orders', False) is not True:
        logger.info("Live order placement is disabled by config. Running in paper mode only.")
    target_bets = compute_multi_kelly_bets(probs_dict, prices_dict, capital, max_per_bucket=0.15, total_max=0.50)
    adjusted_bets = apply_slippage_to_bets(target_bets, token_ids_dict, prices_dict=prices_dict, mock_mode=mock_slippage)

    target_positions = {}
    for label, bet in adjusted_bets.items():
        if bet.get('adjusted_quantity', 0) < PM_MIN_QTY:
            continue
        side = 'YES' if bet['action'] == 'BUY_YES' else 'NO'
        target_positions[label] = {
            'side': side,
            'quantity': bet['adjusted_quantity'],
            'target_price': bet['avg_fill_price']
        }

    return target_positions


def update_paper_positions(target_positions: dict, portfolio_id: str, slug: str, strategy_key: str, prices_dict: dict, strategy_context: dict = None):
    all_pos = load_positions()
    result = reconcile_positions(all_pos, target_positions, portfolio_id, slug, strategy_key, strategy_context)
    save_positions(result.positions_updated)
    audit_events = build_audit_events(result, prices_dict)
    write_audit_log(audit_events)


def run_rebalance_cycle(
    slug: str,
    model_key: str = '9d',
    capital: float = 1000.0,
    mock_slippage: bool = True,
    rain_regime: dict = None,
    target_dt: datetime = None,
    strategy_key: str = None
):
    logger.info(f"=== 啟動再平衡循環: {slug} (model={model_key}) ===")

    market_state = fetch_market_state(slug)
    if not market_state:
        logger.error("無法獲取市場狀態，中止再平衡")
        return

    bucket_validation = validate_bucket_parsing(market_state)
    if not bucket_validation.passed:
        logger.warning(f"桶解析失敗: {bucket_validation.detail}")
        return

    if target_dt is None:
        target_dt = _hkt_now().replace(hour=0, minute=0, second=0, microsecond=0)

    features, meta = build_features_for_date(target_dt)
    if not features:
        logger.error("無法構建特徵，中止再平衡")
        return

    mean, std = predict_distribution(features, 'tmax', meta.get('hko_spread') if meta else None, 1.0)
    is_today = (target_dt.date() == _hkt_now().date())
    probs = predict_bucket_probabilities(mean, std, market_state['buckets'], is_today=False, is_min_temp=False)

    gate_result = evaluate_all_gates(features, market_state, probs, market_state['prices_dict'], rain_regime)
    if not gate_result.passed:
        logger.warning(f"策略閘門阻斷: [{gate_result.reason_code}] {gate_result.detail}")
        return

    rain_regime_label = classify_rain_regime(rain_regime)
    pred_engine = select_prediction_engine(rain_regime_label, rain_regime)

    target_bets = compute_multi_kelly_bets(probs, market_state['prices_dict'], capital, max_per_bucket=0.15, total_max=0.50)
    adjusted_bets = apply_slippage_to_bets(target_bets, market_state['token_ids_dict'], prices_dict=market_state['prices_dict'], mock_mode=mock_slippage)

    target_positions = {}
    for label, bet in adjusted_bets.items():
        if bet.get('adjusted_quantity', 0) < PM_MIN_QTY:
            continue
        side = 'YES' if bet['action'] == 'BUY_YES' else 'NO'
        target_positions[label] = {
            'side': side,
            'quantity': bet['adjusted_quantity'],
            'target_price': bet['avg_fill_price']
        }

    current_positions = load_positions()
    entry_gate_p2 = check_entry_gate_phase2(
        target_positions, adjusted_bets, current_positions,
        model_key, slug, market_state['prices_dict'], capital
    )
    if not entry_gate_p2.passed:
        logger.warning(f"第二階段進場閘門阻斷: {entry_gate_p2.detail}")
        if not target_positions:
            return

    decisions = decide_entry_exit(current_positions, target_positions, model_key, slug)
    entry_count = sum(1 for d in decisions.values() if d['decision'] == 'ENTRY')
    exit_count = sum(1 for d in decisions.values() if d['decision'] == 'EXIT')
    logger.info(f"決策摘要: {entry_count} 進場, {exit_count} 出場, {len(decisions) - entry_count - exit_count} 持有/無交易")

    strategy_context = {
        'strategy_key': strategy_key or model_key,
        'selected_model': pred_engine.get('selected_model', model_key),
        'entry_reason': entry_gate_p2.reason_code if entry_gate_p2.passed else '',
        'exit_reason': '',
        'edge_before_slippage': None,
        'edge_after_slippage': None,
        'rain_regime': rain_regime_label,
        'rain_nc_sum_0_120m': (rain_regime or {}).get('rain_nc_sum_0_120m'),
        'rain_nowcast_age_minutes': (rain_regime or {}).get('nowcast_age_minutes'),
        'data_quality_flags': gate_result.reason_code,
        'reason_code': gate_result.reason_code,
    }

    update_paper_positions(target_positions, model_key, slug, market_state['prices_dict'], strategy_context)
    logger.info(f"再平衡完成，已更新 {model_key} 帳戶")


def run_config_rebalance(
    slug: str,
    model_key: str = 'intra',
    capital: float = 1000.0,
    mock_slippage: bool = True,
    target_probs: dict = None,
    prices_dict: dict = None,
    token_ids_dict: dict = None,
    strategy_key: str = None,
    dt_now: datetime = None,
    current_positions: dict = None,
    temp_now: float = None,
    max_so_far: float = None,
    rain_regime: str = None,
    model_std: float = 1.0,
    recent_price_volatility: float = 0.0,
    hours_to_settlement: float = 24.0,
    nowcast_stale: bool = False,
    data_missing: bool = False,
    drawdown_pct: float = 0.0,
    probs_old: dict = None,
    probs_new: dict = None
) -> dict:
    """Config-driven rebalance cycle entry point with logging and position persistence."""
    if dt_now is None:
        dt_now = _hkt_now()
    if target_probs is None:
        logger.error("target_probs is required")
        return {}
    if prices_dict is None:
        logger.error("prices_dict is required")
        return {}
    if current_positions is None:
        current_positions = load_positions()

    config = load_config_for_strategy(strategy_key)

    logger.info(f"=== Config Rebalance: {slug} (model={model_key}, strategy={strategy_key}) ===")

    summary = run_config_rebalance_cycle(
        slug=slug,
        model_key=model_key,
        capital=capital,
        mock_slippage=mock_slippage,
        target_probs=target_probs,
        prices_dict=prices_dict,
        token_ids_dict=token_ids_dict or {},
        config=config,
        dt_now=dt_now,
        current_positions=current_positions,
        temp_now=temp_now,
        max_so_far=max_so_far,
        rain_regime=rain_regime,
        model_std=model_std,
        recent_price_volatility=recent_price_volatility,
        hours_to_settlement=hours_to_settlement,
        nowcast_stale=nowcast_stale,
        data_missing=data_missing,
        drawdown_pct=drawdown_pct,
        probs_old=probs_old,
        probs_new=probs_new or target_probs,
    )

    entry_count = summary.get('total_entry_count', 0)
    exit_count = summary.get('total_exit_count', 0)
    blocked_count = summary.get('total_blocked_count', 0)
    triggers = summary.get('rebalance_triggers', [])
    logger.info(f"Decisions: {entry_count} entry, {exit_count} exit, {blocked_count} blocked")
    if triggers:
        logger.info(f"Rebalance triggers: {', '.join(triggers)}")

    target_positions = summary.get('target_positions', {})
    if target_positions:
        strategy_context = {
            'strategy_key': strategy_key or model_key,
            'selected_model': model_key,
            'entry_reason': 'config_entry_gate',
            'exit_reason': '',
            'edge_before_slippage': None,
            'edge_after_slippage': None,
            'rain_regime': rain_regime,
            'data_quality_flags': '',
            'reason_code': 'CONFIG_REBALANCE',
            'drawdown_pct': drawdown_pct,
            'hours_to_settlement': hours_to_settlement,
        }
        update_paper_positions(target_positions, model_key, slug, strategy_key or model_key,
                                prices_dict, strategy_context)

    summary['slug'] = slug
    summary['model_key'] = model_key
    return summary


if __name__ == "__main__":
    import pandas as pd

    mock_slug = "highest-temperature-in-hong-kong-on-june-6-2026"
    mock_model_key = "9d"

    mock_positions = {
        mock_model_key: {
            mock_slug: {
                "32°C": {"side": "NO", "quantity": 100.0, "entry_price": 0.60},
                "31°C": {"side": "YES", "quantity": 50.0, "entry_price": 0.30}
            }
        }
    }
    save_positions(mock_positions)

    current_yes_prices = {
        "32°C": 0.30,
        "31°C": 0.40
    }

    print("--- 測試 PnL 計算 (包含 BUY_NO) ---")
    model_pos = load_positions().get(mock_model_key, {})
    records, total_pnl, total_cost = calculate_pnl(model_pos, current_yes_prices)

    print(pd.DataFrame(records).to_string(index=False))
    print(f"總成本: ${total_cost}")
    print(f"總未實現損益: ${total_pnl}")

    print("\n--- 測試粉塵倉位過濾 (PM_MIN_QTY = 5.0) ---")
    target_pos_dust = {
        "33°C": {"side": "YES", "quantity": 3.5, "target_price": 0.20},
        "34°C": {"side": "YES", "quantity": 10.0, "target_price": 0.15}
    }
    update_paper_positions(target_pos_dust, mock_model_key, mock_slug, current_yes_prices)

    updated_pos = load_positions().get(mock_model_key, {}).get(mock_slug, {})
    print("更新後的持倉 (33°C 應不存在，34°C 應存在):")
    for b, data in updated_pos.items():
        print(f"  {b}: {data}")
