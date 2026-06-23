# execution/clob_slippage.py
import math
import requests
import json
import logging
import time
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_API_URL = "https://clob.polymarket.com/book"

# ==========================================
# 1. 獲取 Token IDs
# ==========================================
def get_token_ids_from_slug(slug: str) -> Dict[str, Tuple[str, str]]:
    """
    從 Gamma API 獲取市場的 YES/NO Token IDs。
    Returns: {"28°C": ("yes_token_id", "no_token_id"), ...}
    """
    try:
        resp = requests.get(f"{GAMMA_API_URL}?slug={slug}", timeout=10)
        resp.raise_for_status()
        events = resp.json()
        if not events: return {}
        
        token_map = {}
        markets = events[0].get('markets', [])
        for m in markets:
            title = m.get('groupItemTitle', '')
            import re
            match = re.search(r'(\d+)', title)
            if not match: continue
            label = f"{match.group(1)}°C"
            
            clob_ids_str = m.get('clobTokenIds', '[]')
            try:
                clob_ids = json.loads(clob_ids_str)
                if len(clob_ids) >= 2:
                    token_map[label] = (clob_ids[0], clob_ids[1])
            except json.JSONDecodeError:
                continue
        return token_map
    except Exception as e:
        logger.error(f"獲取 Token IDs 失敗 ({slug}): {e}")
        return {}

# ==========================================
# 2. 獲取訂單簿
# ==========================================
def get_order_book(token_id: str, mock_mode: bool = False, mock_base_price: float = 0.5) -> dict:
    """
    獲取 Polymarket CLOB 訂單簿。
    """
    if mock_mode:
        return _generate_mock_order_book(mock_base_price)

    try:
        resp = requests.get(f"{CLOB_API_URL}?token_id={token_id}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        def parse_side(side_data):
            parsed = []
            for level in side_data:
                parsed.append({'price': float(level['price']), 'size': float(level['size'])})
            return parsed
            
        asks = parse_side(data.get('asks', []))
        bids = parse_side(data.get('bids', []))
        
        asks.sort(key=lambda x: x['price'])
        bids.sort(key=lambda x: x['price'], reverse=True)
        
        return {'bids': bids, 'asks': asks}
    except Exception as e:
        logger.warning(f"獲取訂單簿失敗 (Token: {token_id[:8]}...): {e}")
        return {'bids': [], 'asks': []}

def _generate_mock_order_book(base_price: float) -> dict:
    """生成模擬訂單簿 (用於離線測試)"""
    asks = []
    current_price = base_price
    for i in range(5):
        asks.append({'price': round(current_price, 4), 'size': 100.0 * (i + 1)})
        current_price += 0.02 
        
    bids = []
    current_price = base_price - 0.01
    for i in range(5):
        bids.append({'price': round(current_price, 4), 'size': 100.0 * (i + 1)})
        current_price -= 0.02
        
    return {'bids': bids, 'asks': asks}

# ==========================================
# 3. 按「合約數量」估算執行價格
# ==========================================
def estimate_execution_price(order_type: str, quantity: float, order_book: dict) -> Tuple[Optional[float], float, float, bool]:
    """
    給定目標合約數量，計算加權平均成交價與總成本。
    """
    side = order_book.get('asks', []) if order_type in ['BUY_YES', 'BUY_NO'] else order_book.get('bids', [])
    
    if not side:
        return None, 0.0, 0.0, True
        
    remaining_qty = quantity
    total_cost = 0.0
    filled_qty = 0.0
    warning = False
    
    for level in side:
        p = level['price']
        s = level['size']
        
        if remaining_qty <= s:
            total_cost += remaining_qty * p
            filled_qty += remaining_qty
            remaining_qty = 0
            break
        else:
            total_cost += s * p
            filled_qty += s
            remaining_qty -= s
            
    if remaining_qty > 0:
        warning = True 
        
    avg_price = total_cost / filled_qty if filled_qty > 0 else None
    return avg_price, total_cost, filled_qty, warning

# ==========================================
# 4. 按「USDC 預算」套用滑點並調整下注 (已修正)
# ==========================================
def apply_slippage_to_bets(
    bets_dict: dict, 
    token_ids_dict: dict, 
    prices_dict: dict = None,  
    mock_mode: bool = False
) -> dict:
    """
    根據訂單簿深度，將 Kelly 輸出的 USDC 預算轉換為真實的合約數量與調整後的 Edge。
    """
    adjusted_bets = {}
    
    for label, bet in bets_dict.items():
        is_buy_yes = (bet['action'] == 'BUY_YES')
        
        # [關鍵修復 1] 嚴格區分 YES/NO 的基準價格
        yes_price = prices_dict.get(label, 0.5) if prices_dict else 0.5
        base_price = yes_price if is_buy_yes else (1.0 - yes_price)
        
        # [關鍵修復 2] mock_mode 下不需要真實 token_id
        if mock_mode:
            target_token_id = "mock_token"
        else:
            token_ids = token_ids_dict.get(label) if token_ids_dict else None
            if not token_ids:
                logger.warning(f"Slippage Sim: Missing token_ids for {label}, skipping.")
                continue
            yes_id, no_id = token_ids
            target_token_id = yes_id if is_buy_yes else no_id
        
        ob = get_order_book(target_token_id, mock_mode=mock_mode, mock_base_price=base_price)
        asks = ob.get('asks', [])
        
        if not asks:
            logger.warning(f"Slippage Sim: No asks in order book for {label}, skipping.")
            continue
            
        best_ask = asks[0]['price']
        remaining_budget = bet['amount']
        acquired_contracts = 0.0
        total_cost_actual = 0.0
        
        for level in asks:
            p = level['price']
            s = level['size']
            max_cost_here = p * s
            
            if remaining_budget >= max_cost_here:
                acquired_contracts += s
                total_cost_actual += max_cost_here
                remaining_budget -= max_cost_here
            else:
                qty_here = remaining_budget / p if p > 0 else 0
                acquired_contracts += qty_here
                total_cost_actual += remaining_budget
                remaining_budget = 0
                break
                
        actual_cost = bet['amount'] - remaining_budget
        filled = (remaining_budget < 0.01) 
        
        if acquired_contracts > 0:
            avg_fill_price = total_cost_actual / acquired_contracts
            avg_fill_price = max(0.001, min(0.999, avg_fill_price))
            slippage_pct = ((avg_fill_price - best_ask) / best_ask) * 100 if best_ask > 0 else 0.0
            
            adjusted_bets[label] = {
                **bet,
                'adjusted_quantity': round(acquired_contracts, 2),
                'actual_cost': round(actual_cost, 2),
                'avg_fill_price': round(avg_fill_price, 4),
                'best_ask': best_ask,
                'slippage_pct': round(slippage_pct, 2),
                'filled': filled
            }
            logger.info(f"Slippage Sim OK: {label} | Qty: {acquired_contracts:.2f} | Price: {avg_fill_price:.4f}")
        else:
            logger.warning(f"Slippage Sim: acquired 0 contracts for {label}. Budget: {bet['amount']}, Best Ask: {best_ask}")
            
    return adjusted_bets

# ==========================================
# 測試區塊 (已修正)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("--- 1. 測試 Mock 訂單簿生成 ---")
    mock_ob = get_order_book("dummy_token", mock_mode=True, mock_base_price=0.30)
    print(f"Best Ask: {mock_ob['asks'][0]}")
    
    print("\n--- 2. 測試按「合約數量」估算 (estimate_execution_price) ---")
    avg_p, cost, filled, warn = estimate_execution_price('BUY_YES', 150.0, mock_ob)
    print(f"目標買入 150 張: 均價={avg_p:.4f}, 總成本=${cost:.2f}, 成交={filled}張, 深度不足警告={warn}")
    
    print("\n--- 3. 測試按「USDC 預算」套用滑點 (apply_slippage_to_bets) ---")
    
    # [修正] 移除 bet 中的 target_price，保持與 Kelly 輸出一致
    mock_kelly_bets = {
        "32°C": {'action': 'BUY_YES', 'amount': 50.0, 'fraction': 0.05, 'expected_edge': 0.10},
        "33°C": {'action': 'BUY_NO', 'amount': 200.0, 'fraction': 0.20, 'expected_edge': 0.15}
    }
    
    # [新增] 獨立傳入市場價格字典
    mock_prices_dict = {
        "32°C": 0.30,
        "33°C": 0.40
    }
    
    mock_token_map = {
        "32°C": ("token_yes_32", "token_no_32"),
        "33°C": ("token_yes_33", "token_no_33")
    }
    
    # [修正] 呼叫時傳入 prices_dict
    adjusted = apply_slippage_to_bets(
        mock_kelly_bets, 
        mock_token_map, 
        prices_dict=mock_prices_dict, 
        mock_mode=True
    )
    
    for label, data in adjusted.items():
        print(f"\n[{label}] {data['action']}")
        print(f"  預算: ${data['amount']} -> 實際花費: ${data['actual_cost']}")
        print(f"  獲得合約: {data['adjusted_quantity']} 張")
        print(f"  Best Ask: {data['best_ask']} -> 真實均價: {data['avg_fill_price']} (滑點: {data['slippage_pct']}%)")
        print(f"  完全成交: {data['filled']}")