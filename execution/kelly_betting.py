# execution/kelly_betting.py
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

def compute_multi_kelly_bets(
    probs_dict: dict, 
    prices_dict: dict, 
    capital: float, 
    max_per_bucket: float = 0.15, 
    total_max: float = 0.50
) -> dict:
    """
    基於互斥事件 (Mutually Exclusive) 的嚴格 Kelly 求解器。
    支援 BUY_YES 與 BUY_NO，並精確計算多結果組合下的期望對數財富。
    風險控制完全由 total_max 與 max_per_bucket 約束，不進行人為縮放。
    """
    # 1. 構建候選下注列表 (包含 YES 與 NO)
    candidates = []
    all_labels = sorted(list(probs_dict.keys()))
    
    for label in all_labels:
        p_yes = probs_dict[label]
        price_yes = np.clip(prices_dict.get(label, 0.5), 0.01, 0.99)
        
        edge_yes = p_yes - price_yes
        edge_no = price_yes - p_yes 
        
        # 將正 Edge 的選項加入候選 (同一個桶 YES 和 NO 互斥，只會加入一個)
        if edge_yes > 0:
            candidates.append({
                'type': 'YES', 'label': label,
                'prob': p_yes, 'price': price_yes, 'edge': edge_yes
            })
        elif edge_no > 0:
            candidates.append({
                'type': 'NO', 'label': label,
                'prob': 1.0 - p_yes, 'price': 1.0 - price_yes, 'edge': edge_no
            })
            
    n = len(candidates)
    if n == 0:
        return {}

    # 2. 定義目標函數 (基於原子結果的精確財富計算)
    outcomes = all_labels 
    outcome_probs = np.array([probs_dict[x] for x in outcomes])
    
    def neg_expected_log_wealth(f):
        total_f = np.sum(f)
        expected_log_w = 0.0
        
        for i, x in enumerate(outcomes):
            p_x = outcome_probs[i]
            if p_x <= 1e-9:
                continue
                
            payout = 0.0
            for k, cand in enumerate(candidates):
                if (cand['type'] == 'YES' and cand['label'] == x) or \
                   (cand['type'] == 'NO' and cand['label'] != x):
                    payout += f[k] / cand['price']
                    
            w_x = 1.0 - total_f + payout
            w_x = max(w_x, 1e-9) 
            expected_log_w += p_x * np.log(w_x)
            
        return -expected_log_w

    # 3. 約束與邊界
    bounds = [(0.0, max_per_bucket) for _ in range(n)]
    constraints = [{'type': 'ineq', 'fun': lambda f: total_max - np.sum(f)}]
    f0 = np.full(n, 0.01)

    # 4. 執行數值優化
    try:
        result = minimize(
            neg_expected_log_wealth, 
            f0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        optimal_f = result.x
    except Exception as e:
        logger.error(f"Kelly 優化失敗: {e}")
        optimal_f = np.zeros(n)

    # 5. 格式化輸出 (直接使用優化器結果，無額外縮放)
    bets = {}
    for k, cand in enumerate(candidates):
        f_k = optimal_f[k]
        if f_k > 1e-4:
            action = 'BUY_YES' if cand['type'] == 'YES' else 'BUY_NO'
            
            bets[cand['label']] = {
                'action': action,
                'amount': round(f_k * capital, 2),
                'fraction': f_k,
                'expected_edge': round(cand['edge'], 4)
            }
            
    return bets


def compute_bets_simple(
    probs_dict: dict, 
    prices_dict: dict, 
    capital: float, 
    edge_threshold: float = 0.02,
    kelly_fraction: float = 0.5,
    max_per_bucket: float = 0.10
) -> dict:
    """簡化版獨立 Kelly (未考慮互斥性，容易過度下注)"""
    bets = {}
    for label, p_yes in probs_dict.items():
        price_yes = np.clip(prices_dict.get(label, 0.5), 0.01, 0.99)
        edge_yes = p_yes - price_yes
        
        if edge_yes > edge_threshold:
            odds = (1.0 / price_yes) - 1.0
            f = (edge_yes / odds) * kelly_fraction
            f = min(f, max_per_bucket)
            bets[label] = {'action': 'BUY_YES', 'amount': round(f * capital, 2), 'fraction': f, 'expected_edge': round(edge_yes, 4)}
        else:
            edge_no = price_yes - p_yes
            if edge_no > edge_threshold:
                price_no = 1.0 - price_yes
                odds = (1.0 / price_no) - 1.0
                f = (edge_no / odds) * kelly_fraction
                f = min(f, max_per_bucket)
                bets[label] = {'action': 'BUY_NO', 'amount': round(f * capital, 2), 'fraction': f, 'expected_edge': round(edge_no, 4)}
    return bets

# === 測試區塊 ===
if __name__ == "__main__":
    # 模擬數據：32 和 34 被低估 (YES 有 Edge)，33 被嚴重高估 (NO 有 Edge)
    probs = {"31": 0.1, "32": 0.5, "33": 0.1, "34": 0.3}
    prices = {"31": 0.05, "32": 0.30, "33": 0.40, "34": 0.25}
    capital = 1000.0
    
    print("--- 簡化版獨立 Kelly (忽略互斥性，總倉位容易超標) ---")
    simple_bets = compute_bets_simple(probs, prices, capital)
    total_f_simple = sum(b['fraction'] for b in simple_bets.values())
    print(f"總倉位比例: {total_f_simple:.2%}")
    for k, v in simple_bets.items():
        print(f"{k}: {v}")
        
    print("\n--- 嚴格互斥 Kelly (支援 YES/NO 聯合優化, total_max=0.5) ---")
    multi_bets = compute_multi_kelly_bets(probs, prices, capital, total_max=0.5)
    total_f_multi = sum(b['fraction'] for b in multi_bets.values())
    print(f"總倉位比例: {total_f_multi:.2%}")
    for k, v in multi_bets.items():
        print(f"{k}: {v}")
        
    # 驗證資金分配等式
    print("\n--- 驗證資金分配等式 ---")
    all_match = True
    for k, v in multi_bets.items():
        expected_amount = v['fraction'] * capital
        if abs(v['amount'] - expected_amount) > 0.02:
            print(f"❌ {k} 資金不匹配: {v['amount']} != {expected_amount}")
            all_match = False
            
    if all_match:
        print("✅ 驗證通過: amount 確實等於 fraction * capital，無額外人为縮放。")