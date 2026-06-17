# execution/kelly_stress_test.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from execution.kelly_betting import compute_multi_kelly_bets


def main():
    capital = 1000.0
    # 情境 1: 極度樂觀
    probs = {"31": 0.01, "32": 0.01, "33": 0.01, "34": 0.96, "35": 0.01}
    prices = {"31": 0.01, "32": 0.01, "33": 0.01, "34": 0.90, "35": 0.01}
    bets = compute_multi_kelly_bets(probs, prices, capital, total_max=0.5)
    total = sum(b['fraction'] for b in bets.values())
    print(f"情境1 總倉位: {total:.2%} (預期 ≤ 50%)")

    # 情境 2: 價格偏離
    probs = {"31": 0.1, "32": 0.2, "33": 0.4, "34": 0.3}
    prices = {"31": 0.05, "32": 0.35, "33": 0.35, "34": 0.25}
    bets = compute_multi_kelly_bets(probs, prices, capital, total_max=0.5)
    total = sum(b['fraction'] for b in bets.values())
    print(f"情境2 總倉位: {total:.2%}")

if __name__ == '__main__':
    main()