# Simulation Summary

- **Backtest date**: 2026-07-05 16:41
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-04, 2026-07-05
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1982.82
- **Total return**: 98.28%
- **Sharpe ratio** (daily, annualised): 16.6375
- **Max drawdown**: 10.69%

## Trades

- **Total trades**: 211
- **Total fees**: $117.88
- **Total slippage**: $24.93
- **YES buy trades**: 42
- **NO buy trades**: 53

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $+4.23 | +0.42% |
| 2026-07-03 | $+136.36 | +13.58% |
| 2026-07-04 | $+501.08 | +43.93% |
| 2026-07-05 | $+341.16 | +20.78% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 3

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 652 |
| SKIP_MIN_SHARES | 150 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 0.3333333333333333, 'model_b': 0.3333333333333333, 'model_c': 0.3333333333333333}
- Edge threshold: 0.05
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.88]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05