# Simulation Summary

- **Backtest date**: 2026-07-08 21:10
- **Dates analysed**: 2026-07-08
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $962.49
- **Total return**: -3.75%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 6.23%

## Trades

- **Total trades**: 52
- **Total fees**: $28.07
- **Total slippage**: $5.38
- **YES buy trades**: 18
- **NO buy trades**: 10

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-08 | $-37.51 | -3.75% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 0

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 148 |
| SKIP_MIN_SHARES | 58 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 0.3333333333333333, 'model_b': 0.3333333333333333, 'model_c': 0.3333333333333333}
- Edge threshold: 0.05
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05