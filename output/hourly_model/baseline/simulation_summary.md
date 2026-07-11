# Simulation Summary

- **Backtest date**: 2026-07-10 20:37
- **Dates analysed**: 2026-07-10
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1141.54
- **Total return**: 14.15%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 6.66%

## Trades

- **Total trades**: 97
- **Total fees**: $36.64
- **Total slippage**: $3.16
- **YES buy trades**: 14
- **NO buy trades**: 39

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+141.54 | +14.15% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 2

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 100 |
| SKIP_MIN_SHARES | 70 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 0.3333333333333333, 'model_b': 0.3333333333333333, 'model_c': 0.3333333333333333}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05