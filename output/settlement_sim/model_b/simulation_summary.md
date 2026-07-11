# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2732.41
- **Total return**: 173.24%
- **Sharpe ratio** (daily, annualised): 21.36
- **Max drawdown**: 15.53%

## Trades

- **Total trades**: 303
- **Total fees**: $150.37
- **Total slippage**: $14.41
- **YES buy trades**: 57
- **NO buy trades**: 110

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+1164.73 | +116.47% |
| 2026-07-11 | $+567.67 | +26.22% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 8

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 496 |
| SKIP_MIN_SHARES | 141 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_b': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05