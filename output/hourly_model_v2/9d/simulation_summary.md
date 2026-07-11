# Simulation Summary

- **Backtest date**: 2026-07-11 18:00
- **Dates analysed**: 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1192.78
- **Total return**: 19.28%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 7.40%

## Trades

- **Total trades**: 118
- **Total fees**: $37.05
- **Total slippage**: $3.27
- **YES buy trades**: 6
- **NO buy trades**: 58

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-11 | $+192.78 | +19.28% |

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
| SKIP_COOLDOWN | 99 |
| SKIP_MIN_SHARES | 86 |

## Parameters

- Capital: $1000.0
- Model weights: {'9d': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05