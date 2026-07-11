# Simulation Summary

- **Backtest date**: 2026-07-11 18:06
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1362.33
- **Total return**: 36.23%
- **Sharpe ratio** (daily, annualised): 38.4478
- **Max drawdown**: 5.97%

## Trades

- **Total trades**: 196
- **Total fees**: $58.77
- **Total slippage**: $5.01
- **YES buy trades**: 6
- **NO buy trades**: 103

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+109.42 | +10.94% |
| 2026-07-11 | $+252.91 | +22.80% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 5

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 179 |
| SKIP_MIN_SHARES | 202 |

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