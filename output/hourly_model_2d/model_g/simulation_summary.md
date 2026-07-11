# Simulation Summary

- **Backtest date**: 2026-07-11 18:06
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1392.75
- **Total return**: 39.28%
- **Sharpe ratio** (daily, annualised): 58.4061
- **Max drawdown**: 11.96%

## Trades

- **Total trades**: 236
- **Total fees**: $86.31
- **Total slippage**: $7.99
- **YES buy trades**: 33
- **NO buy trades**: 96

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+139.05 | +13.91% |
| 2026-07-11 | $+253.70 | +22.27% |

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
| SKIP_COOLDOWN | 314 |
| SKIP_MIN_SHARES | 145 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_g': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05