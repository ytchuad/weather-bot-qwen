# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2809.48
- **Total return**: 180.95%
- **Sharpe ratio** (daily, annualised): 27.3788
- **Max drawdown**: 15.41%

## Trades

- **Total trades**: 308
- **Total fees**: $146.58
- **Total slippage**: $14.81
- **YES buy trades**: 57
- **NO buy trades**: 113

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+1064.26 | +106.43% |
| 2026-07-11 | $+745.21 | +36.10% |

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
| SKIP_COOLDOWN | 535 |
| SKIP_MIN_SHARES | 131 |

## Parameters

- Capital: $1000.0
- Model weights: {'rain_nowcast': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05