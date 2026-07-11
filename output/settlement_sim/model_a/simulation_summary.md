# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2660.71
- **Total return**: 166.07%
- **Sharpe ratio** (daily, annualised): 23.5749
- **Max drawdown**: 15.56%

## Trades

- **Total trades**: 309
- **Total fees**: $142.28
- **Total slippage**: $13.50
- **YES buy trades**: 56
- **NO buy trades**: 116

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+1064.34 | +106.43% |
| 2026-07-11 | $+596.37 | +28.89% |

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
| SKIP_COOLDOWN | 490 |
| SKIP_MIN_SHARES | 130 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05