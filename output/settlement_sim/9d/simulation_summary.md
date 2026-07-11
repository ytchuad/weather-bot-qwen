# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2673.62
- **Total return**: 167.36%
- **Sharpe ratio** (daily, annualised): 45.0409
- **Max drawdown**: 15.94%

## Trades

- **Total trades**: 243
- **Total fees**: $87.32
- **Total slippage**: $11.02
- **YES buy trades**: 5
- **NO buy trades**: 137

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+452.65 | +45.27% |
| 2026-07-11 | $+1220.97 | +84.05% |

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
| SKIP_COOLDOWN | 277 |
| SKIP_MIN_SHARES | 191 |

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