# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2733.98
- **Total return**: 173.40%
- **Sharpe ratio** (daily, annualised): 41.7885
- **Max drawdown**: 15.85%

## Trades

- **Total trades**: 245
- **Total fees**: $102.25
- **Total slippage**: $12.06
- **YES buy trades**: 10
- **NO buy trades**: 134

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+451.71 | +45.17% |
| 2026-07-11 | $+1282.27 | +88.33% |

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
| SKIP_COOLDOWN | 272 |
| SKIP_MIN_SHARES | 185 |

## Parameters

- Capital: $1000.0
- Model weights: {'aws': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05