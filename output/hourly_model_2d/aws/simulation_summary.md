# Simulation Summary

- **Backtest date**: 2026-07-11 18:06
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1230.44
- **Total return**: 23.04%
- **Sharpe ratio** (daily, annualised): 1269.6086
- **Max drawdown**: 12.49%

## Trades

- **Total trades**: 200
- **Total fees**: $62.24
- **Total slippage**: $5.51
- **YES buy trades**: 12
- **NO buy trades**: 98

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+108.09 | +10.81% |
| 2026-07-11 | $+122.35 | +11.04% |

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
| SKIP_MIN_SHARES | 190 |

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