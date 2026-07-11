# Simulation Summary

- **Backtest date**: 2026-07-11 18:00
- **Dates analysed**: 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1301.53
- **Total return**: 30.15%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 8.13%

## Trades

- **Total trades**: 137
- **Total fees**: $44.86
- **Total slippage**: $4.50
- **YES buy trades**: 14
- **NO buy trades**: 54

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-11 | $+301.53 | +30.15% |

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
| SKIP_COOLDOWN | 202 |
| SKIP_MIN_SHARES | 84 |

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