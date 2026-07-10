# Simulation Summary

- **Backtest date**: 2026-07-07 21:40
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05, 2026-07-06, 2026-07-07
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1261.24
- **Total return**: 26.12%
- **Sharpe ratio** (daily, annualised): 11.0553
- **Max drawdown**: 11.92%

## Trades

- **Total trades**: 146
- **Total fees**: $65.96
- **Total slippage**: $13.57
- **YES buy trades**: 33
- **NO buy trades**: 35

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $-15.43 | -1.54% |
| 2026-07-03 | $+92.33 | +9.38% |
| 2026-07-05 | $+176.39 | +16.38% |
| 2026-07-06 | $+0.00 | +0.00% |
| 2026-07-07 | $+7.95 | +0.63% |

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
| SKIP_COOLDOWN | 433 |
| SKIP_MIN_SHARES | 109 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 0.3333333333333333, 'model_b': 0.3333333333333333, 'model_c': 0.3333333333333333}
- Edge threshold: 0.05
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05