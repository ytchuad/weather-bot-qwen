# Simulation Summary

- **Backtest date**: 2026-07-11 18:35
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05, 2026-07-06, 2026-07-07, 2026-07-08, 2026-07-09, 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1828.78
- **Total return**: 82.88%
- **Sharpe ratio** (daily, annualised): 17.8056
- **Max drawdown**: 11.92%

## Trades

- **Total trades**: 474
- **Total fees**: $219.98
- **Total slippage**: $26.73
- **YES buy trades**: 88
- **NO buy trades**: 151

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $-15.43 | -1.54% |
| 2026-07-03 | $+92.33 | +9.38% |
| 2026-07-05 | $+176.39 | +16.38% |
| 2026-07-06 | $+0.00 | +0.00% |
| 2026-07-07 | $+40.68 | +3.25% |
| 2026-07-08 | $+78.44 | +6.06% |
| 2026-07-09 | $+31.00 | +2.26% |
| 2026-07-10 | $+150.34 | +10.71% |
| 2026-07-11 | $+275.03 | +17.70% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 12

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 925 |
| SKIP_MIN_SHARES | 373 |

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