# Simulation Summary

- **Backtest date**: 2026-07-11 18:35
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05, 2026-07-06, 2026-07-07, 2026-07-08, 2026-07-09, 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $5066.43
- **Total return**: 406.64%
- **Sharpe ratio** (daily, annualised): 12.6282
- **Max drawdown**: 23.90%

## Trades

- **Total trades**: 1326
- **Total fees**: $1285.15
- **Total slippage**: $141.87
- **YES buy trades**: 242
- **NO buy trades**: 420

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $+5.02 | +0.50% |
| 2026-07-03 | $+131.56 | +13.09% |
| 2026-07-05 | $+346.65 | +30.50% |
| 2026-07-06 | $+0.00 | +0.00% |
| 2026-07-07 | $+53.56 | +3.61% |
| 2026-07-08 | $+194.87 | +12.68% |
| 2026-07-09 | $+108.26 | +6.25% |
| 2026-07-10 | $+688.78 | +37.44% |
| 2026-07-11 | $+2537.71 | +100.36% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 4

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_MIN_SHARES | 794 |

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