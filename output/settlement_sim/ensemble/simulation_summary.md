# Simulation Summary

- **Backtest date**: 2026-07-11 18:22
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2693.34
- **Total return**: 169.33%
- **Sharpe ratio** (daily, annualised): 21.4725
- **Max drawdown**: 15.54%

## Trades

- **Total trades**: 306
- **Total fees**: $147.72
- **Total slippage**: $13.99
- **YES buy trades**: 57
- **NO buy trades**: 115

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+1138.85 | +113.88% |
| 2026-07-11 | $+554.49 | +25.92% |

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
| SKIP_COOLDOWN | 485 |
| SKIP_MIN_SHARES | 136 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_a': 0.3333333333333333, 'model_b': 0.3333333333333333, 'model_c': 0.3333333333333333}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05