# Simulation Summary

- **Backtest date**: 2026-07-11 18:06
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1443.01
- **Total return**: 44.30%
- **Sharpe ratio** (daily, annualised): 75.7399
- **Max drawdown**: 6.76%

## Trades

- **Total trades**: 229
- **Total fees**: $81.46
- **Total slippage**: $7.87
- **YES buy trades**: 33
- **NO buy trades**: 92

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+165.80 | +16.58% |
| 2026-07-11 | $+277.21 | +23.78% |

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
| SKIP_COOLDOWN | 280 |
| SKIP_MIN_SHARES | 143 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_c': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05