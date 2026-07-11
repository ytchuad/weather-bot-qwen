# Simulation Summary

- **Backtest date**: 2026-07-10 20:27
- **Dates analysed**: 2026-07-10
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1233.09
- **Total return**: 23.31%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 17.09%

## Trades

- **Total trades**: 104
- **Total fees**: $42.27
- **Total slippage**: $4.08
- **YES buy trades**: 15
- **NO buy trades**: 44

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+233.09 | +23.31% |

## Risk-Reduction Mode Summary

- Cycles in risk-reduction mode (14:00-15:00): 0

## Trading Window Summary

- **09:00-14:00**: RISK_SEEKING — full Kelly rebalancing
- **14:00-15:00**: RISK_REDUCTION — exit only, no new risk
- **After 15:00**: HARD_FLAT_TARGET — exit all positions

## Breakout Handling Summary

- Deterministic events triggered: 2

## Skipped / Rejected Trades by Reason

| Reason | Count |
|--------|-------|
| SKIP_COOLDOWN | 106 |
| SKIP_MIN_SHARES | 77 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_2a1': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05