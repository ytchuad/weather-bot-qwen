# Simulation Summary

- **Backtest date**: 2026-07-11 18:07
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1423.80
- **Total return**: 42.38%
- **Sharpe ratio** (daily, annualised): 54.2284
- **Max drawdown**: 6.73%

## Trades

- **Total trades**: 224
- **Total fees**: $79.51
- **Total slippage**: $7.64
- **YES buy trades**: 32
- **NO buy trades**: 90

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+145.83 | +14.58% |
| 2026-07-11 | $+277.97 | +24.26% |

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
| SKIP_COOLDOWN | 278 |
| SKIP_MIN_SHARES | 149 |

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