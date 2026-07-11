# Simulation Summary

- **Backtest date**: 2026-07-11 18:00
- **Dates analysed**: 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1269.97
- **Total return**: 27.00%
- **Sharpe ratio** (daily, annualised): 0.0
- **Max drawdown**: 16.17%

## Trades

- **Total trades**: 141
- **Total fees**: $54.17
- **Total slippage**: $4.63
- **YES buy trades**: 25
- **NO buy trades**: 53

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-11 | $+269.97 | +27.00% |

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
| SKIP_COOLDOWN | 193 |
| SKIP_MIN_SHARES | 65 |

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