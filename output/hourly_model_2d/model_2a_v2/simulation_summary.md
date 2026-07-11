# Simulation Summary

- **Backtest date**: 2026-07-11 18:06
- **Dates analysed**: 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1325.82
- **Total return**: 32.58%
- **Sharpe ratio** (daily, annualised): 55.3914
- **Max drawdown**: 13.88%

## Trades

- **Total trades**: 242
- **Total fees**: $91.05
- **Total slippage**: $8.29
- **YES buy trades**: 33
- **NO buy trades**: 91

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-10 | $+189.12 | +18.91% |
| 2026-07-11 | $+136.70 | +11.50% |

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
| SKIP_MIN_SHARES | 140 |

## Parameters

- Capital: $1000.0
- Model weights: {'model_2a_v2': 1.0}
- Edge threshold: 0.08
- Kelly fraction: 0.25
- Max per bucket side: 0.1
- Total exposure cap: 0.5
- Price band: [0.03, 0.8]
- Min shares: 5.0
- Slippage fixed: 0.001
- Fee constant: 0.05