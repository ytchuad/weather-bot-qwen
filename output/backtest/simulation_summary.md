# Simulation Summary

- **Backtest date**: 2026-07-05 17:19
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1452.30
- **Total return**: 45.23%
- **Sharpe ratio** (daily, annualised): 14.3676
- **Max drawdown**: 10.65%

## Trades

- **Total trades**: 141
- **Total fees**: $83.85
- **Total slippage**: $18.58
- **YES buy trades**: 32
- **NO buy trades**: 35

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $-2.98 | -0.30% |
| 2026-07-03 | $+128.98 | +12.94% |
| 2026-07-05 | $+326.30 | +28.98% |

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
| SKIP_COOLDOWN | 416 |
| SKIP_MIN_SHARES | 80 |

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