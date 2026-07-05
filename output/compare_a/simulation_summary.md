# Simulation Summary

- **Backtest date**: 2026-07-05 19:08
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $1484.74
- **Total return**: 48.47%
- **Sharpe ratio** (daily, annualised): 13.9302
- **Max drawdown**: 10.65%

## Trades

- **Total trades**: 135
- **Total fees**: $78.08
- **Total slippage**: $17.49
- **YES buy trades**: 30
- **NO buy trades**: 32

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $-2.98 | -0.30% |
| 2026-07-03 | $+126.28 | +12.67% |
| 2026-07-05 | $+361.43 | +32.18% |

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
| SKIP_COOLDOWN | 125 |
| SKIP_MIN_SHARES | 34 |

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