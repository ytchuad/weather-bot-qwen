# Simulation Summary

- **Backtest date**: 2026-07-11 18:35
- **Dates analysed**: 2026-07-01, 2026-07-02, 2026-07-03, 2026-07-05, 2026-07-06, 2026-07-07, 2026-07-08, 2026-07-09, 2026-07-10, 2026-07-11
- **Strategy**: Ensemble A/B/C (equal-weight)

## Capital & Return

- **Capital start**: $1000.00
- **Capital end**: $2419.84
- **Total return**: 141.98%
- **Sharpe ratio** (daily, annualised): 10.6377
- **Max drawdown**: 11.45%

## Trades

- **Total trades**: 534
- **Total fees**: $394.49
- **Total slippage**: $48.40
- **YES buy trades**: 78
- **NO buy trades**: 188

## Daily Performance

| Date | PnL | Return |
|------|-----|--------|
| 2026-07-01 | $+0.00 | +0.00% |
| 2026-07-02 | $-2.98 | -0.30% |
| 2026-07-03 | $+126.28 | +12.67% |
| 2026-07-05 | $+361.43 | +32.18% |
| 2026-07-06 | $+0.00 | +0.00% |
| 2026-07-07 | $+5.14 | +0.35% |
| 2026-07-08 | $+69.23 | +4.65% |
| 2026-07-09 | $-82.27 | -5.28% |
| 2026-07-10 | $+84.66 | +5.73% |
| 2026-07-11 | $+858.35 | +54.97% |

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
| SKIP_COOLDOWN | 309 |
| SKIP_MIN_SHARES | 89 |

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