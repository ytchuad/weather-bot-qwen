from dataclasses import dataclass, field


@dataclass
class EnsembleParams:
    capital: float = 1000.0

    model_weights: dict = field(default_factory=lambda: {
        "model_a": 1/3,
        "model_b": 1/3,
        "model_c": 1/3,
    })

    edge_threshold: float = 0.05
    kelly_fraction: float = 0.25
    max_per_bucket_side: float = 0.10
    total_exposure_cap: float = 0.50

    min_price: float = 0.03
    max_price: float = 0.80
    min_shares: float = 5.0

    slippage_fixed: float = 0.001
    fee_constant: float = 0.05

    morning_start: float = 9.0
    risk_reduction_start: float = 14.0
    hard_flat_start: float = 15.0

    min_rebalance_interval_minutes: float = 10.0

    hold_behavior: str = "close_on_no_edge"
    exit_behavior: str = "normal"

    # When True, re-walk CLOB book with actual (not estimated) Kelly amount
    # and reject targets that can't be fully filled at the available depth.
    clob_depth_check: bool = False
