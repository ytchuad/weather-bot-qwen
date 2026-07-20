"""Run the read-only Layer A replay smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer_a.replay import replay_layer_a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Layer A export .zip or extracted directory")
    parser.add_argument("--strategy-a-threshold", type=float, default=0.03)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--shares", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(
        replay_layer_a(
            args.input,
            strategy_a_threshold=args.strategy_a_threshold,
            kelly_fraction=args.kelly_fraction,
            requested_shares=args.shares,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
