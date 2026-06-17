# execution/gates/pipeline.py
"""GatePipeline — runs a sequence of gates, short-circuits on block."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from execution.gates.types import GateInput, GateOutput, GateFn, GateConfig

logger = logging.getLogger(__name__)


@dataclass
class GatePipeline:
    """Ordered sequence of gates with their configs.

    On `evaluate()`:
    - Runs each gate in order.
    - For entry/exit pipelines: short-circuits on first `passed=False`.
    - For sizing pipelines: runs ALL gates and returns the product of
      all multipliers (so each sizing gate contributes independently).
    - For rebalance pipelines: short-circuits on first `passed=True`
      (any trigger is sufficient).

    Attributes:
        gates: list of (name, gate_fn, config) triples.
        mode: "entry" (short-circuit on fail), "sizing" (run all, product),
              "rebalance" (short-circuit on pass), or "exit" (run all,
              product of multipliers).
    """

    gates: list[tuple[str, GateFn, GateConfig]]
    mode: str = "entry"  # entry | exit | sizing | rebalance

    def evaluate(self, inp: GateInput) -> tuple[bool, list[GateOutput]]:
        """Run all gates and return (overall_passed, list_of_outputs).

        Entry mode: overall_passed = True only if ALL gates pass.
        Exit mode: overall_passed = True if position should be HELD
                   (i.e. no exit triggered). multiplier is the product.
        Sizing mode: always returns True, multiplier is the product.
        Rebalance mode: overall_passed = True if ANY gate triggers.
        """
        results: list[GateOutput] = []

        if self.mode == "rebalance":
            return self._evaluate_rebalance(inp)

        for name, gate_fn, config in self.gates:
            try:
                out = gate_fn(inp, config)
            except Exception as exc:
                logger.warning("gate %s raised %s", name, exc)
                out = GateOutput(passed=False, reason_code="GATE_ERROR",
                                 detail=str(exc), multiplier=0.0)
            out.metadata["_gate_name"] = name
            results.append(out)

            if self.mode == "entry" and not out.passed:
                return False, results

        if self.mode == "sizing":
            # Always "pass" — sizing gates contribute multipliers only.
            return True, results

        if self.mode == "exit":
            # Exit pipeline: run all, overall = hold if multiplier > 0.
            # multiplier product <= 0 means exit.
            product = 1.0
            for r in results:
                product *= r.multiplier
            # passed=True means HOLD (no exit triggered)
            return product > 0.0, results

        # entry mode: all passed
        return True, results

    def _evaluate_rebalance(self, inp: GateInput) -> tuple[bool, list[GateOutput]]:
        """Rebalance mode: short-circuit on first trigger (passed=True)."""
        results: list[GateOutput] = []
        for name, gate_fn, config in self.gates:
            try:
                out = gate_fn(inp, config)
            except Exception as exc:
                logger.warning("rebalance gate %s raised %s", name, exc)
                continue
            out.metadata["_gate_name"] = name
            results.append(out)
            if out.passed:
                # First trigger is enough.
                return True, results
        return False, results

    @property
    def combined_multiplier(self) -> float:
        """Product of all gate multipliers (call after evaluate)."""
        # This is a convenience — callers typically compute it from results.
        raise NotImplementedError("Use the results from evaluate() instead.")


def product_of_multipliers(results: list[GateOutput]) -> float:
    """Multiply all gate output multipliers together."""
    m = 1.0
    for r in results:
        m *= r.multiplier
        if m <= 0.0:
            return 0.0
    return m
