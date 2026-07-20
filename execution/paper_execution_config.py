"""Explicit rollout configuration for paper execution modes."""

from __future__ import annotations

import os
from pathlib import Path

PAPER_EXECUTION_MODES = (
    "legacy_gamma_mock",
    "clob_depth_shadow",
    "clob_depth",
)
PARTIAL_FILL_POLICIES = (
    "fail_closed",
    "accept_partial",
    "reduce_to_available",
)

# Shadow is the safe rollout default: the existing legacy simulator may keep
# the paper account moving while the CLOB result is observed without a second
# position mutation. Promotion to ``clob_depth`` remains explicit.
DEFAULT_PAPER_EXECUTION_MODE = "clob_depth_shadow"
DEFAULT_PARTIAL_FILL_POLICY = "fail_closed"
DEFAULT_MAX_BOOK_AGE_SECONDS = 60.0


def get_paper_execution_mode(value: str | None = None) -> str:
    """Return one explicit paper execution mode or fail closed."""
    raw = value if value is not None else os.getenv("PAPER_EXECUTION_MODE")
    mode = (raw or DEFAULT_PAPER_EXECUTION_MODE).strip().lower()
    if mode not in PAPER_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported PAPER_EXECUTION_MODE={mode!r}; "
            f"expected one of {PAPER_EXECUTION_MODES}"
        )
    return mode


def get_partial_fill_policy(value: str | None = None) -> str:
    """Return the configured partial-fill policy or fail closed."""
    raw = value if value is not None else os.getenv("PAPER_PARTIAL_FILL_POLICY")
    policy = (raw or DEFAULT_PARTIAL_FILL_POLICY).strip().lower()
    if policy not in PARTIAL_FILL_POLICIES:
        raise ValueError(
            f"Unsupported PAPER_PARTIAL_FILL_POLICY={policy!r}; "
            f"expected one of {PARTIAL_FILL_POLICIES}"
        )
    return policy


def get_max_book_age_seconds(value: float | None = None) -> float:
    """Return the maximum allowed decision-time book age."""
    if value is not None:
        age = float(value)
    else:
        raw = os.getenv("PAPER_CLOB_MAX_BOOK_AGE_SECONDS")
        age = float(raw) if raw else DEFAULT_MAX_BOOK_AGE_SECONDS
    if age <= 0:
        raise ValueError("PAPER_CLOB_MAX_BOOK_AGE_SECONDS must be positive")
    return age


def get_shadow_log_path() -> Path:
    """Return the append-only shadow comparison path."""
    return Path(os.getenv("PAPER_CLOB_SHADOW_LOG", "data/clob_execution_shadow.jsonl"))
