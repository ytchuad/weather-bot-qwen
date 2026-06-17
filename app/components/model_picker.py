# app/components/model_picker.py
"""Pinned model-card strip with +Add popover for model selection."""

from __future__ import annotations

from typing import List

import streamlit as st

from ..config import MODEL_LABELS, COLORS

_DEFAULT_PIN = ["9d", "aws", "baseline"]
_PIN_KEY = "app.model_pin"
_MAX_PIN = 6


def _best_intraday(all_results: dict) -> str:
    """Pick the intraday model with largest absolute edge today."""
    intra_keys = [k for k in all_results if k not in ("9d", "aws")]
    if not intra_keys:
        return "rain_nowcast"
    best_k, best_edge = intra_keys[0], -1.0
    for mk in intra_keys:
        probs = all_results.get(mk, {}).get("probs", {})
        total = sum(abs(p - 0.5) for p in probs.values())
        if total > best_edge:
            best_edge, best_k = total, mk
    return best_k


def _get_pinned(all_results: dict) -> list[str]:
    pinned = st.session_state.get(_PIN_KEY)
    if pinned and isinstance(pinned, list) and len(pinned) > 0:
        return pinned[:]
    return _DEFAULT_PIN + [_best_intraday(all_results)]


def render_model_picker(
    state,
    all_results: dict,
    selected_model: str,
) -> str:
    """Render clickable model cards. Returns the new selected_model key."""
    pinned = _get_pinned(all_results)
    displayed = list(all_results.keys())
    if not displayed:
        st.info("No model predictions available.")
        return selected_model

    new_sel = selected_model

    cols = st.columns(min(len(pinned) + 1, 7))
    for i, mk in enumerate(pinned):
        pred = all_results.get(mk, {})
        mean = pred.get("mean")
        std = pred.get("std")
        label = MODEL_LABELS.get(mk, mk)
        is_active = mk == selected_model

        if mean is not None and std is not None:
            text = f"**{label}**\n{mean:.1f}°C\n±{std:.2f}"
        else:
            text = f"**{label}**\nN/A"

        with cols[i % len(cols)]:
            btn_type = "primary" if is_active else "secondary"
            if st.button(text, key=f"mp_{mk}", type=btn_type, use_container_width=True):
                new_sel = mk

    # +Add popover
    with cols[len(pinned)]:
        with st.popover("+ Add", help="Toggle models on/off"):
            st.caption("Select up to 6 models")
            for mk in displayed:
                is_pinned = mk in pinned
                changed = st.checkbox(
                    MODEL_LABELS.get(mk, mk),
                    value=is_pinned,
                    key=f"mp_toggle_{mk}",
                )
                if changed and not is_pinned and len(pinned) < _MAX_PIN:
                    pinned.append(mk)
                    st.session_state[_PIN_KEY] = pinned[:]
                    st.rerun()
                elif not changed and is_pinned and len(pinned) > 1:
                    pinned.remove(mk)
                    st.session_state[_PIN_KEY] = pinned[:]
                    st.rerun()

            if len(pinned) != len(_DEFAULT_PIN + [_best_intraday(all_results)]):
                if st.button("Reset to defaults", use_container_width=True, key="mp_reset"):
                    st.session_state.pop(_PIN_KEY, None)
                    st.rerun()

    if new_sel != selected_model:
        state.selected_model = new_sel

    return new_sel