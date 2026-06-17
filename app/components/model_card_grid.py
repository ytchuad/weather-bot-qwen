# app/components/model_card_grid.py
"""Grid of model prediction cards with active-model selection."""

import streamlit as st

from ..config import COLORS, MODEL_LABELS


def model_card_grid(
    all_preds: dict[str, dict],
    selected_model: str,
    key_prefix: str = "mc",
) -> str:
    """Render a grid of clickable model cards.

    Args:
        all_preds: {model_key: {mean, std, source}} as returned by ModelService.
        selected_model: currently selected model key.

    Returns:
        The selected model key (may differ if user clicked a different card).
    """
    displayed = list(all_preds.keys())
    if not displayed:
        st.info("No model predictions available.")
        return selected_model

    n_cols = min(len(displayed), 9)
    cols = st.columns(n_cols)

    new_selected = selected_model

    for i, mk in enumerate(displayed):
        pred = all_preds.get(mk, {})
        mean = pred.get("mean")
        std = pred.get("std")
        label = MODEL_LABELS.get(mk, mk)
        is_active = mk == selected_model

        if mean is not None and std is not None:
            text = f"**{label}**  \n{mean:.1f}°C  \n±{std:.2f}"
        else:
            text = f"**{label}**  \nN/A"

        with cols[i % n_cols]:
            btn_type = "primary" if is_active else "secondary"
            if st.button(text, key=f"{key_prefix}_{mk}", type=btn_type, use_container_width=True):
                new_selected = mk

    if new_selected != selected_model:
        st.session_state["app.selected_model"] = new_selected

    return new_selected
