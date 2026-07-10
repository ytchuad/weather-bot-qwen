# app/components/strategy_builder.py
"""Strategy Builder — form-driven UI for creating/editing strategy config.

Replaces the raw-JSON textarea in the old Strategy Registry tab.  Users pick:
  - strategy name, model, capital
  - which entry/exit/sizing/rebalance gates to enable
  - parameter values for each gate
  - test with a one-click Lab run
"""

from __future__ import annotations

import json
import streamlit as st

from execution.strategy_config import (
    DEFAULT_ENTRY_GATES,
    DEFAULT_EXIT_GATES,
    DEFAULT_SIZING_GATES,
    DEFAULT_REBALANCE_GATES,
    GATE_REGISTRY,
)
from execution.strategy_account import StrategyAccount, StrategyAccountStore
from execution.paper_trade_harness import (
    PaperTradeHarness,
    generate_synthetic_scenarios,
)


def strategy_builder_form() -> None:
    """Render the Strategy Builder form."""
    st.subheader("🔧 Create / Edit Strategy")

    store = StrategyAccountStore()
    existing = store.list()
    existing_ids = {a.id for a in existing}

    # ── Strategy identity ──────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        # Option to start from an existing strategy template
        template_options = ["(Blank)"] + sorted(existing_ids)
        selected_template = st.selectbox(
            "Clone from existing",
            template_options,
            help="Start with an existing strategy as template and override its gates.",
        )

        sid = st.text_input(
            "Strategy ID",
            value="",
            placeholder="e.g., my_custom_v1",
            help="Unique identifier used to reference this strategy.",
        )

    with c2:
        model = st.selectbox(
            "Model",
            ["baseline", "rain_nowcast", "rain_observed", "model_a", "model_b",
             "model_c", "model_d", "model_e", "model_g", "model_2a", "model_2a_v2", "9d", "aws"],
            index=0,
        )
        market_template = st.selectbox(
            "Market template",
            ["hk-tmax", "hk-tmin"],
            index=0,
        )

    with c3:
        capital = st.number_input("Capital ($)", min_value=100, value=10_000, step=1000)
        kelly = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05)

    st.markdown("---")

    # ── Gate configuration sections ────────────────────────────────

    entry_gate_names, entry_gate_configs = _gate_section(
        "Entry Gates",
        DEFAULT_ENTRY_GATES,
        st.session_state.get("builder_entry_config", {}),
        "builder_entry",
        default_enabled=DEFAULT_ENTRY_GATES,
    )

    exit_gate_names, exit_gate_configs = _gate_section(
        "Exit Gates",
        DEFAULT_EXIT_GATES,
        st.session_state.get("builder_exit_config", {}),
        "builder_exit",
        default_enabled=DEFAULT_EXIT_GATES,
    )

    sizing_gate_names, sizing_gate_configs = _gate_section(
        "Sizing Gates",
        DEFAULT_SIZING_GATES,
        st.session_state.get("builder_sizing_config", {}),
        "builder_sizing",
        default_enabled=DEFAULT_SIZING_GATES,
    )

    rebalance_gate_names, rebalance_gate_configs = _gate_section(
        "Rebalance Triggers",
        DEFAULT_REBALANCE_GATES,
        st.session_state.get("builder_rebalance_config", {}),
        "builder_rebalance",
        default_enabled=DEFAULT_REBALANCE_GATES,
    )

    # Store in session state for next render
    st.session_state["builder_entry_config"] = entry_gate_configs
    st.session_state["builder_exit_config"] = exit_gate_configs
    st.session_state["builder_sizing_config"] = sizing_gate_configs
    st.session_state["builder_rebalance_config"] = rebalance_gate_configs

    st.markdown("---")

    # ── Action buttons ─────────────────────────────────────────────
    c_save, c_test, c_cancel = st.columns(3)

    with c_save:
        if st.button("💾 Save Strategy", type="primary", use_container_width=True):
            if not sid:
                st.error("Strategy ID is required.")
            else:
                _save_strategy(
                    store, sid, model, market_template, capital, kelly,
                    entry_gate_names, exit_gate_names,
                    sizing_gate_names, rebalance_gate_names,
                    entry_gate_configs, exit_gate_configs,
                    sizing_gate_configs, rebalance_gate_configs,
                )

    with c_test:
        if st.button("🧪 Test on Lab Data", use_container_width=True):
            if not sid:
                st.error("Enter a strategy ID first.")
            else:
                _test_in_lab(
                    sid, capital, kelly,
                    entry_gate_names, entry_gate_configs,
                    exit_gate_names, exit_gate_configs,
                    sizing_gate_names, sizing_gate_configs,
                    rebalance_gate_names, rebalance_gate_configs,
                )

    with c_cancel:
        if st.button("🗑️ Clear Form", use_container_width=True):
            _clear_form()


# ── Gate section helper ──────────────────────────────────────────────

def _gate_section(
    title: str,
    all_gate_names: list[str],
    current_configs: dict,
    prefix: str,
    default_enabled: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Render a collapsible section of gate toggles + param editors.

    Returns
    -------
    enabled_names : list[str]
        Gate names that are toggled ON.
    configs : dict
        Per-gate config dicts (empty if gate has no params).
    """
    if default_enabled is None:
        default_enabled = all_gate_names

    with st.expander(f"⚙️ {title} ({len(all_gate_names)} gates)", expanded=False):
        enabled = []
        configs = {}

        for gname in all_gate_names:
            # Look up the default config hint from the gate itself or defaults
            gate_fn = GATE_REGISTRY.get(gname)
            if gate_fn is None:
                continue

            default_state = gname in default_enabled
            checked = st.toggle(
                gname,
                value=current_configs.get(gname, {}).get("_enabled", default_state),
                key=f"{prefix}_{gname}_toggle",
                help=gate_fn.__doc__,
            )

            if checked:
                enabled.append(gname)
                cfg = _render_gate_params(gname, current_configs.get(gname, {}), prefix)
                configs[gname] = cfg

        return enabled, configs


def _render_gate_params(
    gname: str,
    current: dict,
    prefix: str,
) -> dict:
    """Render parameter controls for a specific gate.

    Returns the config dict with user-adjusted values.
    """
    cfg = dict(current)
    cfg["_enabled"] = True

    # Known gate parameters with sensible defaults
    if gname == "time_gate":
        cfg["min_hour"] = st.slider(
            "Min hour", 0, 23, cfg.get("min_hour", 8), key=f"{prefix}_{gname}_min_hour",
        )
        cfg["blocked_slots"] = st.multiselect(
            "Blocked slots",
            ["morning", "afternoon", "evening", "night"],
            default=cfg.get("blocked_slots", ["evening", "night"]),
            key=f"{prefix}_{gname}_slots",
        )

    elif gname == "regime_edge_gate":
        st.caption("Edge thresholds per regime:")
        thresholds = cfg.setdefault("thresholds", {})
        for regime in ["day_08_12", "rain_08_12", "slot_12_16", "slot_16_24"]:
            rc = thresholds.setdefault(regime, {"min_edge": 0.03, "exposure_cap": 0.5})
            col_a, col_b = st.columns(2)
            with col_a:
                rc["min_edge"] = st.number_input(
                    f"{regime} min_edge",
                    min_value=0.0, max_value=1.0,
                    value=float(rc.get("min_edge", 0.03)),
                    format="%.3f",
                    key=f"{prefix}_{gname}_{regime}_edge",
                )
            with col_b:
                rc["exposure_cap"] = st.slider(
                    f"{regime} exposure_cap",
                    0.0, 1.0, float(rc.get("exposure_cap", 0.5)), 0.05,
                    key=f"{prefix}_{gname}_{regime}_cap",
                )

    elif gname == "confidence_gate":
        cfg["max_model_std"] = st.slider(
            "Max model std", 0.5, 5.0, cfg.get("max_model_std", 2.5), 0.1,
            key=f"{prefix}_{gname}_std",
        )

    elif gname in ("boundary_gate", "boundary_proximity_sizer"):
        cfg["min_standardized_distance"] = st.slider(
            "Min std distance", 0.1, 2.0, cfg.get("min_standardized_distance", 0.5), 0.05,
            key=f"{prefix}_{gname}_min_dist",
        )
        cfg["aggressive_reduction_threshold"] = st.slider(
            "Aggressive reduction threshold", 0.1, 1.0,
            cfg.get("aggressive_reduction_threshold", 0.3), 0.05,
            key=f"{prefix}_{gname}_agg_thresh",
        )
        cfg["aggressive_reduction_multiplier"] = st.slider(
            "Aggressive reduction multiplier", 0.1, 1.0,
            cfg.get("aggressive_reduction_multiplier", 0.5), 0.05,
            key=f"{prefix}_{gname}_agg_mult",
        )

    elif gname in ("drawdown_gate", "drawdown_flatten_gate"):
        cfg["stop_new_entries"] = st.slider(
            "Stop new entries at drawdown", -0.5, 0.0,
            cfg.get("stop_new_entries", -0.10), 0.01,
            key=f"{prefix}_{gname}_stop",
        )
        cfg["hard_flatten"] = st.slider(
            "Hard flatten at drawdown", -0.5, 0.0,
            cfg.get("hard_flatten", -0.15), 0.01,
            key=f"{prefix}_{gname}_hard",
        )

    elif gname in ("time_window_sizer",):
        st.caption("Time window multipliers (e.g., {hours: [8,10], value: 0.6})")
        cfg["multipliers"] = st.text_area(
            "JSON array", value=json.dumps(
                cfg.get("multipliers",
                        [{"hours": [8, 10], "value": 0.6},
                         {"hours": [10, 14], "value": 1.0},
                         {"hours": [14, 16], "value": 0.7},
                         {"hours": [16, 24], "value": 0.3}]), indent=2
            ), height=150, key=f"{prefix}_{gname}_json",
        )

    elif gname == "rain_uncertainty_sizer":
        cfg["multipliers"] = {}
        for regime in ["no_rain", "weak_rain", "moderate_or_heavy_rain"]:
            cfg["multipliers"][regime] = st.slider(
                regime, 0.0, 1.0, cfg.get("multipliers", {}).get(regime, 0.7), 0.05,
                key=f"{prefix}_{gname}_{regime}",
            )

    elif gname == "model_confidence_sizer":
        st.caption("Multipliers per model")
        cfg["multipliers"] = {}
        for mk in ["baseline", "rain_nowcast", "model_a", "model_b", "model_c"]:
            cfg["multipliers"][mk] = st.slider(
                mk, 0.0, 1.0, cfg.get("multipliers", {}).get(mk, 0.8), 0.05,
                key=f"{prefix}_{gname}_{mk}",
            )
        cfg["default_mult"] = st.slider(
            "Default (unknown model)", 0.0, 1.0, cfg.get("default_mult", 0.5), 0.05,
            key=f"{prefix}_{gname}_default",
        )

    return cfg


# ── Save logic ───────────────────────────────────────────────────────

def _save_strategy(
    store: StrategyAccountStore,
    sid: str,
    model: str,
    market_template: str,
    capital: float,
    kelly: float,
    entry_gates: list[str],
    exit_gates: list[str],
    sizing_gates: list[str],
    rebalance_gates: list[str],
    entry_config: dict,
    exit_config: dict,
    sizing_config: dict,
    rebalance_config: dict,
) -> None:
    """Write a strategy account and update the registry."""
    existing = store.load(sid)
    if existing:
        label = existing.label
    else:
        label = sid.replace("_", " ").title()

    acct = StrategyAccount(
        id=sid,
        label=label,
        model=model,
        capital=capital,
        market_template=market_template,
        status="paused",
        scheduler_on=False,
        params={"bias": 0.0, "std_mult": 1.0, "kelly_fraction": kelly},
        from_strategy_key=sid,
        gate_config_override={
            "entry_gates": entry_gates,
            "exit_gates": exit_gates,
            "sizing_gates": sizing_gates,
            "rebalance_gates": rebalance_gates,
            "entry_config": entry_config,
            "exit_config": exit_config,
            "sizing_config": sizing_config,
            "rebalance_config": rebalance_config,
        },
    )
    store.save(acct)

    # Also write to paper_strategies.json V2 format for backward compat
    _update_paper_strategies_json(
        sid, entry_gates, exit_gates, sizing_gates, rebalance_gates,
        entry_config, exit_config, sizing_config, rebalance_config,
    )

    st.success(f"Strategy `{sid}` saved! Go to the **Live** tab to activate it.")


def _update_paper_strategies_json(
    sid: str,
    entry_gates: list[str],
    exit_gates: list[str],
    sizing_gates: list[str],
    rebalance_gates: list[str],
    entry_config: dict,
    exit_config: dict,
    sizing_config: dict,
    rebalance_config: dict,
) -> None:
    """Update config/paper_strategies.json with a new strategy entry."""
    from execution.strategy_factory import DEFAULT_CONFIG_PATH
    from ..services.strategy_service import load_strategy_registry, save_strategy_registry
    from execution.strategy_config import deep_merge

    registry = load_strategy_registry()

    # Build a V2-style override from the gate selections
    override = {}
    if entry_gates or rebalance_gates or sizing_gates or exit_gates:
        override["entry_point"] = "run_config_rebalance_cycle"
        override["module"] = "execution.strategy_engine"
        override["paper_only"] = True
        override["label"] = sid.replace("_", " ").title()
        override["description"] = f"Custom strategy built from {len(entry_gates)} entry gates"
        override["exposure_limits"] = None
        override["entry_rules"] = {"min_hour": 8, "only_on_event_date": False}

    # Set V2 format if we have versioned config
    if registry.get("version") == 2:
        override["entry"] = {}
        override["exit"] = {}
        override["position_sizing"] = {}
        override["rebalance"] = {}

        # Map gate-level configs back to V2 block keys
        if reg_cfg := entry_config.get("regime_edge_gate", {}).get("thresholds"):
            override["entry"]["regime_thresholds"] = reg_cfg
        if prob_cfg := entry_config.get("confidence_gate", {}).get("max_model_std"):
            override["entry"]["probability_confidence"] = {"max_model_std": prob_cfg}
        if time_cfg := entry_config.get("time_gate", {}).get("min_hour"):
            override["entry"]["min_hour"] = time_cfg
        if bound_cfg := (entry_config.get("boundary_gate", {}) or
                          sizing_config.get("boundary_proximity_sizer", {})):
            override["entry"]["boundary_proximity"] = {
                "min_standardized_distance": bound_cfg.get("min_standardized_distance", 0.5),
                "aggressive_reduction_threshold": bound_cfg.get("aggressive_reduction_threshold", 0.3),
                "aggressive_reduction_multiplier": bound_cfg.get("aggressive_reduction_multiplier", 0.5),
            }
        if sizing_cfg := sizing_config.get("rain_uncertainty_sizer", {}).get("multipliers"):
            override["position_sizing"]["rain_uncertainty_multiplier"] = sizing_cfg
        if exit_cfg := exit_config.get("drawdown_flatten_gate", {}):
            override["exit"]["drawdown"] = {
                "hard_flatten": exit_cfg.get("hard_flatten", -0.15),
                "reduce_risk": exit_cfg.get("stop_new_entries", -0.10) * 0.75,
            }

    # Merge into registry
    registry.setdefault("strategies", {})[sid] = deep_merge(
        registry.get("strategies", {}).get(sid, {}),
        override,
    )
    save_strategy_registry(registry)


# ── Lab test ─────────────────────────────────────────────────────────

def _test_in_lab(
    sid: str,
    capital: float,
    kelly: float,
    entry_gate_names: list[str],
    entry_gate_configs: dict,
    exit_gate_names: list[str],
    exit_gate_configs: dict,
    sizing_gate_names: list[str],
    sizing_gate_configs: dict,
    rebalance_gate_names: list[str],
    rebalance_gate_configs: dict,
) -> None:
    """Run a quick Lab backtest with the current builder config."""
    # Build a temporary Strategy from builder config
    from execution.strategy_config import build_pipeline

    entry_pipeline = build_pipeline("entry", entry_gate_names, entry_gate_configs)
    exit_pipeline = build_pipeline("exit", exit_gate_names, exit_gate_configs)
    sizing_pipeline = build_pipeline("sizing", sizing_gate_names, sizing_gate_configs)
    rebalance_pipeline = build_pipeline("rebalance", rebalance_gate_names, rebalance_gate_configs)

    # Synthetic data
    temp_buckets = {
        "25-26": {"market_price": 0.05},
        "26-27": {"market_price": 0.08},
        "27-28": {"market_price": 0.12},
        "28-29": {"market_price": 0.18},
        "29-30": {"market_price": 0.20},
        "30-31": {"market_price": 0.15},
        "31-32": {"market_price": 0.10},
        "32-33": {"market_price": 0.07},
        "33-34": {"market_price": 0.04},
        "34-35": {"market_price": 0.02},
        "35-36": {"market_price": 0.01},
        ">=36": {"market_price": 0.005},
    }

    with st.spinner(f"Testing `{sid}` on 50 synthetic cycles..."):
        scenarios = generate_synthetic_scenarios(temp_buckets, n_cycles=50, seed=42)
        harness = PaperTradeHarness(
            strategy_or_key=entry_pipeline,
            capital=capital,
            kelly_fraction=kelly,
            seed=42,
            _override_pipelines=(entry_pipeline, exit_pipeline, sizing_pipeline, rebalance_pipeline),
        )
        result = harness.run(scenarios)
        summary = result.summary()

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Final Capital", f"${result.final_capital:,.2f}")
    col_b.metric("Total PnL", f"${summary['total_pnl']:+,.2f}")
    col_c.metric("Max DD", f"{summary['max_drawdown']:.2f}%")
    col_d.metric("Sharpe", f"{summary['sharpe_ratio']:.3f}")
    col_e.metric("Trades", summary["num_trades"])


# ── Form clear ───────────────────────────────────────────────────────

def _clear_form():
    """Reset builder form fields in session state."""
    keys = [k for k in st.session_state if k.startswith("builder_")]
    for k in keys:
        del st.session_state[k]
    st.rerun()
