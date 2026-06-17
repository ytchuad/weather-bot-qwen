# app/pages/page_health.py
"""Health page — system checks, smoke tests, feature schema, audit log."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ..state import AppState
from ..config import ROOT_DIR, MODELS_DIR, MODEL_KEYS, DATA_DIR
from ..components.sidebar import render_sidebar
from ..services.weather_service import hkt_now


def _check_module(name: str) -> dict:
    """Check if a module can be imported."""
    try:
        importlib.import_module(name)
        return {"module": name, "status": "✅ OK", "error": ""}
    except Exception as e:
        return {"module": name, "status": "❌ FAIL", "error": str(e)}


def _check_file(path: Path, label: str = "") -> dict:
    """Check if a file exists and report size."""
    try:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            return {"file": label or path.name, "status": "✅ OK", "size": f"{size_kb:.1f} KB"}
        return {"file": label or path.name, "status": "⚠️ Missing", "size": "—"}
    except Exception as e:
        return {"file": label or path.name, "status": "❌ Error", "size": str(e)}


def _check_model_files() -> list[dict]:
    """Check that expected model files exist."""
    results = []
    for mk in MODEL_KEYS:
        if mk == "9d":
            for var in ["tmax", "tmin"]:
                p = MODELS_DIR / f"model_9d_{var}.json"
                results.append(_check_file(p, f"9d_{var}"))
        elif mk == "aws":
            for var in ["tmax", "tmin"]:
                p = MODELS_DIR / f"aws_hf_{var}"
                results.append(_check_file(p, f"aws_{var}"))
        elif mk in ("baseline", "model_a", "model_b", "model_c", "model_d", "model_e"):
            p = MODELS_DIR / f"intraday_model_{mk}.txt"
            results.append(_check_file(p, f"intraday_{mk}"))
        elif mk == "rain_nowcast":
            p = MODELS_DIR / "intraday_model_rain_nowcast.txt"
            results.append(_check_file(p, "rain_nowcast"))
        elif mk == "rain_observed":
            p = MODELS_DIR / "intraday_model_rain_observed.txt"
            results.append(_check_file(p, "rain_observed"))
    return results


def _check_data_files() -> list[dict]:
    """Check that key data files exist."""
    checks = []
    if not DATA_DIR.exists():
        checks.append({"file": "data/", "status": "⚠️ Directory missing", "size": "—"})
        return checks
    for pattern in ["hko_*", "intraday_*", "rain_*"]:
        found = list(DATA_DIR.glob(pattern))
        if found:
            latest = max(found, key=lambda p: p.stat().st_mtime)
            checks.append({"file": pattern, "status": f"✅ {len(found)} files",
                           "size": f"{latest.stat().st_size / 1024:.1f} KB"})
        else:
            checks.append({"file": pattern, "status": "⚠️ None", "size": "—"})
    return checks


def _check_config_files() -> list[dict]:
    """Check config files exist and are valid."""
    checks = [
        _check_file(ROOT_DIR / "config.yaml", "config.yaml"),
        _check_file(ROOT_DIR / "config" / "feature_schema.json", "feature_schema.json"),
        _check_file(ROOT_DIR / "config" / "paper_strategies.json", "paper_strategies.json"),
    ]
    return checks


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ---- sidebar ----
    render_sidebar(state)

    st.header("🏥 System Health Check")

    now = hkt_now()
    st.caption(f"Check run at: {now.isoformat()} HKT")

    tabs = st.tabs(["🔍 Module Imports", "📁 Files & Data", "🛠️ Smoke Tests", "📋 Audit Log"])

    # ========================================================================
    # Tab 1: Module Imports
    # ========================================================================
    with tabs[0]:
        st.subheader("Module Import Status")

        modules_to_check = [
            # Core app
            "app.state", "app.config",
            # Services
            "app.services.market_service", "app.services.weather_service",
            "app.services.model_service", "app.services.strategy_service",
            "app.services.backtest_service",
            # Features / models
            "features.live_feature_builder", "features.dataset_builder",
            "models.model_9d", "models.intraday_models",
            # Execution
            "execution.strategy_runner", "execution.portfolio_manager",
            "execution.paper_adapter", "execution.rebalancer",
        ]
        results = [_check_module(m) for m in modules_to_check]
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary
        ok_count = sum(1 for r in results if "OK" in r["status"])
        fail_count = len(results) - ok_count
        cols = st.columns(4)
        cols[0].metric("Total Modules", len(results))
        cols[1].metric("OK", ok_count)
        cols[2].metric("Failed", fail_count, delta=f"-{fail_count}" if fail_count > 0 else None)

        if fail_count > 0:
            st.warning("Some modules failed to import. Check the error messages above.")

    # ========================================================================
    # Tab 2: Files & Data
    # ========================================================================
    with tabs[1]:
        st.subheader("Model Files")
        model_checks = _check_model_files()
        st.dataframe(pd.DataFrame(model_checks), use_container_width=True, hide_index=True)

        st.subheader("Data Files")
        data_checks = _check_data_files()
        st.dataframe(pd.DataFrame(data_checks), use_container_width=True, hide_index=True)

        st.subheader("Config Files")
        config_checks = _check_config_files()
        st.dataframe(pd.DataFrame(config_checks), use_container_width=True, hide_index=True)

        # Config details
        with st.expander("📋 Feature Schema Details"):
            schema_path = ROOT_DIR / "config" / "feature_schema.json"
            try:
                with open(schema_path) as f:
                    schema = json.load(f)
                st.json(schema.get("features", []))
            except Exception as e:
                st.error(str(e))

        with st.expander("📋 Strategy Config Details"):
            strat_path = ROOT_DIR / "config" / "paper_strategies.json"
            try:
                with open(strat_path) as f:
                    data = json.load(f)
                n_strats = len(data) if isinstance(data, (list, dict)) else 0
                st.caption(f"{n_strats} strategies defined")
                st.json(data)
            except Exception as e:
                st.error(str(e))

    # ========================================================================
    # Tab 3: Smoke Tests
    # ========================================================================
    with tabs[2]:
        st.subheader("Quick Smoke Tests")

        if st.button("🧪 Run Smoke Tests", use_container_width=True):
            results = []

            # Test 1: HKO API reachable
            try:
                from ..services.weather_service import fetch_live_hko_temp_rh
                dt, temp, rh = fetch_live_hko_temp_rh()
                if temp is not None:
                    results.append({"test": "HKO Live Temp", "result": f"✅ {temp}°C at {dt}"})
                else:
                    results.append({"test": "HKO Live Temp", "result": "⚠️ No data returned"})
            except Exception as e:
                results.append({"test": "HKO Live Temp", "result": f"❌ {e}"})

            # Test 2: Polymarket API
            try:
                from ..services.market_service import search_events
                events = search_events("hong-kong-temperature")
                results.append({"test": "Polymarket Search", "result": f"✅ {len(events)} events"})
            except Exception as e:
                results.append({"test": "Polymarket Search", "result": f"❌ {e}"})

            # Test 3: Strategy registry loadable
            try:
                from ..services.strategy_service import load_strategy_registry
                reg = load_strategy_registry()
                n = len(reg) if isinstance(reg, (list, dict)) else 0
                results.append({"test": "Strategy Registry", "result": f"✅ {n} strategies"})
            except Exception as e:
                results.append({"test": "Strategy Registry", "result": f"❌ {e}"})

            # Test 4: Config YAML valid
            try:
                import yaml
                with open(ROOT_DIR / "config.yaml") as f:
                    cfg = yaml.safe_load(f)
                results.append({"test": "config.yaml", "result": f"✅ Loaded (keys: {len(cfg)})"})
            except Exception as e:
                results.append({"test": "config.yaml", "result": f"❌ {e}"})

            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # System info
        st.markdown("---")
        st.subheader("System Info")
        sys_info = {
            "Python": sys.version,
            "Platform": sys.platform,
            "CWD": str(Path.cwd()),
            "ROOT_DIR": str(ROOT_DIR),
        }
        st.json(sys_info)

    # ========================================================================
    # Tab 4: Audit Log
    # ========================================================================
    with tabs[3]:
        st.subheader("Audit Log")

        audit_path = ROOT_DIR / "data" / "audit.log"
        if audit_path.exists():
            try:
                with open(audit_path) as f:
                    lines = f.readlines()
                # Show last 200 lines
                recent = lines[-200:]
                st.code("".join(reversed(recent)), language="log", line_numbers=True)
                st.caption(f"Showing last {len(recent)} of {len(lines)} lines from `{audit_path}`")
            except Exception as e:
                st.error(f"Could not read audit log: {e}")
        else:
            st.info(f"No audit log found at `{audit_path}`.")

        # Also check for any other log files
        data_dir = ROOT_DIR / "data"
        log_files = list(ROOT_DIR.glob("*.log"))
        if data_dir.exists():
            log_files += list(data_dir.glob("*.log"))
        if log_files:
            st.markdown("**Other log files:**")
            for lf in log_files:
                st.caption(f"- `{lf}` ({lf.stat().st_size} bytes)")


if __name__ == "__main__":
    run()
