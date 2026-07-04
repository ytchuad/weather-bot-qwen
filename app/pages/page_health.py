# app/pages/page_health.py
"""Health page — system checks, smoke tests, feature schema, audit log."""

import importlib
import json
import math
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests as _req
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
    # Map model key → (directory relative to MODELS_DIR, indicator file)
    MODEL_DIR_MAP = {
        "baseline":      ("intraday_ml",                         "feature_list.json"),
        "rain_nowcast":  ("intraday_ml_rain_nowcast",             "feature_list.json"),
        "rain_observed": ("intraday_ml_rain_observed",            "feature_list.json"),
        "model_a":       ("intraday_minute_ml",                   "feature_list.json"),
        "model_b":       ("intraday_minute_ml_model_b",           "feature_list.json"),
        "model_c":       ("intraday_minute_ml_model_c",           "feature_list.json"),
        "model_d":       ("intraday_minute_ml_model_d_tmin",      "feature_list.json"),
        "model_e":       ("intraday_minute_ml_model_e_morning_tmin", "feature_list.json"),
        "model_g":       ("intraday_minute_ml_model_g",           "feature_list.json"),
        "model_2a":      ("intraday_minute_ml_model_2a",          "feature_list.json"),
    }
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
        elif mk in MODEL_DIR_MAP:
            subdir, indicator = MODEL_DIR_MAP[mk]
            p = MODELS_DIR / subdir / indicator
            results.append(_check_file(p, f"intraday_{mk}"))
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


def _check_data_sources() -> list[dict]:
    """Check all external data sources — URL reachability + response time."""
    from ..config import (
        HKO_RHRREAD_URL, HKO_AWS_CSV_URL, HKO_MAXMIN_URL,
        HKO_FORECAST_URL_TEMPLATE, PM_SEARCH_URL,
    )
    from ..services.weather_service import (
        HKO_PRESSURE_CSV_URL, WIND_INSTANT_URL,
    )

    sources = [
        ("HKO RHRREAD (Temp/RH)",     HKO_RHRREAD_URL),
        ("HKO AWS CSV (Weather)",      HKO_AWS_CSV_URL),
        ("HKO MaxMin CSV",             HKO_MAXMIN_URL),
        ("HKO Forecast XML",           HKO_FORECAST_URL_TEMPLATE),
        ("HKO Pressure CSV",           HKO_PRESSURE_CSV_URL),
        ("i-Lens DG_WIND",             WIND_INSTANT_URL),
        ("Polymarket Search",          f"{PM_SEARCH_URL}?term=test&limit=1"),
    ]

    results = []
    now_str = hkt_now().strftime("%H:%M:%S")
    for name, url_template in sources:
        url = url_template.format(ts=int(_time.time() * 1000)) if "{ts}" in url_template else url_template
        try:
            t0 = _time.time()
            r = _req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            elapsed = _time.time() - t0
            if r.status_code == 200:
                status = "✅"
                latest = f"{len(r.text):,} bytes"
            else:
                status = "⚠️"
                latest = f"HTTP {r.status_code}"
            results.append({
                "source": name, "url": url,
                "status": status, "response_time": f"{elapsed:.1f}s",
                "last_update": now_str, "latest_value": latest,
            })
        except Exception as e:
            results.append({
                "source": name, "url": url,
                "status": "❌", "response_time": "—",
                "last_update": now_str, "latest_value": str(e)[:80],
            })
    return results


def _get_feature_status() -> list[dict]:
    """Fetch current live data and return per-domain feature status."""
    from ..services.weather_service import (
        fetch_live_hko_temp_rh, fetch_pressure_live, compute_pressure_kwargs,
        fetch_wind_live, compute_wind_kwargs, get_nowcast_rainfall,
    )

    results = []
    now_str = hkt_now().strftime("%H:%M:%S")

    # ── Temp / RH / Dew point ──────────────────────────────────────────
    try:
        _r = fetch_live_hko_temp_rh()
        if isinstance(_r, tuple) and len(_r) >= 3:
            dt, temp, rh = _r[0], _r[1], _r[2]
        else:
            dt, temp, rh = None, None, None
        if temp is not None:
            results.append({"domain": "Temp/RH", "feature": "temp_current",
                            "source": "HKO RHRREAD", "value": f"{temp:.1f} °C",
                            "status": "✅", "updated": now_str})
        else:
            results.append({"domain": "Temp/RH", "feature": "temp_current",
                            "source": "HKO RHRREAD", "value": "None",
                            "status": "⚠️", "updated": now_str})
        if rh is not None:
            results.append({"domain": "Temp/RH", "feature": "rh_current",
                            "source": "HKO RHRREAD", "value": f"{rh:.0f} %",
                            "status": "✅", "updated": now_str})
        else:
            results.append({"domain": "Temp/RH", "feature": "rh_current",
                            "source": "HKO RHRREAD", "value": "None",
                            "status": "⚠️", "updated": now_str})
        if temp is not None and rh is not None:
            a, b = 17.625, 243.04
            alpha = math.log(rh / 100) + a * temp / (b + temp)
            dp = b * alpha / (a - alpha)
            results.append({"domain": "Temp/RH", "feature": "dew_point_current",
                            "source": "Magnus formula", "value": f"{dp:.1f} °C",
                            "status": "✅", "updated": now_str})
        elif temp is not None:
            results.append({"domain": "Temp/RH", "feature": "dew_point_current",
                            "source": "Magnus formula", "value": "N/A (no RH)",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Temp/RH", "feature": "temp/rh/dew_point",
                        "source": "HKO RHRREAD", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    # ── Pressure ────────────────────────────────────────────────────────
    try:
        df_p = fetch_pressure_live()
        if not df_p.empty:
            pk = compute_pressure_kwargs()
            results.append({"domain": "Pressure", "feature": "pressure_current",
                            "source": "HKO Pressure CSV", "value": f"{pk['pressure_current']:.1f} hPa",
                            "status": "✅", "updated": now_str})
            results.append({"domain": "Pressure", "feature": "pressure_change_60m",
                            "source": "HKO Pressure CSV", "value": f"{pk['pressure_change_60m']:+.1f} hPa",
                            "status": "✅", "updated": now_str})
            results.append({"domain": "Pressure", "feature": "pressure_change_180m",
                            "source": "HKO Pressure CSV", "value": f"{pk['pressure_change_180m']:+.1f} hPa",
                            "status": "✅", "updated": now_str})
        else:
            results.append({"domain": "Pressure", "feature": "pressure_*",
                            "source": "HKO Pressure CSV", "value": "Empty DataFrame",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Pressure", "feature": "pressure_*",
                        "source": "HKO Pressure CSV", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    # ── Wind ────────────────────────────────────────────────────────────
    try:
        df_w = fetch_wind_live()
        if not df_w.empty:
            wk = compute_wind_kwargs()
            wind_feats = [
                ("wind_ref_mean", wk.get("wind_ref_mean", 0.0)),
                ("wind_ref_max", wk.get("wind_ref_max", 0.0)),
                ("wind_victoria_harbour_mean", wk.get("wind_victoria_harbour_mean", 0.0)),
                ("wind_victoria_harbour_max", wk.get("wind_victoria_harbour_max", 0.0)),
                ("wind_highland_mean", wk.get("wind_highland_mean", 0.0)),
                ("wind_highland_max", wk.get("wind_highland_max", 0.0)),
                ("wind_all_change_60m", wk.get("wind_all_change_60m", 0.0)),
                ("wind_kings_park_current", wk.get("wind_kings_park_current", 0.0)),
            ]
            for fname, fval in wind_feats:
                results.append({"domain": "Wind", "feature": fname,
                                "source": "i-Lens DG_WIND", "value": f"{fval:.1f} km/h",
                                "status": "✅", "updated": now_str})
        else:
            results.append({"domain": "Wind", "feature": "wind_*",
                            "source": "i-Lens DG_WIND", "value": "No stations parsed",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Wind", "feature": "wind_*",
                        "source": "i-Lens DG_WIND", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    # ── Forecast freshness ──────────────────────────────────────────────
    try:
        from ..config import HKO_FORECAST_URL_TEMPLATE
        url = HKO_FORECAST_URL_TEMPLATE.format(ts=int(_time.time() * 1000))
        r = _req.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        model_time_str = data.get("ModelTime", "")
        forecast_min = data.get("Forecast_Min", {}).get("Value")
        forecast_max = data.get("Forecast_Max", {}).get("Value")
        if model_time_str:
            model_dt = pd.to_datetime(str(model_time_str), format="%Y%m%d%H")
            from ..config import HKT_OFFSET
            model_dt_hkt = model_dt + HKT_OFFSET
            age_min = (hkt_now() - model_dt_hkt).total_seconds() / 60
            results.append({"domain": "Forecast", "feature": "forecast_age_minutes",
                            "source": "HKO XML ModelTime", "value": f"{age_min:.0f} min",
                            "status": "✅", "updated": now_str})
            results.append({"domain": "Forecast", "feature": "forecast_max_temp",
                            "source": "HKO XML", "value": f"{forecast_max} °C" if forecast_max else "None",
                            "status": "✅" if forecast_max else "⚠️", "updated": now_str})
            results.append({"domain": "Forecast", "feature": "forecast_min_temp",
                            "source": "HKO XML", "value": f"{forecast_min} °C" if forecast_min else "None",
                            "status": "✅" if forecast_min else "⚠️", "updated": now_str})
        else:
            results.append({"domain": "Forecast", "feature": "forecast_*",
                            "source": "HKO XML", "value": "No ModelTime field",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Forecast", "feature": "forecast_*",
                        "source": "HKO XML", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    # ── Rainfall (live) ─────────────────────────────────────────────────
    try:
        from ..services.weather_service import fetch_rainfall_live, get_accumulated_rain_today
        df_r = fetch_rainfall_live()
        if not df_r.empty:
            latest_r = float(df_r["rainfall_mm"].iloc[-1])
            total_r = get_accumulated_rain_today()
            results.append({"domain": "Rainfall", "feature": "rainfall_latest",
                            "source": "HKO AWS CSV", "value": f"{latest_r:.1f} mm",
                            "status": "✅", "updated": now_str})
            results.append({"domain": "Rainfall", "feature": "rainfall_accumulated_today",
                            "source": "HKO AWS CSV", "value": f"{total_r:.1f} mm" if total_r else "None",
                            "status": "✅" if total_r else "⚠️", "updated": now_str})
        else:
            results.append({"domain": "Rainfall", "feature": "rainfall_*",
                            "source": "HKO AWS CSV", "value": "Empty DataFrame",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Rainfall", "feature": "rainfall_*",
                        "source": "HKO AWS CSV", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    # ── Gridded nowcast ─────────────────────────────────────────────────
    try:
        nc = get_nowcast_rainfall()
        if nc is not None:
            results.append({"domain": "Rainfall", "feature": "rain_nowcast_value",
                            "source": "HKO Gridded Nowcast", "value": f"{nc:.1f} mm",
                            "status": "✅", "updated": now_str})
        else:
            results.append({"domain": "Rainfall", "feature": "rain_nowcast_value",
                            "source": "HKO Gridded Nowcast", "value": "None",
                            "status": "⚠️", "updated": now_str})
    except Exception as e:
        results.append({"domain": "Rainfall", "feature": "rain_nowcast_value",
                        "source": "HKO Gridded Nowcast", "value": f"Error: {e}",
                        "status": "❌", "updated": now_str})

    return results


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ---- sidebar ----
    render_sidebar(state)

    st.header("🏥 System Health Check")

    now = hkt_now()
    st.caption(f"Check run at: {now.isoformat()} HKT")

    tabs = st.tabs(["🔍 Module Imports", "📁 Files & Data", "🛠️ Smoke Tests", "📋 Audit Log", "📡 Data Sources & Features"])

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
                _r = fetch_live_hko_temp_rh()
                if isinstance(_r, tuple) and len(_r) >= 3:
                    dt, temp, rh = _r[0], _r[1], _r[2]
                else:
                    dt, temp, rh = None, None, None
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

    # ========================================================================
    # Tab 5: Data Sources & Features
    # ========================================================================
    with tabs[4]:
        st.subheader("📡 External Data Sources")

        # Auto-refresh
        if "ds_last_check" not in st.session_state:
            st.session_state.ds_last_check = 0.0
        if "ds_cache" not in st.session_state:
            st.session_state.ds_cache = None
        if "feat_cache" not in st.session_state:
            st.session_state.feat_cache = None

        col1, col2, col3 = st.columns([1.5, 2, 1])
        refresh_clicked = col1.button("🔄 Refresh Now")
        auto_refresh = col2.checkbox("Auto-refresh (60s)", value=True)

        stale = _time.time() - st.session_state.ds_last_check > 60
        if refresh_clicked or (auto_refresh and stale):
            with st.spinner("Checking data sources & fetching features..."):
                st.session_state.ds_cache = _check_data_sources()
                st.session_state.feat_cache = _get_feature_status()
                st.session_state.ds_last_check = _time.time()
            st.rerun()

        # Show last refresh time
        if st.session_state.ds_last_check > 0:
            last_str = datetime.fromtimestamp(st.session_state.ds_last_check).strftime("%H:%M:%S")
            st.caption(f"Last refreshed: {last_str} HKT")
        else:
            st.info("Click **Refresh Now** to check all data sources.")

        # Data sources table
        if st.session_state.ds_cache is not None:
            df_ds = pd.DataFrame(st.session_state.ds_cache)
            df_ds.columns = ["Source", "URL", "Status", "Response", "Checked At", "Latest Value"]
            st.dataframe(df_ds, use_container_width=True, hide_index=True)

            ok = sum(1 for r in st.session_state.ds_cache if r["status"] == "✅")
            warn = sum(1 for r in st.session_state.ds_cache if r["status"] == "⚠️")
            fail = sum(1 for r in st.session_state.ds_cache if r["status"] == "❌")
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Reachable", ok)
            c2.metric("⚠️ Non-200", warn)
            c3.metric("❌ Failed", fail)

        # ── Feature Status ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Live Feature Values")

        if st.session_state.feat_cache is not None:
            df_feat = pd.DataFrame(st.session_state.feat_cache)
            st.dataframe(
                df_feat,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "domain":  st.column_config.Column("Domain", width="small"),
                    "feature": st.column_config.Column("Feature", width="medium"),
                    "source":  st.column_config.Column("Source", width="medium"),
                    "value":   st.column_config.Column("Current Value", width="medium"),
                    "status":  st.column_config.Column("Status", width="small"),
                    "updated": st.column_config.Column("Checked At", width="small"),
                },
            )
        else:
            st.info("Click **Refresh Now** to fetch live feature values.")


if __name__ == "__main__":
    run()
