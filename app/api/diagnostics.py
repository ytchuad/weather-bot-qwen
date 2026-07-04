from fastapi import APIRouter
from datetime import datetime
import math
import logging
import time as _time

import pandas as pd
import requests as _req

router = APIRouter(prefix="/api/diagnostics", tags=["Diagnostics"])
logger = logging.getLogger(__name__)


def _now_str() -> str:
    from app.config import HKT_OFFSET
    return (datetime.utcnow() + HKT_OFFSET).strftime("%H:%M:%S")


def _hkt_now() -> datetime:
    from app.config import HKT_OFFSET
    return datetime.utcnow() + HKT_OFFSET


@router.get("/sources")
def check_data_sources():
    from app.config import (
        HKO_RHRREAD_URL, HKO_AWS_CSV_URL, HKO_MAXMIN_URL,
        HKO_FORECAST_URL_TEMPLATE, PM_SEARCH_URL,
    )
    from app.services.weather_service import (
        fetch_live_hko_temp_rh, fetch_hko_data,
        compute_rain_kwargs_live, _fetch_rainfall_live_uncached,
        hkt_now, get_intraday_state,
        HKO_PRESSURE_CSV_URL, WIND_INSTANT_URL, INSTANT_RAIN_URL,
        fetch_pressure_live, compute_pressure_kwargs,
        fetch_wind_live, compute_wind_kwargs,
        get_nowcast_rainfall,
    )

    all_features_flat = []
    results = {
        "checked_at": _hkt_now().isoformat(),
        "sources": [],
        "features_flat": [],
    }

    # ── 1. HKO AWS CSV (hko.csv) — primary model input ────────────────
    try:
        from app.services.weather_service import fetch_hko_intraday_csv
        df_csv = fetch_hko_intraday_csv(_cache_buster=int(_time.time() // 60))
        features = []
        if not df_csv.empty:
            last_row = df_csv.iloc[-1]
            csv_temp = float(last_row["temp"])
            csv_dt = last_row["datetime"]
            csv_rh = float(last_row["rh"]) if "rh" in last_row else None
            features.append({"name": "temp_current", "value": f"{csv_temp:.1f} °C", "status": "ok"})
            features.append({"name": "record_time", "value": csv_dt.strftime("%H:%M:%S") if hasattr(csv_dt, "strftime") else str(csv_dt), "status": "ok"})
            if csv_rh:
                features.append({"name": "rh_current", "value": f"{csv_rh:.0f} %", "status": "ok"})
            status = "ok"
            msg = f"Temp: {csv_temp:.1f}°C at {csv_dt}"
        else:
            status = "error"
            msg = "Empty DataFrame"
        results["sources"].append({
            "name": "HKO AWS CSV (hko.csv)",
            "url": HKO_AWS_CSV_URL,
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "HKO AWS CSV (hko.csv)",
            "url": HKO_AWS_CSV_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 2. HKO Live Temp/RH (Observatory, RHRREAD API, for reference) ──
    try:
        _r = fetch_live_hko_temp_rh()
        if isinstance(_r, tuple) and len(_r) >= 3:
            dt, temp, rh = _r[0], _r[1], _r[2]
        else:
            dt, temp, rh = None, None, None
        features = []
        record_time = dt.strftime("%H:%M:%S") if dt else "None"
        features.append({"name": "record_time", "value": record_time, "status": "ok" if dt else "warning"})
        if temp is not None:
            features.append({"name": "observatory_temp", "value": f"{temp:.1f} °C", "status": "ok"})
        else:
            features.append({"name": "observatory_temp", "value": "None", "status": "error"})
        if rh is not None:
            features.append({"name": "rh_current", "value": f"{rh:.0f} %", "status": "ok"})
            a, b = 17.625, 243.04
            alpha = math.log(rh / 100) + a * temp / (b + temp)
            dp = b * alpha / (a - alpha)
            features.append({"name": "dew_point_current", "value": f"{dp:.1f} °C", "status": "ok"})
        else:
            features.append({"name": "rh_current", "value": "None", "status": "error"})
        status = "ok" if temp is not None else "error"
        results["sources"].append({
            "name": "HKO Observatory (RHRREAD)",
            "url": HKO_RHRREAD_URL,
            "status": status,
            "message": f"Observatory: {temp}°C (recorded {record_time})" if temp else "No data",
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "HKO Observatory (RHRREAD)",
            "url": HKO_RHRREAD_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 3. HKO Daily Data (AWS CSV) ───────────────────────────────────
    try:
        hko = fetch_hko_data(hkt_now().strftime("%Y%m%d"))
        features = []
        if hko and isinstance(hko, dict):
            max_val = hko.get("max_since_midnight")
            min_val = hko.get("min_since_midnight")
            if max_val is not None:
                features.append({"name": "max_so_far", "value": f"{max_val} °C", "status": "ok"})
            else:
                features.append({"name": "max_so_far", "value": "None", "status": "warning"})
            if min_val is not None:
                features.append({"name": "min_so_far", "value": f"{min_val} °C", "status": "ok"})
        status = "ok" if max_val is not None else "warning"
        results["sources"].append({
            "name": "HKO Daily Data",
            "url": HKO_AWS_CSV_URL,
            "status": status,
            "message": f"Max so far: {max_val}" if max_val else "No max data",
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "HKO Daily Data",
            "url": HKO_AWS_CSV_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 4. i-Lens Rainfall API ────────────────────────────────────────
    try:
        rain_df = _fetch_rainfall_live_uncached()
        features = []
        if rain_df is not None and isinstance(rain_df, pd.DataFrame) and not rain_df.empty:
            latest_time = rain_df.iloc[-1]["datetime"]
            latest_r = float(rain_df.iloc[-1]["rainfall"])
            features.append({"name": "rainfall_latest", "value": f"{latest_r:.1f} mm", "status": "ok"})
            features.append({"name": "rows", "value": str(len(rain_df)), "status": "ok"})
            status = "ok"
            msg = f"Latest data at {latest_time}. Rows: {len(rain_df)}"
        else:
            status = "error"
            msg = "Empty DataFrame - no rainfall data"
        results["sources"].append({
            "name": "i-Lens Rainfall API",
            "url": INSTANT_RAIN_URL,
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "i-Lens Rainfall API",
            "url": INSTANT_RAIN_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 5. Rainfall Features (derived) ────────────────────────────────
    try:
        rk = compute_rain_kwargs_live()
        features = []
        if isinstance(rk, dict):
            r60 = rk.get("rain_60m", 0)
            r120 = rk.get("rain_120m", 0)
            features.append({"name": "rain_60m", "value": f"{r60} mm", "status": "ok"})
            features.append({"name": "rain_120m", "value": f"{r120} mm", "status": "ok"})
            features.append({"name": "rain_data_ok", "value": str(rk.get("rain_data_ok")), "status": "ok" if rk.get("rain_data_ok") else "warning"})
        status = "ok" if rk.get("rain_data_ok") else "warning"
        results["sources"].append({
            "name": "Rainfall Features",
            "url": INSTANT_RAIN_URL,
            "status": status,
            "message": f"60m: {rk.get('rain_60m', 0)}mm, 120m: {rk.get('rain_120m', 0)}mm",
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Rainfall Features",
            "url": INSTANT_RAIN_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ──     6. Gridded Nowcast ────────────────────────────────────────────
    try:
        nc = get_nowcast_rainfall()
        features = []
        if nc is not None:
            features.append({"name": "rain_nowcast", "value": f"{nc:.1f} mm", "status": "ok"})
        status = "ok" if nc is not None else "warning"
        results["sources"].append({
            "name": "Gridded Nowcast",
            "url": "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv",
            "status": status,
            "message": f"Nowcast: {nc:.1f} mm" if nc else "No nowcast data",
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Gridded Nowcast",
            "url": "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv",
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 7. Pressure (HKO CSV) ────────────────────────────────────────
    try:
        df_p = fetch_pressure_live()
        features = []
        if not df_p.empty:
            pk = compute_pressure_kwargs()
            features.append({"name": "pressure_current", "value": f"{pk['pressure_current']:.1f} hPa", "status": "ok"})
            features.append({"name": "pressure_change_60m", "value": f"{pk['pressure_change_60m']:+.1f} hPa", "status": "ok"})
            features.append({"name": "pressure_change_180m", "value": f"{pk['pressure_change_180m']:+.1f} hPa", "status": "ok"})
            status = "ok"
            msg = f"Current: {pk['pressure_current']:.1f} hPa"
        else:
            status = "warning"
            msg = "Empty DataFrame"
        results["sources"].append({
            "name": "Pressure (HKO CSV)",
            "url": HKO_PRESSURE_CSV_URL,
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Pressure (HKO CSV)",
            "url": HKO_PRESSURE_CSV_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 8. Wind (i-Lens DG_WIND) ──────────────────────────────────────
    try:
        df_w = fetch_wind_live()
        features = []
        if not df_w.empty:
            wk = compute_wind_kwargs()
            for fname in ["wind_ref_mean", "wind_ref_max", "wind_victoria_harbour_mean",
                          "wind_highland_mean", "wind_all_change_60m", "wind_kings_park_current"]:
                val = wk.get(fname, 0.0)
                features.append({"name": fname, "value": f"{val:.1f} km/h", "status": "ok"})
            status = "ok"
            msg = f"Ref mean: {wk.get('wind_ref_mean', 0.0):.1f} km/h"
        else:
            status = "warning"
            msg = "No stations parsed"
        results["sources"].append({
            "name": "Wind (i-Lens DG_WIND)",
            "url": WIND_INSTANT_URL,
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Wind (i-Lens DG_WIND)",
            "url": WIND_INSTANT_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 9. Forecast Freshness (HKO XML) ───────────────────────────────
    try:
        url = HKO_FORECAST_URL_TEMPLATE.format(ts=int(_time.time() * 1000))
        r = _req.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        model_time_str = data.get("ModelTime", "")
        daily = data.get("DailyForecast", [])
        forecast_max = daily[0].get("ForecastMaximumTemperature") if daily else None
        forecast_min = daily[0].get("ForecastMinimumTemperature") if daily else None
        features = []
        if model_time_str:
            from app.config import HKT_OFFSET
            model_dt = pd.to_datetime(str(model_time_str), format="%Y%m%d%H") + HKT_OFFSET
            age_min = (hkt_now() - model_dt).total_seconds() / 60
            age_stale = age_min > 180
            features.append({"name": "forecast_age_minutes", "value": f"{age_min:.0f} min", "status": "warning" if age_stale else "ok"})
            features.append({"name": "forecast_max_temp", "value": f"{forecast_max} °C" if forecast_max else "None", "status": "ok" if forecast_max else "warning"})
            features.append({"name": "forecast_min_temp", "value": f"{forecast_min} °C" if forecast_min else "None", "status": "ok" if forecast_min else "warning"})
            status = "warning" if age_stale else "ok"
            msg = f"Age: {age_min:.0f}min, Max: {forecast_max}°C"
        else:
            status = "warning"
            msg = "No ModelTime field"
        results["sources"].append({
            "name": "Forecast Freshness",
            "url": HKO_FORECAST_URL_TEMPLATE,
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Forecast Freshness",
            "url": HKO_FORECAST_URL_TEMPLATE,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 10. Polymarket API ─────────────────────────────────────────────
    try:
        from app.services.market_service import fetch_today_event
        event = fetch_today_event(hkt_now().strftime("%Y-%m-%d"))
        features = []
        if event:
            features.append({"name": "event", "value": event.get("slug", ""), "status": "ok"})
        status = "ok" if event else "error"
        results["sources"].append({
            "name": "Polymarket API",
            "url": PM_SEARCH_URL,
            "status": status,
            "message": f"Event: {event.get('slug')}" if event else "No event found",
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Polymarket API",
            "url": PM_SEARCH_URL,
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # ── 11. Minute Buffer (df_today) ──────────────────────────────────
    try:
        state = get_intraday_state(hkt_now().strftime("%Y%m%d"))
        features = []
        if state and state.get("df_today") is not None:
            df = state["df_today"]
            row_count = len(df)
            latest_time = df["datetime"].max() if not df.empty else None
            temp_std = float(df["temp"].std()) if not df.empty else 0.0
            temp_val = float(state.get("temp_now", df["temp"].iloc[-1] if not df.empty else 0))
            features.append({"name": "temp_current", "value": f"{temp_val:.1f} °C", "status": "ok"})
            features.append({"name": "rows", "value": str(row_count), "status": "ok" if row_count >= 30 else "warning"})
            features.append({"name": "latest_timestamp", "value": str(latest_time), "status": "ok"})
            features.append({"name": "temp_std", "value": f"{temp_std:.2f}", "status": "ok" if temp_std > 0.1 else "warning"})
            if row_count >= 30 and temp_std > 0.1:
                status = "ok"
            elif row_count > 0:
                status = "warning"
            else:
                status = "error"
            msg = f"Rows: {row_count}, Latest: {latest_time}, Temp Std: {temp_std:.2f}"
        else:
            status = "error"
            msg = "No intraday state available"
        results["sources"].append({
            "name": "Minute Buffer (df_today)",
            "url": "",
            "status": status, "message": msg,
            "last_update": _now_str(),
            "features": features,
        })
    except Exception as e:
        results["sources"].append({
            "name": "Minute Buffer (df_today)",
            "url": "",
            "status": "error", "message": str(e),
            "last_update": _now_str(), "features": [],
        })

    # Flatten all features into features_flat table
    for src in results["sources"]:
        domain = src["name"]
        source_url = src["url"]
        updated = src["last_update"]
        for feat in src.get("features", []):
            results["features_flat"].append({
                "domain": domain,
                "feature": feat["name"],
                "source": source_url,
                "value": feat["value"],
                "status": feat["status"],
                "updated": updated,
            })

    return results
