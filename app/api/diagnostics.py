from fastapi import APIRouter
from datetime import datetime
import logging

import pandas as pd

router = APIRouter(prefix="/api/diagnostics", tags=["Diagnostics"])
logger = logging.getLogger(__name__)


@router.get("/sources")
def check_data_sources():
    from app.services.weather_service import (
        fetch_live_hko_temp_rh,
        fetch_hko_data,
        compute_rain_kwargs_live,
        _fetch_rainfall_live_uncached,
        hkt_now,
        get_intraday_state,
    )

    results = {
        "checked_at": datetime.now().isoformat(),
        "sources": []
    }

    # 1. HKO Live Temp/RH
    try:
        temp_rh = fetch_live_hko_temp_rh()
        if temp_rh and isinstance(temp_rh, tuple) and len(temp_rh) >= 3:
            status = "ok"
            msg = f"Temp: {temp_rh[1]}°C, RH: {temp_rh[2]}%"
        else:
            status = "error"
            msg = "Failed to fetch or invalid format"
        results["sources"].append({"name": "HKO Live Temp/RH", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "HKO Live Temp/RH", "status": "error", "message": str(e)})

    # 2. HKO Daily Data
    try:
        hko = fetch_hko_data(hkt_now().strftime("%Y%m%d"))
        if hko and isinstance(hko, dict):
            max_val = hko.get('max_since_midnight')
            status = "ok" if max_val is not None else "warning"
            msg = f"Max so far: {max_val}" if max_val is not None else "No data"
        else:
            status = "error"
            msg = "No data or invalid format"
        results["sources"].append({"name": "HKO Daily Data", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "HKO Daily Data", "status": "error", "message": str(e)})

    # 3. i-Lens Rainfall API (use uncached version to avoid cache pollution)
    try:
        rain_df = _fetch_rainfall_live_uncached()
        if rain_df is None or not isinstance(rain_df, pd.DataFrame):
            status = "error"
            msg = f"Invalid type: {type(rain_df).__name__}"
        elif rain_df.empty:
            status = "error"
            msg = "Empty DataFrame - no rainfall data"
        else:
            latest_time = rain_df.iloc[-1]['datetime']
            status = "ok"
            msg = f"Latest data at {latest_time}. Rows: {len(rain_df)}"
        results["sources"].append({"name": "i-Lens Rainfall API", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "i-Lens Rainfall API", "status": "error", "message": str(e)})

    # 4. Rainfall Features
    try:
        rain_kwargs = compute_rain_kwargs_live()
        if isinstance(rain_kwargs, dict):
            status = "ok" if rain_kwargs.get("rain_data_ok") else "warning"
            msg = f"60m: {rain_kwargs.get('rain_60m', 0)}mm, 120m: {rain_kwargs.get('rain_120m', 0)}mm"
        else:
            status = "error"
            msg = f"Unexpected type: {type(rain_kwargs).__name__}"
        results["sources"].append({"name": "Rainfall Features", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "Rainfall Features", "status": "error", "message": str(e)})

    # 5. Polymarket API
    try:
        from app.services.market_service import fetch_today_event
        event = fetch_today_event(hkt_now().strftime("%Y-%m-%d"))
        status = "ok" if event else "error"
        msg = f"Event: {event.get('slug')}" if event else "No event found"
        results["sources"].append({"name": "Polymarket API", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "Polymarket API", "status": "error", "message": str(e)})

    # 6. Minute Buffer (df_today) - intraday history data
    try:
        state = get_intraday_state(hkt_now().strftime("%Y%m%d"))

        if state and state.get("df_today") is not None:
            df = state["df_today"]
            row_count = len(df)
            latest_time = df["datetime"].max() if not df.empty else None
            temp_std = float(df["temp"].std()) if not df.empty else 0.0

            # Determine health status
            if row_count >= 30 and temp_std > 0.1:
                status = "ok"
            elif row_count > 0:
                status = "warning"
            else:
                status = "error"

            msg = f"Rows: {row_count}, Latest: {latest_time}, Temp Std: {temp_std:.2f}"
            results["sources"].append({
                "name": "Minute Buffer (df_today)",
                "status": status,
                "message": msg
            })
        else:
            results["sources"].append({
                "name": "Minute Buffer (df_today)",
                "status": "error",
                "message": "No intraday state available"
            })
    except Exception as e:
        results["sources"].append({
            "name": "Minute Buffer (df_today)",
            "status": "error",
            "message": str(e)
        })

    return results