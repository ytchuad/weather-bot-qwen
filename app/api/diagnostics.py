from fastapi import APIRouter
from datetime import datetime
import logging

from app.services.weather_service import (
    fetch_live_hko_temp_rh,
    fetch_hko_data,
    hkt_now,
    compute_rain_kwargs_live,
    fetch_rainfall_live,
)

router = APIRouter(prefix="/api/diagnostics", tags=["Diagnostics"])
logger = logging.getLogger(__name__)


@router.get("/sources")
def check_data_sources():
    results = {
        "checked_at": datetime.now().isoformat(),
        "sources": []
    }

    # 1. HKO Live Temp/RH
    try:
        temp_rh = fetch_live_hko_temp_rh()
        status = "ok" if temp_rh else "error"
        msg = f"Temp: {temp_rh[1]}°C, RH: {temp_rh[2]}%" if temp_rh else "Failed to fetch"
        results["sources"].append({"name": "HKO Live Temp/RH", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "HKO Live Temp/RH", "status": "error", "message": str(e)})

    # 2. HKO Daily Data
    try:
        hko = fetch_hko_data(hkt_now().strftime("%Y%m%d"))
        status = "ok" if hko else "error"
        msg = f"Max so far: {hko.get('max_since_midnight')}" if hko else "No data"
        results["sources"].append({"name": "HKO Daily Data", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "HKO Daily Data", "status": "error", "message": str(e)})

    # 3. i-Lens Rainfall API
    try:
        rain_df = fetch_rainfall_live()
        if rain_df is not None and not rain_df.empty:
            latest_time = rain_df.iloc[-1]['datetime']
            status = "ok"
            msg = f"Latest data at {latest_time}. Rows: {len(rain_df)}"
        else:
            status = "error"
            msg = "Failed to fetch or empty data"
        results["sources"].append({"name": "i-Lens Rainfall API", "status": status, "message": msg})
    except Exception as e:
        results["sources"].append({"name": "i-Lens Rainfall API", "status": "error", "message": str(e)})

    # 4. Rainfall Features
    try:
        rain_kwargs = compute_rain_kwargs_live()
        status = "ok" if rain_kwargs.get("rain_data_ok") else "warning"
        msg = f"60m: {rain_kwargs['rain_60m']}mm, 120m: {rain_kwargs['rain_120m']}mm"
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

    return results