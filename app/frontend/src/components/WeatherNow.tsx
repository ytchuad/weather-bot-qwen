import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Thermometer, Droplets, Calendar, CloudRain } from "lucide-react"
import { fetchWeatherNow } from "../api/client"

type VisibilityState = {
  rain_60m: boolean
  rain_120m: boolean
  rain_accumulated: boolean
  rain_nowcast: boolean
}

export default function WeatherNow({ date }: { date: string }) {
  const [visibility, setVisibility] = useState<VisibilityState>({
    rain_60m: true,
    rain_120m: true,
    rain_accumulated: true,
    rain_nowcast: true,
  })

  const { data, isLoading } = useQuery({
    queryKey: ["weatherNow", date],
    queryFn: () => fetchWeatherNow(date),
    refetchInterval: 60_000,
  })

  if (isLoading || !data)
    return (
      <div className="h-24 rounded-xl bg-white/[0.03] animate-pulse" />
    )

  const toggleVisibility = (key: keyof VisibilityState) => {
    setVisibility(v => ({ ...v, [key]: !v[key] }))
  }

  return (
    <div className="flex flex-wrap gap-6 rounded-xl bg-white/[0.03] border border-white/5 px-5 py-4">
      <div className="flex items-center gap-3">
        <Calendar size={20} className="text-[#00E5FF]" />
        <div>
          <div className="text-xs text-white/40 uppercase tracking-wider">
            Date
          </div>
          <div className="text-sm font-semibold tabular-nums">
            {data.date ?? date}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Thermometer size={20} className="text-[#F43F5E]" />
        <div>
          <div className="text-xs text-white/40 uppercase tracking-wider">
            Current
          </div>
          <div className="text-xl font-semibold tabular-nums">
            {data.temp != null ? `${data.temp.toFixed(1)}°C` : "--"}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Droplets size={20} className="text-[#3B82F6]" />
        <div>
          <div className="text-xs text-white/40 uppercase tracking-wider">
            Humidity
          </div>
          <div className="text-xl font-semibold tabular-nums">
            {data.humidity != null ? `${data.humidity.toFixed(0)}%` : "--"}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Thermometer size={20} className="text-[#22D3EE]" />
        <div>
          <div className="text-xs text-white/40 uppercase tracking-wider">
            Today
          </div>
          <div className="text-sm tabular-nums text-white/70">
            {data.min_today != null ? `${data.min_today.toFixed(1)}` : "--"} /{" "}
            {data.max_today != null ? `${data.max_today.toFixed(1)}°C` : "--"}
          </div>
        </div>
      </div>
      {data.forecast != null && (
        <div className="flex items-center gap-3">
          <Thermometer size={20} className="text-[#A78BFA]" />
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">
              Forecast
            </div>
            <div className="text-sm tabular-nums text-white/70">
              {data.forecast.toFixed(1)}°C
            </div>
          </div>
        </div>
      )}
      {visibility.rain_60m && data.rain_60m != null && (
        <div className="flex items-center gap-3">
          <CloudRain size={20} className="text-cyan-400 cursor-pointer" onClick={() => toggleVisibility("rain_60m")} />
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">
              Rain 60m
            </div>
            <div className="text-sm font-semibold tabular-nums text-cyan-400">
              {data.rain_60m.toFixed(1)} mm
            </div>
          </div>
        </div>
      )}
      {visibility.rain_120m && data.rain_120m != null && (
        <div className="flex items-center gap-3">
          <CloudRain size={20} className="text-cyan-400 cursor-pointer" onClick={() => toggleVisibility("rain_120m")} />
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">
              Rain 120m
            </div>
            <div className="text-sm font-semibold tabular-nums text-cyan-400">
              {data.rain_120m.toFixed(1)} mm
            </div>
          </div>
        </div>
      )}
      {visibility.rain_accumulated && data.rain_accumulated_today != null && (
        <div className="flex items-center gap-3">
          <CloudRain size={20} className="text-cyan-400 cursor-pointer" onClick={() => toggleVisibility("rain_accumulated")} />
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">
              Rain Today
            </div>
            <div className="text-sm font-semibold tabular-nums text-cyan-400">
              {data.rain_accumulated_today.toFixed(1)} mm
            </div>
          </div>
        </div>
      )}
      {visibility.rain_nowcast && data.rain_nowcast != null && (
        <div className="flex items-center gap-3">
          <CloudRain size={20} className="text-cyan-400 cursor-pointer" onClick={() => toggleVisibility("rain_nowcast")} />
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">
              Nowcast
            </div>
            <div className="text-sm font-semibold tabular-nums text-cyan-400">
              {data.rain_nowcast.toFixed(1)} mm
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
