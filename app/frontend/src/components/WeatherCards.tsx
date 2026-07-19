import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { ArrowDown, ArrowUp, CloudRain, CloudSun, Droplets, Eye, EyeOff, Gauge, Thermometer } from "lucide-react"
import { fetchWeatherNow } from "../api/client"

type CardKey = "temp" | "humidity" | "max_so_far" | "min_so_far" | "rain_60m" | "rain_120m" | "rain_accumulated" | "rain_nowcast"

export default function WeatherCards({ date }: { date: string }) {
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [visibleKeys, setVisibleKeys] = useState<Set<CardKey>>(() => {
    const defaults = new Set<CardKey>(["temp", "humidity", "max_so_far", "min_so_far", "rain_60m"])
    try {
      const saved = localStorage.getItem("weatherCardVisibility")
      return saved ? new Set(JSON.parse(saved)) : defaults
    } catch {
      return defaults
    }
  })

  const { data, isLoading } = useQuery({
    queryKey: ["weatherNow", date],
    queryFn: () => fetchWeatherNow(date),
    refetchInterval: 60_000,
  })

  const cards = useMemo(() => {
    if (!data) return []
    return [
      { key: "temp" as CardKey, label: "Current", value: data.temp != null ? `${data.temp.toFixed(1)}°C` : "--", color: "text-cyan-300", Icon: Thermometer },
      { key: "humidity" as CardKey, label: "Humidity", value: data.humidity != null ? `${data.humidity}%` : "--", color: "text-emerald-300", Icon: Droplets },
      { key: "max_so_far" as CardKey, label: "Max Today", value: data.max_today != null ? `${data.max_today.toFixed(1)}°C` : "--", color: "text-rose-300", Icon: ArrowUp },
      { key: "min_so_far" as CardKey, label: "Min Today", value: data.min_today != null ? `${data.min_today.toFixed(1)}°C` : "--", color: "text-sky-300", Icon: ArrowDown },
      { key: "rain_60m" as CardKey, label: "Rain 60m", value: data.rain_60m != null ? `${data.rain_60m.toFixed(1)}mm` : "--", color: "text-white/55", Icon: CloudRain },
      { key: "rain_120m" as CardKey, label: "Rain 120m", value: data.rain_120m != null ? `${data.rain_120m.toFixed(1)}mm` : "--", color: "text-white/55", Icon: CloudRain },
      { key: "rain_accumulated" as CardKey, label: "Rain Today", value: data.rain_accumulated_today != null ? `${data.rain_accumulated_today.toFixed(1)}mm` : "--", color: "text-sky-300", Icon: CloudSun },
      { key: "rain_nowcast" as CardKey, label: "Nowcast", value: data.rain_nowcast != null ? `${data.rain_nowcast.toFixed(1)}mm` : "--", color: "text-violet-300", Icon: Gauge },
    ]
  }, [data])

  const handleToggleVisible = (key: CardKey) => {
    setVisibleKeys((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      localStorage.setItem("weatherCardVisibility", JSON.stringify([...next]))
      return next
    })
  }

  if (isLoading) return <div className="panel h-[58px] animate-pulse" />

  return (
    <section className="panel px-2 py-1">
      <div className="flex flex-wrap items-center justify-between gap-y-2">
        <div className="flex flex-1 flex-wrap items-center justify-between gap-y-1">
          {cards.filter((card) => visibleKeys.has(card.key)).map((card) => {
            const Icon = card.Icon
            return (
              <div key={card.key} className="flex items-center gap-2 px-3 py-2.5 sm:px-4">
                <Icon className={`h-4 w-4 opacity-80 ${card.color}`} />
                <span className={`mono tnum text-[13px] font-semibold ${card.color}`}>{card.value}</span>
                <span className="text-[11.5px] font-medium text-white/35">{card.label}</span>
              </div>
            )
          })}
          {cards.length > 0 && cards.every((card) => !visibleKeys.has(card.key)) && <span className="px-3 py-2 text-xs text-white/35">No weather elements selected.</span>}
        </div>
        <div className="relative ml-auto">
          <button onClick={() => setDropdownOpen((open) => !open)} className="mono flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] px-2.5 py-1.5 text-[10px] font-semibold tracking-[0.1em] text-white/55 transition-colors hover:text-white/85" title="Toggle weather elements">
            <Eye className="h-3 w-3" /> Elements
          </button>
          {dropdownOpen && (
            <>
              <div className="fixed inset-0 z-[100]" aria-hidden="true" onClick={() => setDropdownOpen(false)} />
              <div className="absolute right-0 top-full z-[110] mt-2 w-52 rounded-xl border border-white/[0.12] bg-[#070b13] p-2 shadow-[0_18px_50px_rgba(0,0,0,0.7)] backdrop-blur-xl">
                <div className="eyebrow px-2 py-1.5">Toggle visibility</div>
                {cards.map((card) => {
                  const visible = visibleKeys.has(card.key)
                  const Icon = card.Icon
                  return (
                    <button key={card.key} onClick={() => handleToggleVisible(card.key)} className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-xs text-white/70 transition-colors hover:bg-white/[0.05]">
                      {visible ? <Eye className="h-3 w-3 text-cyan-300" /> : <EyeOff className="h-3 w-3 text-white/25" />}
                      <Icon className={`h-3.5 w-3.5 ${visible ? card.color : "text-white/25"}`} />
                      <span className={visible ? "text-white/75" : "text-white/35"}>{card.label}</span>
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}
