import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { Thermometer, Droplets, CloudRain, Eye, EyeOff, ChevronDown, Gauge, ArrowUp, ArrowDown, CloudSun } from "lucide-react"
import { fetchWeatherNow } from "../api/client"

type CardKey = 
  | "temp" 
  | "humidity" 
  | "max_so_far" 
  | "min_so_far" 
  | "rain_60m" 
  | "rain_120m" 
  | "rain_accumulated" 
  | "rain_nowcast"

export default function WeatherCards({ date }: { date: string }) {
  const [dropdownOpen, setDropdownOpen] = useState(false)
  
  const [visibleKeys, setVisibleKeys] = useState<Set<CardKey>>(() => {
    try {
      const saved = localStorage.getItem("weatherCardVisibility")
      return saved ? new Set(JSON.parse(saved)) : new Set(["temp", "humidity", "max_so_far", "min_so_far", "rain_60m"])
    } catch {
      return new Set(["temp", "humidity", "max_so_far", "min_so_far", "rain_60m"])
    }
  })

  const { data, isLoading } = useQuery({
    queryKey: ["weatherNow", date],
    queryFn: () => fetchWeatherNow(date),
    refetchInterval: 60_000,
  })

  const handleToggleVisible = (key: CardKey) => {
    setVisibleKeys(prev => {
      const current = new Set(prev)
      if (current.has(key)) {
        current.delete(key)
      } else {
        current.add(key)
      }
      localStorage.setItem("weatherCardVisibility", JSON.stringify(Array.from(current)))
      return current
    })
  }

  const cards = useMemo(() => {
    if (!data) return []
    
    return [
      { 
        key: "temp" as CardKey, 
        label: "Current", 
        value: data.temp != null ? `${data.temp.toFixed(1)}°C` : "--", 
        color: "text-cyan-400", 
        border: "border-cyan-500/50",
        Icon: Thermometer 
      },
      { 
        key: "humidity" as CardKey, 
        label: "Humidity", 
        value: data.humidity != null ? `${data.humidity}%` : "--", 
        color: "text-emerald-400", 
        border: "border-emerald-500/50",
        Icon: Droplets 
      },
      { 
        key: "max_so_far" as CardKey, 
        label: "Max So Far", 
        value: data.max_today != null ? `${data.max_today.toFixed(1)}°C` : "--", 
        color: "text-rose-400", 
        border: "border-rose-500/50",
        Icon: ArrowUp 
      },
      { 
        key: "min_so_far" as CardKey, 
        label: "Min So Far", 
        value: data.min_today != null ? `${data.min_today.toFixed(1)}°C` : "--", 
        color: "text-blue-400", 
        border: "border-blue-500/50",
        Icon: ArrowDown 
      },
      { 
        key: "rain_60m" as CardKey, 
        label: "Rain 60m", 
        value: data.rain_60m != null ? `${data.rain_60m.toFixed(1)}mm` : "--", 
        color: "text-sky-400", 
        border: "border-sky-500/50",
        Icon: CloudRain 
      },
      { 
        key: "rain_120m" as CardKey, 
        label: "Rain 120m", 
        value: data.rain_120m != null ? `${data.rain_120m.toFixed(1)}mm` : "--", 
        color: "text-sky-400", 
        border: "border-sky-500/50",
        Icon: CloudRain 
      },
      { 
        key: "rain_accumulated" as CardKey, 
        label: "Rain Today", 
        value: data.rain_accumulated_today != null ? `${data.rain_accumulated_today.toFixed(1)}mm` : "--", 
        color: "text-indigo-400", 
        border: "border-indigo-500/50",
        Icon: CloudSun 
      },
      { 
        key: "rain_nowcast" as CardKey, 
        label: "Nowcast", 
        value: data.rain_nowcast != null ? `${data.rain_nowcast.toFixed(1)}mm` : "--", 
        color: "text-indigo-400", 
        border: "border-indigo-500/50",
        Icon: Gauge 
      },
    ]
  }, [data])

  // 根據卡片數量動態決定網格欄數
  const getGridColsClass = (count: number) => {
    if (count <= 1) return "grid-cols-1"
    if (count === 2) return "grid-cols-2"
    if (count === 3) return "grid-cols-1 sm:grid-cols-3"
    if (count === 4) return "grid-cols-2 sm:grid-cols-4"
    return "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5"
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 rounded-xl bg-white/[0.03] animate-pulse" />
        ))}
      </div>
    )
  }

  const visibleCards = cards.filter(c => visibleKeys.has(c.key))
  const gridClass = getGridColsClass(visibleCards.length)

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Current Weather</h2>
        
        <div className="relative">
          <button 
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700"
          >
            <Eye size={12} />
            Elements
            <ChevronDown size={12} />
          </button>
          
          {dropdownOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setDropdownOpen(false)} />
              <div className="absolute right-0 top-full mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 p-2 max-h-80 overflow-y-auto">
                <div className="text-xs text-slate-500 px-2 py-1 border-b border-slate-700 mb-1">Show Elements</div>
                {cards.map(card => {
                  const isVisible = visibleKeys.has(card.key)
                  const Icon = card.Icon
                  return (
                    <button
                      key={card.key}
                      onClick={() => handleToggleVisible(card.key)}
                      className="flex items-center gap-2 w-full text-left px-2 py-1.5 rounded hover:bg-slate-700 text-sm text-slate-300"
                    >
                      {isVisible ? <Eye size={14} className="text-cyan-400" /> : <EyeOff size={14} className="text-slate-500" />}
                      <Icon size={14} className={isVisible ? card.color : "text-slate-600"} />
                      {card.label}
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
      
      {visibleCards.length === 0 ? (
        <div className="text-center text-slate-500 text-sm py-4 border border-dashed border-slate-800 rounded-lg">
          No weather elements selected. Click "Elements" to display cards.
        </div>
      ) : (
        <div className={`grid ${gridClass} gap-4 justify-items-center`}>
          {visibleCards.map(card => {
            const Icon = card.Icon
            return (
              <div 
                key={card.key} 
                className={`w-full flex items-center gap-3 bg-slate-900/40 border-t-2 ${card.border} rounded-xl p-3 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1`}
              >
                <Icon size={18} className={card.color} />
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wider">{card.label}</div>
                  <div className={`text-sm font-semibold tabular-nums ${card.color}`}>
                    {card.value}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}