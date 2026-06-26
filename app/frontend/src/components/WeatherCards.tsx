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
    // ✅ 修正 1：顯式指定 Set 的泛型型別為 <CardKey>
    const defaultKeys = new Set<CardKey>(["temp", "humidity", "max_so_far", "min_so_far", "rain_60m"])
    
    try {
      const saved = localStorage.getItem("weatherCardVisibility")
      if (saved) {
        const parsed: string[] = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) {
          // ✅ 修正 2：將 parsed 斷言為 CardKey[]，並過濾掉無效的 key
          const validKeys = parsed.filter((key): key is CardKey => 
            defaultKeys.has(key as CardKey)
          )
          return validKeys.length > 0 ? new Set(validKeys) : defaultKeys
        }
        return defaultKeys
      }
      return defaultKeys
    } catch {
      return defaultKeys
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
      { key: "temp" as CardKey, label: "Current Temp", value: data.temp != null ? `${data.temp.toFixed(1)}°C` : "--", color: "text-cyan-400", Icon: Thermometer },
      { key: "humidity" as CardKey, label: "Humidity", value: data.humidity != null ? `${data.humidity}%` : "--", color: "text-emerald-400", Icon: Droplets },
      { key: "max_so_far" as CardKey, label: "Max So Far", value: data.max_today != null ? `${data.max_today.toFixed(1)}°C` : "--", color: "text-rose-400", Icon: ArrowUp },
      { key: "min_so_far" as CardKey, label: "Min So Far", value: data.min_today != null ? `${data.min_today.toFixed(1)}°C` : "--", color: "text-blue-400", Icon: ArrowDown },
      { key: "rain_60m" as CardKey, label: "Rain 60m", value: data.rain_60m != null ? `${data.rain_60m.toFixed(1)}mm` : "--", color: "text-sky-400", Icon: CloudRain },
      { key: "rain_120m" as CardKey, label: "Rain 120m", value: data.rain_120m != null ? `${data.rain_120m.toFixed(1)}mm` : "--", color: "text-sky-400", Icon: CloudRain },
      { key: "rain_accumulated" as CardKey, label: "Rain Today", value: data.rain_accumulated_today != null ? `${data.rain_accumulated_today.toFixed(1)}mm` : "--", color: "text-indigo-400", Icon: CloudSun },
      { key: "rain_nowcast" as CardKey, label: "Nowcast", value: data.rain_nowcast != null ? `${data.rain_nowcast.toFixed(1)}mm` : "--", color: "text-indigo-400", Icon: Gauge },
    ]
  }, [data])

  if (isLoading) {
    return <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">{[1, 2, 3, 4].map((i) => <div key={i} className="h-20 rounded-md bg-white/[0.02] animate-pulse" />)}</div>
  }

  const visibleCards = cards.filter(c => visibleKeys.has(c.key))

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        {/* 提亮字体至 text-slate-300 */}
        <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2">
          <span className="w-4 h-px bg-slate-500"></span> Live Conditions
        </h2>
        
        <div className="relative">
          <button 
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-md bg-white/[0.03] border border-white/[0.06] text-slate-300 hover:bg-white/[0.06] transition-colors"
          >
            <Eye size={12} />
            Elements
            <ChevronDown size={12} className={`transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
          </button>
          
          {dropdownOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setDropdownOpen(false)} />
              <div className="absolute right-0 top-full mt-2 w-48 bg-[#0f1013] border border-white/[0.08] rounded-md shadow-xl z-20 p-2 max-h-80 overflow-y-auto custom-scrollbar">
                <div className="text-[10px] text-slate-400 px-2 py-1 border-b border-white/5 mb-1 uppercase tracking-widest">Toggle Visibility</div>
                {cards.map(card => {
                  const isVisible = visibleKeys.has(card.key)
                  const Icon = card.Icon
                  return (
                    <button
                      key={card.key}
                      onClick={() => handleToggleVisible(card.key)}
                      className="flex items-center gap-3 w-full text-left px-2 py-2 rounded-sm hover:bg-white/5 text-xs text-slate-300 transition-colors"
                    >
                      {isVisible ? <Eye size={12} className="text-cyan-400" /> : <EyeOff size={12} className="text-slate-500" />}
                      <Icon size={14} className={isVisible ? card.color : "text-slate-500"} />
                      <span className={isVisible ? "text-slate-300" : "text-slate-500"}>{card.label}</span>
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
      
      {visibleCards.length === 0 ? (
        <div className="text-center text-slate-400 text-xs py-6 border border-dashed border-white/5 rounded-md">
          No weather elements selected. Click "Elements" to display cards.
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {visibleCards.map(card => {
            const Icon = card.Icon
            return (
              <div 
                key={card.key} 
                className="bg-[#0f1013] border border-white/[0.06] rounded-md p-4 flex items-center gap-4 transition-all duration-300 hover:border-white/10 group"
                style={{ boxShadow: "0 10px 40px -10px rgba(0,0,0,0.8), inset 0 1px 0 0 rgba(255,255,255,0.08)" }}
              >
                <div className={`p-2.5 rounded-md bg-white/[0.03] border border-white/[0.04] ${card.color}`}>
                  <Icon size={18} />
                </div>
                <div className="flex flex-col">
                  {/* 提亮标签字体至 text-slate-400 */}
                  <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">{card.label}</div>
                  <div className={`text-xl font-light tabular-nums mono ${card.color}`}>
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