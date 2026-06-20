import { useCallback, useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchEvent, fetchTodayEvent, fetchPredictions, fetchWeatherNow } from "../api/client"
import ModelGrid, { LABEL_MAP } from "../components/ModelGrid"
import BucketChart from "../components/BucketChart"
import ComparisonChart from "../components/ComparisonChart"
import type { ModelPrediction } from "../types"
import { Eye, EyeOff, ChevronDown, GitCompare } from "lucide-react"

const TODAY = new Date().toISOString().slice(0, 10)

export default function Hub() {
  const [date, setDate] = useState(TODAY)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [order, setOrder] = useState<string[] | null>(null)
  const [isMinTemp, setIsMinTemp] = useState(false)
    const [dropdownOpen, setDropdownOpen] = useState(false)
    const [weatherDropdownOpen, setWeatherDropdownOpen] = useState(false)
    const [compareMode, setCompareMode] = useState(false)

   const [visibleKeys, setVisibleKeys] = useState<Set<string> | null>(() => {
     try {
       const saved = localStorage.getItem("visibleModelKeys");
       return saved ? new Set(JSON.parse(saved)) : null;
     } catch {
       return null;
     }
   });

const [weatherElements, setWeatherElements] = useState<Set<string>>(() => {
      try {
        const saved = localStorage.getItem("weatherElements");
        return saved ? new Set(JSON.parse(saved)) : new Set(["current_temp", "max_so_far", "min_so_far", "rain_60m", "rain_120m", "rain_accumulated", "rain_nowcast"]);
      } catch {
        return new Set(["current_temp", "max_so_far", "min_so_far", "rain_60m", "rain_120m", "rain_accumulated", "rain_nowcast"]);
      }
    });

  const { data, isLoading } = useQuery({
    queryKey: ["predictions", date],
    queryFn: () => fetchPredictions(date, false),
    refetchInterval: 120_000,
  })

  const { data: weatherData } = useQuery({
    queryKey: ["weather", date],
    queryFn: () => fetchWeatherNow(date),
    enabled: !!date,
    refetchInterval: 120_000,
  })



  const { data: todayEventData } = useQuery({
    queryKey: ["today-event", date],
    queryFn: () => fetchTodayEvent(date),
    enabled: !!date,
    refetchInterval: 120_000,
  })

  const eventSlug = todayEventData?.event?.slug ?? null
  const { data: eventData } = useQuery({
    queryKey: ["event", eventSlug],
    queryFn: () => fetchEvent(eventSlug!, isMinTemp),
    enabled: !!eventSlug,
    refetchInterval: 120_000,
  })

  const marketPrices = useMemo(() => {
    const p = eventData?.prices ?? {}
    const decoded: Record<string, number> = {}
    for (const [k, v] of Object.entries(p)) {
      const decodedKey = k.replace(/^&lt;/, "<").replace(/^&gt;/, ">").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      decoded[decodedKey] = v
    }
    return decoded
  }, [eventData?.prices])

  const models = data?.models
  const entries = useMemo(() => (models ? Object.entries(models) : []), [models])

  useEffect(() => {
    if (activeKey && models && !models[activeKey]) setActiveKey(null)
  }, [activeKey, models])

  const sorted = useCallback(
    (entries: [string, ModelPrediction][]) => {
      if (!order) return entries
      const map = Object.fromEntries(entries)
      const ordered = order
        .filter((k) => k in map)
        .map((k) => [k, map[k]] as [string, ModelPrediction])
      const rest = entries.filter(([k]) => !order.includes(k))
      return [...ordered, ...rest]
    },
    [order],
  )(entries)

  const visibleModels = useMemo(() => {
    if (!visibleKeys) return sorted
    return sorted.filter(([k]) => visibleKeys.has(k))
  }, [sorted, visibleKeys])

  const currentActiveIsVisible = activeKey && (visibleKeys?.has(activeKey) ?? true);
  const finalActiveKey = currentActiveIsVisible ? activeKey : visibleModels[0]?.[0] ?? null;
  const activeModel = finalActiveKey && models?.[finalActiveKey] ? models[finalActiveKey] : undefined;

  const handleToggleVisible = (key: string) => {
    setVisibleKeys((prev) => {
      const current = prev ? new Set(prev) : new Set(entries.map(([k]) => k))
      if (current.has(key)) {
        current.delete(key)
      } else {
        current.add(key)
      }
      localStorage.setItem("visibleModelKeys", JSON.stringify(Array.from(current)));
      return current
    })
  }

  const expectedValue = activeModel?.mean?.toFixed(1) ?? "--"
  const marketAvg = useMemo(() => {
    const vals = Object.values(marketPrices ?? {})
    if (vals.length === 0) return "--"
    return (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2)
  }, [marketPrices])

  const subtitle = useMemo(() => {
    if (!eventData?.title) return "Weather predictions"
    return eventData.title.replace("highest", "TMAX").replace("lowest", "TMIN")
  }, [eventData?.title])

  const weatherElems = {
    current_temp: { label: "Current", value: `${weatherData?.temp?.toFixed(1) ?? "--"}°C`, color: "text-cyan-400", border: "border-cyan-500/50" },
    max_so_far: { label: "Max Today", value: `${weatherData?.max_today?.toFixed(1) ?? "--"}°C`, color: "text-rose-400", border: "border-rose-500/50" },
    min_so_far: { label: "Min Today", value: `${weatherData?.min_today?.toFixed(1) ?? "--"}°C`, color: "text-blue-400", border: "border-blue-500/50" },
    humidity: { label: "Humidity", value: `${weatherData?.humidity ?? "--"}%`, color: "text-emerald-400", border: "border-emerald-500/50" },
    rain_60m: { label: "Rain 60m", value: weatherData?.rain_60m != null ? `${weatherData.rain_60m.toFixed(1)}mm` : "--", color: "text-sky-400", border: "border-sky-500/50" },
    rain_120m: { label: "Rain 120m", value: weatherData?.rain_120m != null ? `${weatherData.rain_120m.toFixed(1)}mm` : "--", color: "text-sky-400", border: "border-sky-500/50" },
    rain_accumulated: { label: "Rain Today", value: weatherData?.rain_accumulated_today != null ? `${weatherData.rain_accumulated_today.toFixed(1)}mm` : "--", color: "text-sky-400", border: "border-sky-500/50" },
    rain_nowcast: { label: "Nowcast", value: weatherData?.rain_nowcast != null ? `${weatherData.rain_nowcast.toFixed(1)}mm` : "--", color: "text-sky-400", border: "border-sky-500/50" },
  }

  const handleToggleWeatherElement = (key: string) => {
    setWeatherElements((prev) => {
      const current = new Set(prev)
      if (current.has(key)) current.delete(key)
      else current.add(key)
      localStorage.setItem("weatherElements", JSON.stringify(Array.from(current)))
      return current
    })
  }

  const allBuckets = useMemo(() => {
    const set = new Set<string>()
    entries.forEach(([_, pred]) => {
      Object.keys(pred.probs ?? {}).forEach(b => set.add(b))
    })
    Object.keys(marketPrices ?? {}).forEach(b => set.add(b))
    return Array.from(set).sort((a, b) => {
      const parse = (s: string) => {
        if (s.startsWith("<")) return -999
        if (s.startsWith(">=")) return 999
        if (s.startsWith(">")) return 999
        const num = parseFloat(s.split("-")[0])
        return isNaN(num) ? 0 : num
      }
      return parse(a) - parse(b)
    })
  }, [entries, marketPrices])

return (
    <div className="flex h-full w-full overflow-hidden">
      <aside className="w-1/3 min-w-[300px] max-w-[400px] h-full flex flex-col border-r border-slate-800 bg-slate-950/50 backdrop-blur-sm">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between relative">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Model Matrix</h2>
          
          <div className="relative">
            <button 
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700"
            >
              <Eye size={12} />
              Models
              <ChevronDown size={12} />
            </button>
            
            {dropdownOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setDropdownOpen(false)} />
                
                <div className="absolute right-0 top-full mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 p-2">
                  <div className="text-xs text-slate-500 px-2 py-1 border-b border-slate-700 mb-1">Show on Dashboard</div>
                  {entries.map(([k, _]) => {
                    const isVisible = visibleKeys ? visibleKeys.has(k) : true;
                    return (
                      <button
                        key={k}
                        onClick={() => handleToggleVisible(k)}
                        className="flex items-center gap-2 w-full text-left px-2 py-1.5 rounded hover:bg-slate-700 text-sm text-slate-300"
                      >
                        {isVisible ? <Eye size={14} className="text-cyan-400" /> : <EyeOff size={14} className="text-slate-500" />}
                        {LABEL_MAP[k] ?? k}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>
          
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={["flex items-center gap-1 text-xs px-2 py-1 rounded-md transition-colors",
              compareMode ? "bg-cyan-500/20 text-cyan-400" : "bg-slate-800 text-slate-400 hover:bg-slate-700"
            ].join(" ")}
            title="Compare multiple models"
          >
            <GitCompare size={12} />
            <span className="hidden sm:inline">Compare</span>
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
           {isLoading ? (
             <div className="space-y-3">
               {[1, 2, 3].map((i) => (
                 <div key={i} className="h-24 rounded-lg shimmer-card" />
               ))}
             </div>
           ) : (
             <ModelGrid
               models={visibleModels}
               activeKey={finalActiveKey}
               visibleKeys={visibleKeys}
               onSelect={(k) => setActiveKey(k)}
               onReorder={(keys) => setOrder(keys)}
               onToggleVisible={handleToggleVisible}
             />
           )}
         </div>
      </aside>

      <main className="flex-1 h-full overflow-y-auto p-8">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold text-slate-50">Weather Hub</h1>
            <p className="text-sm text-slate-500">{subtitle}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative">
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="bg-slate-900/50 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-500 transition-colors"
              />
            </div>
            <div className="flex gap-1 bg-slate-900/50 p-1 rounded-full border border-slate-800">
              <button
                className={[
                  "px-4 py-1.5 rounded-full text-xs font-semibold transition-all",
                  !isMinTemp ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-slate-200",
                ].join(" ")}
                onClick={() => setIsMinTemp(false)}
              >
                TMAX
              </button>
              <button
                className={[
                  "px-4 py-1.5 rounded-full text-xs font-semibold transition-all",
                  isMinTemp ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-slate-200",
                ].join(" ")}
                onClick={() => setIsMinTemp(true)}
              >
                TMIN
              </button>
            </div>
          </div>
        </header>

        <section className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Current Weather</h2>
            <div className="relative">
              <button
                onClick={() => setWeatherDropdownOpen(!weatherDropdownOpen)}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                <Eye size={12} />
                Elements
                <ChevronDown size={12} />
              </button>
              {weatherDropdownOpen && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setWeatherDropdownOpen(false)} />
                  <div className="absolute right-0 top-full mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 p-2">
                    <div className="text-xs text-slate-500 px-2 py-1 border-b border-slate-700 mb-1">Show Elements</div>
                    {Object.entries(weatherElems).map(([key, { label }]) => {
                      const isVisible = weatherElements.has(key)
                      return (
                        <button
                          key={key}
                          onClick={() => handleToggleWeatherElement(key)}
                          className="flex items-center gap-2 w-full text-left px-2 py-1.5 rounded hover:bg-slate-700 text-sm text-slate-300"
                        >
                          {isVisible ? <Eye size={14} className="text-cyan-400" /> : <EyeOff size={14} className="text-slate-500" />}
                          {label}
                        </button>
                      )
                    })}
                  </div>
                </>
              )}
            </div>
          </div>
<div className="grid grid-cols-2 md:grid-cols-8 gap-4">
             {Object.entries(weatherElems).map(([key, { label, value, color, border }]) => {
               if (!weatherElements.has(key)) return null
               return (
                 <div key={key} className={`bg-slate-900/40 border-t-2 ${border} rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1`}>
                   <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</div>
                   <div className={`text-2xl font-bold ${color} tabular-nums`}>{value}</div>
                 </div>
               )
             })}
           </div>
        </section>

        <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-900/40 border-t-2 border-cyan-500/50 rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Expected Value</div>
            <div className="text-2xl font-bold text-cyan-400 tabular-nums">{expectedValue}°C</div>
          </div>
          <div className="bg-slate-900/40 border-t-2 border-violet-500/50 rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Market Avg</div>
            <div className="text-2xl font-bold text-violet-400 tabular-nums">${marketAvg}</div>
          </div>
          <div className="bg-slate-900/40 border-t-2 border-emerald-500/50 rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Edge</div>
            <div className="text-2xl font-bold text-emerald-400 tabular-nums">--</div>
          </div>
          <div className="bg-slate-900/40 border-t-2 border-slate-600/50 rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Kelly Bet</div>
            <div className="text-2xl font-bold text-slate-100 tabular-nums">--</div>
          </div>
        </section>

        <section className="bg-slate-900/40 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-[0_0_30px_-5px_rgba(6,182,212,0.1)] transition-all duration-300">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-lg font-semibold text-slate-100">
                {compareMode ? "Model Comparison" : "Probability by Bucket"}
              </h2>
              {compareMode && (
                <button
                  onClick={() => setCompareMode(false)}
                  className="text-xs text-slate-400 hover:text-slate-200 mt-1"
                >
                  Exit Compare
                </button>
              )}
            </div>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-cyan-500"></span>
                <span className="text-slate-400">Model Probability</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-violet-500"></span>
                <span className="text-slate-400">Market Price</span>
              </div>
            </div>
          </div>
          
          {compareMode ? (
            <ComparisonChart
              models={visibleModels}
              marketPrices={marketPrices}
              allBuckets={allBuckets}
            />
          ) : activeModel && activeModel.probs ? (
            <BucketChart
              modelProbs={activeModel.probs}
              marketPrices={marketPrices}
              allBuckets={allBuckets}
            />
          ) : (
            <div className="h-80 flex items-center justify-center text-slate-500">
              Select a model to view probabilities
            </div>
          )}
        </section>

      </main>
    </div>
  )
}