import { useState, useMemo, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import ModelGrid from "../components/ModelGrid"
import WeatherCards from "../components/WeatherCards"
import BucketChart from "../components/BucketChart"
import ComparisonChart from "../components/ComparisonChart"
import ModelsComparisonChart from "../components/ModelsComparisonChart"
import { fetchEvent, fetchTodayEvent, fetchPredictions } from "../api/client"
import { GitCompare } from "lucide-react"

// 解析溫度桶的中間值，用於計算市場加權平均溫度
function parseBucketMidpoint(bucket: string): number | null {
  if (bucket.startsWith("<") || bucket.includes("or below")) {
    const match = bucket.match(/(\d+)/)
    return match ? parseFloat(match[1]) - 0.5 : null
  }
  if (bucket.startsWith(">=") || bucket.includes("or higher")) {
    const match = bucket.match(/(\d+)/)
    return match ? parseFloat(match[1]) + 0.5 : null
  }
  const match = bucket.match(/(\d+)-(\d+)/)
  if (match) return (parseFloat(match[1]) + parseFloat(match[2])) / 2
  const singleMatch = bucket.match(/(\d+)°?C/)
  if (singleMatch) return parseFloat(singleMatch[1])
  return null
}

export default function Hub() {
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10))
  const [isMinTemp, setIsMinTemp] = useState(false)
  const [compareMode, setCompareMode] = useState(false)

  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [visibleKeys, setVisibleKeys] = useState<Set<string> | null>(null)
  const [order, setOrder] = useState<string[]>([])

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["predictions", date, isMinTemp],
    queryFn: () => fetchPredictions(date, isMinTemp),
    refetchInterval: 120_000,
    retry: 1,
  })

  const { data: todayEventData } = useQuery({
    queryKey: ["today-event", date, isMinTemp],
    queryFn: () => fetchTodayEvent(date, isMinTemp),
    enabled: !!date,
    refetchInterval: 120_000,
  })

  const eventSlug = todayEventData?.event?.slug ?? null
  const { data: eventData, isError: isMarketError } = useQuery({
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

  useEffect(() => {
    if (data?.models) {
      const keys = Object.keys(data.models)
      if (keys.length > 0) {
        setOrder(prev => {
          const existingOrder = prev.filter(k => keys.includes(k))
          const newKeys = keys.filter(k => !prev.includes(k))
          return [...existingOrder, ...newKeys]
        })
        setActiveKey(prev => (prev && keys.includes(prev)) ? prev : keys[0])
        setVisibleKeys(prev => {
          if (!prev) return new Set(keys)
          const current = new Set(prev)
          keys.forEach(k => { if (!prev.has(k)) current.add(k) })
          return current
        })
      }
    }
  }, [data])

  const displayModels = useMemo(() => {
    if (!data?.models || order.length === 0) return []
    return order.filter(k => data.models[k]).map(k => [k, data.models[k]] as [string, typeof data.models[string]])
  }, [data, order])

  const activeModelProbs = useMemo(() => {
    if (!activeKey || !data?.models?.[activeKey]) return {}
    return data.models[activeKey].probs || {}
  }, [activeKey, data])

  const allBuckets = useMemo(() => {
    const set = new Set<string>()
    Object.values(data?.models || {}).forEach(m => Object.keys(m.probs || {}).forEach(b => set.add(b)))
    Object.keys(marketPrices || {}).forEach(b => set.add(b))
    return Array.from(set).sort((a, b) => {
      const parse = (s: string) => {
        if (s.startsWith("<") || s.includes(" or below")) return -999
        if (s.startsWith(">=") || s.includes(" or higher")) return 999
        if (s.startsWith(">")) return 999
        const num = parseFloat(s.split("-")[0])
        return isNaN(num) ? 0 : num
      }
      return parse(a) - parse(b)
    })
  }, [data, marketPrices])

  // 計算模型預期溫度
  const modelExpectedTemp = useMemo(() => {
    if (!activeKey || !data?.models?.[activeKey]) return null
    return data.models[activeKey].mean?.toFixed(1) ?? null
  }, [activeKey, data])

  // 計算市場加權預期溫度
  const marketExpectedTemp = useMemo(() => {
    if (!marketPrices || Object.keys(marketPrices).length === 0) return null
    let totalProb = 0
    let weightedSum = 0
    for (const [bucket, price] of Object.entries(marketPrices)) {
      const mid = parseBucketMidpoint(bucket)
      if (mid !== null) {
        weightedSum += mid * price
        totalProb += price
      }
    }
    return totalProb > 0 ? (weightedSum / totalProb).toFixed(1) : null
  }, [marketPrices])

  // 計算最高機率桶 (Polymarket)
  const expectedBucket = useMemo(() => {
    if (!marketPrices || Object.keys(marketPrices).length === 0) return null
    let maxProb = 0
    let bucketName = ""
    for (const [bucket, prob] of Object.entries(marketPrices)) {
      if (prob > maxProb) {
        maxProb = prob
        bucketName = bucket
      }
    }
    return maxProb > 0 ? { name: bucketName, prob: (maxProb * 100).toFixed(0) } : null
  }, [marketPrices])

  // 計算所有可見模型的溫度範圍，用於 ModelGrid 的置信區間軌
  const tempRange = useMemo(() => {
    if (!displayModels || displayModels.length === 0) return { min: 20, max: 35 }
    let min = 100, max = -100
    displayModels.forEach(([_, pred]) => {
      min = Math.min(min, pred.mean - (pred.std || 1))
      max = Math.max(max, pred.mean + (pred.std || 1))
    })
    return { min: min - 1, max: max + 1 }
  }, [displayModels])

  const handleReorder = (newOrder: string[]) => setOrder(newOrder)
  const handleToggleVisible = (key: string) => {
    setVisibleKeys(prev => {
      const current = new Set(prev || [])
      if (current.has(key)) {
        current.delete(key)
        if (key === activeKey) setActiveKey(order.find(k => current.has(k)) || null)
      } else {
        current.add(key)
      }
      return current
    })
  }

  return (
    <div className="flex flex-col w-full max-w-[1600px] mx-auto">
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4 p-4 md:p-8 pb-0">
        <div>
          <p className="text-[10px] text-cyan-400/80 mono tracking-[0.2em] uppercase mb-2">Institutional Terminal</p>
          <h1 className="text-3xl font-light text-white tracking-tight">Hong Kong Market Forecast</h1>
        </div>
        <div className="flex items-center gap-4 w-full sm:w-auto">
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className="w-full bg-white/[0.03] border border-white/[0.06] text-slate-200 rounded-md px-3 py-2 text-sm outline-none focus:border-cyan-400/50 transition-colors mono" />
          <div className="flex gap-1 bg-[#0f1013] border border-white/[0.06] p-1 rounded-md shrink-0">
            <button onClick={() => setIsMinTemp(false)} className={`px-4 py-1.5 rounded-sm text-[10px] font-mono uppercase tracking-widest transition-all ${!isMinTemp ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]" : "text-slate-400 hover:text-slate-200"}`}>TMAX</button>
            <button onClick={() => setIsMinTemp(true)} className={`px-4 py-1.5 rounded-sm text-[10px] font-mono uppercase tracking-widest transition-all ${isMinTemp ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]" : "text-slate-400 hover:text-slate-200"}`}>TMIN</button>
          </div>
        </div>
      </header>

      <div className="px-4 md:px-8"><WeatherCards date={date} /></div>

      {/* KPI 摘要卡片 */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 px-4 md:px-8 mb-6 mt-6">
        <div className="obsidian-card rounded-md p-4 flex justify-between items-center">
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">Model Expected</p>
            <p className="text-xl font-light text-cyan-400 mono">{modelExpectedTemp ? `${modelExpectedTemp}°C` : "--"}</p>
          </div>
          <div className="text-cyan-400/20"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg></div>
        </div>
        <div className="obsidian-card rounded-md p-4 flex justify-between items-center">
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">Market Expected</p>
            <p className="text-xl font-light text-violet-400 mono">{marketExpectedTemp ? `${marketExpectedTemp}°C` : "--"}</p>
          </div>
          <div className="text-violet-400/20"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg></div>
        </div>
        <div className="obsidian-card rounded-md p-4 flex justify-between items-center">
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">Expected Bucket</p>
            <p className="text-xl font-light text-emerald-400 mono">{expectedBucket ? `${expectedBucket.name} (${expectedBucket.prob}%)` : "--"}</p>
          </div>
          <div className="text-emerald-400/20"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
        </div>
      </div>

      {/* 歷史溫度追蹤圖 (全寬) */}
      <div className="px-4 md:px-8 mb-6">
        <div className="obsidian-card rounded-md p-6">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-[10px] text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-2"><span className="w-4 h-px bg-slate-500"></span> Models vs Market — Temperature Tracking</h2>
              <p className="text-lg font-light text-white tracking-tight">Historical Prediction Trajectory</p>
            </div>
          </div>
          <ModelsComparisonChart date={date} />
        </div>
      </div>

      {/* 底部雙欄：左模型列表，右 Edge/Prob 分佈 */}
      <div className="flex flex-col lg:flex-row gap-6 px-4 md:px-8 pb-8">
        <aside className="w-full lg:w-1/3 lg:min-w-[300px] lg:max-w-[400px] flex flex-col border border-white/[0.06] bg-[#09090b] rounded-md max-h-[600px] overflow-hidden">
          <div className="p-4 border-b border-white/[0.06] flex items-center justify-between shrink-0">
            <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2"><span className="w-4 h-px bg-slate-500"></span> Model Matrix</h2>
            <button onClick={() => setCompareMode(!compareMode)} className={["flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-md transition-colors", compareMode ? "bg-cyan-500/20 text-cyan-400" : "bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:bg-white/[0.06]"].join(" ")} title="Compare multiple models">
              <GitCompare size={12} /> Compare
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
            {isLoading ? <div className="text-center text-slate-400 text-xs py-4">Loading models...</div> : isError ? <div className="text-center text-rose-400 text-xs py-4 px-2">Failed to load models.<br/><span className="text-slate-500 text-[10px] mt-1 block">{(error as Error)?.message || "API Error"}</span></div> : displayModels.length > 0 ? <ModelGrid models={displayModels} activeKey={activeKey} visibleKeys={visibleKeys} tempRange={tempRange} onSelect={(key) => setActiveKey(key)} onReorder={handleReorder} onToggleVisible={handleToggleVisible} /> : <div className="text-center text-slate-400 text-xs py-4">No models available for this date.</div>}
          </div>
        </aside>

        <section className="flex-1 w-full bg-[#0f1013] border border-white/[0.06] rounded-md p-6 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.8)]">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-[10px] text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-2"><span className="w-4 h-px bg-slate-500"></span> Market Analysis</h2>
              <p className="text-xl font-light text-white tracking-tight">Model vs Market Dynamics</p>
            </div>
          </div>
          <div className="h-[400px] w-full">
            {isLoading ? <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm">Loading chart data...</div> : isError ? <div className="w-full h-full flex flex-col items-center justify-center text-rose-400 text-sm text-center px-4">Failed to load chart data.<span className="text-slate-500 text-xs mt-2">{(error as Error)?.message || "API Error"}</span></div> : isMarketError ? <div className="w-full h-full flex flex-col items-center justify-center text-amber-400 text-sm text-center px-4">Polymarket data unavailable.<span className="text-slate-500 text-xs mt-2">Displaying model probabilities only.</span></div> : compareMode ? <ComparisonChart models={displayModels} marketPrices={marketPrices} allBuckets={allBuckets} /> : <BucketChart modelProbs={activeModelProbs} marketPrices={marketPrices} allBuckets={allBuckets} />}
          </div>
        </section>
      </div>
    </div>
  )
}