import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { CalendarDays, Download, GitCompare, SlidersHorizontal } from "lucide-react"
import ModelGrid from "../components/ModelGrid"
import WeatherCards from "../components/WeatherCards"
import BucketChart from "../components/BucketChart"
import BucketProbsChart from "../components/BucketProbsChart"
import ComparisonChart from "../components/ComparisonChart"
import ModelsComparisonChart from "../components/ModelsComparisonChart"
import MinuteHistoryPanel from "../components/MinuteHistoryPanel"
import { fetchEvent, fetchPredictions, fetchTodayEvent } from "../api/client"

const MODEL_LABELS: Record<string, string> = {
  "9d": "9-Day XGBoost",
  aws: "AWS High-Freq",
  baseline: "Baseline Intraday",
  model_a: "Model A",
  model_b: "Model B (Rain)",
  model_c: "Model C (Nowcast)",
  model_g: "Model G (Gap+Max)",
  model_2a: "Model 2A (Core+Wind)",
  model_2a1: "Model 2A1 (i-lens)",
  model_2a_v2: "Model 2A v2 (Offshore+Highland)",
}

function parseBucketMidpoint(bucket: string): number | null {
  if (bucket.startsWith("<") || bucket.includes("or below")) {
    const match = bucket.match(/(\d+)/)
    return match ? Number(match[1]) - 0.5 : null
  }
  if (bucket.startsWith(">=") || bucket.includes("or higher")) {
    const match = bucket.match(/(\d+)/)
    return match ? Number(match[1]) + 0.5 : null
  }
  const range = bucket.match(/(\d+)-(\d+)/)
  if (range) return (Number(range[1]) + Number(range[2])) / 2
  const single = bucket.match(/(\d+)°?C/)
  return single ? Number(single[1]) : null
}

function sortBucketValue(bucket: string): number {
  if (bucket.startsWith("<") || bucket.includes("or below")) return -999
  if (bucket.startsWith(">") || bucket.includes("or higher")) return 999
  const match = bucket.match(/(\d+)/)
  return match ? Number(match[1]) : 0
}

function getHKTDateString(): string {
  const parts = new Intl.DateTimeFormat("en-HK", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date())
  const year = parts.find((part) => part.type === "year")?.value || ""
  const month = parts.find((part) => part.type === "month")?.value || ""
  const day = parts.find((part) => part.type === "day")?.value || ""
  return `${year}-${month}-${day}`
}

export default function Hub() {
  const [date, setDate] = useState(getHKTDateString())
  const [isMinTemp, setIsMinTemp] = useState(false)
  const [compareMode, setCompareMode] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [viewMode, setViewMode] = useState<"trajectory" | "bucket">("trajectory")
  const [marketViewMode, setMarketViewMode] = useState<"edge" | "prob">("prob")
  const [selectedBucket, setSelectedBucket] = useState("")
  const [activeKey, setActiveKey] = useState<string | null>(null)

  const [visibleKeys, setVisibleKeys] = useState<Set<string> | null>(() => {
    try {
      const saved = localStorage.getItem("visibleKeys")
      return saved ? new Set(JSON.parse(saved)) : null
    } catch {
      return null
    }
  })

  const [order, setOrder] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem("modelOrder")
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["predictions", date, isMinTemp],
    queryFn: () => fetchPredictions(date, isMinTemp),
    refetchInterval: 120_000,
    retry: 1,
  })

  const { data: todayEventData } = useQuery({
    queryKey: ["today-event", date, isMinTemp],
    queryFn: () => fetchTodayEvent(date, isMinTemp),
    enabled: Boolean(date),
    refetchInterval: 120_000,
  })

  const eventSlug = todayEventData?.event?.slug ?? null
  const { data: eventData, isError: isMarketError } = useQuery({
    queryKey: ["event", eventSlug],
    queryFn: () => fetchEvent(eventSlug!, isMinTemp),
    enabled: Boolean(eventSlug),
    refetchInterval: 120_000,
  })

  const marketPrices = useMemo(() => {
    const prices = eventData?.prices ?? {}
    return Object.fromEntries(
      Object.entries(prices).map(([bucket, price]) => [
        bucket.replace(/^&lt;/, "<").replace(/^&gt;/, ">").replace(/&lt;/g, "<").replace(/&gt;/g, ">"),
        price,
      ]),
    )
  }, [eventData?.prices])

  useEffect(() => {
    if (!data?.models) return
    const keys = Object.keys(data.models)
    if (keys.length === 0) return

    setOrder((previous) => {
      const existing = previous.filter((key) => keys.includes(key))
      const additions = keys.filter((key) => !previous.includes(key))
      const merged = [...existing, ...additions]
      localStorage.setItem("modelOrder", JSON.stringify(merged))
      return merged
    })
    const firstVisible = keys.find((key) => !visibleKeys || visibleKeys.has(key)) || keys[0]
    setActiveKey((previous) => previous && keys.includes(previous) && (!visibleKeys || visibleKeys.has(previous)) ? previous : firstVisible)
    setVisibleKeys((previous) => {
      if (previous) return previous
      const initial = new Set(keys)
      localStorage.setItem("visibleKeys", JSON.stringify([...initial]))
      return initial
    })
  }, [data])

  const displayModels = useMemo(() => {
    if (!data?.models || order.length === 0) return []
    return order.filter((key) => data.models[key]).map((key) => [key, data.models[key]] as [string, typeof data.models[string]])
  }, [data, order])

  const visibleModels = useMemo(() => {
    if (!visibleKeys) return displayModels
    return displayModels.filter(([key]) => visibleKeys.has(key))
  }, [displayModels, visibleKeys])

  const activeModelProbs = useMemo(() => {
    if (!activeKey || !data?.models?.[activeKey]) return {}
    return data.models[activeKey].probs || {}
  }, [activeKey, data])

  const allBuckets = useMemo(() => {
    const buckets = new Set<string>()
    Object.values(data?.models || {}).forEach((model) => Object.keys(model.probs || {}).forEach((bucket) => buckets.add(bucket)))
    Object.keys(marketPrices).forEach((bucket) => buckets.add(bucket))
    return [...buckets].sort((a, b) => sortBucketValue(a) - sortBucketValue(b))
  }, [data, marketPrices])

  const modelExpectedTemp = useMemo(() => {
    if (!data?.models || !visibleKeys) return null
    let weightedSum = 0
    let totalWeight = 0
    for (const [key, prediction] of Object.entries(data.models)) {
      if (!visibleKeys.has(key) || prediction.mean == null) continue
      const weight = prediction.std > 0 ? 1 / prediction.std : 10
      weightedSum += prediction.mean * weight
      totalWeight += weight
    }
    return totalWeight > 0 ? (weightedSum / totalWeight).toFixed(1) : null
  }, [data, visibleKeys])

  const marketExpectedTemp = useMemo(() => {
    let totalProbability = 0
    let weightedSum = 0
    for (const [bucket, price] of Object.entries(marketPrices)) {
      const midpoint = parseBucketMidpoint(bucket)
      if (midpoint == null) continue
      weightedSum += midpoint * price
      totalProbability += price
    }
    return totalProbability > 0 ? (weightedSum / totalProbability).toFixed(1) : null
  }, [marketPrices])

  const expectedBucket = useMemo(() => {
    let bucketName = ""
    let maxProbability = 0
    for (const [bucket, probability] of Object.entries(marketPrices)) {
      if (probability > maxProbability) {
        bucketName = bucket
        maxProbability = probability
      }
    }
    return maxProbability > 0 ? { name: bucketName, prob: Math.round(maxProbability * 100) } : null
  }, [marketPrices])

  const tempRange = useMemo(() => {
    if (!data?.models || !visibleKeys) return { min: 20, max: 35 }
    let min = 100
    let max = -100
    for (const [key, prediction] of Object.entries(data.models)) {
      if (!visibleKeys.has(key)) continue
      min = Math.min(min, prediction.mean - (prediction.std || 1))
      max = Math.max(max, prediction.mean + (prediction.std || 1))
    }
    return { min: min - 1, max: max + 1 }
  }, [data, visibleKeys])

  const handleReorder = (newOrder: string[]) => {
    const visibleSet = new Set(newOrder)
    let visibleIndex = 0
    const mergedOrder = order.map((key) => visibleSet.has(key) ? newOrder[visibleIndex++] : key)
    setOrder(mergedOrder)
    localStorage.setItem("modelOrder", JSON.stringify(mergedOrder))
  }

  const handleToggleVisible = (key: string) => {
    setVisibleKeys((previous) => {
      const current = new Set(previous || [])
      if (current.has(key)) {
        current.delete(key)
        if (key === activeKey) setActiveKey(order.find((candidate) => current.has(candidate)) || null)
      } else {
        current.add(key)
      }
      localStorage.setItem("visibleKeys", JSON.stringify([...current]))
      return current
    })
  }

  const edge = modelExpectedTemp && marketExpectedTemp
    ? (Number(modelExpectedTemp) - Number(marketExpectedTemp)).toFixed(1)
    : null

  return (
    <div className="hub-page mx-auto w-full max-w-[1480px] space-y-5 px-4 pb-10 pt-4 sm:px-6 lg:px-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="eyebrow eyebrow-accent">Hub — Hong Kong · {isMinTemp ? "TMIN" : "TMAX"}</div>
          <h1 className="mt-2 text-[28px] font-bold leading-tight tracking-[-0.02em] text-white/95">Hong Kong Market Forecast</h1>
          <p className="mono mt-1 text-[11px] tracking-[0.08em] text-white/35">{date} · LIVE MODEL SNAPSHOT · SRC HKO / POLYMARKET</p>
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <label className="relative flex flex-1 items-center gap-2 sm:flex-none" title="Forecast date">
            <CalendarDays className="pointer-events-none absolute left-3 h-3.5 w-3.5 text-white/35" />
            <input className="hub-input pl-9 pr-3 py-2" type="date" value={date} onChange={(event) => setDate(event.target.value)} />
          </label>
          <a
            className="mono inline-flex items-center gap-1.5 rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-[10px] font-semibold tracking-[0.1em] text-white/55 transition-colors hover:border-cyan-400/30 hover:bg-cyan-400/10 hover:text-cyan-300"
            href={`/api/data/export-csv?date=${encodeURIComponent(date)}`}
            download={`snapshots-${date}.csv`}
            aria-label="Download selected day CSV"
            title="Download selected day snapshot CSV"
          >
            <Download className="h-3.5 w-3.5" />
            <span>CSV</span>
          </a>
          <div className="seg shrink-0">
            <button className={`seg-item ${!isMinTemp ? "seg-item-active" : ""}`} onClick={() => setIsMinTemp(false)}>TMAX</button>
            <button className={`seg-item ${isMinTemp ? "seg-item-active" : ""}`} onClick={() => setIsMinTemp(true)}>TMIN</button>
          </div>
          <button className="rounded-xl border border-white/[0.08] bg-white/[0.03] p-2.5 text-white/40 transition-colors hover:text-cyan-300" onClick={() => setShowSettings(true)} title="Model visibility settings">
            <SlidersHorizontal className="h-3.5 w-3.5" />
          </button>
        </div>
      </header>

      <section className="panel overflow-hidden">
        <div className="grid grid-cols-2 divide-x divide-y divide-white/[0.05] lg:grid-cols-4 lg:divide-y-0">
          <HubStat label="MODEL EXPECTED" value={modelExpectedTemp ? `${modelExpectedTemp}°C` : "--"} valueClass="text-cyan-300" foot={`${displayModels.filter(([key]) => !visibleKeys || visibleKeys.has(key)).length || 0} visible models · precision weighted`} />
          <HubStat label="MARKET EXPECTED" value={marketExpectedTemp ? `${marketExpectedTemp}°C` : "--"} valueClass="text-violet-300" foot="Polymarket bucket price midpoint" />
          <HubStat label="EDGE" value={edge ? `${Number(edge) >= 0 ? "+" : ""}${edge}°C` : "--"} valueClass={edge && Number(edge) >= 0 ? "text-emerald-300" : "text-rose-300"} foot="Model minus market expectation" />
          <div className="px-5 py-5 sm:px-6">
            <div className="eyebrow">EXPECTED BUCKET</div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="tnum text-[27px] font-bold leading-none tracking-[-0.02em] text-white/95">{expectedBucket?.name || "--"}</span>
              <span className="mono tnum text-[13px] font-semibold text-emerald-300">{expectedBucket ? `${expectedBucket.prob}%` : ""}</span>
            </div>
            <div className="mt-2.5 h-1 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400/70 to-emerald-400/90" style={{ width: `${expectedBucket?.prob || 0}%` }} /></div>
            <div className="mono mt-2 text-[10px] tracking-wide text-white/30">Market-implied bucket probability</div>
          </div>
        </div>
      </section>

      <WeatherCards date={date} />

      <MinuteHistoryPanel date={date} />

      <section className="panel p-4 sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="eyebrow">MODELS VS MARKET · TEMPERATURE TRACKING</div>
            <h2 className="mt-1.5 text-lg font-semibold tracking-tight text-white/90">{viewMode === "trajectory" ? "Historical Prediction Trajectory" : "Per-Bucket Probability"}</h2>
          </div>
          <div className="seg">
            <button className={`seg-item ${viewMode === "trajectory" ? "seg-item-active" : ""}`} onClick={() => setViewMode("trajectory")}>Trajectory</button>
            <button className={`seg-item ${viewMode === "bucket" ? "seg-item-active" : ""}`} onClick={() => setViewMode("bucket")}>Bucket</button>
          </div>
        </div>
        <div className="mt-4 min-h-[320px]">
          {viewMode === "trajectory" ? <ModelsComparisonChart date={date} visibleKeys={visibleKeys ?? undefined} /> : <BucketProbsChart date={date} bucket={selectedBucket} onBucketChange={setSelectedBucket} visibleKeys={visibleKeys ?? undefined} />}
        </div>
        <p className="mono mt-1 text-[10.5px] tracking-wide text-white/25">Click model controls below to change the visible series · all values shown in °C</p>
      </section>

      <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-12">
        <section className="panel overflow-hidden p-4 sm:p-6 xl:col-span-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="eyebrow">MODEL MATRIX</div>
              <h2 className="mt-1.5 text-lg font-semibold tracking-tight text-white/90">Forecast Ensemble</h2>
            </div>
            <button onClick={() => setCompareMode(!compareMode)} className={`mono flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[10px] font-semibold tracking-[0.1em] transition-colors ${compareMode ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-300" : "border-white/[0.08] bg-white/[0.03] text-white/55 hover:text-white/85"}`} title="Compare multiple models">
              <GitCompare className="h-3 w-3" /> Compare
            </button>
          </div>
          <div className="mt-2 max-h-[380px] overflow-y-auto pr-1 custom-scrollbar">
            {isLoading ? <div className="py-8 text-center text-xs text-white/40">Loading models...</div> : isError ? <div className="py-8 text-center text-xs text-rose-300">Failed to load models.<span className="mt-1 block text-[10px] text-white/35">{(error as Error)?.message || "API error"}</span></div> : visibleModels.length > 0 ? <ModelGrid models={visibleModels} activeKey={activeKey} tempRange={tempRange} onSelect={(key) => setActiveKey(key)} onReorder={handleReorder} /> : <div className="py-8 text-center text-xs text-white/40">No visible models. Use the settings control to show models.</div>}
          </div>
          <p className="mono mt-4 text-[10.5px] tracking-wide text-white/25">Drag the handle to reorder · click a row to inspect its distribution</p>
        </section>

        <section className="panel p-4 sm:p-6 xl:col-span-8">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="eyebrow">MARKET ANALYSIS</div>
              <h2 className="mt-1.5 text-lg font-semibold tracking-tight text-white/90">Model vs Market Dynamics</h2>
            </div>
            <div className="seg shrink-0">
              <button className={`seg-item ${marketViewMode === "edge" ? "seg-item-active" : ""}`} onClick={() => setMarketViewMode("edge")}>Edge</button>
              <button className={`seg-item ${marketViewMode === "prob" ? "seg-item-active" : ""}`} onClick={() => setMarketViewMode("prob")}>Prob</button>
            </div>
          </div>
          <div className="mt-4 h-[360px] w-full sm:h-[400px]">
            {isLoading ? <div className="flex h-full items-center justify-center text-sm text-white/40">Loading chart data...</div> : isError ? <div className="flex h-full flex-col items-center justify-center text-center text-sm text-rose-300">Failed to load chart data.<span className="mt-2 text-xs text-white/35">{(error as Error)?.message || "API error"}</span></div> : isMarketError ? <div className="flex h-full flex-col items-center justify-center text-center text-sm text-amber-300">Polymarket data unavailable.<span className="mt-2 text-xs text-white/35">Displaying model probabilities only.</span></div> : compareMode ? <ComparisonChart models={visibleModels} marketPrices={marketPrices} allBuckets={allBuckets} /> : <BucketChart viewMode={marketViewMode} modelProbs={activeModelProbs} marketPrices={marketPrices} allBuckets={allBuckets} />}
          </div>
          <p className="mono mt-1 text-[10.5px] tracking-wide text-white/25">Positive edge indicates the model assigns more probability than the market price.</p>
        </section>
      </div>

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm" onClick={() => setShowSettings(false)}>
          <div className="panel w-full max-w-md p-5 sm:p-6" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between">
              <div><div className="eyebrow">MODEL CONTROLS</div><h3 className="mt-1 text-lg font-semibold text-white/90">Visibility & ordering</h3></div>
              <button onClick={() => setShowSettings(false)} className="text-xl text-white/35 hover:text-white/80" aria-label="Close model controls">×</button>
            </div>
            <div className="mt-4 space-y-1">
              {order.map((key, index) => (
                <label key={key} className="flex cursor-pointer items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-white/[0.04]">
                  <span className="mono w-5 text-[10px] text-white/25">{String(index + 1).padStart(2, "0")}</span>
                  <input type="checkbox" checked={visibleKeys?.has(key) ?? true} onChange={() => handleToggleVisible(key)} className="accent-cyan-400" />
                  <span className="text-sm text-white/75">{MODEL_LABELS[key] || key}</span>
                </label>
              ))}
            </div>
            <div className="mt-4 flex gap-2 border-t border-white/[0.06] pt-4">
              <button onClick={() => { const all = new Set(order); setVisibleKeys(all); localStorage.setItem("visibleKeys", JSON.stringify([...all])) }} className="flex-1 rounded-lg bg-cyan-400/12 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-300 transition-colors hover:bg-cyan-400/20">Show all</button>
              <button onClick={() => { setVisibleKeys(new Set()); localStorage.setItem("visibleKeys", JSON.stringify([])) }} className="flex-1 rounded-lg bg-white/[0.04] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-white/55 transition-colors hover:bg-white/[0.08]">Hide all</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function HubStat({ label, value, valueClass, foot }: { label: string; value: string; valueClass: string; foot: string }) {
  return (
    <div className="px-5 py-5 sm:px-6">
      <div className="eyebrow">{label}</div>
      <div className={`tnum mt-1.5 text-[27px] font-bold leading-none tracking-[-0.02em] ${valueClass}`}>{value}</div>
      <div className="mono mt-2 text-[10.5px] tracking-wide text-white/30">{foot}</div>
    </div>
  )
}
