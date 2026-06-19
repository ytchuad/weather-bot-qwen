import { useCallback, useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchEvent, fetchTodayEvent, fetchPredictions } from "../api/client"
import ModelGrid from "../components/ModelGrid"
import BucketChart from "../components/BucketChart"
import type { ModelPrediction } from "../types"

const TODAY = new Date().toISOString().slice(0, 10)

export default function Hub() {
  const [date, setDate] = useState(TODAY)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [order, setOrder] = useState<string[] | null>(null)
  const [isMinTemp, setIsMinTemp] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ["predictions", date],
    queryFn: () => fetchPredictions(date, false),
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

  const activeModel = activeKey && models?.[activeKey] ? models[activeKey] : entries[0]?.[1]

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <aside className="w-1/3 min-w-[300px] max-w-[400px] h-full flex flex-col border-r border-slate-800 bg-slate-950/50 backdrop-blur-sm">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Model Matrix</h2>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-24 rounded-lg bg-slate-800/50 animate-pulse" />
              ))}
            </div>
          ) : (
            <ModelGrid
              models={sorted}
              activeKey={activeKey ?? (activeModel ? entries[0]?.[0] : null)}
              onSelect={(k) => setActiveKey(k)}
              onReorder={(keys) => setOrder(keys)}
            />
          )}
        </div>
      </aside>

      <main className="flex-1 h-full overflow-y-auto p-8">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold text-slate-50">Weather Hub</h1>
            <p className="text-sm text-slate-500">{eventData?.title?.replace("highest", "TMAX Prediction").replace("TMAX", "") ?? "Weather predictions"}</p>
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

        {activeModel && activeModel.probs && (
          <BucketChart
            modelProbs={activeModel.probs}
            marketPrices={marketPrices}
          />
        )}
      </main>
    </div>
  )
}