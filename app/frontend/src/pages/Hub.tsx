import { useCallback, useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchEvent, fetchTodayEvent, fetchPredictions } from "../api/client"
import WeatherNow from "../components/WeatherNow"
import ModelGrid from "../components/ModelGrid"
import BucketChart from "../components/BucketChart"
import type { ModelPrediction } from "../types"

const TODAY = new Date().toISOString().slice(0, 10)

export default function Hub() {
  const [date, setDate] = useState(TODAY)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [order, setOrder] = useState<string[] | null>(null)

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
    queryFn: () => fetchEvent(eventSlug!, false),
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
  const activeModelKey = activeKey && models?.[activeKey] ? activeKey : entries[0]?.[0]

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-xs font-semibold tracking-wider uppercase text-white/40">
            Hub
          </div>
          <div className="text-sm text-white/30 mt-0.5">
            {eventData?.title ?? "Weather predictions"}
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs text-white/50">
          <span>Date</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="rounded-lg border border-white/5 bg-white/[0.03] px-3 py-2 text-sm text-white outline-none focus:border-[#00E5FF]/50"
          />
        </label>
      </div>

      <WeatherNow date={date} />

      <div>
        <div className="text-xs font-semibold tracking-wider uppercase text-white/40 mb-3">
          Model Predictions
        </div>
        {isLoading ? (
          <div className="grid grid-cols-3 gap-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-24 rounded-xl bg-white/[0.03] animate-pulse"
              />
            ))}
          </div>
        ) : (
          <ModelGrid
            models={sorted}
            activeKey={activeModelKey ?? null}
            onSelect={(k) => setActiveKey(k)}
            onReorder={(keys) => setOrder(keys)}
          />
        )}
      </div>

      {activeModel && activeModel.probs && (
        <BucketChart
          modelProbs={activeModel.probs}
          marketPrices={marketPrices}
        />
      )}
    </div>
  )
}
