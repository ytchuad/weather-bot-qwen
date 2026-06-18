import { useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchPredictions } from "../api/client"
import WeatherNow from "../components/WeatherNow"
import ModelGrid from "../components/ModelGrid"
import BucketChart from "../components/BucketChart"

const TODAY = new Date().toISOString().slice(0, 10)

export default function Hub() {
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [order, setOrder] = useState<string[] | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ["predictions", TODAY],
    queryFn: () => fetchPredictions(TODAY, false),
    refetchInterval: 120_000,
  })

  const models = data?.models
  const entries = models ? Object.entries(models) : []

  const sorted = useCallback(
    (entries: [string, import("../types").ModelPrediction][]) => {
      if (!order) return entries
      const map = Object.fromEntries(entries)
      const ordered = order.filter((k) => k in map).map((k) => [k, map[k]] as [string, import("../types").ModelPrediction])
      const rest = entries.filter(([k]) => !order.includes(k))
      return [...ordered, ...rest]
    },
    [order],
  )(entries)

  const activeModel = activeKey && models?.[activeKey] ? models[activeKey] : entries[0]?.[1]
  const activeModelKey = activeKey && models?.[activeKey] ? activeKey : entries[0]?.[0]

  return (
    <div className="space-y-5">
      <WeatherNow />

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
          marketPrices={{}}
        />
      )}
    </div>
  )
}
