import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import { fetchBucketProbs } from "../api/client"
import type { BucketProbsData } from "../types"

const MODEL_COLORS: Record<string, string> = {
  "9d": "#38bdf8",
  aws: "#fb923c",
  baseline: "#94a3b8",
  model_a: "#34d399",
  model_b: "#c084fc",
  model_c: "#fbbf24",
  model_g: "#fb923c",
  model_2a: "#f472b6",
  model_2a1: "#2dd4bf",
  model_2a_v2: "#e879f9",
  model_2b: "#64748b",
  model_3a: "#818cf8",
  model_3b: "#a78bfa",
  model_4: "#fda4af",
  model_4_restricted: "#67e8f9",
  rain_nowcast: "#f59e0b",
}

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
  model_2b: "Model 2B",
  model_3a: "Model 3A",
  model_3b: "Model 3B",
  model_4: "Model 4",
  model_4_restricted: "Model 4 Restricted",
  rain_nowcast: "Rain Nowcast",
}

function sortBuckets(a: string, b: string) {
  const parse = (value: string) => {
    if (value.startsWith("<") || value.includes("or below")) return -999
    if (value.startsWith(">") || value.includes("or higher")) return 999
    const match = value.match(/(\d+)/)
    return match ? Number(match[1]) : 0
  }
  return parse(a) - parse(b)
}

function parseTime(timestamp: string): string {
  try {
    const parsed = new Date(timestamp)
    if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleTimeString("en-HK", { hour: "2-digit", minute: "2-digit", hour12: false })
  } catch {
    // Fall through to source timestamp.
  }
  return timestamp.length >= 16 ? timestamp.slice(11, 16) : timestamp.slice(0, 5)
}

function padArray(values: (number | null)[] | undefined, length: number): (number | null)[] {
  if (!values) return new Array(length).fill(null)
  if (values.length === length) return values
  if (values.length > length) return values.slice(0, length)
  return [...values, ...new Array(length - values.length).fill(null)]
}

export default function BucketProbsChart({ date, bucket: selectedBucket, onBucketChange, visibleKeys }: { date: string; bucket: string; onBucketChange: (bucket: string) => void; visibleKeys?: Set<string> }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["bucketProbs", date, selectedBucket],
    queryFn: () => fetchBucketProbs(date, selectedBucket || undefined),
    refetchInterval: 120_000,
  })

  const availableBuckets = useMemo(() => ((data as BucketProbsData | undefined)?.available_buckets ?? []).slice().sort(sortBuckets), [data])
  const activeBucket = selectedBucket || data?.bucket || availableBuckets[0] || ""

  const option = useMemo<EChartsOption>(() => {
    if (!data || !data.timestamps?.length) return {}
    const timestamps = data.timestamps.map(parseTime)
    const visibleModels = Object.entries(data.models).filter(([key]) => !visibleKeys || visibleKeys.has(key))
    const series: any[] = [
      { name: "Polymarket Price", type: "line", data: padArray(data.market_prices, timestamps.length), step: "end", symbol: "none", connectNulls: true, lineStyle: { width: 2.2, color: "#a78bfa", type: "dashed" }, itemStyle: { color: "#a78bfa" }, z: 5 },
      ...visibleModels.map(([key, values]) => ({ name: MODEL_LABELS[key] || key, type: "line", data: padArray(values, timestamps.length), step: "end", symbol: "none", connectNulls: true, lineStyle: { width: 1.2, color: MODEL_COLORS[key] || "#94a3b8", opacity: 0.48 }, itemStyle: { color: MODEL_COLORS[key] || "#94a3b8" }, z: 1 })),
    ]

    return {
      animationDuration: 500,
      grid: { left: 48, right: 18, top: 34, bottom: 32 },
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(9,12,20,0.96)",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        padding: [10, 14],
        textStyle: { color: "rgba(255,255,255,0.85)", fontSize: 11, fontFamily: "ui-monospace, monospace" },
        axisPointer: { type: "line", lineStyle: { color: "rgba(255,255,255,0.2)", type: "dashed" } },
        formatter: (params: any) => {
          const entries = Array.isArray(params) ? params : [params]
          const title = entries[0]?.axisValue || ""
          const rows = entries.filter((entry: any) => entry.value != null).map((entry: any) => `<div style="display:flex;justify-content:space-between;gap:20px;margin-top:3px"><span style="color:${entry.color}">${entry.seriesName}</span><span style="color:#fff;font-weight:600">${(Number(entry.value) * 100).toFixed(1)}%</span></div>`).join("")
          return `<div style="font-family:ui-monospace,monospace;font-size:10px;color:rgba(255,255,255,.55);margin-bottom:6px">${title} · ${activeBucket}</div>${rows}`
        },
      },
      legend: { data: ["Polymarket Price", ...visibleModels.map(([key]) => MODEL_LABELS[key] || key)], top: 0, type: "scroll", textStyle: { color: "rgba(255,255,255,0.55)", fontSize: 10, fontFamily: "ui-monospace, monospace" } },
      xAxis: { type: "category", data: timestamps, boundaryGap: false, axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } }, axisTick: { show: false }, axisLabel: { color: "rgba(255,255,255,0.35)", fontSize: 10, fontFamily: "ui-monospace, monospace", interval: Math.max(1, Math.floor(timestamps.length / 8)) } },
      yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "rgba(255,255,255,0.35)", fontSize: 10, fontFamily: "ui-monospace, monospace", formatter: (value: number) => `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.045)" } } },
      series,
    }
  }, [activeBucket, data, visibleKeys])

  if (isLoading) return <div className="flex h-[360px] w-full items-center justify-center text-sm text-white/40">Loading bucket probabilities...</div>
  if (isError) return <div className="flex h-[360px] w-full items-center justify-center text-sm text-rose-300">Failed to load bucket probabilities.</div>
  if (!data || !data.timestamps?.length) return <div className="flex h-[360px] w-full items-center justify-center text-xs text-white/35">No bucket probability data available yet.</div>

  return (
    <div className="flex h-[360px] w-full flex-col">
      <div className="mb-2 flex shrink-0 items-center gap-2">
        <label className="eyebrow" htmlFor="hub-bucket-select">Bucket</label>
        <select id="hub-bucket-select" className="hub-input h-7 px-2" value={activeBucket} onChange={(event) => onBucketChange(event.target.value)}>
          {availableBuckets.map((bucket) => <option key={bucket} value={bucket}>{bucket}</option>)}
        </select>
      </div>
      <div className="min-h-0 flex-1"><ReactECharts option={option} notMerge={true} style={{ height: "100%", width: "100%" }} /></div>
    </div>
  )
}
