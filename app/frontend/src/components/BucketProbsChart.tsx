import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import { fetchBucketProbs } from "../api/client"
import type { BucketProbsData } from "../types"

const MODEL_COLORS: Record<string, string> = {
  "9d": "#1f77b4", "aws": "#ff7f0e", "baseline": "#636EFA", "model_a": "#00CC96", "model_b": "#AB63FA", "model_c": "#FFA15A", "model_g": "#FFB86C", "model_2a": "#FF2C97", "model_2a1": "#0d9488",
}
const MODEL_LABELS: Record<string, string> = {
  "9d": "9-Day XGBoost", "aws": "AWS High-Freq", "baseline": "Baseline Intraday", "model_a": "Model A", "model_b": "Model B (Rain)", "model_c": "Model C (Nowcast)", "model_g": "Model G (Gap+Max)", "model_2a": "Model 2A (Core+Wind)", "model_2a1": "Model 2A1 (i-lens)",
}

function parseTime(ts: string): string {
  try { const d = new Date(ts); if (!isNaN(d.getTime())) return d.toLocaleTimeString("en-HK", { hour: "2-digit", minute: "2-digit", hour12: false }) } catch { /* */ }
  if (ts.length >= 16) return ts.slice(11, 16)
  if (ts.length >= 5) return ts.slice(0, 5)
  return ts
}

function padArray(arr: (number | null)[] | undefined, len: number): (number | null)[] {
  if (!arr) return new Array(len).fill(null)
  if (arr.length === len) return arr
  if (arr.length > len) return arr.slice(0, len)
  return [...arr, ...new Array(len - arr.length).fill(null)]
}

export default function BucketProbsChart({ date, bucket: selectedBucket, onBucketChange }: { date: string; bucket: string; onBucketChange: (b: string) => void }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["bucketProbs", date, selectedBucket],
    queryFn: () => fetchBucketProbs(date, selectedBucket),
    refetchInterval: 120_000,
  })

  const option = useMemo<EChartsOption>(() => {
    if (!data || !data.timestamps?.length) return {}
    const timestamps = data.timestamps.map(parseTime)
    const maxLen = timestamps.length
    const series: any[] = [
      { name: "Polymarket Price", type: "line", data: padArray(data.market_prices, maxLen), smooth: true, symbol: "diamond", symbolSize: 6, connectNulls: true, lineStyle: { width: 2.5, color: "#fbbf24", type: "dashed", shadowBlur: 8, shadowColor: "rgba(251, 191, 36, 0.4)" }, itemStyle: { color: "#fbbf24" }, z: 10 },
    ]
    Object.entries(data.models).forEach(([key, values]) => {
      series.push({ name: MODEL_LABELS[key] || key, type: "line", data: padArray(values, maxLen), smooth: true, symbol: "none", connectNulls: true, lineStyle: { width: 1.5, color: MODEL_COLORS[key] || "#94a3b8", opacity: 0.8 }, itemStyle: { color: MODEL_COLORS[key] || "#94a3b8" }, z: 1 })
    })

    const allValues: (number | null)[] = [...(data.market_prices ?? []), ...Object.values(data.models).flat()]
    const maxVal = allValues.reduce<number>((m, v) => (v != null && v > m ? v : m), 0)
    const yMax = maxVal > 0 ? Math.min(maxVal * 1.15, 1) : 1

    return {
      backgroundColor: "transparent", grid: { left: "2%", right: "2%", bottom: "5%", top: "15%", containLabel: true },
      tooltip: { trigger: "axis", backgroundColor: "#0f1013", borderColor: "rgba(255,255,255,0.08)", borderWidth: 1, borderRadius: 4, textStyle: { color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: 11 }, extraCssText: "box-shadow: 0 10px 40px -10px rgba(0,0,0,0.8); backdrop-filter: blur(4px);", formatter: (params: any) => {
        if (!Array.isArray(params)) return ""
        let html = `<div style="font-weight:600;margin-bottom:6px;font-size:12px;color:#e2e8f0">${params[0]?.axisValue || ""}</div>`
        params.forEach((p: any) => { if (p.value != null) html += `<div style="display:flex;justify-content:space-between;gap:24px"><span style="color:${p.color}">${p.seriesName}</span><span style="color:#e2e8f0;font-weight:600">${(p.value * 100).toFixed(1)}%</span></div>` })
        return html
      } },
      legend: { data: ["Polymarket Price", ...Object.keys(data.models).map(k => MODEL_LABELS[k] || k)], textStyle: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono" }, top: 0, type: "scroll" },
      xAxis: { type: "category", boundaryGap: false, data: timestamps, axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } }, axisTick: { show: false }, axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", margin: 12 } },
      yAxis: { type: "value", min: 0, max: yMax, axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", formatter: (v: number) => `${(v * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } }, axisLine: { show: false }, axisTick: { show: false } },
      series,
    }
  }, [data])

  const availableBuckets = (data as BucketProbsData | undefined)?.available_buckets ?? []

  if (isLoading) return <div className="w-full h-[300px] flex items-center justify-center text-slate-500 text-sm">Loading bucket probabilities...</div>
  if (isError) return <div className="w-full h-[300px] flex items-center justify-center text-rose-400 text-sm">Failed to load bucket probabilities.</div>
  if (!data || !data.timestamps?.length) return <div className="w-full h-[300px] flex items-center justify-center text-slate-500 text-xs">No bucket probability data yet.</div>

  return (
    <div className="w-full h-[300px] flex flex-col">
      <div className="flex items-center gap-2 mb-2 shrink-0">
        <label className="text-xs text-slate-400 font-mono">Bucket</label>
        <select
          className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono focus:outline-none focus:ring-1 focus:ring-cyan-500"
          value={selectedBucket}
          onChange={(e) => onBucketChange(e.target.value)}
        >
          {availableBuckets.map((b) => (
            <option key={b} value={b}>{b}°C</option>
          ))}
        </select>
      </div>
      <div className="flex-1 min-h-0">
        <ReactECharts option={option} notMerge={true} style={{ height: "100%", width: "100%" }} />
      </div>
    </div>
  )
}
