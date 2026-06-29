import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import { fetchModelsComparison } from "../api/client"

const MODEL_COLORS: Record<string, string> = {
  "9d": "#1f77b4",
  "aws": "#ff7f0e",
  "baseline": "#636EFA",
  "model_a": "#00CC96",
  "model_b": "#AB63FA",
  "model_c": "#FFA15A",
  "model_g": "#FFB86C",
  "model_2a": "#FF2C97",
}

const MODEL_LABELS: Record<string, string> = {
  "9d": "9-Day XGBoost",
  "aws": "AWS High-Freq",
  "baseline": "Baseline Intraday",
  "model_a": "Model A",
  "model_b": "Model B (Rain)",
  "model_c": "Model C (Nowcast)",
  "model_g": "Model G (Gap+Max)",
  "model_2a": "Model 2A (Core+Wind)",
}

function parseTime(ts: string): string {
  try {
    const d = new Date(ts)
    if (!isNaN(d.getTime())) {
      return d.toLocaleTimeString("en-HK", { hour: "2-digit", minute: "2-digit", hour12: false })
    }
  } catch { /* fallthrough */ }
  if (ts.length >= 16) return ts.slice(11, 16)
  if (ts.length >= 5) return ts.slice(0, 5)
  return ts
}

// 強制將數據陣列補齊到指定長度，防止 ECharts 數據錯位
function padArray(arr: (number | null)[] | undefined, len: number): (number | null)[] {
  if (!arr) return new Array(len).fill(null)
  if (arr.length === len) return arr
  if (arr.length > len) return arr.slice(0, len)
  return [...arr, ...new Array(len - arr.length).fill(null)]
}

export default function ModelsComparisonChart({ date }: { date: string }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["modelsComparison", date],
    queryFn: () => fetchModelsComparison(date),
    refetchInterval: 120_000,
  })

  const option = useMemo<EChartsOption>(() => {
    if (!data || !data.timestamps?.length) return {}

    const timestamps = data.timestamps.map(parseTime)
    const maxLen = timestamps.length

    const series: any[] = [
      {
        name: "實際氣溫",
        type: "line",
        data: padArray(data.actual_temps, maxLen),
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        connectNulls: true,
        lineStyle: { width: 3, color: "#fb7185", shadowBlur: 10, shadowColor: "rgba(251, 113, 133, 0.6)" },
        itemStyle: { color: "#fb7185" },
        z: 10,
      },
      {
        name: "Polymarket 加權",
        type: "line",
        data: padArray(data.market_temps, maxLen),
        smooth: true,
        symbol: "diamond",
        symbolSize: 6,
        connectNulls: true,
        lineStyle: { width: 2.5, color: "#a78bfa", type: "dashed", shadowBlur: 10, shadowColor: "rgba(167, 139, 250, 0.4)" },
        itemStyle: { color: "#a78bfa" },
        z: 9,
      },
    ]

    Object.entries(data.models).forEach(([key, values]) => {
      series.push({
        name: MODEL_LABELS[key] || key,
        type: "line",
        data: padArray(values, maxLen),
        smooth: true,
        symbol: "none",
        connectNulls: true,
        lineStyle: { width: 1.5, color: MODEL_COLORS[key] || "#94a3b8", opacity: 0.8 },
        itemStyle: { color: MODEL_COLORS[key] || "#94a3b8" },
        z: 1,
      })
    })

    return {
      backgroundColor: "transparent",
      grid: { left: "2%", right: "2%", bottom: "5%", top: "15%", containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0f1013",
        borderColor: "rgba(255,255,255,0.08)",
        borderWidth: 1,
        borderRadius: 4,
        textStyle: { color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: 11 },
        extraCssText: "box-shadow: 0 10px 40px -10px rgba(0,0,0,0.8); backdrop-filter: blur(4px);",
      },
      legend: {
        data: ["實際氣溫", "Polymarket 加權", ...Object.keys(data.models).map(k => MODEL_LABELS[k] || k)],
        textStyle: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono" },
        top: 0,
        type: "scroll",
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: timestamps,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
        axisTick: { show: false },
        axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", margin: 12 },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", formatter: "{value}°C" },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series,
    }
  }, [data])

  if (isLoading) {
    return <div className="w-full h-[400px] flex items-center justify-center text-slate-500 text-sm">Loading temperature tracking data...</div>
  }

  if (isError) {
    return <div className="w-full h-[400px] flex items-center justify-center text-rose-400 text-sm">Failed to load tracking data.</div>
  }

  if (!data || !data.timestamps?.length) {
    return <div className="w-full h-[400px] flex items-center justify-center text-slate-500 text-xs">No snapshot data available yet. Waiting for the next strategy cycle...</div>
  }

  return (
    <div className="w-full h-[400px]">
      <ReactECharts option={option} notMerge={true} style={{ height: "100%", width: "100%" }} />
    </div>
  )
}