import { useMemo } from "react"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import type { ModelPrediction } from "../types"

const MODEL_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fbbf24", "#fb923c", "#818cf8", "#2dd4bf"]

function sortBuckets(a: string, b: string) {
  const parse = (value: string) => {
    if (value.startsWith("<") || value.includes("or below")) return -999
    if (value.startsWith(">") || value.includes("or higher")) return 999
    const match = value.match(/(\d+)/)
    return match ? Number(match[1]) : 0
  }
  return parse(a) - parse(b)
}

export default function ComparisonChart({ models, marketPrices, allBuckets }: {
  models: [string, ModelPrediction][]
  marketPrices: Record<string, number>
  allBuckets: string[]
}) {
  const option = useMemo<EChartsOption>(() => {
    const buckets = [...allBuckets].sort(sortBuckets)
    const series = models.map(([key, prediction], index) => ({
      name: key,
      type: "line",
      smooth: false,
      step: "end",
      symbol: "none",
      data: buckets.map((bucket) => prediction.probs?.[bucket] != null ? +(prediction.probs[bucket] * 100).toFixed(1) : 0),
      lineStyle: { width: 1.3, color: MODEL_COLORS[index % MODEL_COLORS.length], opacity: 0.72 },
      itemStyle: { color: MODEL_COLORS[index % MODEL_COLORS.length] },
    })) as any[]

    series.push({
      name: "Market",
      type: "bar",
      data: buckets.map((bucket) => marketPrices[bucket] != null ? +(marketPrices[bucket] * 100).toFixed(1) : 0),
      barWidth: "34%",
      itemStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(167,139,250,0.75)" }, { offset: 1, color: "rgba(167,139,250,0.12)" }] }, borderRadius: [4, 4, 0, 0] },
    })

    return {
      animationDuration: 500,
      grid: { left: 48, right: 18, top: 34, bottom: 32 },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(255,255,255,0.03)" } }, backgroundColor: "rgba(9,12,20,0.96)", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1, padding: [10, 14], textStyle: { color: "rgba(255,255,255,0.85)", fontSize: 11, fontFamily: "ui-monospace, monospace" } },
      legend: { data: [...models.map(([key]) => key), "Market"], top: 0, type: "scroll", textStyle: { color: "rgba(255,255,255,0.55)", fontSize: 10, fontFamily: "ui-monospace, monospace" } },
      xAxis: { type: "category", data: buckets, axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } }, axisTick: { show: false }, axisLabel: { color: "rgba(255,255,255,0.35)", fontSize: 10, fontFamily: "ui-monospace, monospace" } },
      yAxis: { type: "value", max: 100, axisLabel: { color: "rgba(255,255,255,0.35)", fontSize: 10, fontFamily: "ui-monospace, monospace", formatter: (value: number) => `${value}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.045)" } } },
      series,
    }
  }, [allBuckets, marketPrices, models])

  return <div className="h-full min-h-[350px] w-full"><ReactECharts option={option} notMerge={true} style={{ height: "100%", width: "100%" }} /></div>
}
