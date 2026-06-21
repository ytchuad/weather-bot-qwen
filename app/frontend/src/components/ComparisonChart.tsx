import { useMemo } from "react"
import ReactECharts from "echarts-for-react"
import type { ModelPrediction } from "../types"

const sortBuckets = (a: string, b: string) => {
  const parseBucket = (s: string) => {
    if (s.startsWith("<")) return -999
    if (s.startsWith(">=")) return 999
    if (s.startsWith(">")) return 999
    const num = parseFloat(s.split("-")[0])
    return isNaN(num) ? 0 : num
  }
  return parseBucket(a) - parseBucket(b)
}

export default function ComparisonChart({
  models,
  marketPrices,
  allBuckets,
}: {
  models: [string, ModelPrediction][]
  marketPrices: Record<string, number>
  allBuckets: string[]
}) {
  const colors = ["#06b6d4", "#8b5cf6", "#10b981", "#f43f5e", "#f59e0b"]

  const option = useMemo(() => {
    const sortedBuckets = [...allBuckets].sort(sortBuckets)

    const series = models.map(([key, pred], idx) => ({
      name: key,
      type: "line",
      smooth: true,
      data: sortedBuckets.map(b => pred.probs?.[b] != null ? +(pred.probs[b] * 100).toFixed(1) : 0),
      itemStyle: { color: colors[idx % colors.length] },
      lineStyle: { width: 2, color: colors[idx % colors.length] },
      areaStyle: { opacity: 0.1 },
    })) as any[]

    ;(series as any).push({
      name: "Market",
      type: "bar",
      barWidth: 40,
      data: sortedBuckets.map(b => marketPrices[b] != null ? +(marketPrices[b] * 100).toFixed(1) : 0),
      itemStyle: { color: 'rgba(100, 116, 139, 0.3)' },
    })

    return {
      animation: true,
      tooltip: { trigger: "axis" },
      legend: { data: [...models.map(m => m[0]), "Market"], textStyle: { color: "#94a3b8" }, top: 0 },
      grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
      xAxis: {
        type: "category",
        data: sortedBuckets,
        axisLine: { lineStyle: { color: "#1e293b" } },
        axisLabel: { color: "#64748b", fontSize: 12 },
      },
      yAxis: {
        type: "value",
        max: 100, // 關鍵修正：強制 Y 軸最大值為 100
        axisLabel: { color: "#64748b", fontSize: 12, formatter: "{value}%" },
        splitLine: { lineStyle: { color: "#1e293b", type: "dashed" } },
      },
      series,
    }
  }, [models, marketPrices, allBuckets])

  return (
    <div style={{ height: 350 }}>
      <ReactECharts 
        option={option} 
        notMerge={true} 
        style={{ height: "100%", width: "100%" }} 
      />
    </div>
  )
}