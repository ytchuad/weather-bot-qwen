import { useMemo } from "react"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"

function sortBuckets(a: string, b: string) {
  const parse = (value: string) => {
    if (value.startsWith("<") || value.includes("or below")) return -999
    if (value.startsWith(">") || value.includes("or higher")) return 999
    const match = value.match(/(\d+)/)
    return match ? Number(match[1]) : 0
  }
  return parse(a) - parse(b)
}

const AXIS_LABEL = {
  color: "rgba(255,255,255,0.35)",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 10,
}

export default function BucketChart({ viewMode, modelProbs, marketPrices, allBuckets }: {
  viewMode: "edge" | "prob"
  modelProbs: Record<string, number>
  marketPrices: Record<string, number>
  allBuckets: string[]
}) {

  const { sortedBuckets, edgeData, probData, topBucket } = useMemo(() => {
    const buckets = allBuckets.length > 0 ? allBuckets : [...new Set([...Object.keys(modelProbs), ...Object.keys(marketPrices)])]
    const sorted = [...buckets].sort(sortBuckets)
    const edge = sorted.map((bucket) => +(((modelProbs[bucket] ?? 0) - (marketPrices[bucket] ?? 0)) * 100).toFixed(1))
    const probs = sorted.map((bucket) => ({ model: +((modelProbs[bucket] ?? 0) * 100).toFixed(1), market: +((marketPrices[bucket] ?? 0) * 100).toFixed(1) }))
    const highest = probs.reduce<{ bucket: string; value: number }>((best, point, index) => point.market > best.value ? { bucket: sorted[index], value: point.market } : best, { bucket: "", value: 0 })
    return { sortedBuckets: sorted, edgeData: edge, probData: probs, topBucket: highest.bucket }
  }, [allBuckets, modelProbs, marketPrices])

  const option = useMemo<EChartsOption>(() => {
    const base: EChartsOption = {
      animationDuration: 500,
      grid: { left: 48, right: 18, top: 34, bottom: 32 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow", shadowStyle: { color: "rgba(255,255,255,0.03)" } },
        backgroundColor: "rgba(9,12,20,0.96)",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        padding: [10, 14],
        textStyle: { color: "rgba(255,255,255,0.85)", fontSize: 11, fontFamily: "ui-monospace, monospace" },
      },
      xAxis: { type: "category", data: sortedBuckets, axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } }, axisTick: { show: false }, axisLabel: { ...AXIS_LABEL, interval: 0 } },
    }

    if (viewMode === "edge") {
      return {
        ...base,
        grid: { ...base.grid, top: 30 },
        yAxis: { type: "value", axisLabel: { ...AXIS_LABEL, formatter: (value: number) => `${value > 0 ? "+" : ""}${value}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.045)" } } },
        series: [{
          name: "Model Edge",
          type: "bar",
          data: edgeData.map((value) => {
            const positive = value > 0
            const negative = value < 0
            const color = positive ? "rgba(52,211,153,0.82)" : negative ? "rgba(251,113,133,0.82)" : "rgba(148,163,184,0.5)"
            return {
              value,
              itemStyle: {
                color,
                borderRadius: positive ? [4, 4, 0, 0] : [0, 0, 4, 4],
                shadowBlur: 12,
                shadowColor: positive ? "rgba(52,211,153,0.35)" : negative ? "rgba(251,113,133,0.35)" : "transparent",
              },
            }
          }),
          barWidth: "52%",
          label: { show: true, position: "top", fontFamily: "ui-monospace, monospace", fontSize: 9.5, color: "rgba(255,255,255,0.55)", formatter: (params: any) => Math.abs(Number(params.value)) >= 1 ? `${Number(params.value) > 0 ? "+" : ""}${Number(params.value).toFixed(1)}%` : "" },
          markLine: { symbol: "none", lineStyle: { color: "rgba(255,255,255,0.2)", width: 1 }, label: { show: false }, data: [{ yAxis: 0 }] },
        }],
      }
    }

    return {
      ...base,
      legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, icon: "roundRect", textStyle: { color: "rgba(255,255,255,0.55)", fontSize: 11, fontFamily: "ui-monospace, monospace" } },
      yAxis: { type: "value", max: 100, axisLabel: { ...AXIS_LABEL, formatter: (value: number) => `${value}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.045)" } } },
      series: [
        {
          name: "Model",
          type: "bar",
          data: probData.map((point) => point.model),
          barWidth: "30%",
          itemStyle: { borderRadius: [4, 4, 0, 0], shadowBlur: 16, shadowColor: "rgba(34,211,238,0.42)", shadowOffsetY: 2, color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(34,211,238,0.98)" }, { offset: 1, color: "rgba(34,211,238,0.32)" }] } },
          emphasis: { itemStyle: { shadowBlur: 24, shadowColor: "rgba(34,211,238,0.62)" } },
          label: { show: true, position: "top", fontFamily: "ui-monospace, monospace", fontSize: 9.5, color: "rgba(103,232,249,0.85)", formatter: (params: any) => Number(params.value) >= 1 ? `${Number(params.value).toFixed(1)}%` : "" },
          markArea: topBucket ? { silent: true, itemStyle: { color: "rgba(34,211,238,0.045)" }, data: [[{ xAxis: topBucket }, { xAxis: topBucket }]] } : undefined,
        },
        {
          name: "Market",
          type: "bar",
          data: probData.map((point) => point.market),
          barWidth: "30%",
          itemStyle: { borderRadius: [4, 4, 0, 0], shadowBlur: 16, shadowColor: "rgba(167,139,250,0.42)", shadowOffsetY: 2, color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(167,139,250,0.95)" }, { offset: 1, color: "rgba(167,139,250,0.3)" }] } },
          emphasis: { itemStyle: { shadowBlur: 24, shadowColor: "rgba(167,139,250,0.62)" } },
          label: { show: true, position: "top", fontFamily: "ui-monospace, monospace", fontSize: 9.5, color: "rgba(196,181,253,0.9)", formatter: (params: any) => Number(params.value) >= 1 ? `${Number(params.value).toFixed(1)}%` : "" },
        },
      ],
    }
  }, [edgeData, marketPrices, modelProbs, probData, sortedBuckets, topBucket, viewMode])

  return (
    <div className="relative h-full min-h-[300px] w-full">
      <ReactECharts option={option} notMerge={true} style={{ position: "absolute", inset: 0, height: "100%", width: "100%" }} />
    </div>
  )
}
