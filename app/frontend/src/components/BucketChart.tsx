import { useMemo } from "react"
import ReactECharts from "echarts-for-react"

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

export default function BucketChart({
  modelProbs,
  marketPrices,
  allBuckets,
}: {
  modelProbs: Record<string, number>
  marketPrices: Record<string, number>
  allBuckets: string[]
}) {
  const data = useMemo(() => {
    const buckets = allBuckets && allBuckets.length > 0
      ? allBuckets
      : Array.from(new Set([...Object.keys(modelProbs), ...Object.keys(marketPrices)]))

    const sortedBuckets = [...buckets].sort(sortBuckets)

    return sortedBuckets.map((bucket) => ({
      bucket,
      Model: modelProbs[bucket] != null ? +(modelProbs[bucket] * 100).toFixed(1) : 0,
      Market: marketPrices[bucket] != null ? +(marketPrices[bucket] * 100).toFixed(1) : 0,
    }))
  }, [modelProbs, marketPrices, allBuckets])

  const option = useMemo(() => {
    return {
      animation: true,
      animationDuration: 500,
      animationDurationUpdate: 500,
      animationEasing: "cubicOut",
      animationEasingUpdate: "cubicOut",

      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        containLabel: true,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        backgroundColor: "#0f172a",
        borderColor: "#1e293b",
        borderRadius: 4,
        textStyle: {
          color: "#e2e8f0",
        },
        extraCssText: "backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); box-shadow: 0 4px 20px rgba(0,0,0,0.5);",
      },
      legend: {
        data: ["Model Probability", "Market Price"],
        textStyle: { color: "#94a3b8" },
        top: 0,
      },
      xAxis: {
        type: "category",
        data: data.map(d => d.bucket),
        axisLine: { lineStyle: { color: "#1e293b" } },
        axisTick: { show: false },
        axisLabel: { color: "#64748b", fontSize: 12 },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          color: "#64748b",
          fontSize: 12,
          formatter: "{value}%",
        },
        splitLine: { lineStyle: { color: "#1e293b", type: "dashed" } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: [
        {
          name: "Model Probability",
          type: "bar",
          data: data.map(d => d.Model),
          itemStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(6, 182, 212, 0.9)" },
                { offset: 1, color: "rgba(6, 182, 212, 0.4)" },
              ],
            },
            borderRadius: [4, 4, 0, 0],
          },
        },
        {
          name: "Market Price",
          type: "bar",
          data: data.map(d => d.Market),
          itemStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(139, 92, 246, 0.9)" },
                { offset: 1, color: "rgba(139, 92, 246, 0.4)" },
              ],
            },
            borderRadius: [4, 4, 0, 0],
          },
        },
      ],
    }
  }, [data])

  return (
    <div style={{ height: 350 }}>
      <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
    </div>
  )
}