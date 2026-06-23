import { useMemo, useState } from "react"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"

const sortBuckets = (a: string, b: string) => {
  const parseBucket = (s: string) => {
    if (s.startsWith("<") || s.includes(" or below")) return -999
    if (s.startsWith(">=") || s.includes(" or higher")) return 999
    const singleMatch = s.match(/(\d+)°?C/)
    if (singleMatch) return parseFloat(singleMatch[1])
    const rangeMatch = s.match(/(\d+)-(\d+)/)
    if (rangeMatch) return parseFloat(rangeMatch[1])
    return 0
  }
  return parseBucket(a) - parseBucket(b)
}

export default function BucketChart({
  modelProbs, marketPrices, allBuckets,
}: {
  modelProbs: Record<string, number>; marketPrices: Record<string, number>; allBuckets: string[];
}) {
  const [viewMode, setViewMode] = useState<"edge" | "prob">("edge")

  const { sortedBuckets, edgeData, probData } = useMemo(() => {
    const buckets = allBuckets && allBuckets.length > 0
      ? allBuckets
      : Array.from(new Set([...Object.keys(modelProbs), ...Object.keys(marketPrices)]))

    const sorted = [...buckets].sort(sortBuckets)

    const edge = sorted.map((bucket) => {
      const modelProb = modelProbs[bucket] != null ? modelProbs[bucket] * 100 : 0
      const marketPrice = marketPrices[bucket] != null ? marketPrices[bucket] * 100 : 0
      return +(modelProb - marketPrice).toFixed(1)
    })

    const prob = sorted.map((bucket) => ({
      model: modelProbs[bucket] != null ? +(modelProbs[bucket] * 100).toFixed(1) : 0,
      market: marketPrices[bucket] != null ? +(marketPrices[bucket] * 100).toFixed(1) : 0,
    }))

    return { sortedBuckets: sorted, edgeData: edge, probData: prob }
  }, [modelProbs, marketPrices, allBuckets])

  const option = useMemo<EChartsOption>(() => {
    const isEdge = viewMode === "edge"

    const baseTooltip = {
      trigger: "axis" as const,
      axisPointer: { type: "shadow" as const },
      backgroundColor: "#0f1013",
      borderColor: "rgba(255,255,255,0.08)",
      borderWidth: 1,
      borderRadius: 4,
      padding: [8, 12],
      textStyle: { color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: 11 },
      extraCssText: "box-shadow: 0 10px 40px -10px rgba(0,0,0,0.8); backdrop-filter: blur(4px); z-index: 100;",
    }

    if (isEdge) {
      return {
        animation: true,
        animationDuration: 600,
        grid: { left: "2%", right: "2%", bottom: "3%", top: "15%", containLabel: true },
        tooltip: {
          ...baseTooltip,
          formatter: (params: any) => {
            const bucket = params[0].name
            const edge = params[0].value
            const modelProb = modelProbs[bucket] != null ? (modelProbs[bucket] * 100).toFixed(1) : "0.0"
            const marketPrice = marketPrices[bucket] != null ? (marketPrices[bucket] * 100).toFixed(1) : "0.0"
            const action = edge > 0 ? "BUY YES" : edge < 0 ? "BUY NO" : "PASS"
            const color = edge > 0 ? "#34d399" : edge < 0 ? "#fb7185" : "#94a3b8"
            
            return `
              <div style="font-size:10px; color:#38bdf8; letter-spacing:0.1em; margin-bottom:6px;">${bucket} BUCKET</div>
              <div style="display:flex; justify-content:space-between; gap:16px; margin-bottom:2px;">
                <span style="color:#94a3b8;">Model Prob</span>
                <span style="color:#fff;">${modelProb}%</span>
              </div>
              <div style="display:flex; justify-content:space-between; gap:16px; margin-bottom:6px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="color:#94a3b8;">Market Price</span>
                <span style="color:#fff;">${marketPrice}%</span>
              </div>
              <div style="display:flex; justify-content:space-between; gap:16px; margin-bottom:4px;">
                <span style="color:${color};">Model Edge</span>
                <span style="color:${color}; font-weight:600;">${edge > 0 ? "+" : ""}${edge.toFixed(1)}%</span>
              </div>
              <div style="margin-top:4px; padding-top:4px; border-top:1px solid rgba(255,255,255,0.05); text-align:center;">
                <span style="font-size:9px; color:${color}; background:${color}15; padding:2px 6px; border-radius:2px;">ACTION: ${action}</span>
              </div>
            `
          }
        },
        xAxis: {
          type: "category",
          data: sortedBuckets,
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
          axisTick: { show: false },
          axisLabel: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono", margin: 12 },
        },
        yAxis: {
          type: "value",
          axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", formatter: "{value}%" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLine: { show: false },
          axisTick: { show: false },
        },
        series: [
          {
            name: "Edge",
            type: "bar",
            data: edgeData.map((val) => ({
              value: val,
              itemStyle: {
                color: val > 0 ? "#34d399" : val < 0 ? "#fb7185" : "#475569",
                shadowBlur: 15,
                shadowColor: val > 0 ? "rgba(52, 211, 153, 0.6)" : val < 0 ? "rgba(251, 113, 133, 0.6)" : "transparent",
                borderRadius: val > 0 ? [2, 2, 0, 0] : [0, 0, 2, 2],
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 25,
                  shadowColor: val > 0 ? "rgba(52, 211, 153, 0.9)" : val < 0 ? "rgba(251, 113, 133, 0.9)" : "transparent",
                }
              }
            })),
            barWidth: "40%",
            markLine: {
              symbol: "none",
              lineStyle: { color: "rgba(255,255,255,0.15)", type: "solid", width: 1 },
              data: [{ yAxis: 0 }],
              label: { show: false }
            }
          },
        ],
      }
    } else {
      return {
        animation: true,
        animationDuration: 600,
        grid: { left: "2%", right: "2%", bottom: "3%", top: "15%", containLabel: true },
        tooltip: {
          ...baseTooltip,
          formatter: (params: any) => {
            const colors = ["#38bdf8", "#94a3b8"]
            return params.map((p: any, idx: number) => `
              <div style="display:flex; justify-content:space-between; gap:16px;">
                <span style="color:${colors[idx]};">● ${p.seriesName}</span>
                <span style="color:#fff;">${p.value}%</span>
              </div>
            `).join("")
          }
        },
        legend: {
          data: ["Model", "Market"],
          textStyle: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono" },
          top: 0,
          right: 10,
          icon: "rect",
          itemWidth: 8,
          itemHeight: 8,
        },
        xAxis: {
          type: "category",
          data: sortedBuckets,
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
          axisTick: { show: false },
          axisLabel: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono", margin: 12 },
        },
        yAxis: {
          type: "value",
          max: 100,
          axisLabel: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono", formatter: "{value}%" },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLine: { show: false },
          axisTick: { show: false },
        },
        series: [
          {
            name: "Model",
            type: "bar",
            data: probData.map(d => d.model),
            itemStyle: {
              color: {
                type: "linear", x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: "rgba(56, 189, 248, 1)" },
                  { offset: 1, color: "rgba(56, 189, 248, 0.2)" },
                ],
              },
              borderRadius: [2, 2, 0, 0],
              shadowBlur: 10,
              shadowColor: "rgba(56, 189, 248, 0.4)"
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 20,
                shadowColor: "rgba(56, 189, 248, 0.8)"
              }
            },
            label: {
              show: true,
              position: "top",
              color: "#38bdf8",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 10,
              formatter: "{c}%"
            }
          },
          {
            name: "Market",
            type: "bar",
            data: probData.map(d => d.market),
            itemStyle: {
              color: {
                type: "linear", x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: "rgba(148, 163, 184, 1)" },
                  { offset: 1, color: "rgba(148, 163, 184, 0.2)" },
                ],
              },
              borderRadius: [2, 2, 0, 0],
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 20,
                shadowColor: "rgba(148, 163, 184, 0.6)"
              }
            },
            label: {
              show: true,
              position: "top",
              color: "#94a3b8",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 10,
              formatter: "{c}%"
            }
          },
        ],
      }
    }
  }, [viewMode, sortedBuckets, edgeData, probData, modelProbs, marketPrices])

  return (
    <div className="relative w-full h-full min-h-[300px]">
      <div className="absolute top-0 right-0 z-20 flex items-center gap-1 p-1 bg-white/[0.03] border border-white/[0.06] rounded-md backdrop-blur-sm">
        <button
          onClick={() => setViewMode("edge")}
          className={`px-3 py-1 text-[10px] font-mono uppercase tracking-widest rounded-sm transition-all ${
            viewMode === "edge"
              ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Edge
        </button>
        <button
          onClick={() => setViewMode("prob")}
          className={`px-3 py-1 text-[10px] font-mono uppercase tracking-widest rounded-sm transition-all ${
            viewMode === "prob"
              ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Prob
        </button>
      </div>

      {/* 关键修复：增加 notMerge={true}，防止切换视图时配置合并导致渲染异常 */}
      <ReactECharts 
        option={option} 
        notMerge={true}
        style={{ position: 'absolute', top: 0, left: 0, height: '100%', width: '100%' }} 
      />
    </div>
  )
}