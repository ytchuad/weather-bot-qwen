import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import { fetchStrategyChart } from "../api/client"
import type { StrategyChartTrade } from "../types"

function parseTime(ts: string): string {
  try {
    const d = new Date(ts)
    if (!isNaN(d.getTime())) {
      return d.toLocaleTimeString("en-HK", { hour: "2-digit", minute: "2-digit", hour12: false })
    }
  } catch {
    // fallthrough
  }
  if (ts.length >= 16) return ts.slice(11, 16)
  if (ts.length >= 5) return ts.slice(0, 5)
  return ts
}

const CHART_HEIGHT = 320;

export default function StrategyChart({
  strategyId,
  date,
  slug,
}: {
  strategyId: string
  date: string
  slug?: string
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["strategyChart", strategyId, date, slug],
    queryFn: () => fetchStrategyChart(strategyId, date, slug),
    refetchInterval: 120_000,
  })

  const option = useMemo<EChartsOption>(() => {
    if (!data || !data.timestamps?.length) return {}

    const timestamps = data.timestamps.map(parseTime)
    const maxLen = timestamps.length

    const tradeMarkPoints: any[] = []
    if (data.trades?.length) {
      data.trades.forEach((t: StrategyChartTrade) => {
        // 統一解析時間以進行匹配
        const tradeTime = parseTime(t.time || "")
        const idx = timestamps.findIndex((ts) => ts === tradeTime)
        
        if (idx >= 0) {
          const yVal = data.actual_temps[idx]
          if (yVal != null) {
            const isBuy = t.action === "NEW" || t.action === "INCREASE"
            tradeMarkPoints.push({
              name: `${t.bucket} ${t.action}`,
              coord: [timestamps[idx], yVal],
              itemStyle: { color: isBuy ? "#34d399" : "#fb7185" },
              symbol: isBuy ? "pin" : "arrow",
              symbolSize: 14,
              label: { formatter: t.bucket, fontSize: 9, color: "#94a3b8", position: "top" },
            })
          }
        }
      })
    }

    return {
      animation: true,
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0f1013",
        borderColor: "rgba(255,255,255,0.08)",
        borderWidth: 1,
        padding: [8, 12],
        textStyle: { color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: 11 },
      },
      legend: {
        data: ["Polymarket 加權", "策略預測", "實際氣溫"],
        textStyle: { color: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono" },
        top: 0,
        right: 10,
        icon: "roundRect",
        itemWidth: 12,
        itemHeight: 3,
      },
      grid: { left: "3%", right: "4%", bottom: "3%", top: "15%", containLabel: true },
      xAxis: {
        type: "category",
        data: timestamps,
        boundaryGap: false,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
        axisTick: { show: false },
        axisLabel: { color: "#64748b", fontSize: 9, fontFamily: "JetBrains Mono" },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        name: "°C",
        nameTextStyle: { color: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" },
        axisLabel: { color: "#64748b", fontSize: 9, fontFamily: "JetBrains Mono" },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: [
        {
          name: "Polymarket 加權",
          type: "line",
          smooth: true,
          symbol: "none",
          connectNulls: true,
          data: _padArray(data.market_temps, maxLen),
          lineStyle: { width: 2, color: "#a78bfa" },
          itemStyle: { color: "#a78bfa" },
          areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(167, 139, 250, 0.15)" }, { offset: 1, color: "rgba(167, 139, 250, 0)" }] } },
        },
        {
          name: "策略預測",
          type: "line",
          smooth: true,
          symbol: "none",
          connectNulls: true,
          data: _padArray(data.model_temps, maxLen),
          lineStyle: { width: 2, color: "#10b981" },
          itemStyle: { color: "#10b981" },
        },
        {
          name: "實際氣溫",
          type: "line",
          smooth: true,
          symbol: "circle",
          symbolSize: 4,
          connectNulls: true,
          data: _padArray(data.actual_temps, maxLen),
          lineStyle: { width: 2, color: "#f43f5e" },
          itemStyle: { color: "#f43f5e" },
          markPoint: tradeMarkPoints.length > 0 ? {
            data: tradeMarkPoints,
            symbol: "pin",
            symbolSize: 14,
            label: { fontSize: 9, color: "#f8fafc" },
          } : undefined,
        },
      ],
    }
  }, [data])

  // 統一所有狀態的高度，防止佈局抖動
  if (isLoading) {
    return <div style={{ height: CHART_HEIGHT }} className="w-full rounded-md bg-white/[0.02] animate-pulse" />
  }

  if (isError) {
    return (
      <div style={{ height: CHART_HEIGHT }} className="w-full flex items-center justify-center text-xs text-rose-400">
        Failed to load chart data.
      </div>
    )
  }

  if (!data || !data.timestamps?.length) {
    return (
      <div style={{ height: CHART_HEIGHT }} className="w-full flex items-center justify-center text-xs text-slate-500">
        No snapshot data available yet.
      </div>
    )
  }

  return (
    <div style={{ height: CHART_HEIGHT, width: "100%" }}>
      <ReactECharts option={option} notMerge style={{ height: "100%", width: "100%" }} />
    </div>
  )
}

function _padArray(arr: (number | null)[] | undefined, len: number): (number | null)[] {
  if (!arr) return new Array(len).fill(null)
  if (arr.length >= len) return arr.slice(0, len)
  return [...arr, ...new Array(len - arr.length).fill(null)]
}