import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption, SeriesOption } from "echarts"
import { fetchModelsComparison } from "../api/client"

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

const AXIS_LABEL = {
  color: "rgba(255,255,255,0.35)",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 10,
}

type Point = number | null

function parseTime(timestamp: string): string {
  try {
    const parsed = new Date(timestamp)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleTimeString("en-HK", { hour: "2-digit", minute: "2-digit", hour12: false })
    }
  } catch {
    // Fall through to the source timestamp format.
  }
  if (timestamp.length >= 16) return timestamp.slice(11, 16)
  if (timestamp.length >= 5) return timestamp.slice(0, 5)
  return timestamp
}

function padArray(values: Point[] | undefined, length: number): Point[] {
  if (!values) return new Array(length).fill(null)
  if (values.length === length) return values
  if (values.length > length) return values.slice(0, length)
  return [...values, ...new Array(length - values.length).fill(null)]
}

function averageAt(series: Point[][], index: number): number | null {
  const values = series.map((values) => values[index]).filter((value): value is number => value != null && Number.isFinite(value))
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null
}

function SeriesChip({ color, label, dash, dimmed, active, onClick }: {
  color: string
  label: string
  dash?: boolean
  dimmed?: boolean
  active?: boolean
  onClick: () => void
}) {
  return (
    <button onClick={onClick} className={`group flex items-center gap-2 rounded-full border px-2.5 py-1 transition-all duration-150 ${active ? "border-white/20 bg-white/[0.07]" : "border-transparent hover:border-white/10 hover:bg-white/[0.04]"} ${dimmed ? "opacity-30" : "opacity-100"}`} title={active ? "Clear focus" : "Focus this series"}>
      {dash ? (
        <span className="flex w-4 items-center gap-[2px]">
          <span className="h-[2px] w-[4px] rounded-sm" style={{ backgroundColor: color }} />
          <span className="h-[2px] w-[4px] rounded-sm" style={{ backgroundColor: color }} />
          <span className="h-[2px] w-[4px] rounded-sm" style={{ backgroundColor: color }} />
        </span>
      ) : <span className="h-[7px] w-[7px] rounded-full" style={{ backgroundColor: color, boxShadow: dimmed ? "none" : `0 0 8px ${color}55` }} />}
      <span className="mono text-[10px] font-medium tracking-wide text-white/65 group-hover:text-white/90">{label}</span>
    </button>
  )
}

export default function ModelsComparisonChart({ date, isMinTemp, visibleKeys }: { date: string; isMinTemp: boolean; visibleKeys?: Set<string> }) {
  const [isolated, setIsolated] = useState<string | null>(null)
  const { data, isLoading, isError } = useQuery({
    queryKey: ["modelsComparison", date, isMinTemp],
    queryFn: ({ signal }) => fetchModelsComparison(date, isMinTemp, signal),
    refetchInterval: 60_000,
    refetchIntervalInBackground: true,
    refetchOnMount: "always",
    staleTime: 0,
    retry: 1,
  })

  const chart = useMemo(() => {
    if (!data || !data.timestamps?.length) return null

    const timestamps = data.timestamps.map(parseTime)
    const length = timestamps.length
    const actual = padArray(data.actual_temps, length)
    const market = padArray(data.market_temps, length)
    const visibleModels = Object.entries(data.models).filter(([key]) => !visibleKeys || visibleKeys.has(key))
    const modelValues = visibleModels.map(([, values]) => padArray(values, length))
    const consensus = timestamps.map((_, index) => averageAt(modelValues, index))

    const allValues = [...actual, ...market, ...consensus, ...modelValues.flat()].filter((value): value is number => value != null && Number.isFinite(value))
    const rawMin = allValues.length > 0 ? Math.min(...allValues) : 20
    const rawMax = allValues.length > 0 ? Math.max(...allValues) : 35
    const spread = Math.max(rawMax - rawMin, 1)
    const min = Math.floor((rawMin - spread * 0.08) * 2) / 2
    const max = Math.ceil((rawMax + spread * 0.08) * 2) / 2
    let lastActualIndex = -1
    actual.forEach((value, index) => { if (value != null) lastActualIndex = index })

    const isDimmed = (id: string) => Boolean(isolated && isolated !== id)
    const line = (id: string, name: string, values: Point[], color: string, options: { dash?: boolean; area?: boolean; z?: number } = {}): SeriesOption => {
      const dimmed = isDimmed(id)
      const focused = isolated === id
      return {
        name,
        type: "line",
        step: "end",
        symbol: "none",
        data: values,
        connectNulls: true,
        lineStyle: { width: focused ? 3 : options.area ? 2.4 : options.dash ? 2 : 1.15, color, opacity: dimmed ? 0.1 : options.area || options.dash || focused ? 1 : 0.42, type: options.dash ? "dashed" : "solid" },
        itemStyle: { color, opacity: dimmed ? 0.1 : 1 },
        emphasis: { focus: "series", lineStyle: { width: 3, opacity: 1 } },
        areaStyle: options.area ? { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${color}1c` }, { offset: 1, color: `${color}00` }] } } : undefined,
        z: options.z ?? 1,
        markLine: id === "actual" && lastActualIndex >= 0 ? { silent: true, symbol: "none", animation: false, lineStyle: { color: "rgba(255,255,255,0.28)", width: 1, type: "dashed" }, label: { formatter: "NOW", position: "insideEndTop", color: "rgba(255,255,255,0.45)", fontFamily: "ui-monospace, monospace", fontSize: 9.5 }, data: [{ xAxis: timestamps[lastActualIndex] }] } : undefined,
      }
    }

    const series: SeriesOption[] = [
      line("actual", "Actual Temperature", actual, "#fb7185", { area: true, z: 4 }),
      line("market", "Polymarket Weighted", market, "#a78bfa", { dash: true, z: 3 }),
      line("consensus", "Consensus (Mean)", consensus, "#22d3ee", { area: true, z: 3 }),
      ...visibleModels.map(([key, values]) => line(`model:${key}`, MODEL_LABELS[key] || key, padArray(values, length), MODEL_COLORS[key] || "#94a3b8")),
    ]

    const option: EChartsOption = {
      animationDuration: 500,
      grid: { left: 52, right: 20, top: 18, bottom: 32 },
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
          const rows = entries.filter((entry: any) => entry.value != null).map((entry: any) => `<div style="display:flex;justify-content:space-between;gap:20px;margin-top:3px"><span style="color:${entry.color}">${entry.seriesName}</span><span style="color:#fff;font-weight:600">${Number(entry.value).toFixed(2)}°C</span></div>`).join("")
          return `<div style="font-family:ui-monospace,monospace;font-size:10px;color:rgba(255,255,255,.55);margin-bottom:6px">${title}</div>${rows}`
        },
      },
      xAxis: {
        type: "category",
        data: timestamps,
        boundaryGap: false,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
        axisTick: { show: false },
        axisLabel: { ...AXIS_LABEL, interval: Math.max(1, Math.floor(length / 8)) },
      },
      yAxis: {
        type: "value",
        min,
        max,
        interval: 1,
        axisLabel: { ...AXIS_LABEL, formatter: (value: number) => `${value}°C` },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.045)" } },
      },
      series,
    }

    const chips = [
      { id: "actual", label: "Actual", color: "#fb7185" },
      { id: "market", label: "Polymarket", color: "#a78bfa", dash: true },
      { id: "consensus", label: "Consensus", color: "#22d3ee" },
      ...visibleModels.map(([key]) => ({ id: `model:${key}`, label: MODEL_LABELS[key] || key, color: MODEL_COLORS[key] || "#94a3b8" })),
    ]
    return {
      option,
      chips,
      visibleCount: visibleModels.length,
      timestampCount: length,
      granularity: data.granularity || "minute",
    }
  }, [data, isolated, visibleKeys])

  if (isLoading && !data) return <div className="flex h-[360px] w-full items-center justify-center text-sm text-white/40">Loading temperature tracking data...</div>
  if (isError && !data) return <div className="flex h-[360px] w-full items-center justify-center text-sm text-rose-300">Failed to load tracking data.</div>
  if (!chart) return <div className="flex h-[360px] w-full items-center justify-center text-xs text-white/35">No trajectory data available yet.</div>

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1">
        {chart.chips.map((chip) => (
          <SeriesChip key={chip.id} color={chip.color} label={chip.label} dash={chip.dash} active={isolated === chip.id} dimmed={Boolean(isolated && isolated !== chip.id)} onClick={() => setIsolated(isolated === chip.id ? null : chip.id)} />
        ))}
        {isolated && <button onClick={() => setIsolated(null)} className="mono ml-1 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-2.5 py-1 text-[10px] font-semibold text-cyan-300 transition-colors hover:bg-cyan-400/20">Show all</button>}
      </div>
      <div className="h-[330px] w-full sm:h-[360px]">
        <ReactECharts option={chart.option} notMerge={false} replaceMerge={["series"]} lazyUpdate={true} style={{ height: "100%", width: "100%" }} />
      </div>
      <div className="mono text-[10.5px] tracking-wide text-white/25">Click a series to focus it · {chart.visibleCount} model series · {chart.timestampCount} observations · dashed line = market weighted temperature</div>
    </div>
  )
}
