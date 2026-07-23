import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import ReactECharts from "echarts-for-react"
import type { EChartsOption, SeriesOption } from "echarts"
import { fetchModelsComparison } from "../api/client"
import {
  buildTrajectoryTooltip,
  canConnectRealModelCycles,
  formatHktTime,
  segmentCoordinatesByCadence,
  timestampToEpochMillis,
} from "../lib/trajectoryFormatting.js"
import type { TrajectoryPointMetadata } from "../types"

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
type Coordinate = [number, number]
type ChartPoint = {
  timestamp: string
  epoch: number
  metadata: TrajectoryPointMetadata
}

function padArray(values: Point[] | undefined, length: number): Point[] {
  if (!values) return new Array(length).fill(null)
  if (values.length === length) return values
  if (values.length > length) return values.slice(0, length)
  return [...values, ...new Array(length - values.length).fill(null)]
}

function coordinate(timestamp: string | null | undefined, value: Point): Coordinate | null {
  if (value == null || !Number.isFinite(value)) return null
  const epoch = timestampToEpochMillis(timestamp)
  return epoch == null ? null : [epoch, value]
}

function segmentRealCycles(values: Point[], points: ChartPoint[], expectedIntervalSeconds: number | null): Coordinate[][] {
  const segments: Coordinate[][] = []
  let segment: Coordinate[] = []
  let previous: ChartPoint | null = null
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index]
    const realCycle = point.metadata.model_cycle_is_real === true
    const decisionTimestamp = point.metadata.prediction_decision_timestamp
    const value = values[index]
    const next = realCycle ? coordinate(decisionTimestamp, value) : null
    if (!next || !decisionTimestamp) continue
    if (!previous || !canConnectRealModelCycles(previous.metadata.prediction_decision_timestamp, decisionTimestamp, expectedIntervalSeconds)) {
      if (segment.length) segments.push(segment)
      segment = [next]
    } else {
      segment.push(next)
    }
    previous = point
  }
  if (segment.length) segments.push(segment)
  return segments
}

function averageCoordinates(modelSegments: Coordinate[][][]): Coordinate[] {
  const valuesByTime = new Map<number, number[]>()
  for (const segments of modelSegments) {
    for (const segment of segments) {
      for (const [timestamp, value] of segment) {
        const values = valuesByTime.get(timestamp) || []
        values.push(value)
        valuesByTime.set(timestamp, values)
      }
    }
  }
  return [...valuesByTime.entries()]
    .map(([timestamp, values]) => [timestamp, values.reduce((sum, value) => sum + value, 0) / values.length] as Coordinate)
    .sort(([left], [right]) => left - right)
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

    const length = data.timestamps.length
    const rawMetadata = data.point_metadata || []
    const points: ChartPoint[] = data.timestamps.flatMap((timestamp, index) => {
      const epoch = timestampToEpochMillis(timestamp)
      if (epoch == null) return []
      const supplied = rawMetadata[index]
      // Legacy rows retain one stored strategy-cycle timestamp per row.  They
      // are not expanded into minute points, and modern Layer A rows must
      // explicitly identify a real canonical cycle before a model is drawn.
      const isLegacyStrategyCycle = data.granularity === "strategy_cycle" && supplied?.model_cycle_is_real == null
      return [{
        timestamp,
        epoch,
        metadata: {
          timestamp: supplied?.timestamp ?? timestamp,
          actual_observation_timestamp: supplied?.actual_observation_timestamp ?? null,
          actual_first_seen_timestamp: supplied?.actual_first_seen_timestamp ?? null,
          actual_source_release_timestamp: supplied?.actual_source_release_timestamp ?? null,
          actual_release_lag_seconds: supplied?.actual_release_lag_seconds ?? null,
          prediction_decision_timestamp: supplied?.prediction_decision_timestamp ?? (isLegacyStrategyCycle ? timestamp : null),
          weather_data_through: supplied?.weather_data_through ?? null,
          weather_first_seen_timestamp: supplied?.weather_first_seen_timestamp ?? null,
          weather_age_seconds: supplied?.weather_age_seconds ?? null,
          weather_snapshot_id: supplied?.weather_snapshot_id ?? null,
          model_cycle_id: supplied?.model_cycle_id ?? null,
          model_age_seconds: supplied?.model_age_seconds ?? null,
          model_cycle_is_real: supplied?.model_cycle_is_real ?? isLegacyStrategyCycle,
        },
      }]
    })
    if (!points.length) return null

    const actual = padArray(data.actual_temps, length)
    const market = padArray(data.market_temps, length)
    const actualCoordinates = points.flatMap((point, index) => {
      const sourceTimestamp = point.metadata.actual_observation_timestamp || (data.granularity === "strategy_cycle" ? point.timestamp : null)
      const value = coordinate(sourceTimestamp, actual[index])
      return value ? [value] : []
    })
    const marketCoordinates = points.flatMap((point, index) => {
      const value = coordinate(point.timestamp, market[index])
      return value ? [value] : []
    })
    const visibleModels = Object.entries(data.models).filter(([key]) => !visibleKeys || visibleKeys.has(key))
    const expectedIntervalSeconds = Number.isFinite(data.expected_model_cycle_interval_seconds) && (data.expected_model_cycle_interval_seconds || 0) > 0
      ? data.expected_model_cycle_interval_seconds || null
      : null
    const modelSegments = visibleModels.map(([, values]) => segmentRealCycles(padArray(values, length), points, expectedIntervalSeconds))
    const consensusSegments = segmentCoordinatesByCadence(
      averageCoordinates(modelSegments),
      expectedIntervalSeconds,
    )

    const allValues = [
      ...actualCoordinates,
      ...marketCoordinates,
      ...consensusSegments.flat(),
      ...modelSegments.flat(2),
    ].map(([, value]) => value)
    const rawMin = allValues.length > 0 ? Math.min(...allValues) : 20
    const rawMax = allValues.length > 0 ? Math.max(...allValues) : 35
    const spread = Math.max(rawMax - rawMin, 1)
    const min = Math.floor((rawMin - spread * 0.08) * 2) / 2
    const max = Math.ceil((rawMax + spread * 0.08) * 2) / 2
    const lastActual = actualCoordinates.at(-1)
    const metadataByEpoch = new Map(points.map((point) => [point.epoch, point]))

    const isDimmed = (id: string) => Boolean(isolated && isolated !== id)
    const line = (id: string, name: string, values: Coordinate[], color: string, options: { dash?: boolean; area?: boolean; z?: number; step?: boolean; showSymbol?: boolean } = {}): SeriesOption => {
      const dimmed = isDimmed(id)
      const focused = isolated === id
      return {
        name,
        type: "line",
        step: options.step ? "end" : false,
        symbol: options.showSymbol ? "circle" : "none",
        symbolSize: options.showSymbol ? 5 : 0,
        showSymbol: options.showSymbol ?? false,
        data: values,
        lineStyle: { width: focused ? 3 : options.area ? 2.4 : options.dash ? 2 : 1.15, color, opacity: dimmed ? 0.1 : options.area || options.dash || focused ? 1 : 0.42, type: options.dash ? "dashed" : "solid" },
        itemStyle: { color, opacity: dimmed ? 0.1 : 1 },
        emphasis: { focus: "series", lineStyle: { width: 3, opacity: 1 } },
        areaStyle: options.area ? { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${color}1c` }, { offset: 1, color: `${color}00` }] } } : undefined,
        z: options.z ?? 1,
        markLine: id === "actual" && lastActual ? { silent: true, symbol: "none", animation: false, lineStyle: { color: "rgba(255,255,255,0.28)", width: 1, type: "dashed" }, label: { formatter: "LATEST ACTUAL", position: "insideEndTop", color: "rgba(255,255,255,0.45)", fontFamily: "ui-monospace, monospace", fontSize: 9.5 }, data: [{ xAxis: lastActual[0] }] } : undefined,
      }
    }

    const series: SeriesOption[] = [
      line("actual", "Actual Temperature", actualCoordinates, "#fb7185", { area: true, z: 4 }),
      line("market", "Polymarket Weighted", marketCoordinates, "#a78bfa", { dash: true, z: 3 }),
      ...consensusSegments.map((segment) => line("consensus", "Consensus (Mean)", segment, "#22d3ee", { area: true, z: 3, step: true, showSymbol: true })),
      ...visibleModels.flatMap(([key], index) => modelSegments[index].map((segment) => line(`model:${key}`, MODEL_LABELS[key] || key, segment, MODEL_COLORS[key] || "#94a3b8", { step: true, showSymbol: true }))),
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
          const first = entries[0]
          const axisEpoch = Array.isArray(first?.value) ? first.value[0] : first?.axisValue
          const point = metadataByEpoch.get(Number(axisEpoch))
          return buildTrajectoryTooltip({
            timestamp: point?.timestamp || axisEpoch,
            metadata: point?.metadata,
            entries: entries.map((entry: any) => ({
              seriesName: entry.seriesName,
              color: entry.color,
              value: Array.isArray(entry.value) ? entry.value[1] : entry.value,
            })),
            expectedCycleIntervalSeconds: expectedIntervalSeconds,
          })
        },
      },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
        axisTick: { show: false },
        axisLabel: { ...AXIS_LABEL, formatter: (value: number) => formatHktTime(value) },
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
      timestampCount: points.length,
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
      <div className="mono text-[10.5px] tracking-wide text-white/25">All times HKT · model markers are real decision cycles only · {chart.visibleCount} model series · {chart.timestampCount} source timestamps</div>
    </div>
  )
}
