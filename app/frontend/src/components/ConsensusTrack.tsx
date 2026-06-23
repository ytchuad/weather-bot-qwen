import { useMemo } from "react"
import ReactECharts from "echarts-for-react"
import type { EChartsOption } from "echarts"
import type { ModelPrediction } from "../types"

function generateNormalData(mu: number, sigma: number, minX: number, maxX: number) {
  const points: [number, number][] = []
  const step = (maxX - minX) / 100
  for (let x = minX; x <= maxX; x += step) {
    const y = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma, 2)))
    points.push([x, y])
  }
  return points
}

export default function ConsensusTrack({
  models, activeKey, visibleKeys,
}: {
  models: [string, ModelPrediction][]; activeKey: string | null; visibleKeys: Set<string> | null;
}) {
  const { option, range, visibleCount } = useMemo(() => {
    const visibleModels = models.filter(([k]) => !visibleKeys || visibleKeys.has(k))
    if (visibleModels.length === 0) return { option: {}, range: [0, 1], visibleCount: 0 }

    const means = visibleModels.map(([, p]) => p.mean)
    const stds = visibleModels.map(([, p]) => p.std)
    
    const minMean = Math.min(...means) - Math.max(...stds) - 1
    const maxMean = Math.max(...means) + Math.max(...stds) + 1

    const series = visibleModels.map(([key, pred]) => {
      const isActive = key === activeKey
      const data = generateNormalData(pred.mean, pred.std || 0.5, minMean, maxMean)
      
      return {
        name: key,
        type: "line",
        smooth: true,
        showSymbol: false,
        data: data,
        lineStyle: {
          width: isActive ? 3 : 1.5,
          color: isActive ? "#38bdf8" : "rgba(148, 163, 184, 0.6)",
          shadowBlur: isActive ? 20 : 0,
          shadowColor: isActive ? "rgba(56, 189, 248, 0.8)" : "transparent"
        },
        areaStyle: {
          color: isActive ? {
            type: "linear", x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(56, 189, 248, 0.4)" },
              { offset: 1, color: "rgba(56, 189, 248, 0)" }
            ]
          } : "rgba(148, 163, 184, 0.1)",
        },
        z: isActive ? 10 : 1
      }
    })

    const option: EChartsOption = {
      animation: true,
      animationDuration: 600,
      grid: { left: "2%", right: "2%", bottom: "20%", top: "10%", containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0f1013",
        borderColor: "rgba(255,255,255,0.08)",
        borderWidth: 1,
        borderRadius: 4,
        textStyle: { color: "#e2e8f0", fontFamily: "JetBrains Mono, monospace", fontSize: 11 },
        extraCssText: "box-shadow: 0 10px 40px -10px rgba(0,0,0,0.8); backdrop-filter: blur(4px);",
        formatter: (params: any) => `Temp: ${parseFloat(params[0].axisValue).toFixed(1)}°C`
      },
      xAxis: {
        type: "value",
        min: minMean,
        max: maxMean,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
        axisTick: { show: false },
        axisLabel: {
          color: "#94a3b8",
          fontSize: 10,
          fontFamily: "JetBrains Mono",
          formatter: (val: number) => `${val.toFixed(1)}°C`
        },
        splitLine: { show: false }
      },
      yAxis: {
        type: "value",
        show: false,
      },
      series: series as any,
    }

    return { option, range: [minMean, maxMean], visibleCount: visibleModels.length }
  }, [models, activeKey, visibleKeys])

  return (
    <div className="bg-[#0f1013] border border-white/[0.06] rounded-md p-6 mb-6 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.8)]">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-[10px] text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2">
          <span className="w-4 h-px bg-slate-500"></span> Model Consensus Distribution
        </h2>
        <div className="flex gap-8 text-xs mono">
          <div className="flex items-center gap-2 text-slate-300">
            <span className="text-slate-400">VISIBLE:</span> <span className="text-white">{visibleCount}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-300">
            <span className="text-slate-400">RANGE:</span> <span className="text-white">{range[0].toFixed(1)} - {range[1].toFixed(1)}°C</span>
          </div>
        </div>
      </div>
      {/* 关键修复：增加高度至 180px，并添加 notMerge */}
      <div className="h-[180px] w-full">
        {visibleCount > 0 ? (
          <ReactECharts 
            option={option} 
            notMerge={true}
            style={{ height: "100%", width: "100%" }} 
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs">No visible models</div>
        )}
      </div>
    </div>
  )
}