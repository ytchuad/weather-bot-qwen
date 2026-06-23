import { useState, useMemo, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import ModelGrid from "../components/ModelGrid"
import WeatherCards from "../components/WeatherCards"
import BucketChart from "../components/BucketChart"
import ComparisonChart from "../components/ComparisonChart"
import ConsensusTrack from "../components/ConsensusTrack"
import { fetchEvent, fetchTodayEvent, fetchPredictions } from "../api/client"
import { GitCompare } from "lucide-react"

export default function Hub() {
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10))
  const [isMinTemp, setIsMinTemp] = useState(false)
  const [compareMode, setCompareMode] = useState(false)

  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [visibleKeys, setVisibleKeys] = useState<Set<string> | null>(null)
  const [order, setOrder] = useState<string[]>([])

  const { data, isLoading } = useQuery({
    queryKey: ["predictions", date, isMinTemp],
    queryFn: () => fetchPredictions(date, isMinTemp),
    refetchInterval: 120_000,
  })

  // 获取 Polymarket 事件数据
  const { data: todayEventData } = useQuery({
    queryKey: ["today-event", date, isMinTemp],
    queryFn: () => fetchTodayEvent(date, isMinTemp),
    enabled: !!date,
    refetchInterval: 120_000,
  })

  const eventSlug = todayEventData?.event?.slug ?? null
  const { data: eventData } = useQuery({
    queryKey: ["event", eventSlug],
    queryFn: () => fetchEvent(eventSlug!, isMinTemp),
    enabled: !!eventSlug,
    refetchInterval: 120_000,
  })

  // 解码并提取市场概率
  const marketPrices = useMemo(() => {
    const p = eventData?.prices ?? {}
    const decoded: Record<string, number> = {}
    for (const [k, v] of Object.entries(p)) {
      const decodedKey = k.replace(/^&lt;/, "<").replace(/^&gt;/, ">").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      decoded[decodedKey] = v
    }
    return decoded
  }, [eventData?.prices])

  useEffect(() => {
    if (data?.models) {
      const keys = Object.keys(data.models)
      if (keys.length > 0) {
        // 1. 动态更新排序：保留原有顺序，将新出现的模型追加到末尾，并过滤掉已不存在的模型
        setOrder(prev => {
          const existingOrder = prev.filter(k => keys.includes(k))
          const newKeys = keys.filter(k => !prev.includes(k))
          return [...existingOrder, ...newKeys]
        })

        // 2. 动态更新激活模型：如果当前激活的模型不在新数据里，则自动切换到第一个
        setActiveKey(prev => (prev && keys.includes(prev)) ? prev : keys[0])

        // 3. 动态更新可见模型：如果之前没设置过，设为全部可见；如果设置过，保留原有状态并将新模型设为可见
        setVisibleKeys(prev => {
          if (!prev) return new Set(keys)
          const current = new Set(prev)
          keys.forEach(k => {
            if (!prev.has(k)) {
              current.add(k) // 新出现的模型默认设为可见
            }
          })
          return current
        })
      }
    }
  }, [data])

  const displayModels = useMemo(() => {
    if (!data?.models || order.length === 0) return []
    return order
      .filter(k => data.models[k])
      .map(k => [k, data.models[k]] as [string, typeof data.models[string]])
  }, [data, order])

  const activeModelProbs = useMemo(() => {
    if (!activeKey || !data?.models?.[activeKey]) return {}
    return data.models[activeKey].probs || {}
  }, [activeKey, data])

  const allBuckets = useMemo(() => {
    const set = new Set<string>()
    Object.values(data?.models || {}).forEach(m => {
      Object.keys(m.probs || {}).forEach(b => set.add(b))
    })
    Object.keys(marketPrices || {}).forEach(b => set.add(b))
    return Array.from(set).sort((a, b) => {
      const parse = (s: string) => {
        if (s.startsWith("<") || s.includes(" or below")) return -999
        if (s.startsWith(">=") || s.includes(" or higher")) return 999
        if (s.startsWith(">")) return 999
        const num = parseFloat(s.split("-")[0])
        return isNaN(num) ? 0 : num
      }
      return parse(a) - parse(b)
    })
  }, [data, marketPrices])

  const handleReorder = (newOrder: string[]) => setOrder(newOrder)

  const handleToggleVisible = (key: string) => {
    setVisibleKeys(prev => {
      const current = new Set(prev || [])
      if (current.has(key)) {
        current.delete(key)
        if (key === activeKey) {
          const firstVisible = order.find(k => current.has(k))
          setActiveKey(firstVisible || null)
        }
      } else {
        current.add(key)
      }
      return current
    })
  }

  return (
    <div className="flex flex-col h-full w-full max-w-[1600px] mx-auto overflow-y-auto custom-scrollbar">
      {/* 顶部标题与切换 */}
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4 p-4 md:p-8 pb-0">
        <div>
          <p className="text-[10px] text-cyan-400/80 mono tracking-[0.2em] uppercase mb-2">Institutional Terminal</p>
          <h1 className="text-3xl font-light text-white tracking-tight">Hong Kong Market Forecast</h1>
        </div>
        <div className="flex items-center gap-4 w-full sm:w-auto">
          <div className="relative flex-1 sm:flex-none">
            <input 
              type="date" 
              value={date} 
              onChange={(e) => setDate(e.target.value)} 
              className="w-full bg-white/[0.03] border border-white/[0.06] text-slate-200 rounded-md px-3 py-2 text-sm outline-none focus:border-cyan-400/50 transition-colors mono"
            />
          </div>
          <div className="flex gap-1 bg-[#0f1013] border border-white/[0.06] p-1 rounded-md shrink-0">
            <button
              onClick={() => setIsMinTemp(false)}
              className={`px-4 py-1.5 rounded-sm text-[10px] font-mono uppercase tracking-widest transition-all ${
                !isMinTemp ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              TMAX
            </button>
            <button
              onClick={() => setIsMinTemp(true)}
              className={`px-4 py-1.5 rounded-sm text-[10px] font-mono uppercase tracking-widest transition-all ${
                isMinTemp ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              TMIN
            </button>
          </div>
        </div>
      </header>

      <div className="px-4 md:px-8">
        <WeatherCards date={date} />
      </div>

      {/* 模型共识轨 (横跨全宽) */}
      <div className="px-4 md:px-8">
        <ConsensusTrack 
          models={displayModels} 
          activeKey={activeKey} 
          visibleKeys={visibleKeys} 
        />
      </div>

      {/* 下部双栏：左模型矩阵，右市场分析 */}
      <div className="flex flex-col lg:flex-row gap-6 px-4 md:px-8 pb-8">
        {/* 左侧：模型矩阵 */}
        <aside className="w-full lg:w-1/3 lg:min-w-[300px] lg:max-w-[400px] flex flex-col border border-white/[0.06] bg-[#09090b] rounded-md max-h-[600px] overflow-hidden">
          <div className="p-4 border-b border-white/[0.06] flex items-center justify-between shrink-0">
            <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2">
              <span className="w-4 h-px bg-slate-500"></span> Model Matrix
            </h2>
            {/* 恢复原有的 Compare 模式按钮 */}
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={["flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-md transition-colors",
                compareMode ? "bg-cyan-500/20 text-cyan-400" : "bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:bg-white/[0.06]"
              ].join(" ")}
              title="Compare multiple models"
            >
              <GitCompare size={12} />
              Compare
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
            {isLoading ? (
              <div className="text-center text-slate-400 text-xs py-4">Loading models...</div>
            ) : displayModels.length > 0 ? (
              <ModelGrid
                models={displayModels}
                activeKey={activeKey}
                visibleKeys={visibleKeys}
                onSelect={(key) => setActiveKey(key)}
                onReorder={handleReorder}
                onToggleVisible={handleToggleVisible}
              />
            ) : (
              <div className="text-center text-slate-400 text-xs py-4">No models available</div>
            )}
          </div>
        </aside>

        {/* 右侧：市场分析图表 */}
        <section className="flex-1 w-full bg-[#0f1013] border border-white/[0.06] rounded-md p-6 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.8)]">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-[10px] text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-2">
                <span className="w-4 h-px bg-slate-500"></span> Market Analysis
              </h2>
              <p className="text-xl font-light text-white tracking-tight">Model vs Market Dynamics</p>
            </div>
          </div>
          <div className="h-[400px] w-full">
            {isLoading || !data ? (
              <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm">Loading chart data...</div>
            ) : compareMode ? (
              <ComparisonChart
                models={displayModels}
                marketPrices={marketPrices}
                allBuckets={allBuckets}
              />
            ) : (
              <BucketChart
                modelProbs={activeModelProbs}
                marketPrices={marketPrices}
                allBuckets={allBuckets}
              />
            )}
          </div>
        </section>
      </div>
    </div>
  )
}