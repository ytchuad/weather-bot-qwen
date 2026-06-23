import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { 
  suggestStrategy, 
  getPortfolio, 
  getStrategies, 
  createStrategy, 
  updateStrategy,
  resetStrategy,
  deleteStrategy
} from "../api/client"
import type { Suggestion, StrategyCreate, Strategy, StrategyTrade } from "../types"
import { TrendingUp, Wallet, Percent, Activity, Plus, Beaker, Settings2 } from "lucide-react"

const TODAY = new Date().toISOString().slice(0, 10)

// ── 子元件：投資組合概覽 ─────────────────────────────────────────────────────
function PortfolioOverview() {
  const { data } = useQuery({
    queryKey: ["portfolio"],
    queryFn: getPortfolio,
    refetchInterval: 60_000,
  })

  const pnl = data?.total_pnl ?? 0
  const returnPct = data?.total_return_pct ?? 0

  const stats = [
    { label: "Total Capital", value: data ? `$${data.total_capital.toLocaleString()}` : "--", icon: Wallet, color: "text-cyan-400" },
    { label: "Total PnL", value: data ? `$${pnl.toLocaleString()}` : "--", icon: TrendingUp, color: pnl >= 0 ? "text-emerald-400" : "text-rose-400" },
    { label: "Return %", value: data ? `${returnPct.toFixed(2)}%` : "--", icon: Percent, color: returnPct >= 0 ? "text-emerald-400" : "text-rose-400" },
    { label: "Active", value: data ? `${data.active_strategies}/${data.count}` : "--", icon: Activity, color: "text-violet-400" },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <div key={stat.label} className="bg-[#0f1013] border border-white/[0.06] rounded-md p-4 flex items-center gap-4 transition-all duration-300 hover:border-white/10">
            <div className={`p-2.5 rounded-md bg-white/[0.03] border border-white/[0.04] ${stat.color}`}>
              <Icon size={18} />
            </div>
            <div className="flex flex-col">
              <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">{stat.label}</div>
              <div className={`text-xl font-light tabular-nums mono ${stat.color}`}>{stat.value}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── 子元件：單一策略卡片 ─────────────────────────────────────────────────────
function StrategyCard({ strategy }: { strategy: Strategy }) {
  const queryClient = useQueryClient()
  const [isExpanded, setIsExpanded] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  
  // 编辑表单状态
  const [editCapital, setEditCapital] = useState(strategy.capital)
  const [editInitialCapital, setEditInitialCapital] = useState(strategy.initial_capital)
  const [editKelly, setEditKelly] = useState(strategy.params.kelly_fraction)

  const toggleMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => 
      updateStrategy(id, { status }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["strategies"] }),
  })

  const resetMutation = useMutation({
    mutationFn: (id: string) => resetStrategy(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["strategies"] }),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteStrategy(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["strategies"] }),
  })

  // 更新策略参数的 Mutation
  const updateMutation = useMutation({
    mutationFn: (payload: { id: string; capital?: number; initial_capital?: number; params?: Record<string, unknown> }) => 
      // 使用 as any 绕过 TypeScript 严格检查，确保 initial_capital 能发送到后端
      updateStrategy(payload.id, { 
        capital: payload.capital, 
        initial_capital: payload.initial_capital, 
        params: payload.params 
      } as any),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] })
      setIsEditing(false)
    },
  })

  const displayModel = strategy.from_strategy_key?.startsWith("enhanced_v") 
    ? "Auto (C→B→A)" 
    : strategy.model

  const isRunning = strategy.status === "running"
  const pnl = strategy.capital - strategy.initial_capital
  const roi = strategy.initial_capital > 0 ? (pnl / strategy.initial_capital) * 100 : 0

  const { data: tradesData, isLoading: tradesLoading } = useQuery({
    queryKey: ["strategyTrades", strategy.id],
    queryFn: () => fetch(`/api/strategies/${strategy.id}/trades`).then(r => r.json()),
    enabled: isExpanded,
  })

  const handleSaveChanges = () => {
    updateMutation.mutate({
      id: strategy.id,
      capital: Number(editCapital),
      initial_capital: Number(editInitialCapital),
      params: {
        ...strategy.params,
        kelly_fraction: Number(editKelly)
      }
    })
  }

  return (
    <div className="bg-[#0f1013] border border-white/[0.06] rounded-md p-5 transition-all duration-300 hover:border-white/10">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h3 className="text-base font-medium text-white tracking-wide">{strategy.label}</h3>
          <p className="text-[10px] text-slate-400 mt-1 mono uppercase tracking-widest">Model: {displayModel}</p>
        </div>
        <button 
          onClick={() => toggleMutation.mutate({ id: strategy.id, status: isRunning ? "paused" : "running" })}
          className={`relative w-11 h-5 rounded-full transition-colors ${isRunning ? "bg-cyan-500/80" : "bg-slate-700"}`}
        >
          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${isRunning ? "translate-x-6 shadow-[0_0_8px_rgba(56,189,248,0.5)]" : "translate-x-0"}`} />
        </button>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">Capital</div>
          <div className="text-xl font-light text-white tabular-nums mono">${strategy.capital.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-1">ROI</div>
          <div className={`text-xl font-light tabular-nums mono ${roi >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {roi >= 0 ? "+" : ""}{roi.toFixed(2)}%
          </div>
        </div>
      </div>
      
      <div className="pt-4 border-t border-white/[0.04] flex justify-between items-center">
        <span className="text-[10px] text-slate-400 mono">
          Init: ${strategy.initial_capital.toLocaleString()} | Kelly: {(strategy.params.kelly_fraction * 100).toFixed(0)}%
        </span>
        <div className="flex gap-1">
          <button 
            onClick={() => { setIsExpanded(!isExpanded); setIsEditing(false); }}
            className="text-[10px] px-2 py-1 rounded-sm bg-white/[0.03] border border-white/[0.04] text-slate-400 hover:text-cyan-400 transition-colors"
          >
            {isExpanded ? "Hide" : "Trades"}
          </button>
          <button 
            onClick={() => { setIsEditing(!isEditing); setIsExpanded(false); }}
            className="text-[10px] px-2 py-1 rounded-sm bg-white/[0.03] border border-white/[0.04] text-slate-400 hover:text-cyan-400 transition-colors"
          >
            {isEditing ? "Cancel" : "Edit"}
          </button>
          <button 
            onClick={() => resetMutation.mutate(strategy.id)}
            className="text-[10px] px-2 py-1 rounded-sm bg-white/[0.03] border border-white/[0.04] text-slate-400 hover:text-cyan-400 transition-colors"
          >
            Reset
          </button>
          <button 
            onClick={() => {
              if (confirm(`Delete strategy "${strategy.label}"?`)) {
                deleteMutation.mutate(strategy.id)
              }
            }}
            className="text-[10px] px-2 py-1 rounded-sm bg-white/[0.03] border border-white/[0.04] text-slate-400 hover:text-rose-400 transition-colors"
          >
            Del
          </button>
        </div>
      </div>

      {/* 编辑面板 */}
      {isEditing && (
        <div className="mt-4 pt-4 border-t border-white/[0.04]">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="text-[10px] text-slate-400 uppercase tracking-[0.2em] block mb-2">Current Capital ($)</label>
              <input
                type="number"
                value={editCapital}
                onChange={(e) => setEditCapital(Number(e.target.value))}
                className="w-full bg-white/[0.03] border border-white/[0.06] text-white rounded-md px-3 py-2 text-sm outline-none focus:border-cyan-400/50 transition-colors mono"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400 uppercase tracking-[0.2em] block mb-2">Initial Capital ($)</label>
              <input
                type="number"
                value={editInitialCapital}
                onChange={(e) => setEditInitialCapital(Number(e.target.value))}
                className="w-full bg-white/[0.03] border border-white/[0.06] text-white rounded-md px-3 py-2 text-sm outline-none focus:border-cyan-400/50 transition-colors mono"
              />
            </div>
          </div>
          
          <div className="mb-4">
            <label className="text-[10px] text-slate-400 uppercase tracking-[0.2em] block mb-2">Kelly Fraction</label>
            <div className="flex items-center gap-2 bg-white/[0.03] border border-white/[0.06] rounded-md px-3 py-2">
              <input
                type="range"
                min="0.05"
                max="1.0"
                step="0.05"
                value={editKelly}
                onChange={(e) => setEditKelly(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <span className="text-sm text-white mono w-10 text-right">{(editKelly * 100).toFixed(0)}%</span>
            </div>
          </div>

          <button
            onClick={handleSaveChanges}
            disabled={updateMutation.isPending}
            className="w-full py-2 rounded-md bg-cyan-500/80 text-slate-950 font-semibold text-sm disabled:opacity-30 hover:bg-cyan-400 transition-colors"
          >
            {updateMutation.isPending ? "Saving..." : "Save Changes"}
          </button>
        </div>
      )}

      {/* 交易紀錄展開視圖 */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-white/[0.04] max-h-60 overflow-y-auto custom-scrollbar pr-1">
          {tradesLoading ? (
            <div className="text-center text-xs text-slate-400 py-4">Loading...</div>
          ) : tradesData?.trades?.length > 0 ? (
            tradesData.trades.map((trade: StrategyTrade, idx: number) => (
              <div key={idx} className="bg-white/[0.02] p-3 rounded-sm text-xs space-y-1 mb-2 border border-white/[0.03]">
                <div className="flex justify-between">
                  <span className="font-mono text-slate-400">{trade.timestamp || trade.entry_time}</span>
                  <span className={`font-semibold mono ${trade.pnl != null && trade.pnl > 0 ? "text-emerald-400" : trade.pnl != null && trade.pnl < 0 ? "text-rose-400" : "text-slate-400"}`}>
                    {trade.action || trade.side}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400 mono">
                  <span>{trade.bucket}</span>
                  <span>PnL: {trade.pnl != null ? `$${trade.pnl.toFixed(2)}` : "--"}</span>
                </div>
                <div className="text-slate-500 italic mt-1 pt-1 border-t border-white/[0.03] space-y-0.5 text-[10px]">
                  {trade.selected_model && <div>Model: {trade.selected_model}</div>}
                  {trade.entry_reason && <div>Reason: {trade.entry_reason}</div>}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-xs text-slate-500 py-4">No recent trades</div>
          )}
        </div>
      )}
    </div>
  )
}

// ── 分頁：Live ──────────────────────────────────────────────────────────────
function LiveTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["strategies"],
    queryFn: getStrategies,
    refetchInterval: 60_000,
  })

  return (
    <div>
      <PortfolioOverview />
      <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-4">
        <span className="w-4 h-px bg-slate-500"></span> Active Strategies
      </h2>
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map(i => <div key={i} className="h-40 rounded-md bg-white/[0.02] animate-pulse" />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {data?.strategies.map(s => <StrategyCard key={s.id} strategy={s} />)}
        </div>
      )}
    </div>
  )
}

// ── 分頁：Builder ───────────────────────────────────────────────────────────
function BuilderTab() {
  const [showCreate, setShowCreate] = useState(false)
  const queryClient = useQueryClient()
  
  const [formData, setFormData] = useState<StrategyCreate>({
    id: "",
    label: "",
    model: "baseline",
    capital: 10000,
    market_template: "hk-tmax"
  })

  const createMutation = useMutation({
    mutationFn: createStrategy,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] })
      setShowCreate(false)
      setFormData({ id: "", label: "", model: "baseline", capital: 10000, market_template: "hk-tmax" })
    },
  })

  const inputClass = "mt-1 w-full bg-white/[0.03] border border-white/[0.06] text-slate-300 rounded-md px-3 py-2 text-sm outline-none focus:border-cyan-400/50 transition-colors"
  const labelClass = "text-[10px] text-slate-400 uppercase tracking-[0.2em]"

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-[#0f1013] border border-white/[0.06] rounded-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-base font-medium text-white tracking-wide flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-cyan-400" />
            Create New Strategy
          </h2>
          <button 
            onClick={() => setShowCreate(!showCreate)}
            className="flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-sm bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
          >
            <Plus size={12} />
            New
          </button>
        </div>

        {showCreate && (
          <div className="space-y-4 mb-6 p-4 bg-white/[0.02] rounded-md border border-white/[0.04]">
            <div>
              <label className={labelClass}>Strategy ID</label>
              <input value={formData.id} onChange={(e) => setFormData({...formData, id: e.target.value})} placeholder="e.g., my_aggressive_v2" className={inputClass} />
            </div>
            <div>
              <label className={labelClass}>Display Name</label>
              <input value={formData.label} onChange={(e) => setFormData({...formData, label: e.target.value})} placeholder="My Aggressive V2" className={inputClass} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Base Model</label>
                <select value={formData.model} onChange={(e) => setFormData({...formData, model: e.target.value})} className={inputClass}>
                  <option value="baseline">Baseline</option>
                  <option value="rain_observed">Rain Observed</option>
                  <option value="rain_nowcast">Rain Nowcast</option>
                  <option value="gated_ensemble">Gated Ensemble</option>
                  <option value="enhanced_v1">Enhanced V1</option>
                  <option value="enhanced_v2">Enhanced V2</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>Initial Capital</label>
                <input type="number" value={formData.capital} onChange={(e) => setFormData({...formData, capital: Number(e.target.value)})} className={inputClass} />
              </div>
            </div>
            <button
              onClick={() => createMutation.mutate(formData)}
              disabled={!formData.id || !formData.label}
              className="w-full mt-2 py-2 rounded-md bg-cyan-500/80 text-slate-950 font-semibold text-sm disabled:opacity-30 hover:bg-cyan-400 transition-colors"
            >
              Create Strategy
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// ── 分頁：Lab ───────────────────────────────────────────────────────────────
function LabTab() {
  const [date] = useState(TODAY)
  const [capital, setCapital] = useState(10000)
  const [kellyFrac, setKellyFrac] = useState(0.25)
  const [isMinTemp, setIsMinTemp] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ["suggestions", date, capital, kellyFrac, isMinTemp],
    queryFn: () => suggestStrategy(date, capital, kellyFrac, isMinTemp),
    refetchInterval: 120_000,
  })

  const signals = data?.suggestions.filter(s => s.action !== "pass") ?? []
  const passes = data?.suggestions.filter(s => s.action === "pass") ?? []

  return (
    <div className="space-y-8">
      <div className="bg-[#0f1013] border border-white/[0.06] rounded-md p-6">
        <div className="flex items-center gap-2 mb-6">
          <Beaker className="w-4 h-4 text-violet-400" />
          <h2 className="text-base font-medium text-white tracking-wide">Strategy Lab</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 bg-white/[0.02] p-3 rounded-md border border-white/[0.04]">
            <input type="checkbox" checked={isMinTemp} onChange={(e) => setIsMinTemp(e.target.checked)} className="w-4 h-4 rounded accent-cyan-500" />
            <span className="text-xs text-slate-300">Min Temperature (TMIN)</span>
          </div>
          
          <div className="bg-white/[0.02] p-3 rounded-md border border-white/[0.04]">
            <label className="text-[10px] text-slate-400 uppercase tracking-[0.2em] block mb-1">Capital</label>
            <input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} className="w-full bg-transparent text-slate-200 text-sm outline-none mono" />
          </div>
          
          <div className="bg-white/[0.02] p-3 rounded-md border border-white/[0.04]">
            <label className="text-[10px] text-slate-400 uppercase tracking-[0.2em] block mb-1">Kelly Fraction</label>
            <div className="flex items-center gap-2">
              <input type="range" min="0.05" max="1.0" step="0.05" value={kellyFrac} onChange={(e) => setKellyFrac(Number(e.target.value))} className="w-full accent-cyan-500" />
              <span className="text-sm text-slate-300 font-mono w-10 text-right">{(kellyFrac * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => <div key={i} className="h-20 rounded-md bg-white/[0.02] animate-pulse" />)}
        </div>
      ) : (
        <>
          {signals.length > 0 && (
            <div>
              <h3 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-4">
                <span className="w-4 h-px bg-slate-500"></span> Actionable Signals ({signals.length})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {signals.map((s) => <SuggestionCard key={s.bucket} s={s} />)}
              </div>
            </div>
          )}
          
          {passes.length > 0 && (
            <div>
              <h3 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-4">
                <span className="w-4 h-px bg-slate-500"></span> No Edge ({passes.length})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {passes.map((s) => <SuggestionCard key={s.bucket} s={s} compact />)}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function SuggestionCard({ s, compact = false }: { s: Suggestion; compact?: boolean }) {
  const isBuy = s.action === "buy_yes"
  const color = isBuy ? "text-emerald-400" : "text-rose-400"
  const bg = isBuy 
    ? "bg-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/30" 
    : "bg-rose-500/5 border-rose-500/20 hover:border-rose-500/30"
  
  if (compact) {
    return (
      <div className="bg-[#0f1013] border border-white/[0.06] rounded-md p-3 flex justify-between items-center">
        <span className="text-xs text-slate-400 mono">{s.bucket}</span>
        <span className="text-[10px] text-slate-500 uppercase tracking-widest">No Edge</span>
      </div>
    )
  }

  return (
    <div className={`rounded-md border p-5 transition-all ${bg}`}>
      <div className="flex justify-between items-center mb-6">
        <span className="text-base font-medium text-white mono">{s.bucket}</span>
        <span className={`text-[10px] font-bold uppercase tracking-[0.2em] px-2.5 py-1 rounded-sm ${color} bg-black/30 border border-white/5`}>
          {s.action.replace("_", " ")}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-2">Model</div>
          <div className="text-lg font-light text-white mono">{(s.model_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-2">Market</div>
          <div className="text-lg font-light text-slate-400 mono">{(s.market_price * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 uppercase tracking-[0.2em] mb-2">Kelly</div>
          <div className={`text-lg font-light mono ${color}`}>{(s.kelly_fraction * 100).toFixed(1)}%</div>
        </div>
      </div>
    </div>
  )
}

// ── 主頁面 ──────────────────────────────────────────────────────────────────
export default function Strategies() {
  const [tab, setTab] = useState<"live" | "builder" | "lab">("live")

  const tabs = [
    { key: "live" as const, label: "Live", icon: Activity },
    { key: "builder" as const, label: "Builder", icon: Settings2 },
    { key: "lab" as const, label: "Lab", icon: Beaker },
  ]

  return (
    <div className="p-4 md:p-8 h-full overflow-y-auto custom-scrollbar">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-10 gap-4 max-w-[1600px] mx-auto w-full">
        <div>
          <p className="text-[10px] text-cyan-400/80 mono tracking-[0.2em] uppercase mb-2">Strategy Center</p>
          <h1 className="text-3xl font-light text-white tracking-tight">Manage & Backtest</h1>
        </div>
        <div className="flex gap-1 p-1 bg-[#0f1013] border border-white/[0.06] rounded-full">
          {tabs.map(t => {
            const Icon = t.icon
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-[11px] font-medium uppercase tracking-[0.15em] transition-all ${
                  tab === t.key 
                    ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.4)]" 
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon size={12} />
                {t.label}
              </button>
            )
          })}
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto w-full">
        {tab === "live" && <LiveTab />}
        {tab === "builder" && <BuilderTab />}
        {tab === "lab" && <LabTab />}
      </div>
    </div>
  )
}