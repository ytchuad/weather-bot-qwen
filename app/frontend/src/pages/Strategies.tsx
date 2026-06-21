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
import type { Suggestion, StrategyCreate, Strategy } from "../types"
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
    { label: "Total Capital", value: data ? `$${data.total_capital.toLocaleString()}` : "--", icon: Wallet, color: "text-cyan-400", border: "border-cyan-500/50" },
    { label: "Total PnL", value: data ? `$${pnl.toLocaleString()}` : "--", icon: TrendingUp, color: pnl >= 0 ? "text-emerald-400" : "text-rose-400", border: pnl >= 0 ? "border-emerald-500/50" : "border-rose-500/50" },
    { label: "Return %", value: data ? `${returnPct.toFixed(2)}%` : "--", icon: Percent, color: returnPct >= 0 ? "text-emerald-400" : "text-rose-400", border: returnPct >= 0 ? "border-emerald-500/50" : "border-rose-500/50" },
    { label: "Active", value: data ? `${data.active_strategies}/${data.count}` : "--", icon: Activity, color: "text-violet-400", border: "border-violet-500/50" },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <div key={stat.label} className={`bg-slate-900/40 border-t-2 ${stat.border} rounded-xl p-4 backdrop-blur-sm transition-all duration-300 hover:bg-slate-800/40 hover:-translate-y-1`}>
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs text-slate-500 uppercase tracking-wider">{stat.label}</div>
              <Icon size={16} className={stat.color} />
            </div>
            <div className={`text-xl md:text-2xl font-bold ${stat.color} tabular-nums`}>{stat.value}</div>
          </div>
        )
      })}
    </div>
  )
}

// ── 子元件：單一策略卡片 ─────────────────────────────────────────────────────
function StrategyCard({ strategy }: { strategy: Strategy }) {
  const queryClient = useQueryClient()
  
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

  const isRunning = strategy.status === "running"
  const pnl = strategy.capital - strategy.initial_capital
  const roi = strategy.initial_capital > 0 ? (pnl / strategy.initial_capital) * 100 : 0

  return (
    <div className="bg-slate-900/40 rounded-xl border border-slate-800 p-5 backdrop-blur-md transition-all duration-300 hover:bg-slate-800/40">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">{strategy.label}</h3>
          <p className="text-xs text-slate-500 mt-1">Model: {strategy.model}</p>
        </div>
        <button
          onClick={() => toggleMutation.mutate({ id: strategy.id, status: isRunning ? "paused" : "running" })}
          className={`relative w-12 h-6 rounded-full transition-colors ${isRunning ? "bg-emerald-500" : "bg-slate-600"}`}
        >
          <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${isRunning ? "translate-x-6" : "translate-x-0"}`} />
        </button>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Capital</div>
          <div className="text-xl font-bold text-slate-100 tabular-nums">${strategy.capital.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">ROI</div>
          <div className={`text-xl font-bold tabular-nums ${roi >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {roi >= 0 ? "+" : ""}{roi.toFixed(2)}%
          </div>
        </div>
      </div>
      
      <div className="pt-4 border-t border-slate-800 flex justify-between items-center">
        <span className="text-xs text-slate-500">
          Init: ${strategy.initial_capital.toLocaleString()} | Kelly: {(strategy.params.kelly_fraction * 100).toFixed(0)}%
        </span>
        <div className="flex gap-2">
          <button 
            onClick={() => resetMutation.mutate(strategy.id)}
            className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400 hover:text-cyan-400"
            title="Reset to initial capital"
          >
            Reset
          </button>
          <button 
            onClick={() => {
              if (confirm(`Delete strategy "${strategy.label}"?`)) {
                deleteMutation.mutate(strategy.id)
              }
            }}
            className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400 hover:text-rose-400"
            title="Delete strategy"
          >
            Delete
          </button>
        </div>
      </div>
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
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Active Strategies</h2>
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map(i => <div key={i} className="h-40 rounded-xl bg-slate-800/50 animate-pulse" />)}
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

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-slate-900/40 rounded-2xl border border-slate-800 p-6 backdrop-blur-md">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <Settings2 className="w-5 h-5 text-cyan-400" />
            Create New Strategy
          </h2>
          <button 
            onClick={() => setShowCreate(!showCreate)}
            className="flex items-center gap-1 text-xs px-3 py-1.5 rounded-md bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30"
          >
            <Plus size={14} />
            New
          </button>
        </div>

        {showCreate && (
          <div className="space-y-4 mb-6 p-4 bg-slate-800/30 rounded-xl border border-slate-700">
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wider">Strategy ID</label>
              <input
                value={formData.id}
                onChange={(e) => setFormData({...formData, id: e.target.value})}
                placeholder="e.g., my_aggressive_v2"
                className="mt-1 w-full bg-slate-900/50 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wider">Display Name</label>
              <input
                value={formData.label}
                onChange={(e) => setFormData({...formData, label: e.target.value})}
                placeholder="My Aggressive V2"
                className="mt-1 w-full bg-slate-900/50 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-500"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-slate-500 uppercase tracking-wider">Base Model</label>
                <select
                  value={formData.model}
                  onChange={(e) => setFormData({...formData, model: e.target.value})}
                  className="mt-1 w-full bg-slate-900/50 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-500"
                >
                  <option value="baseline">Baseline</option>
                  <option value="rain_observed">Rain Observed</option>
                  <option value="rain_nowcast">Rain Nowcast</option>
                  <option value="gated_ensemble">Gated Ensemble</option>
                  <option value="enhanced_v1">Enhanced V1</option>
                  <option value="enhanced_v2">Enhanced V2</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-slate-500 uppercase tracking-wider">Initial Capital</label>
                <input
                  type="number"
                  value={formData.capital}
                  onChange={(e) => setFormData({...formData, capital: Number(e.target.value)})}
                  className="mt-1 w-full bg-slate-900/50 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-cyan-500"
                />
              </div>
            </div>
            <button
              onClick={() => createMutation.mutate(formData)}
              disabled={!formData.id || !formData.label}
              className="w-full mt-2 py-2 rounded-lg bg-cyan-500 text-slate-950 font-semibold text-sm disabled:opacity-50 hover:bg-cyan-400 transition-colors"
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
    <div className="space-y-6">
      {/* 控制面板 */}
      <div className="bg-slate-900/40 rounded-2xl border border-slate-800 p-6 backdrop-blur-md">
        <div className="flex items-center gap-2 mb-4">
          <Beaker className="w-5 h-5 text-violet-400" />
          <h2 className="text-lg font-semibold text-slate-100">Strategy Lab</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 bg-slate-800/30 p-3 rounded-lg border border-slate-700">
            <input
              type="checkbox"
              checked={isMinTemp}
              onChange={(e) => setIsMinTemp(e.target.checked)}
              className="w-4 h-4 rounded accent-violet-500"
            />
            <span className="text-sm text-slate-300">Min Temperature (TMIN)</span>
          </div>
          
          <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700">
            <label className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Capital</label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              className="w-full bg-transparent text-slate-200 text-sm outline-none"
            />
          </div>
          
          <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700">
            <label className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Kelly Fraction</label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0.05"
                max="1.0"
                step="0.05"
                value={kellyFrac}
                onChange={(e) => setKellyFrac(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <span className="text-sm text-slate-300 font-mono w-10 text-right">{(kellyFrac * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* 結果區 */}
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => <div key={i} className="h-20 rounded-xl bg-slate-800/50 animate-pulse" />)}
        </div>
      ) : (
        <>
          {signals.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Signals ({signals.length})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {signals.map((s) => <SuggestionCard key={s.bucket} s={s} />)}
              </div>
            </div>
          )}
          
          {passes.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3 mt-6">
                No Edge ({passes.length})
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
  const bg = isBuy ? "bg-emerald-900/20 border-emerald-500/30" : "bg-rose-900/20 border-rose-500/30"
  
  if (compact) {
    return (
      <div className="bg-slate-900/30 border border-slate-800 rounded-lg p-3 flex justify-between items-center">
        <span className="text-sm text-slate-400">{s.bucket}</span>
        <span className="text-xs text-slate-600">No Edge</span>
      </div>
    )
  }

  return (
    <div className={`rounded-xl border p-4 backdrop-blur-sm ${bg}`}>
      <div className="flex justify-between items-center mb-3">
        <span className="text-base font-semibold text-slate-100">{s.bucket}</span>
        <span className={`text-xs font-bold uppercase tracking-wider px-2 py-1 rounded ${color} bg-slate-900/50`}>
          {s.action.replace("_", " ")}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-xs text-slate-500 mb-1">Model</div>
          <div className="text-sm font-bold text-slate-200">{(s.model_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Market</div>
          <div className="text-sm font-bold text-slate-200">{(s.market_price * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Kelly</div>
          <div className={`text-sm font-bold ${color}`}>{(s.kelly_fraction * 100).toFixed(1)}%</div>
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
    <div className="p-4 md:p-8 h-full overflow-y-auto">
      {/* 頁面標題與分頁切換 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">Strategy Center</h1>
          <p className="text-sm text-slate-500 mt-1">Manage and backtest your trading strategies</p>
        </div>
        <div className="flex gap-1 bg-slate-900/50 p-1 rounded-full border border-slate-800">
          {tabs.map(t => {
            const Icon = t.icon
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  tab === t.key 
                    ? "bg-cyan-500 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.3)]" 
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon size={14} />
                {t.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* 分頁內容 */}
      {tab === "live" && <LiveTab />}
      {tab === "builder" && <BuilderTab />}
      {tab === "lab" && <LabTab />}
    </div>
  )
}