import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { 
  suggestStrategy, 
  getPortfolio, 
  getStrategies, 
  createStrategy, 
  updateStrategy
} from "../api/client"
import type { Suggestion, StrategyCreate } from "../types"

const TODAY = new Date().toISOString().slice(0, 10)

function SuggestionRow({ s }: { s: Suggestion }) {
  const bg =
    s.action === "buy_yes"
      ? "bg-[#00D68F]/10 border-[#00D68F]/20"
      : s.action === "buy_no"
        ? "bg-[#FF4D6D]/10 border-[#FF4D6D]/20"
        : "bg-white/[0.02] border-white/5"

  const badge =
    s.action === "buy_yes"
      ? "text-[#00D68F]"
      : s.action === "buy_no"
        ? "text-[#FF4D6D]"
        : "text-white/30"

  return (
    <div className={`rounded-xl border px-4 py-3 ${bg}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium">{s.bucket}</span>
        <span className={`text-xs font-semibold uppercase tracking-wider ${badge}`}>
          {s.action === "pass" ? "—" : s.action.replace("_", " ")}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-4 text-xs text-white/50">
        <div>
          <span className="block text-white/30">Model</span>
          <span className="font-mono font-semibold text-white/80">
            {(s.model_prob * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="block text-white/30">Market</span>
          <span className="font-mono font-semibold text-white/80">
            {(s.market_price * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="block text-white/30">Kelly</span>
          <span className="font-mono font-semibold text-white/80">
            {s.kelly_fraction > 0 ? `${(s.kelly_fraction * 100).toFixed(1)}%` : "—"}
          </span>
        </div>
      </div>
    </div>
  )
}

function LiveTab() {
  const { data } = useQuery({
    queryKey: ["portfolio"],
    queryFn: getPortfolio,
    refetchInterval: 60_000,
  })

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <div className="rounded-xl bg-white/[0.02] p-4">
          <div className="text-xs text-white/40">Total Capital</div>
          <div className="text-xl font-semibold">${data?.total_capital?.toLocaleString() ?? "—"}</div>
        </div>
        <div className="rounded-xl bg-white/[0.02] p-4">
          <div className="text-xs text-white/40">Total PnL</div>
          <div className={`text-xl font-semibold ${data?.total_pnl && data.total_pnl > 0 ? "text-[#00D68F]" : "text-[#FF4D6D]"}`}>
            ${data?.total_pnl?.toLocaleString() ?? "—"}
          </div>
        </div>
        <div className="rounded-xl bg-white/[0.02] p-4">
          <div className="text-xs text-white/40">Return %</div>
          <div className={`text-xl font-semibold ${data?.total_return_pct && data.total_return_pct > 0 ? "text-[#00D68F]" : "text-[#FF4D6D]"}`}>
            {data?.total_return_pct?.toFixed(2) ?? "—"}%
          </div>
        </div>
        <div className="rounded-xl bg-white/[0.02] p-4">
          <div className="text-xs text-white/40">Active Strategies</div>
          <div className="text-xl font-semibold">{data?.active_strategies ?? 0}/{data?.count ?? 0}</div>
        </div>
      </div>

      <h3 className="text-sm font-semibold tracking-wider uppercase text-white/40 pt-2">Active Strategies</h3>
      <StrategiesList />
    </div>
  )
}

function StrategiesList() {
  const queryClient = useQueryClient()
  const { data } = useQuery({
    queryKey: ["strategies"],
    queryFn: getStrategies,
    refetchInterval: 60_000,
  })

  const toggleMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => 
      updateStrategy(id, { status }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["strategies"] }),
  })

  const strategies = data?.strategies ?? []

  if (strategies.length === 0) {
    return <div className="text-sm text-white/30 text-center py-8">No strategies configured.</div>
  }

  return (
    <div className="space-y-2">
      {strategies.map((s) => (
        <div key={s.id} className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">{s.label || s.id}</div>
              <div className="text-xs text-white/40 mt-0.5">
                Model: {s.model} | Capital: ${s.capital.toLocaleString()}
              </div>
            </div>
            <button
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                s.status === "running" 
                  ? "bg-[#00D68F]/20 text-[#00D68F]" 
                  : "bg-white/10 text-white/50"
              }`}
              onClick={() => toggleMutation.mutate({
                id: s.id,
                status: s.status === "running" ? "paused" : "running"
              })}
            >
              {s.status === "running" ? "Running" : "Paused"}
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

function BuilderTab() {
  const [showCreate, setShowCreate] = useState(false)
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: createStrategy,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] })
      setShowCreate(false)
    },
  })

  return (
    <div className="space-y-4">
      <button
        onClick={() => setShowCreate(true)}
        className="rounded-lg bg-white/5 px-4 py-2 text-sm font-medium hover:bg-white/10"
      >
        + Create Strategy
      </button>

      {showCreate && (
        <CreateStrategyForm 
          onClose={() => setShowCreate(false)} 
          onSubmit={(req) => createMutation.mutate(req)} 
        />
      )}

      <h3 className="text-sm font-semibold tracking-wider uppercase text-white/40 pt-2">All Strategies</h3>
      <StrategiesList />
    </div>
  )
}

function CreateStrategyForm({ 
  onClose, 
  onSubmit 
}: { 
  onClose: () => void
  onSubmit: (req: StrategyCreate) => void 
}) {
  const [id, setId] = useState("")
  const [label, setLabel] = useState("")
  const [model, setModel] = useState("baseline")
  const [capital, setCapital] = useState(10000)

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="space-y-3">
        <div>
          <label className="text-xs text-white/40">Strategy ID</label>
          <input
            value={id}
            onChange={(e) => setId(e.target.value)}
            placeholder="e.g., my_strategy"
            className="mt-1 w-full rounded-lg bg-white/5 px-3 py-2 text-sm outline-none focus:bg-white/10"
          />
        </div>
        <div>
          <label className="text-xs text-white/40">Label</label>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Human readable name"
            className="mt-1 w-full rounded-lg bg-white/5 px-3 py-2 text-sm outline-none focus:bg-white/10"
          />
        </div>
        <div>
          <label className="text-xs text-white/40">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="mt-1 w-full rounded-lg bg-white/5 px-3 py-2 text-sm outline-none focus:bg-white/10"
          >
            <option value="baseline">baseline</option>
            <option value="rain_observed">rain_observed</option>
            <option value="rain_nowcast">rain_nowcast</option>
            <option value="gated_ensemble">gated_ensemble</option>
            <option value="enhanced_v1">enhanced_v1</option>
            <option value="enhanced_v2">enhanced_v2</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-white/40">Capital</label>
          <input
            type="number"
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value))}
            className="mt-1 w-full rounded-lg bg-white/5 px-3 py-2 text-sm outline-none focus:bg-white/10"
          />
        </div>
        <div className="flex gap-2 pt-2">
          <button
            onClick={onClose}
            className="flex-1 rounded-lg bg-white/5 py-2 text-sm hover:bg-white/10"
          >
            Cancel
          </button>
          <button
            onClick={() => onSubmit({ id, label, model, capital })}
            disabled={!id || !label}
            className="flex-1 rounded-lg bg-[#00D68F]/20 py-2 text-sm font-medium text-[#00D68F] disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  )
}

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

  const pass: Suggestion[] = []
  const signals: Suggestion[] = []
  for (const s of data?.suggestions ?? []) {
    if (s.action === "pass") pass.push(s)
    else signals.push(s)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={isMinTemp}
            onChange={(e) => setIsMinTemp(e.target.checked)}
            className="h-4 w-4"
          />
          Min Temperature
        </label>
        <label className="text-sm">
          Capital:
          <input
            type="number"
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value))}
            className="ml-2 w-24 rounded bg-white/5 px-2 py-1 text-sm"
          />
        </label>
        <label className="text-sm">
          Kelly %:
          <input
            type="number"
            step="0.05"
            max="1"
            value={kellyFrac}
            onChange={(e) => setKellyFrac(Number(e.target.value))}
            className="ml-2 w-16 rounded bg-white/5 px-2 py-1 text-sm"
          />
        </label>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 rounded-xl bg-white/[0.03] animate-pulse" />
          ))}
        </div>
      ) : (
        <>
          {signals.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold tracking-wider uppercase text-white/40">
                Signals ({signals.length})
              </div>
              {signals.map((s) => (
                <SuggestionRow key={s.bucket} s={s} />
              ))}
            </div>
          )}
          {pass.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold tracking-wider uppercase text-white/30 mt-4">
                Pass ({pass.length})
              </div>
              {pass.map((s) => (
                <SuggestionRow key={s.bucket} s={s} />
              ))}
            </div>
          )}
          {!data?.suggestions?.length && !isLoading && (
            <div className="text-sm text-white/30 text-center py-12">
              No suggestions available.
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function Strategies() {
  const [tab, setTab] = useState<"live" | "builder" | "lab">("live")

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs font-semibold tracking-wider uppercase text-white/40">
            Strategy Management
          </div>
          <div className="mt-2 flex gap-2">
            {(["live", "builder", "lab"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-all ${
                  tab === t 
                    ? "bg-[#00D68F]/20 text-[#00D68F]" 
                    : "text-white/40 hover:bg-white/5"
                }`}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div>
        {tab === "live" && <LiveTab />}
        {tab === "builder" && <BuilderTab />}
        {tab === "lab" && <LabTab />}
      </div>
    </div>
  )
}