import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { suggestStrategy } from "../api/client"
import type { Suggestion } from "../types"

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

export default function Strategies() {
  const [capital] = useState(10000)
  const [kellyFrac] = useState(0.25)

  const { data, isLoading } = useQuery({
    queryKey: ["suggestions", TODAY, capital, kellyFrac],
    queryFn: () => suggestStrategy(TODAY, capital, kellyFrac, false),
    refetchInterval: 120_000,
  })

  const pass: Suggestion[] = []
  const signals: Suggestion[] = []
  for (const s of data?.suggestions ?? []) {
    if (s.action === "pass") pass.push(s)
    else signals.push(s)
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs font-semibold tracking-wider uppercase text-white/40">
            Strategy Suggestions
          </div>
          <div className="text-sm text-white/30 mt-0.5">
            {TODAY} · Kelly fraction {(kellyFrac * 100).toFixed(0)}%
          </div>
        </div>
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
