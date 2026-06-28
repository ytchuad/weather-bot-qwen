import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Activity, RefreshCw, CheckCircle2, XCircle, AlertTriangle, ChevronDown, ChevronRight, ExternalLink } from "lucide-react"

export default function Diagnostics() {
  const [expanded, setExpanded] = useState<string | null>(null)

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["diagnostics"],
    queryFn: async () => {
      const res = await fetch("/api/diagnostics/sources")
      if (!res.ok) {
        throw new Error(`API Error: ${res.status}`)
      }
      return res.json()
    },
    refetchInterval: 60_000,
  })

  const toggleSource = (name: string) => {
    setExpanded(expanded === name ? null : name)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-50 flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan-500" />
            Data Source Diagnostics
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Last checked: {data?.checked_at ? new Date(data.checked_at).toLocaleTimeString() : "--"}
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 rounded-xl bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      ) : isError ? (
        <div className="p-4 rounded-xl bg-rose-900/20 border border-rose-500/30 text-rose-400">
          <h3 className="font-semibold flex items-center gap-2">
            <XCircle className="w-5 h-5" />
            Failed to load diagnostics
          </h3>
          <p className="text-sm mt-1 font-mono">{(error as Error)?.message || "Unknown error"}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {data?.sources?.map((source: any) => {
            const isExpanded = expanded === source.name
            return (
              <div
                key={source.name}
                className={`rounded-xl border backdrop-blur-md overflow-hidden ${
                  source.status === "ok"
                    ? "bg-emerald-900/20 border-emerald-500/30"
                    : source.status === "warning"
                    ? "bg-amber-900/20 border-amber-500/30"
                    : "bg-rose-900/20 border-rose-500/30"
                }`}
              >
                {/* Header (always visible) */}
                <button
                  onClick={() => toggleSource(source.name)}
                  className="w-full p-4 flex items-start gap-3 text-left hover:bg-white/5 transition-colors"
                >
                  <div className="mt-0.5 shrink-0">
                    {source.status === "ok" ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : source.status === "warning" ? (
                      <AlertTriangle className="w-5 h-5 text-amber-400" />
                    ) : (
                      <XCircle className="w-5 h-5 text-rose-400" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="font-semibold text-slate-100">{source.name}</h3>
                      <span
                        className={`text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${
                          source.status === "ok"
                            ? "bg-emerald-500/20 text-emerald-400"
                            : source.status === "warning"
                            ? "bg-amber-500/20 text-amber-400"
                            : "bg-rose-500/20 text-rose-400"
                        }`}
                      >
                        {source.status}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 mt-1 font-mono">{source.message}</p>
                    {source.last_update && (
                      <p className="text-xs text-slate-500 mt-1">
                        Checked: {source.last_update}
                      </p>
                    )}
                  </div>
                  <div className="shrink-0 text-slate-500 mt-1">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </div>
                </button>

                {/* Expanded details */}
                {isExpanded && (
                  <div className="px-4 pb-4 pt-0 border-t border-white/5">
                    {/* URL */}
                    {source.url && (
                      <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-500">
                        <ExternalLink className="w-3 h-3 shrink-0" />
                        <span className="truncate font-mono">{source.url}</span>
                      </div>
                    )}

                    {/* Features table */}
                    {source.features && source.features.length > 0 && (
                      <div className="mt-3 overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-slate-500 border-b border-white/5">
                              <th className="text-left py-1.5 pr-4 font-medium">Feature</th>
                              <th className="text-left py-1.5 pr-4 font-medium">Value</th>
                              <th className="text-left py-1.5 font-medium">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {source.features.map((f: any) => (
                              <tr key={f.name} className="border-b border-white/5 last:border-0">
                                <td className="py-1.5 pr-4 text-slate-300 font-mono">{f.name}</td>
                                <td className="py-1.5 pr-4 text-slate-400 font-mono">{f.value}</td>
                                <td className="py-1.5">
                                  <span
                                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                                      f.status === "ok"
                                        ? "bg-emerald-500/15 text-emerald-400"
                                        : f.status === "warning"
                                        ? "bg-amber-500/15 text-amber-400"
                                        : "bg-rose-500/15 text-rose-400"
                                    }`}
                                  >
                                    {f.status === "ok" ? (
                                      <CheckCircle2 className="w-2.5 h-2.5" />
                                    ) : f.status === "warning" ? (
                                      <AlertTriangle className="w-2.5 h-2.5" />
                                    ) : (
                                      <XCircle className="w-2.5 h-2.5" />
                                    )}
                                    {f.status}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {(!source.features || source.features.length === 0) && !source.url && (
                      <p className="mt-3 text-xs text-slate-500 italic">No additional details</p>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
