import { useQuery } from "@tanstack/react-query"
import { Activity, RefreshCw, CheckCircle2, XCircle, AlertTriangle } from "lucide-react"

export default function Diagnostics() {
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

  const statusIcon = (s: string) =>
    s === "ok" ? (
      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
    ) : s === "warning" ? (
      <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
    ) : (
      <XCircle className="w-3.5 h-3.5 text-rose-400" />
    )

  const statusBadge = (s: string) => (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
        s === "ok"
          ? "bg-emerald-500/15 text-emerald-400"
          : s === "warning"
          ? "bg-amber-500/15 text-amber-400"
          : "bg-rose-500/15 text-rose-400"
      }`}
    >
      {statusIcon(s)}
      {s}
    </span>
  )

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-50 flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan-500" />
            Data Source Diagnostics
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Last checked: {data?.checked_at ? new Date(data.checked_at).toLocaleTimeString("en-HK", { timeZone: "Asia/Hong_Kong", hour12: false }) : "--"} HKT
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
            <div key={i} className="h-20 rounded-xl bg-slate-800/50 animate-pulse" />
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
        <>
          {/* ── Source Cards (compact overview) ─────────────────────── */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-10">
            {data?.sources?.map((source: any) => (
              <div
                key={source.name}
                className={`p-3 rounded-xl border backdrop-blur-md flex items-start gap-3 ${
                  source.status === "ok"
                    ? "bg-emerald-900/20 border-emerald-500/30"
                    : source.status === "warning"
                    ? "bg-amber-900/20 border-amber-500/30"
                    : "bg-rose-900/20 border-rose-500/30"
                }`}
              >
                <div className="mt-0.5 shrink-0">
                  {statusIcon(source.status)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="font-semibold text-slate-100 text-sm">{source.name}</h3>
                    {statusBadge(source.status)}
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono">{source.message}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">{source.last_update}</p>
                </div>
              </div>
            ))}
          </div>

          {/* ── Feature Status Table ────────────────────────────────── */}
          <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-500" />
            Feature Status
          </h2>

          <div className="overflow-x-auto rounded-xl border border-slate-700/50">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-800/80 text-slate-400 text-xs uppercase tracking-wider">
                  <th className="text-left py-3 px-4 font-medium">Domain</th>
                  <th className="text-left py-3 px-4 font-medium">Feature</th>
                  <th className="text-left py-3 px-4 font-medium">Source</th>
                  <th className="text-left py-3 px-4 font-medium">Value</th>
                  <th className="text-left py-3 px-4 font-medium">Status</th>
                  <th className="text-left py-3 px-4 font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {data?.features_flat?.map((f: any, i: number) => (
                  <tr
                    key={`${f.domain}-${f.feature}`}
                    className={`border-t border-slate-700/30 text-slate-300 ${
                      i % 2 === 0 ? "bg-slate-800/20" : "bg-slate-800/40"
                    }`}
                  >
                    <td className="py-2.5 px-4 font-medium text-slate-200 whitespace-nowrap">
                      {f.domain}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-xs whitespace-nowrap">
                      {f.feature}
                    </td>
                    <td className="py-2.5 px-4 text-xs text-slate-500 max-w-[200px] truncate font-mono">
                      {f.source || "—"}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-xs whitespace-nowrap">
                      {f.value}
                    </td>
                    <td className="py-2.5 px-4">{statusBadge(f.status)}</td>
                    <td className="py-2.5 px-4 text-xs text-slate-500 whitespace-nowrap">
                      {f.updated}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(!data?.features_flat || data.features_flat.length === 0) && (
            <p className="text-sm text-slate-500">No feature data available.</p>
          )}
        </>
      )}
    </div>
  )
}
