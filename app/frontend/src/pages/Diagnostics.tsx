import { useQuery } from "@tanstack/react-query"
import { Activity, RefreshCw, CheckCircle2, XCircle, AlertTriangle } from "lucide-react"

export default function Diagnostics() {
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["diagnostics"],
    queryFn: () => fetch("/api/diagnostics/sources").then((res) => res.json()),
    refetchInterval: 60_000,
  })

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
            <div key={i} className="h-20 rounded-xl bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {data?.sources.map((source: any) => (
            <div
              key={source.name}
              className={`p-4 rounded-xl border backdrop-blur-md flex items-start gap-4 ${
                source.status === "ok"
                  ? "bg-emerald-900/20 border-emerald-500/30"
                  : source.status === "warning"
                  ? "bg-amber-900/20 border-amber-500/30"
                  : "bg-rose-900/20 border-rose-500/30"
              }`}
            >
              <div className="mt-1">
                {source.status === "ok" ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : source.status === "warning" ? (
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-rose-400" />
                )}
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-slate-100">{source.name}</h3>
                <p className="text-sm text-slate-400 mt-1 font-mono">{source.message}</p>
              </div>
              <span
                className={`text-xs font-semibold uppercase tracking-wider px-2 py-1 rounded ${
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
          ))}
        </div>
      )}
    </div>
  )
}