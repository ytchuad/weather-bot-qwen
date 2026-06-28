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

  const statusColor = (s: string) =>
    s === "ok" ? "text-emerald-400" : s === "warning" ? "text-amber-400" : "text-rose-400"

  const statusBg = (s: string) =>
    s === "ok"
      ? "bg-emerald-500/5 border-emerald-500/20"
      : s === "warning"
      ? "bg-amber-500/5 border-amber-500/20"
      : "bg-rose-500/5 border-rose-500/20"

  const StatusIcon = ({ s, size = 16 }: { s: string; size?: number }) => {
    if (s === "ok") return <CheckCircle2 size={size} className="text-emerald-400" />
    if (s === "warning") return <AlertTriangle size={size} className="text-amber-400" />
    return <XCircle size={size} className="text-rose-400" />
  }

  const statusBadge = (s: string) => (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm text-[10px] font-mono uppercase tracking-widest border ${statusColor(s)} border-current/30 bg-current/10`}
    >
      {s}
    </span>
  )

  return (
    <div className="p-4 md:p-8 max-w-[1600px] mx-auto w-full h-full overflow-y-auto custom-scrollbar">
      {/* 页面头部 */}
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-10 gap-4">
        <div>
          <p className="text-[10px] text-cyan-400/80 mono tracking-[0.2em] uppercase mb-2">System Health</p>
          <h1 className="text-3xl font-light text-white tracking-tight">Data Source Diagnostics</h1>
          <p className="text-xs text-slate-400 mt-2 mono">
            Last checked: {data?.checked_at ? new Date(data.checked_at).toLocaleTimeString("en-HK", { timeZone: "Asia/Hong_Kong", hour12: false }) : "--"} HKT
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 rounded-md hover:bg-cyan-500/20 transition-colors disabled:opacity-50 text-[10px] font-mono uppercase tracking-widest"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </header>

      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 rounded-md bg-white/[0.02] animate-pulse" />
          ))}
        </div>
      ) : isError ? (
        <div className="p-6 rounded-md bg-rose-500/5 border border-rose-500/20 text-rose-400 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.8)]">
          <h3 className="font-medium flex items-center gap-2 text-sm uppercase tracking-widest">
            <XCircle className="w-4 h-4" />
            Failed to load diagnostics
          </h3>
          <p className="text-xs mt-2 font-mono text-rose-300/80">{(error as Error)?.message || "Unknown error"}</p>
        </div>
      ) : (
        <>
          {/* 数据源概览卡片 */}
          <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-4">
            <span className="w-4 h-px bg-slate-500"></span> Live Sources
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
            {data?.sources?.map((source: any) => (
              <div
                key={source.name}
                className={`p-4 rounded-md border backdrop-blur-md flex items-start gap-4 transition-colors ${statusBg(source.status)}`}
              >
                <div className={`p-2 rounded-md bg-black/30 border border-white/5 ${statusColor(source.status)}`}>
                  <StatusIcon s={source.status} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-medium text-white text-sm tracking-wide">{source.name}</h3>
                    {statusBadge(source.status)}
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono break-all">{source.message}</p>
                  <p className="text-[10px] text-slate-500 mt-2 font-mono uppercase tracking-widest">Updated: {source.last_update || "--"}</p>
                </div>
              </div>
            ))}
          </div>

          {/* 功能状态矩阵表格 */}
          <h2 className="text-[10px] font-medium text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2 mb-4">
            <span className="w-4 h-px bg-slate-500"></span> Feature Status Matrix
          </h2>

          <div className="overflow-x-auto rounded-md border border-white/[0.06] bg-[#0f1013] shadow-[0_10px_40px_-10px_rgba(0,0,0,0.8)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-white/[0.02] text-slate-400 text-[10px] uppercase tracking-[0.2em] border-b border-white/[0.06]">
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
                    className="border-t border-white/[0.03] text-slate-300 hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="py-3 px-4 font-medium text-slate-200 whitespace-nowrap text-xs">
                      {f.domain}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs whitespace-nowrap text-cyan-400">
                      {f.feature}
                    </td>
                    <td className="py-3 px-4 text-xs text-slate-400 max-w-[200px] truncate font-mono">
                      {f.source || "—"}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs whitespace-nowrap text-white">
                      {f.value}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <StatusIcon s={f.status} size={14} />
                        <span className={`text-[10px] font-mono uppercase ${statusColor(f.status)}`}>{f.status}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-xs text-slate-500 whitespace-nowrap font-mono">
                      {f.updated}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(!data?.features_flat || data.features_flat.length === 0) && (
            <div className="text-center text-slate-500 text-xs py-6 mt-2 border border-dashed border-white/5 rounded-md">
              No feature data available.
            </div>
          )}
        </>
      )}
    </div>
  )
}