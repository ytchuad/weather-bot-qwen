import { useQuery } from "@tanstack/react-query"
import { fetchMinuteHistory } from "../api/client"

function formatTime(value: string) {
  return new Intl.DateTimeFormat("en-HK", {
    timeZone: "Asia/Hong_Kong",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value))
}

export default function MinuteHistoryPanel({ date }: { date: string }) {
  const { data, isLoading, isError, error, dataUpdatedAt } = useQuery({
    queryKey: ["layer-a-minute-history", date],
    queryFn: ({ signal }) => fetchMinuteHistory(date, 1000, signal),
    refetchInterval: 60_000,
    refetchIntervalInBackground: true,
    refetchOnMount: "always",
    staleTime: 0,
    retry: 1,
  })

  const rows = data?.minutes ?? []
  const visibleRows = rows.slice(-180)
  const errorMessage = error instanceof Error ? error.message : "API error"
  const refreshedAt = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString("en-HK", { timeZone: "Asia/Hong_Kong" })
    : null

  return (
    <section className="panel overflow-hidden p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="eyebrow">LAYER A · MINUTE HISTORY</div>
          <h2 className="mt-1.5 text-lg font-semibold tracking-tight text-white/90">Minute history</h2>
        </div>
        <div className="mono text-[10px] tracking-wide text-white/35">
          {data?.remote_history?.status === "loading"
            ? "REMOTE HISTORY LOADING"
            : data?.remote_history?.status === "degraded"
              ? "REMOTE HISTORY DEGRADED"
              : `${rows.length} MINUTES`}
        </div>
      </div>
      <div className="mt-4 max-h-[360px] overflow-auto rounded-xl border border-white/[0.06]">
        {isLoading ? (
          <div className="p-8 text-center text-xs text-white/40">Loading minute history...</div>
        ) : isError ? (
          <div className="p-8 text-center text-xs text-amber-300">Minute history unavailable: {errorMessage}</div>
        ) : visibleRows.length === 0 ? (
          <div className="p-8 text-center text-xs text-white/40">No minute records for this date.</div>
        ) : (
          <table className="min-w-full text-left">
            <thead className="sticky top-0 bg-[#14171F]">
              <tr className="mono text-[9px] uppercase tracking-[0.12em] text-white/35">
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Actual</th>
                <th className="px-3 py-2">Max / Min</th>
                <th className="px-3 py-2">Weather</th>
                <th className="px-3 py-2">Model cycle / age</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => (
                <tr key={row.timestamp} className="border-t border-white/[0.045] mono text-[10px] text-white/65">
                  <td className="px-3 py-2 text-white/85">{formatTime(row.timestamp)}</td>
                  <td className="px-3 py-2 text-rose-300">{row.actual_temperature == null ? "--" : `${row.actual_temperature.toFixed(1)}°C`}</td>
                  <td className="px-3 py-2">{row.max_so_far == null ? "--" : row.max_so_far.toFixed(1)} / {row.min_so_far == null ? "--" : row.min_so_far.toFixed(1)}</td>
                  <td className="px-3 py-2"><span className={row.weather_quality_status === "observed" ? "text-emerald-300" : "text-amber-300"}>{row.weather_quality_status}</span> · {row.weather_age_seconds == null ? "--" : `${Math.round(row.weather_age_seconds)}s`}</td>
                  <td className="px-3 py-2">{row.model_cycle_timestamp ? `${formatTime(row.model_cycle_timestamp)} · ${Math.round(row.model_age_seconds ?? 0)}s` : "--"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="mono mt-2 text-[10.5px] tracking-wide text-white/25">
        {data?.sources?.includes("legacy_csv") ? "Legacy CSV fallback · " : ""}
        Model cycles are backward as-of; a minute row never implies new model inference.
        {refreshedAt ? ` Updated ${refreshedAt}.` : ""}
      </p>
    </section>
  )
}
