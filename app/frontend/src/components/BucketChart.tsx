import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"

export default function BucketChart({
  modelProbs,
  marketPrices,
}: {
  modelProbs: Record<string, number>
  marketPrices: Record<string, number>
}) {
  const allBuckets = Array.from(
    new Set([...Object.keys(modelProbs), ...Object.keys(marketPrices)]),
  ).sort()

  const data = allBuckets.map((b) => ({
    bucket: b,
    Model: modelProbs[b] != null ? +(modelProbs[b] * 100).toFixed(1) : null,
    Market: marketPrices[b] != null ? +(marketPrices[b] * 100).toFixed(1) : null,
  }))

  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
      <div className="text-xs font-semibold tracking-wider uppercase text-white/40 mb-4">
        Probability by Bucket
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} barGap={2}>
          <XAxis
            dataKey="bucket"
            tick={{ fill: "#6B7280", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#6B7280", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(20,19,50,0.95)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              fontSize: 12,
              color: "#E6E9EF",
            }}
            formatter={(v: unknown) => {
              const n = typeof v === "number" ? v : 0
              return `${n.toFixed(1)}%`
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#8F9BB7" }}
          />
          <Bar dataKey="Model" fill="#22C55E" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Market" fill="#A78BFA" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
