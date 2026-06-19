import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
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
  const buckets = Object.keys(modelProbs).sort()

  const data = buckets.map((bucket) => ({
    bucket,
    Model: modelProbs[bucket] != null ? +(modelProbs[bucket] * 100).toFixed(1) : null,
    Market: marketPrices[bucket] != null ? +(marketPrices[bucket] * 100).toFixed(1) : null,
  }))

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={data} margin={{ top: 20, right: 0, left: -10, bottom: 0 }}>
        <defs>
          <linearGradient id="modelGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.9} />
            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.4} />
          </linearGradient>
          <linearGradient id="marketGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.9} />
            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.4} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis
          dataKey="bucket"
          stroke="#64748b"
          fontSize={12}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          stroke="#64748b"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: "0.5rem",
            color: "#e2e8f0",
          }}
          cursor={{ fill: "#1e293b50" }}
        />
        <Legend
          wrapperStyle={{ fontSize: "12px", color: "#94a3b8" }}
          iconType="circle"
        />
        <Bar dataKey="Model" fill="url(#modelGrad)" radius={[4, 4, 0, 0]} />
        <Bar dataKey="Market" fill="url(#marketGrad)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}