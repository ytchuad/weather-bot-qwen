import { useMemo } from "react"
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

const sortBuckets = (a: string, b: string) => {
  const parseBucket = (s: string) => {
    if (s.startsWith("<")) return -999
    if (s.startsWith(">=")) return 999
    if (s.startsWith(">")) return 999
    const num = parseFloat(s.split("-")[0])
    return isNaN(num) ? 0 : num
  }
  return parseBucket(a) - parseBucket(b)
}

export default function BucketChart({
  modelProbs,
  marketPrices,
}: {
  modelProbs: Record<string, number>
  marketPrices: Record<string, number>
}) {
  const data = useMemo(() => {
    const buckets = Array.from(new Set([...Object.keys(modelProbs), ...Object.keys(marketPrices)]))
    const sortedBuckets = [...buckets].sort(sortBuckets)

    return sortedBuckets.map((bucket) => ({
      bucket,
      Model: modelProbs[bucket] != null ? +(modelProbs[bucket] * 100).toFixed(1) : 0,
      Market: marketPrices[bucket] != null ? +(marketPrices[bucket] * 100).toFixed(1) : 0,
    }))
  }, [modelProbs, marketPrices])

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
        <XAxis dataKey="bucket" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
        <YAxis
          stroke="#64748b"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "rgba(15, 23, 42, 0.8)",
            border: "1px solid rgba(148, 163, 184, 0.2)",
            borderRadius: "0.5rem",
            color: "#e2e8f0",
            backdropFilter: "blur(8px)",
            WebkitBackdropFilter: "blur(8px)",
            boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
          }}
          cursor={{ fill: "#1e293b50" }}
        />
        <Legend wrapperStyle={{ fontSize: "12px", color: "#94a3b8" }} iconType="circle" />
        <Bar dataKey="Model" fill="url(#modelGrad)" radius={[4, 4, 0, 0]} />
        <Bar dataKey="Market" fill="url(#marketGrad)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}