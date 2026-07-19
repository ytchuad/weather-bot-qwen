import type { CSSProperties } from "react"
import type { ModelPrediction } from "../types"
import { DndContext, closestCenter } from "@dnd-kit/core"
import { SortableContext, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { GripVertical } from "lucide-react"

interface ModelGridProps {
  models: [string, ModelPrediction][]
  activeKey: string | null
  tempRange: { min: number; max: number }
  onSelect: (key: string) => void
  onReorder: (keys: string[]) => void
}

const MODEL_COLORS: Record<string, string> = {
  "9d": "#38bdf8",
  aws: "#fb923c",
  baseline: "#94a3b8",
  model_a: "#34d399",
  model_b: "#c084fc",
  model_c: "#fbbf24",
  model_g: "#fb923c",
  model_2a: "#f472b6",
  model_2a1: "#2dd4bf",
  model_2a_v2: "#e879f9",
  model_2b: "#94a3b8",
  model_3a: "#64748b",
  model_3b: "#818cf8",
  model_4: "#fda4af",
  model_4_restricted: "#67e8f9",
  rain_nowcast: "#f59e0b",
}

function SortableCard({ id, rank, pred, label, active, tempRange, onClick }: {
  id: string
  rank: number
  pred: ModelPrediction
  label: string
  active: boolean
  tempRange: { min: number; max: number }
  onClick: () => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })
  const modelColor = MODEL_COLORS[id] ?? "#94a3b8"
  const style: CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
  }
  if (active) {
    style.background = `${modelColor}12`
    style.boxShadow = `inset 0 0 0 1px ${modelColor}66`
  }
  if (isDragging) style.zIndex = 10

  const std = pred.std || 0.5
  const minVal = pred.mean - std
  const maxVal = pred.mean + std
  const range = tempRange.max - tempRange.min
  const percent = (value: number) => range > 0 ? Math.max(0, Math.min(100, ((value - tempRange.min) / range) * 100)) : 50
  const left = percent(minVal)
  const right = percent(maxVal)
  const center = percent(pred.mean)
  const topBucket = Object.entries(pred.probs || {}).reduce<{ name: string; probability: number }>((best, [name, probability]) => probability > best.probability ? { name, probability } : best, { name: "", probability: 0 })

  return (
    <div ref={setNodeRef} style={style} {...attributes} onClick={onClick} className={`model-row group flex cursor-pointer items-center gap-2.5 px-2 py-3 sm:gap-3 sm:px-1 ${isDragging ? "model-row-dragging" : ""}`}>
      <span className="mono w-5 shrink-0 text-center text-[10px] font-semibold text-white/25">{String(rank).padStart(2, "0")}</span>
      <button className="shrink-0 cursor-grab touch-none rounded p-1 text-white/30 transition-colors hover:bg-white/[0.06] hover:text-white/75" {...listeners} onClick={(event) => event.stopPropagation()} title="Drag to reorder">
        <GripVertical className="h-3.5 w-3.5" />
      </button>
      <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: modelColor, boxShadow: `0 0 8px ${modelColor}88` }} />
      <div className="min-w-0 w-[108px] shrink-0 sm:w-[122px]">
        <div className={`truncate text-[12px] font-semibold ${active ? "text-white/95" : "text-white/75"}`}>{label}</div>
        <div className="mono mt-0.5 truncate text-[10px] tracking-wide text-white/35">{topBucket.name ? `${topBucket.name} (${Math.round(topBucket.probability * 100)}%)` : "No bucket data"}</div>
      </div>
      <div className="flex shrink-0 items-baseline gap-1 whitespace-nowrap">
        <span className={`tnum text-[19px] font-bold tracking-[-0.01em] ${active ? "text-white/95" : "text-white/80"}`}>{pred.mean.toFixed(1)}{"\u00b0C"}</span>
        <span className="mono tnum text-[10px] text-white/35">{"\u00b1"}{std.toFixed(2)}</span>
      </div>
      <div className="relative mx-1 hidden h-[5px] min-w-8 flex-1 overflow-hidden rounded-full bg-white/[0.06] sm:block">
        <div className="absolute top-0 h-full rounded-full opacity-45" style={{ left: `${left}%`, width: `${Math.max(1, right - left)}%`, backgroundColor: modelColor }} />
        <div className="absolute top-1/2 h-2.5 w-[3px] -translate-y-1/2 rounded-full" style={{ left: `${center}%`, backgroundColor: modelColor, boxShadow: `0 0 7px ${modelColor}` }} />
      </div>
    </div>
  )
}

export const LABEL_MAP: Record<string, string> = {
  "9d": "9-Day XGBoost",
  aws: "AWS High-Freq",
  baseline: "Baseline",
  rain_nowcast: "Rain Nowcast",
  model_a: "Model A",
  model_b: "Model B",
  model_c: "Model C",
  model_g: "Model G",
  model_2a: "Model 2A",
  model_2a1: "Model 2A1",
  model_2a_v2: "Model 2A v2",
  model_2b: "model_2b",
  model_3a: "model_3a",
  model_3b: "model_3b",
  model_4: "Model 4",
  model_4_restricted: "Model 4 Restricted",
}

export default function ModelGrid({ models, activeKey, tempRange, onSelect, onReorder }: ModelGridProps) {
  const keys = models.map(([key]) => key)
  return (
    <DndContext collisionDetection={closestCenter} onDragEnd={({ active, over }) => {
      if (!over || active.id === over.id) return
      const oldIndex = keys.indexOf(String(active.id))
      const newIndex = keys.indexOf(String(over.id))
      if (oldIndex === -1 || newIndex === -1) return
      const next = [...keys]
      next.splice(oldIndex, 1)
      next.splice(newIndex, 0, String(active.id))
      onReorder(next)
    }}>
      <SortableContext items={keys} strategy={verticalListSortingStrategy}>
        <div>
          {models.map(([key, prediction], index) => (
            <SortableCard key={key} id={key} rank={index + 1} pred={prediction} label={LABEL_MAP[key] ?? key} active={key === activeKey} tempRange={tempRange} onClick={() => onSelect(key)} />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  )
}
