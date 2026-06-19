import type { ModelPrediction } from "../types"
import { DndContext, closestCenter } from "@dnd-kit/core"
import { SortableContext, useSortable, rectSortingStrategy } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { GripVertical } from "lucide-react"

function SortableCard({
  id,
  pred,
  label,
  active,
  onClick,
}: {
  id: string
  pred: ModelPrediction
  label: string
  active: boolean
  onClick: () => void
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })

  const maxProb = Math.max(...Object.values(pred.probs ?? {}))
  const maxPct = Math.round(maxProb * 100)

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      onClick={onClick}
      className={[
        "model-card group relative bg-slate-900/50 p-4 rounded-lg cursor-pointer border-l-4 transition-all duration-200",
        active
          ? "border-cyan-500 bg-slate-800/50 shadow-[0_0_15px_rgba(6,182,212,0.1)]"
          : "border-transparent hover:bg-slate-800/50 hover:translate-x-1",
      ].join(" ")}
    >
      <div className="flex items-center justify-between mb-2">
        <span
          className={[
            "text-sm font-semibold",
            active ? "text-slate-100" : "text-slate-300",
          ].join(" ")}
        >
          {label}
        </span>
        <div {...listeners} className="cursor-grab">
          <GripVertical size={14} className="text-slate-600 group-hover:text-slate-400" />
        </div>
      </div>
      <div className="flex items-baseline gap-2 mb-3">
        <span
          className={[
            "text-3xl font-bold tabular-nums",
            active ? "text-cyan-400" : "text-slate-300",
          ].join(" ")}
        >
          {pred.mean.toFixed(1)}°C
        </span>
        <span className="text-xs text-slate-500">±{pred.std.toFixed(2)} std</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500 w-12">Max Prob</span>
        <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-violet-500 rounded-full" style={{ width: `${maxPct}%` }} />
        </div>
        <span className="text-xs text-violet-400 font-mono">{maxPct}%</span>
      </div>
    </div>
  )
}

const LABEL_MAP: Record<string, string> = {
  "9d": "9-Day XGBoost",
  aws: "AWS High-Freq",
  baseline: "Baseline",
  rain_nowcast: "Rain Nowcast",
  model_a: "Model A",
  model_b: "Model B",
  model_c: "Model C",
}

export default function ModelGrid({
  models,
  activeKey,
  onSelect,
  onReorder,
}: {
  models: [string, ModelPrediction][]
  activeKey: string | null
  onSelect: (key: string) => void
  onReorder: (keys: string[]) => void
}) {
  const keys = models.map(([k]) => k)

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragEnd={(e) => {
        const { active, over } = e
        if (!over || active.id === over.id) return
        const oldIdx = keys.indexOf(String(active.id))
        const newIdx = keys.indexOf(String(over.id))
        if (oldIdx === -1 || newIdx === -1) return
        const next = [...keys]
        next.splice(oldIdx, 1)
        next.splice(newIdx, 0, String(active.id))
        onReorder(next)
      }}
    >
      <SortableContext items={keys} strategy={rectSortingStrategy}>
        <div className="space-y-3">
          {models.map(([k, pred]) => (
            <SortableCard
              key={k}
              id={k}
              pred={pred}
              label={LABEL_MAP[k] ?? k}
              active={k === activeKey}
              onClick={() => onSelect(k)}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  )
}
