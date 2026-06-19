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
      className={[
        "rounded-xl border px-4 py-3 cursor-pointer select-none transition-all duration-150 relative",
        active
          ? "border-[#00E5FF]/50 bg-[#00E5FF]/5 shadow-[0_0_12px_rgba(0,229,255,0.15)]"
          : "border-white/5 bg-white/[0.02] hover:bg-white/[0.04]",
      ].join(" ")}
    >
      <div
        {...listeners}
        className="absolute top-2 left-2 cursor-grab hover:text-white/60"
      >
        <GripVertical size={14} className="text-white/20" />
      </div>
      <div className="pl-6" onClick={onClick}>
        <div className="text-[11px] font-semibold tracking-wider uppercase text-white/40 mb-1">
          {label}
        </div>
        <div className="font-mono text-2xl font-bold tabular-nums">
          {pred.mean.toFixed(1)}°C
        </div>
        <div className="text-[11px] text-white/30 font-mono mt-0.5">
          ±{pred.std.toFixed(2)}
        </div>
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
        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.max(keys.length, 1)}, 1fr)` }}>
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
