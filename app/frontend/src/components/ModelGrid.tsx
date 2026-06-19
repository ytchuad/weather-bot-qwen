import type { ModelPrediction } from "../types"
import { DndContext, closestCenter } from "@dnd-kit/core"
import { SortableContext, useSortable, rectSortingStrategy } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { GripVertical, Eye, EyeOff } from "lucide-react"

function SortableCard({
  id,
  pred,
  label,
  active,
  isVisible,
  onClick,
  onToggleVisible,
}: {
  id: string
  pred: ModelPrediction
  label: string
  active: boolean
  isVisible: boolean
  onClick: () => void
  onToggleVisible: () => void
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })

  const probEntries = Object.entries(pred.probs ?? {}) as [string, number][]
  const topBucket: [string, number] = probEntries.length > 0 
    ? probEntries.reduce((max, curr) => curr[1] > max[1] ? curr : max) 
    : ["N/A", 0]
  const topPct = Math.round(topBucket[1] * 100)

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : (isVisible ? 1 : 0.4),
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      onClick={() => {
        if (!isVisible) {
          onToggleVisible();
        }
        onClick();
      }}
      className={[
        "model-card group relative bg-slate-900/50 p-3 rounded-lg cursor-pointer border-l-4 transition-all duration-200",
        active
          ? "border-cyan-500 bg-slate-800/50 shadow-[0_0_15px_rgba(6,182,212,0.1)]"
          : "border-transparent hover:bg-slate-800/50 hover:translate-x-1",
      ].join(" ")}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className={["text-xs font-semibold", active ? "text-slate-100" : "text-slate-300"].join(" ")}>
          {label}
        </span>
        <div className="flex items-center gap-2">
          <button 
            onClick={(e) => { e.stopPropagation(); onToggleVisible(); }} 
            className="text-slate-600 hover:text-cyan-400 transition-colors"
            title={isVisible ? "Hide from dashboard" : "Show on dashboard"}
          >
            {isVisible ? <Eye size={12} /> : <EyeOff size={12} />}
          </button>
          <div {...listeners} className="cursor-grab text-slate-600 group-hover:text-slate-400">
            <GripVertical size={12} />
          </div>
        </div>
      </div>
      
      <div className="flex items-baseline gap-2 mb-2">
        <span className={["text-2xl font-bold tabular-nums", active ? "text-cyan-400" : "text-slate-300"].join(" ")}>
          {pred.mean.toFixed(1)}°C
        </span>
        <span className="text-[10px] text-slate-500">±{pred.std.toFixed(2)}</span>
      </div>
      
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Most Likely</span>
        <span className="text-[10px] text-violet-400 font-mono font-semibold">{topBucket[0]} ({topPct}%)</span>
      </div>
      
      <div className="h-1 w-full bg-slate-700/50 rounded-full overflow-hidden">
        <div className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-full transition-all duration-500 ease-out" style={{ width: `${topPct}%` }} />
      </div>
    </div>
  )
}

// 確保這裡有 export
export const LABEL_MAP: Record<string, string> = {
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
  visibleKeys,
  onSelect,
  onReorder,
  onToggleVisible,
}: {
  models: [string, ModelPrediction][]
  activeKey: string | null
  visibleKeys: Set<string> | null
  onSelect: (key: string) => void
  onReorder: (keys: string[]) => void
  onToggleVisible: (key: string) => void
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
              isVisible={!visibleKeys || visibleKeys.has(k)}
              onClick={() => onSelect(k)}
              onToggleVisible={() => onToggleVisible(k)}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  )
}