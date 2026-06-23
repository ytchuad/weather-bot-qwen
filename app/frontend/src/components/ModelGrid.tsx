import type { ModelPrediction } from "../types"
import { DndContext, closestCenter } from "@dnd-kit/core"
import { SortableContext, useSortable, rectSortingStrategy } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { GripVertical, Eye, EyeOff } from "lucide-react"

function SortableCard({
  id, pred, label, active, isVisible, onClick, onToggleVisible,
}: {
  id: string; pred: ModelPrediction; label: string; active: boolean; isVisible: boolean; onClick: () => void; onToggleVisible: () => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : (isVisible ? 1 : 0.4),
  }

  // 激活状态强制内联渲染发光阴影
  if (active && isVisible) {
    style.boxShadow = "inset 0 0 30px -10px rgba(56, 189, 248, 0.6)";
    style.borderColor = "#22d3ee";
    style.backgroundColor = "rgba(56, 189, 248, 0.1)";
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      onClick={() => { if (!isVisible) onToggleVisible(); onClick(); }}
      className={[
        "relative px-4 py-4 cursor-pointer flex items-center justify-between group transition-all duration-200 border-l-2 border-b border-white/[0.02]",
        !active || !isVisible 
          ? isVisible 
            ? "border-transparent hover:bg-white/[0.02] hover:border-slate-500" 
            : "border-transparent hover:bg-white/[0.02]"
          : "" // 激活态样式由内联控制
      ].join(" ")}
    >
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <button className="text-slate-400 hover:text-slate-200 cursor-grab touch-none" {...listeners}>
          <GripVertical size={12} />
        </button>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              pred.degraded 
                ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)]" 
                : active 
                  ? "bg-cyan-400 shadow-[0_0_10px_rgba(56,189,248,0.8)]" 
                  : "bg-slate-400"
            }`}></span>
            <p className={`text-xs font-medium tracking-wide truncate ${active ? "text-white" : "text-slate-200"}`}>
              {label}
            </p>
            {pred.degraded && (
              <span className="text-[9px] text-amber-400 mono border border-amber-400/30 px-1 rounded-sm flex-shrink-0">
                DEGRADED
              </span>
            )}
          </div>
          
          <div className="flex items-baseline gap-2">
            <span className={`text-3xl font-light mono ${active ? "text-white" : "text-slate-100"}`}>
              {pred.mean.toFixed(1)}
              <span className={`text-lg ${active ? "text-slate-300" : "text-slate-400"}`}>°C</span>
            </span>
            <span className="text-[10px] text-slate-400 mono">±{pred.std.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <button 
        onClick={(e) => { e.stopPropagation(); onToggleVisible(); }} 
        className={`transition-colors p-2 rounded-md flex-shrink-0 ${isVisible ? "text-slate-400 hover:text-cyan-400" : "text-slate-600 hover:text-cyan-400"}`}
        title={isVisible ? "Hide from dashboard" : "Show on dashboard"}
      >
        {isVisible ? <Eye size={14} /> : <EyeOff size={14} />}
      </button>
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
}

export default function ModelGrid({
  models, activeKey, visibleKeys, onSelect, onReorder, onToggleVisible,
}: {
  models: [string, ModelPrediction][]; activeKey: string | null; visibleKeys: Set<string> | null; onSelect: (key: string) => void; onReorder: (keys: string[]) => void; onToggleVisible: (key: string) => void;
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
        <div className="flex flex-col">
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