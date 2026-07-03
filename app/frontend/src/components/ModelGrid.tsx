import type { ModelPrediction } from "../types"
import { DndContext, closestCenter } from "@dnd-kit/core"
import { SortableContext, useSortable, rectSortingStrategy } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { GripVertical, Eye, EyeOff } from "lucide-react"

interface ModelGridProps {
  models: [string, ModelPrediction][]
  activeKey: string | null
  visibleKeys: Set<string> | null
  tempRange: { min: number; max: number }
  onSelect: (key: string) => void
  onReorder: (keys: string[]) => void
  onToggleVisible: (key: string) => void
}

function SortableCard({ id, pred, label, active, isVisible, tempRange, onClick, onToggleVisible }: any) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })
  const style: React.CSSProperties = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : (isVisible ? 1 : 0.4) }
  if (active && isVisible) {
    style.boxShadow = "inset 0 0 30px -10px rgba(56, 189, 248, 0.6)"
    style.borderColor = "#22d3ee"
    style.backgroundColor = "rgba(56, 189, 248, 0.1)"
  }

  // 計算置信區間軌的位置
  const std = pred.std || 0.5
  const minVal = pred.mean - std
  const maxVal = pred.mean + std
  const range = tempRange.max - tempRange.min
  const leftPercent = range > 0 ? ((minVal - tempRange.min) / range) * 100 : 0
  const widthPercent = range > 0 ? ((maxVal - minVal) / range) * 100 : 100
  const dotPercent = range > 0 ? ((pred.mean - tempRange.min) / range) * 100 : 50

  // 獲取最高機率桶
  const probs = pred.probs || {}
  const topBucket = Object.entries(probs).reduce((max: any, curr: any) => curr[1] > max[1] ? curr : max, ["N/A", 0])

  return (
    <div ref={setNodeRef} style={style} {...attributes} onClick={() => { if (!isVisible) onToggleVisible(); onClick(); }} className={["relative px-4 py-4 cursor-pointer flex items-center gap-4 group transition-all duration-200 border-l-2 border-b border-white/[0.02]", !active || !isVisible ? (isVisible ? "border-transparent hover:bg-white/[0.02] hover:border-slate-500" : "border-transparent hover:bg-white/[0.02]") : ""].join(" ")}>
      <button className="text-slate-400 hover:text-slate-200 cursor-grab touch-none" {...listeners}><GripVertical size={12} /></button>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${pred.degraded ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)]" : active ? "bg-cyan-400 shadow-[0_0_10px_rgba(56,189,248,0.8)]" : "bg-slate-400"}`}></span>
            <p className={`text-xs font-medium tracking-wide truncate ${active ? "text-white" : "text-slate-200"}`}>{label}</p>
          </div>
          {topBucket[0] !== "N/A" && <span className={`text-[10px] mono ${active ? "text-cyan-400" : "text-slate-500"}`}>{topBucket[0]} ({(topBucket[1] * 100).toFixed(0)}%)</span>}
        </div>
        <div className="flex items-baseline gap-2 mb-2">
          <span className={`text-3xl font-light mono ${active ? "text-white" : "text-slate-100"}`}>{pred.mean.toFixed(1)}<span className={`text-lg ${active ? "text-slate-300" : "text-slate-400"}`}>°C</span></span>
          <span className="text-[10px] text-slate-400 mono">±{pred.std.toFixed(2)}</span>
        </div>
        {/* 置信區間軌 */}
        <div className="ci-track relative h-1 w-full bg-slate-700/50 rounded-full">
          <div className={`absolute h-full rounded-full ${active ? "bg-cyan-500/40" : "bg-slate-600/40"}`} style={{ left: `${Math.max(0, leftPercent)}%`, width: `${Math.min(100, widthPercent)}%` }}></div>
          <div className={`absolute top-1/2 w-2 h-2 rounded-full transform -translate-y-1/2 -translate-x-1/2 ${active ? "bg-cyan-400 shadow-[0_0_8px_#38bdf8]" : "bg-slate-400"}`} style={{ left: `${Math.max(0, Math.min(100, dotPercent))}%` }}></div>
        </div>
      </div>
      <button onClick={(e) => { e.stopPropagation(); onToggleVisible(); }} className={`transition-colors p-2 rounded-md flex-shrink-0 ${isVisible ? "text-slate-400 hover:text-cyan-400" : "text-slate-600 hover:text-cyan-400"}`} title={isVisible ? "Hide" : "Show"}>
        {isVisible ? <Eye size={14} /> : <EyeOff size={14} />}
      </button>
    </div>
  )
}

export const LABEL_MAP: Record<string, string> = {
  "9d": "9-Day XGBoost", aws: "AWS High-Freq", baseline: "Baseline", rain_nowcast: "Rain Nowcast", model_a: "Model A", model_b: "Model B", model_c: "Model C", model_g: "Model G", model_2a: "Model 2A", model_2a1: "Model 2A1",
}

export default function ModelGrid({ models, activeKey, visibleKeys, tempRange, onSelect, onReorder, onToggleVisible }: ModelGridProps) {
  const keys = models.map(([k]) => k)
  return (
    <DndContext collisionDetection={closestCenter} onDragEnd={(e) => {
      const { active, over } = e
      if (!over || active.id === over.id) return
      const oldIdx = keys.indexOf(String(active.id))
      const newIdx = keys.indexOf(String(over.id))
      if (oldIdx === -1 || newIdx === -1) return
      const next = [...keys]; next.splice(oldIdx, 1); next.splice(newIdx, 0, String(active.id)); onReorder(next)
    }}>
      <SortableContext items={keys} strategy={rectSortingStrategy}>
        <div className="flex flex-col">
          {models.map(([k, pred]) => (
            <SortableCard key={k} id={k} pred={pred} label={LABEL_MAP[k] ?? k} active={k === activeKey} isVisible={!visibleKeys || visibleKeys.has(k)} tempRange={tempRange} onClick={() => onSelect(k)} onToggleVisible={() => onToggleVisible(k)} />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  )
}