import { Link, useLocation } from "react-router-dom"
import { Cloud, BarChart3, Activity, RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"
import type { ReactNode } from "react"

const navItems = [
  { to: "/", label: "Hub", icon: Cloud },
  { to: "/strategies", label: "Strategies", icon: BarChart3 },
  { to: "/diagnostics", label: "Diagnostics", icon: Activity },
]

export default function Layout({ children }: { children: ReactNode }) {
  const loc = useLocation()
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date())
      setIsLive(true)
    }, 120_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200">
      <nav className="h-16 flex items-center justify-between px-8 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-2">
          <Cloud className="w-5 h-5 text-cyan-500" />
          <span className="text-lg font-bold text-slate-100">Weather Quant</span>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-500">
            <div className="relative">
              <div className={`w-2 h-2 rounded-full ${isLive ? "bg-emerald-500" : "bg-amber-500"}`}>
                {isLive && (
                  <div className="absolute inset-0 rounded-full bg-emerald-500 animate-ping opacity-75" />
                )}
              </div>
            </div>
            <span>Last sync: {lastUpdate.toLocaleTimeString("zh-HK", { hour: "2-digit", minute: "2-digit" })}</span>
            <RefreshCw
              className="w-3 h-3 hover:text-cyan-400 cursor-pointer transition-colors"
              onClick={() => { setLastUpdate(new Date()); setIsLive(true) }}
            />
          </div>
          <div className="flex items-center gap-2">
            {navItems.map((item) => {
              const active = loc.pathname === item.to
              const Icon = item.icon
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={[
                    "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all",
                    active
                      ? "bg-slate-800 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]"
                      : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200",
                  ].join(" ")}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </nav>
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  )
}