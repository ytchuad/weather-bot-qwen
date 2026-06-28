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
    <div className="relative flex flex-col h-screen bg-[#09090b] text-slate-400 overflow-hidden">
      {/* 全局环境光与材质背景层 */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div 
          className="absolute inset-0 opacity-[0.03]" 
          style={{ 
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.4'/%3E%3C/svg%3E")` 
          }}
        ></div>
        
        <div 
          className="absolute w-[600px] h-[600px] rounded-full blur-[120px] opacity-[0.2] bg-[#0284c7] top-[-200px] left-[-100px]"
          style={{ animation: "pulseGlow 8s infinite alternate" }}
        ></div>
        
        <div 
          className="absolute w-[500px] h-[500px] rounded-full blur-[120px] opacity-[0.2] bg-[#155e75] bottom-[-150px] right-[-100px]"
          style={{ animation: "pulseGlow 10s infinite alternate-reverse" }}
        ></div>
      </div>

      <style>{`
        @keyframes pulseGlow {
          0% { opacity: 0.1; transform: scale(1); }
          100% { opacity: 0.25; transform: scale(1.1); }
        }
      `}</style>

      {/* 悬浮黑曜石导航栏 */}
      <nav className="sticky top-4 z-50 px-4 mt-4">
        <div className="obsidian-nav mx-auto max-w-5xl flex items-center justify-between px-4 sm:px-6 py-2.5 rounded-full">
          <div className="flex items-center gap-4 sm:gap-8">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 border border-white/10 rounded-md flex items-center justify-center bg-[#09090b] shadow-inner shrink-0">
                <div className="w-2.5 h-2.5 border border-cyan-400/50 rotate-45 bg-cyan-400/10"></div>
              </div>
              {/* 手机端隐藏 Logo 文字，腾出空间 */}
              <span className="hidden sm:block text-white font-medium text-sm uppercase tracking-[0.2em]">Weather Quant</span>
            </div>
            
            {/* 桌面端导航 (带文字) */}
            <div className="hidden md:flex items-center gap-2">
              {navItems.map((item) => {
                const active = loc.pathname === item.to
                const Icon = item.icon
                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    className={[
                      "flex items-center gap-2 px-4 py-1.5 rounded-full text-[11px] font-medium uppercase tracking-[0.15em] transition-all duration-300",
                      active
                        ? "bg-cyan-500/10 text-cyan-400 shadow-[0_0_10px_-2px_rgba(56,189,248,0.2)]"
                        : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                    ].join(" ")}
                  >
                    <Icon className="w-3 h-3" />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </div>

            {/* 手机端导航 (纯图标) */}
            <div className="flex md:hidden items-center gap-1">
              {navItems.map((item) => {
                const active = loc.pathname === item.to
                const Icon = item.icon
                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    className={[
                      "flex items-center justify-center p-2 rounded-full transition-all duration-300",
                      active
                        ? "bg-cyan-500/10 text-cyan-400"
                        : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                    ].join(" ")}
                  >
                    <Icon className="w-4 h-4" />
                  </Link>
                )
              })}
            </div>
          </div>

          <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400">
            {/* 手机端隐藏 Live 时间，腾出空间 */}
            <div className="hidden md:flex items-center gap-2">
              <span className={`w-1 h-1 rounded-full ${isLive ? "bg-emerald-400 shadow-[0_0_6px_#34d399]" : "bg-amber-500"}`}></span>
              <span>LIVE • {lastUpdate.toLocaleTimeString("zh-HK", { hour: "2-digit", minute: "2-digit" })}</span>
            </div>
            <button 
              onClick={() => { setLastUpdate(new Date()); setIsLive(true) }}
              className="p-1.5 hover:bg-white/5 rounded-md transition-colors text-slate-400 hover:text-cyan-400"
              title="Force refresh"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </nav>

      <div className="relative z-10 flex-1 overflow-hidden overflow-y-auto custom-scrollbar mt-6">
        {children}
      </div>

      <style>{`
        .obsidian-nav {
          background: #0f1013;
          border: 1px solid rgba(255, 255, 255, 0.06);
          box-shadow: 
            0 10px 40px -10px rgba(0, 0, 0, 0.8), 
            inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
        }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; height: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #4a4a4a; }
      `}</style>
    </div>
  )
}