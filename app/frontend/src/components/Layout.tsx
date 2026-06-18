import { Link, useLocation } from "react-router-dom"
import { CloudSun, LineChart } from "lucide-react"
import clsx from "clsx"
import type { ReactNode } from "react"

const navItems = [
  { to: "/", label: "Hub", icon: CloudSun },
  { to: "/strategies", label: "Strategies", icon: LineChart },
]

export default function Layout({ children }: { children: ReactNode }) {
  const loc = useLocation()

  return (
    <div className="min-h-screen bg-[#0B0E14] text-[#E6E9EF] font-sans">
      <nav className="flex items-center gap-6 px-6 h-14 border-b border-white/5 bg-[#0B0E14]/80 backdrop-blur-md sticky top-0 z-50">
        <span className="text-sm font-semibold tracking-wide text-white/70">
          Weather Quant
        </span>
        {navItems.map((item) => {
          const active = loc.pathname === item.to
          return (
            <Link
              key={item.to}
              to={item.to}
              className={clsx(
                "flex items-center gap-1.5 text-xs font-medium tracking-wider uppercase transition-colors",
                active
                  ? "text-[#00E5FF]"
                  : "text-white/40 hover:text-white/70",
              )}
            >
              <item.icon size={14} />
              {item.label}
            </Link>
          )
        })}
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">{children}</main>
    </div>
  )
}
