import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Hand,
  Database,
  FileText,
  Settings,
  Menu,
  X,
  Activity,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { getModelStatus } from '../services/api'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/live', icon: Hand, label: 'Live Recognition' },
  { to: '/dataset', icon: Database, label: 'Dataset Manager' },
  { to: '/logs', icon: FileText, label: 'Logs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export default function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [darkMode, setDarkMode] = useState(() => {
    // Initialize from localStorage or current DOM state
    try {
      const saved = localStorage.getItem('slr_settings')
      if (saved) return JSON.parse(saved).darkMode ?? false
    } catch { /* ignore */ }
    return document.documentElement.classList.contains('dark')
  })
  const location = useLocation()

  // Sync dark mode class on mount and when it changes
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  // Listen for dark mode changes from Settings page
  useEffect(() => {
    const handler = () => {
      setDarkMode(document.documentElement.classList.contains('dark'))
    }
    window.addEventListener('darkmode-changed', handler)
    return () => window.removeEventListener('darkmode-changed', handler)
  }, [])

  const toggleDark = () => {
    const next = !darkMode
    setDarkMode(next)
    // Persist to localStorage
    try {
      const saved = JSON.parse(localStorage.getItem('slr_settings') || '{}')
      saved.darkMode = next
      localStorage.setItem('slr_settings', JSON.stringify(saved))
    } catch { /* ignore */ }
    window.dispatchEvent(new Event('darkmode-changed'))
  }

  return (
    <div className={`flex h-screen ${darkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        <motion.aside
          initial={false}
          animate={{ width: sidebarOpen ? 240 : 68 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          className="flex flex-col bg-[#081f5c] text-white flex-shrink-0 overflow-hidden z-30"
        >
          {/* Logo area */}
          <div className="flex items-center h-16 px-4 gap-3 border-b border-white/10">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#334eac] to-[#7096d1] flex items-center justify-center flex-shrink-0">
              <span className="text-lg">🤟</span>
            </div>
            {sidebarOpen && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="font-bold text-lg whitespace-nowrap"
              >
                SignLens
              </motion.span>
            )}
          </div>

          {/* Nav items */}
          <nav className="flex-1 py-4 px-2 space-y-1">
            {/* eslint-disable-next-line no-unused-vars */}
            {navItems.map(({ to, icon: NavIcon, label }) => {
              const isActive = location.pathname === to
              return (
                <NavLink
                  key={to}
                  to={to}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group
                    ${isActive
                      ? 'bg-gradient-to-r from-[#334eac] to-[#7096d1] text-white shadow-lg shadow-[#334eac]/30'
                      : 'text-white/70 hover:bg-white/10 hover:text-white'
                    }`}
                >
                  <NavIcon size={20} className="flex-shrink-0" />
                  {sidebarOpen && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-sm font-medium whitespace-nowrap"
                    >
                      {label}
                    </motion.span>
                  )}
                </NavLink>
              )
            })}
          </nav>

          {/* Collapse toggle */}
          <div className="p-3 border-t border-white/10">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-lg hover:bg-white/10 transition"
            >
              {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </motion.aside>
      </AnimatePresence>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#ebf2f7] dark:bg-[#0f172a]">
        {/* Top nav bar */}
        <header className="h-16 flex items-center justify-between px-6 bg-white/80 dark:bg-[#1e293b]/80 backdrop-blur-lg border-b border-gray-200/50 dark:border-gray-700/50 z-20">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-[#081f5c] dark:text-white">
              {navItems.find((n) => n.to === location.pathname)?.label || 'SignLens'}
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <ModelStatusBadge />
            <button
              onClick={toggleDark}
              className="w-9 h-9 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-600 transition"
              title="Toggle theme"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

function ModelStatusBadge() {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    getModelStatus()
      .then(setStatus)
      .catch(() => setStatus({ loaded: false }))
  }, [])

  if (!status) return null

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
        status.loaded
          ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
          : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
      }`}
    >
      <Activity size={12} />
      {status.loaded ? `Model Ready (${status.classes} classes)` : 'Model Not Loaded'}
    </div>
  )
}
