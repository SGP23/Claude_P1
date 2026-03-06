import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Filter } from 'lucide-react'
import GradientCard from '../components/GradientCard'
import { getLogs } from '../services/api'
import { getSocket } from '../services/websocket'

const FILTERS = ['All', 'ERROR', 'INFO', 'WARNING']

export default function Logs() {
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('All')
  const scrollRef = useRef(null)

  // Fetch initial logs
  useEffect(() => {
    getLogs()
      .then((data) => setLogs(data.logs || []))
      .catch(console.error)
  }, [])

  // Listen for live log events
  useEffect(() => {
    const socket = getSocket()

    const handler = (entry) => {
      setLogs((prev) => [...prev, entry].slice(-500))
    }

    socket.on('log_message', handler)
    return () => socket.off('log_message', handler)
  }, [])

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const filteredLogs =
    filter === 'All' ? logs : logs.filter((l) => l.level === filter)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Filter controls */}
      <div className="flex items-center gap-3">
        <Filter size={16} className="text-gray-400" />
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
              filter === f
                ? 'bg-gradient-to-r from-[#334eac] to-[#7096d1] text-white shadow-md'
                : 'bg-white dark:bg-[#1e293b] text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700'
            }`}
          >
            {f}
          </button>
        ))}
        <span className="ml-auto text-xs text-gray-400">
          {filteredLogs.length} entries
        </span>
      </div>

      {/* Log console */}
      <GradientCard className="overflow-hidden">
        <div className="bg-[#0f172a] rounded-2xl">
          <div className="flex items-center gap-2 px-4 py-2.5 bg-[#1e293b] border-b border-gray-700/50">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="ml-2 text-xs text-gray-400 font-mono">System Logs</span>
          </div>
          <div
            ref={scrollRef}
            className="h-[500px] overflow-y-auto p-4 font-mono text-sm space-y-1"
          >
            {filteredLogs.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No log entries found</p>
            ) : (
              filteredLogs.map((log, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`flex gap-3 py-1 px-2 rounded ${
                    log.level === 'ERROR'
                      ? 'bg-red-500/10 text-red-400'
                      : log.level === 'WARNING'
                      ? 'bg-yellow-500/10 text-yellow-400'
                      : 'text-gray-300'
                  }`}
                >
                  <span className="text-gray-500 flex-shrink-0 w-20">
                    {log.timestamp}
                  </span>
                  <span
                    className={`flex-shrink-0 w-16 font-semibold ${
                      log.level === 'ERROR'
                        ? 'text-red-400'
                        : log.level === 'WARNING'
                        ? 'text-yellow-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    [{log.level}]
                  </span>
                  <span className="break-all">{log.message}</span>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </GradientCard>
    </div>
  )
}
