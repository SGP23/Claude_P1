import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Camera, Sliders, Sun, Moon, Save } from 'lucide-react'
import GradientCard from '../components/GradientCard'

export default function Settings() {
  const [devices, setDevices] = useState([])
  const [selectedDevice, setSelectedDevice] = useState('')
  const [threshold, setThreshold] = useState(0.4)
  const [darkMode, setDarkMode] = useState(
    document.documentElement.classList.contains('dark')
  )
  const [saved, setSaved] = useState(false)

  // Enumerate cameras
  useEffect(() => {
    async function enumerate() {
      try {
        // Need to request access first to get labels
        await navigator.mediaDevices.getUserMedia({ video: true }).then((s) => {
          s.getTracks().forEach((t) => t.stop())
        })
        const allDevices = await navigator.mediaDevices.enumerateDevices()
        const videoCams = allDevices.filter((d) => d.kind === 'videoinput')
        setDevices(videoCams)
        // Load saved settings
        const savedSettings = localStorage.getItem('slr_settings')
        if (savedSettings) {
          const s = JSON.parse(savedSettings)
          if (s.deviceId) setSelectedDevice(s.deviceId)
          if (s.threshold) setThreshold(s.threshold)
        }
      } catch (e) {
        console.error('Cannot enumerate devices:', e)
      }
    }
    enumerate()
  }, [])

  const toggleDarkMode = () => {
    const next = !darkMode
    setDarkMode(next)
    document.documentElement.classList.toggle('dark', next)
    window.dispatchEvent(new Event('darkmode-changed'))
  }

  const saveSettings = () => {
    localStorage.setItem(
      'slr_settings',
      JSON.stringify({
        deviceId: selectedDevice,
        threshold,
        darkMode,
      })
    )
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-2xl font-bold text-[#081f5c] dark:text-white mb-1">
          Settings
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Configure your Sign Language Recognition experience
        </p>
      </motion.div>

      {/* Webcam selector */}
      <GradientCard className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#334eac] to-[#7096d1] flex items-center justify-center">
            <Camera size={18} className="text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-[#081f5c] dark:text-white">
              Webcam Device
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Select which camera to use for recognition
            </p>
          </div>
        </div>
        <select
          value={selectedDevice}
          onChange={(e) => setSelectedDevice(e.target.value)}
          className="w-full px-4 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-[#334eac] transition"
        >
          <option value="">Default Camera</option>
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Camera ${d.deviceId.slice(0, 8)}`}
            </option>
          ))}
        </select>
      </GradientCard>

      {/* Confidence threshold */}
      <GradientCard className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#334eac] to-[#7096d1] flex items-center justify-center">
            <Sliders size={18} className="text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-[#081f5c] dark:text-white">
              Display Threshold
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Minimum confidence for visual feedback (predictions always show)
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full appearance-none cursor-pointer accent-[#334eac]"
          />
          <span className="w-16 text-center text-sm font-bold text-[#334eac] bg-[#334eac]/10 rounded-lg py-1">
            {(threshold * 100).toFixed(0)}%
          </span>
        </div>
      </GradientCard>

      {/* Theme toggle */}
      <GradientCard className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#334eac] to-[#7096d1] flex items-center justify-center">
              {darkMode ? (
                <Moon size={18} className="text-white" />
              ) : (
                <Sun size={18} className="text-white" />
              )}
            </div>
            <div>
              <h3 className="font-semibold text-[#081f5c] dark:text-white">
                Dark Mode
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Toggle between light and dark theme
              </p>
            </div>
          </div>
          <button
            onClick={toggleDarkMode}
            className={`relative w-14 h-7 rounded-full transition-colors ${
              darkMode ? 'bg-[#334eac]' : 'bg-gray-300'
            }`}
          >
            <motion.div
              className="absolute top-0.5 w-6 h-6 bg-white rounded-full shadow-md"
              animate={{ left: darkMode ? 30 : 2 }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            />
          </button>
        </div>
      </GradientCard>

      {/* Save button */}
      <motion.button
        onClick={saveSettings}
        whileTap={{ scale: 0.97 }}
        className={`w-full py-3 rounded-xl font-medium text-white transition-all flex items-center justify-center gap-2 ${
          saved
            ? 'bg-emerald-500 shadow-lg shadow-emerald-500/20'
            : 'bg-gradient-to-r from-[#334eac] to-[#7096d1] shadow-lg shadow-[#334eac]/20 hover:shadow-xl'
        }`}
      >
        <Save size={16} />
        {saved ? 'Settings Saved!' : 'Save Settings'}
      </motion.button>
    </div>
  )
}
