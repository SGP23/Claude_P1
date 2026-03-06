import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  Layers,
  Clock,
  Cpu,
  ArrowRight,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import StatCard from '../components/StatCard'
import GradientCard from '../components/GradientCard'
import LiveChart from '../components/LiveChart'
import { getModelStatus, getDatasetInfo } from '../services/api'

export default function Dashboard() {
  const [modelStatus, setModelStatus] = useState(null)
  const [datasetInfo, setDatasetInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    async function fetchData() {
      try {
        const [ms, ds] = await Promise.all([getModelStatus(), getDatasetInfo()])
        setModelStatus(ms)
        setDatasetInfo(ds)
      } catch (e) {
        console.error('Failed to fetch dashboard data:', e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const classChartData = datasetInfo?.class_names?.map((name) => ({
    name,
    value: 1,
  })) || []

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin w-10 h-10 border-4 border-[#334eac] border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Welcome banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-[#334eac] to-[#7096d1] rounded-2xl p-6 text-white shadow-xl shadow-[#334eac]/20"
      >
        <h2 className="text-2xl font-bold mb-1">Welcome to SignLens</h2>
        <p className="text-white/80 mb-4">
          Real-time American Sign Language recognition powered by deep learning.
        </p>
        <button
          onClick={() => navigate('/live')}
          className="inline-flex items-center gap-2 bg-white/20 hover:bg-white/30 px-4 py-2 rounded-xl text-sm font-medium transition"
        >
          Start Recognizing <ArrowRight size={16} />
        </button>
      </motion.div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Activity}
          label="Model Status"
          value={modelStatus?.loaded ? 'Ready' : 'Not Loaded'}
          sub={modelStatus?.device}
          color={modelStatus?.loaded ? 'from-emerald-500 to-emerald-600' : 'from-red-500 to-red-600'}
        />
        <StatCard
          icon={Layers}
          label="Total Classes"
          value={modelStatus?.classes || 0}
          sub="Sign language letters"
        />
        <StatCard
          icon={Cpu}
          label="Device"
          value={modelStatus?.device?.toUpperCase() || 'N/A'}
          sub="Inference device"
        />
        <StatCard
          icon={Clock}
          label="Load Time"
          value={`${modelStatus?.load_time_seconds?.toFixed(2) || '—'}s`}
          sub="Model initialization"
        />
      </div>

      {/* Chart + recent info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <GradientCard className="lg:col-span-2 p-6">
          <h3 className="text-lg font-semibold text-[#081f5c] dark:text-white mb-4">
            Recognized Classes
          </h3>
          <LiveChart data={classChartData} title="" />
        </GradientCard>

        <GradientCard className="p-6">
          <h3 className="text-lg font-semibold text-[#081f5c] dark:text-white mb-4">
            Quick Info
          </h3>
          <div className="space-y-3">
            <InfoRow label="Model Type" value={modelStatus?.model_type === 'landmark' ? 'Landmark MLP' : 'CNN'} />
            <InfoRow label="Prediction Mode" value={modelStatus?.prediction_mode || 'N/A'} />
            <InfoRow label="Smoothing" value={modelStatus?.temporal_smoothing?.enabled ? 'Enabled' : 'Disabled'} />
            <InfoRow label="Classes" value={datasetInfo?.class_names?.join(', ') || 'N/A'} />
          </div>
          <button
            onClick={() => navigate('/live')}
            className="mt-6 w-full py-2.5 bg-gradient-to-r from-[#334eac] to-[#7096d1] text-white rounded-xl font-medium hover:shadow-lg hover:shadow-[#334eac]/20 transition-all"
          >
            Go to Live Recognition
          </button>
        </GradientCard>
      </div>
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div className="flex justify-between items-start py-2 border-b border-gray-100 dark:border-gray-700/50 last:border-0">
      <span className="text-sm text-gray-500 dark:text-gray-400">{label}</span>
      <span className="text-sm font-medium text-[#081f5c] dark:text-white text-right max-w-[180px] truncate">
        {value}
      </span>
    </div>
  )
}
