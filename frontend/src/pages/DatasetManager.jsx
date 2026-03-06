import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Database, Layers, Image } from 'lucide-react'
import StatCard from '../components/StatCard'
import GradientCard from '../components/GradientCard'
import LiveChart from '../components/LiveChart'
import { getDatasetInfo } from '../services/api'

export default function DatasetManager() {
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDatasetInfo()
      .then(setInfo)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-10 h-10 border-4 border-[#334eac] border-t-transparent rounded-full" />
      </div>
    )
  }

  const chartData = info?.class_names?.map((name, i) => ({
    name,
    value: 1,
    index: i,
  })) || []

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Info banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-[#334eac] to-[#7096d1] rounded-2xl p-5 text-white shadow-lg"
      >
        <h2 className="text-lg font-bold mb-1">Dataset Overview</h2>
        <p className="text-white/80 text-sm">
          Read-only view of the classes the pre-trained model recognizes. The model
          has already been trained — no modifications to the dataset are needed.
        </p>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          icon={Layers}
          label="Total Classes"
          value={info?.num_classes || 0}
          sub="Trained sign language letters"
        />
        <StatCard
          icon={Image}
          label="Input Size"
          value={`${info?.image_size || 224}×${info?.image_size || 224}`}
          sub="Pixels per frame crop"
        />
        <StatCard
          icon={Database}
          label="Model File"
          value="CNN"
          sub={info?.model_file || 'N/A'}
        />
      </div>

      {/* Class distribution chart */}
      <GradientCard className="p-6">
        <h3 className="text-lg font-semibold text-[#081f5c] dark:text-white mb-4">
          Class Distribution
        </h3>
        <LiveChart data={chartData} />
      </GradientCard>

      {/* Class grid */}
      <GradientCard className="p-6">
        <h3 className="text-lg font-semibold text-[#081f5c] dark:text-white mb-4">
          Class Labels
        </h3>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-3">
          {info?.class_names?.map((name, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.02 }}
              className="flex flex-col items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-[#334eac] hover:shadow-md transition-all"
            >
              <span className="text-2xl font-black text-[#334eac]">{name}</span>
              <span className="text-[10px] text-gray-400 mt-1">idx {i}</span>
            </motion.div>
          ))}
        </div>
      </GradientCard>
    </div>
  )
}
