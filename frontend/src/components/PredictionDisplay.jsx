import { motion, AnimatePresence } from 'framer-motion'

export default function PredictionDisplay({ letter, confidence, word }) {
  return (
    <div className="flex flex-col items-center gap-4">
      {/* Detected Letter */}
      <div className="text-center">
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
          Detected Letter
        </p>
        <AnimatePresence mode="wait">
          <motion.div
            key={letter || 'empty'}
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.5, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="w-28 h-28 rounded-2xl bg-gradient-to-br from-[#334eac] to-[#7096d1] flex items-center justify-center shadow-xl shadow-[#334eac]/20"
          >
            <span className="text-5xl font-black text-white">
              {letter || '—'}
            </span>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Confidence bar */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between text-xs font-medium mb-1">
          <span className="text-gray-500 dark:text-gray-400">Confidence</span>
          <span className="text-[#334eac] dark:text-[#7096d1]">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-[#334eac] to-[#7096d1] rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(confidence * 100, 100)}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Current Word */}
      <div className="mt-2 text-center">
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
          Current Word
        </p>
        <div className="min-h-[48px] px-6 py-2 rounded-xl bg-gray-100 dark:bg-gray-800 border-2 border-dashed border-[#7096d1]/40">
          <span className="text-2xl font-bold tracking-widest text-[#081f5c] dark:text-white">
            {word || '\u00A0'}
          </span>
        </div>
      </div>
    </div>
  )
}
