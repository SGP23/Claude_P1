import { motion, AnimatePresence } from 'framer-motion'

export default function WordSuggestions({ suggestions, onSelect }) {
  if (!suggestions || suggestions.every((s) => !s)) {
    return null
  }

  return (
    <div className="mt-4">
      <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
        Suggestions
      </p>
      <div className="grid grid-cols-2 gap-2">
        <AnimatePresence mode="popLayout">
          {suggestions.map((suggestion, idx) => (
            <motion.button
              key={suggestion || `empty-${idx}`}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.15, delay: idx * 0.03 }}
              onClick={() => suggestion && onSelect(suggestion)}
              disabled={!suggestion}
              className={`px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                suggestion
                  ? 'bg-[#334eac]/10 text-[#334eac] hover:bg-[#334eac] hover:text-white dark:bg-[#334eac]/20 dark:text-[#7096d1] dark:hover:bg-[#334eac] dark:hover:text-white cursor-pointer'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-300 dark:text-gray-600 cursor-not-allowed'
              }`}
            >
              {suggestion || '—'}
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}
