import { motion } from 'framer-motion'

export default function GradientCard({ children, className = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className={`bg-white dark:bg-[#1e293b] rounded-2xl shadow-lg border border-gray-100 dark:border-gray-700/50 overflow-hidden ${className}`}
    >
      {children}
    </motion.div>
  )
}
