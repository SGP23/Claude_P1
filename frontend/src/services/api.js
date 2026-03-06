import axios from 'axios'

// In dev, Vite proxy rewrites /api/* → http://localhost:8000/*
// In production, set VITE_API_URL to the backend origin
const API_BASE = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
})

export async function getModelStatus() {
  const { data } = await api.get('/model-status')
  return data
}

export async function getDatasetInfo() {
  const { data } = await api.get('/dataset-info')
  return data
}

export async function getLogs(level = 'all') {
  const { data } = await api.get('/logs', { params: { level } })
  return data
}

export async function postSpeak(text) {
  const { data } = await api.post('/speak', { text })
  return data
}

// ─── Available but not currently used by the UI ─────────
// Uncomment when implementing the corresponding features.

// export async function getHealth() {
//   const { data } = await api.get('/health')
//   return data
// }

// export async function postPredict(file) {
//   const formData = new FormData()
//   formData.append('file', file)
//   const { data } = await api.post('/predict', formData)
//   return data
// }

// export async function getTrainingStatus() {
//   const { data } = await api.get('/training-status')
//   return data
// }

// export async function startTraining(config = {}) {
//   const { data } = await api.post('/start-training', config)
//   return data
// }

// export async function stopTraining() {
//   const { data } = await api.post('/stop-training')
//   return data
// }

// export async function collectData(letter, frameBase64) {
//   const { data } = await api.post('/collect-data', { letter, frame: frameBase64 })
//   return data
// }

// export async function getConfidenceSettings() {
//   const { data } = await api.get('/confidence-settings')
//   return data
// }
