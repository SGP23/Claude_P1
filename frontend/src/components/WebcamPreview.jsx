import { useRef, useEffect, useCallback, useState } from 'react'

// MediaPipe hand landmark connections for drawing skeleton
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17],            // Palm
]

// Finger tip indices (for highlighting)
const FINGER_TIPS = [4, 8, 12, 16, 20]

export default function WebcamPreview({ onFrame, isActive = true, deviceId = '', landmarks = null }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const [error, setError] = useState(null)

  const startWebcam = useCallback(async () => {
    setError(null)
    try {
      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
          : { width: 640, height: 480, facingMode: 'user' },
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error('Webcam access failed:', err)
      let errorMsg
      if (err.name === 'NotAllowedError') {
        errorMsg = 'Camera access denied. Please allow camera access in your browser settings.'
      } else if (err.name === 'NotFoundError') {
        errorMsg = 'No camera found. Please connect a camera and try again.'
      } else if (err.name === 'NotReadableError') {
        errorMsg = 'Camera is in use by another application.'
      } else {
        errorMsg = `Camera error: ${err.message}`
      }
      setError(errorMsg)
    }
  }, [deviceId])

  const stopWebcam = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  const captureFrame = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
    if (onFrame) onFrame(dataUrl)
  }, [onFrame])

  // Draw hand landmarks on overlay canvas
  const drawLandmarks = useCallback(() => {
    const overlay = overlayRef.current
    const video = videoRef.current
    if (!overlay || !video) return

    const ctx = overlay.getContext('2d')
    const cw = overlay.width
    const ch = overlay.height

    // Clear previous frame
    ctx.clearRect(0, 0, cw, ch)

    if (!landmarks || landmarks.length !== 21) return

    // ── object-cover coordinate mapping ──────────────────────
    // The video uses CSS object-cover, which scales the native frame to fill
    // the 16:9 container and crops the overflow (typically top/bottom for a
    // 4:3 webcam). Landmarks are normalised 0-1 relative to the FULL native
    // frame, so we must map them to the container accounting for scale+crop.
    const vw = video.videoWidth  || cw
    const vh = video.videoHeight || ch

    const containerAspect = cw / ch
    const videoAspect     = vw / vh

    let scale, offsetX, offsetY
    if (containerAspect > videoAspect) {
      // Container wider than video → match width, crop height
      scale   = cw / vw
      offsetX = 0
      offsetY = (ch - vh * scale) / 2          // negative = top/bottom cropped
    } else {
      // Container taller than video → match height, crop width
      scale   = ch / vh
      offsetX = (cw - vw * scale) / 2          // negative = left/right cropped
      offsetY = 0
    }

    // Map normalised landmarks → canvas pixel coordinates.
    // NO manual x-mirror here: the overlay has CSS scaleX(-1) which mirrors
    // identically to the video element, so raw coordinates align correctly.
    const mapped = landmarks.map(lm => ({
      x: lm.x * vw * scale + offsetX,
      y: lm.y * vh * scale + offsetY,
    }))

    // Draw connections (skeleton lines)
    ctx.strokeStyle = '#00ff88'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    
    for (const [start, end] of HAND_CONNECTIONS) {
      const s = mapped[start]
      const e = mapped[end]
      
      ctx.beginPath()
      ctx.moveTo(s.x, s.y)
      ctx.lineTo(e.x, e.y)
      ctx.stroke()
    }

    // Draw landmark points
    mapped.forEach((pt, idx) => {
      const isTip = FINGER_TIPS.includes(idx)
      const isWrist = idx === 0
      
      // Outer glow
      ctx.beginPath()
      ctx.arc(pt.x, pt.y, isTip ? 10 : isWrist ? 12 : 6, 0, Math.PI * 2)
      ctx.fillStyle = isWrist ? 'rgba(255, 100, 100, 0.4)' : 
                      isTip ? 'rgba(100, 200, 255, 0.4)' : 
                      'rgba(0, 255, 136, 0.3)'
      ctx.fill()
      
      // Inner dot
      ctx.beginPath()
      ctx.arc(pt.x, pt.y, isTip ? 5 : isWrist ? 6 : 3, 0, Math.PI * 2)
      ctx.fillStyle = isWrist ? '#ff6464' : isTip ? '#64c8ff' : '#00ff88'
      ctx.fill()
    })

    // Draw palm center indicator
    const palmCenter = {
      x: (mapped[0].x + mapped[5].x + mapped[17].x) / 3,
      y: (mapped[0].y + mapped[5].y + mapped[17].y) / 3,
    }
    ctx.beginPath()
    ctx.arc(palmCenter.x, palmCenter.y, 15, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.lineWidth = 2
    ctx.stroke()

  }, [landmarks])

  // Start/stop webcam based on isActive prop
  useEffect(() => {
    if (isActive) {
      // startWebcam is async and only sets error state on failure
      // eslint-disable-next-line react-hooks/set-state-in-effect
      startWebcam()
    }
    return () => stopWebcam()
  }, [isActive, startWebcam, stopWebcam])

  useEffect(() => {
    if (isActive && onFrame) {
      intervalRef.current = setInterval(captureFrame, 150) // ~6.7 FPS for smoother tracking
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [isActive, captureFrame, onFrame])

  // Update overlay canvas size when video loads
  useEffect(() => {
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!video || !overlay) return

    const handleResize = () => {
      overlay.width = video.offsetWidth
      overlay.height = video.offsetHeight
    }

    video.addEventListener('loadedmetadata', handleResize)
    window.addEventListener('resize', handleResize)
    handleResize()

    return () => {
      video.removeEventListener('loadedmetadata', handleResize)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  // Draw landmarks whenever they update
  useEffect(() => {
    drawLandmarks()
  }, [drawLandmarks])

  return (
    <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
        style={{ transform: 'scaleX(-1)' }}
      />
      <canvas ref={canvasRef} className="hidden" />
      <canvas 
        ref={overlayRef} 
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ transform: 'scaleX(-1)' }}
      />
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-900/80 text-white p-6">
          <svg className="w-12 h-12 mb-3 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="text-lg font-medium text-center">{error}</p>
          <button 
            onClick={() => startWebcam()} 
            className="mt-4 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition"
          >
            Try Again
          </button>
        </div>
      )}
      {!isActive && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-white">
          <p className="text-lg font-medium">Camera Paused</p>
        </div>
      )}
    </div>
  )
}
