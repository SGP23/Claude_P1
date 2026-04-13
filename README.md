<p align="center">
  <img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" alt="Status: Live" />
  <img src="https://img.shields.io/badge/Model-24_ASL_Classes-blue?style=for-the-badge" alt="24 Classes" />
  <img src="https://img.shields.io/badge/Backend-Render_(Docker)-purple?style=for-the-badge" alt="Render Docker" />
  <img src="https://img.shields.io/badge/Frontend-GitHub_Pages-orange?style=for-the-badge" alt="GitHub Pages" />
</p>

# SignLens — Real-Time ASL Recognition

> A real-time American Sign Language (ASL) alphabet recognition system powered by MediaPipe hand tracking and deep learning. Translate hand gestures into text through your webcam with live inference, word suggestions, and speech synthesis.

## 🌐 Live Demo

| | Link |
|:---|:---|
| **Frontend App** | [https://sgp23.github.io/Signlens/](https://sgp23.github.io/Signlens/) |
| **Backend API** | [https://signlens-api.onrender.com](https://signlens-api.onrender.com) |
| **API Docs (Swagger)** | [https://signlens-api.onrender.com/docs](https://signlens-api.onrender.com/docs) |
| **Health Check** | [https://signlens-api.onrender.com/health](https://signlens-api.onrender.com/health) |

> **Note:** The Render free tier spins down after inactivity. The first request may take ~30 seconds while the container cold-starts.

---

## ✨ Features

- **Real-Time Recognition** — Live webcam ASL alphabet detection at ~10 FPS
- **Hand Landmark Visualization** — 21-point MediaPipe hand skeleton overlay on video feed
- **Temporal Smoothing** — Confidence-weighted sliding window eliminates prediction flickering
- **Geometric Disambiguation** — Rule-based refinement for visually similar letters (M/N, U/V, etc.)
- **Word Builder** — Accumulates recognized letters into words with hold-to-confirm logic
- **Dictionary Suggestions** — Auto-complete word suggestions based on partial input
- **Text-to-Speech** — Speak recognized words and sentences aloud
- **24 ASL Classes** — A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND (GitHub Pages)                         │
│  React + Vite + Tailwind CSS                                               │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Webcam   │→│ Frame Capture │→│  Socket.IO   │→│ Prediction Display  │  │
│  │  Preview  │  │  (Base64)    │  │  Emit        │  │ + Word Builder     │  │
│  └──────────┘  └──────────────┘  └──────┬───────┘  └─────────────────────┘  │
└──────────────────────────────────────────┼──────────────────────────────────┘
                                           │ WebSocket
┌──────────────────────────────────────────┼──────────────────────────────────┐
│                         BACKEND (Render — Docker)                          │
│  FastAPI + Socket.IO + PyTorch + MediaPipe                                 │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ Base64     │→│  MediaPipe    │→│ Landmark  │→│ MLP Classifier        │  │
│  │ Decode     │  │ HandLandmark │  │ Normalize │  │ (24 classes)         │  │
│  └───────────┘  └──────────────┘  └──────────┘  └───────────┬───────────┘  │
│                                                              │              │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┴───────────┐  │
│  │ Word Suggestions  │←│  Smoothing    │←│ Geometric Disambiguation     │  │
│  │ (Dictionary)      │  │  Buffer       │  │ (M/N, U/V, etc.)           │  │
│  └──────────────────┘  └──────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Signlens/
├── frontend/                    # React + Vite SPA
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── WebcamPreview.jsx      # Camera feed + landmark overlay
│   │   │   ├── PredictionDisplay.jsx  # Letter + confidence display
│   │   │   ├── WordSuggestions.jsx    # Dictionary auto-complete
│   │   │   └── GradientCard.jsx       # Styled card wrapper
│   │   ├── pages/               # Application pages
│   │   │   ├── Dashboard.jsx          # System status overview
│   │   │   ├── LiveRecognition.jsx    # Main recognition interface
│   │   │   ├── DatasetManager.jsx     # Dataset inspection
│   │   │   ├── Logs.jsx               # Real-time server logs
│   │   │   └── Settings.jsx           # Confidence/device configuration
│   │   ├── services/            # API and WebSocket clients
│   │   │   ├── socketClient.js        # Socket.IO connection manager
│   │   │   └── apiClient.js           # REST API client (axios)
│   │   └── layouts/             # Page layout wrappers
│   ├── .env.production          # Production API URL
│   └── vite.config.js           # Vite build configuration
│
├── backend/                     # FastAPI server
│   ├── server.py                # Main server (REST + Socket.IO + inference)
│   ├── requirements.txt         # Python dependencies
│   ├── api/                     # Additional REST route modules
│   ├── models/                  # Disambiguation logic
│   │   └── disambiguation.py         # Geometric rules for M/N, U/V, etc.
│   ├── prediction/              # Word prediction engine
│   │   └── word_predictor.py         # Dictionary-based suggestions
│   └── tests/                   # Test suite
│       └── test_pipeline.py          # End-to-end inference tests
│
├── models/                      # ML model definitions + weights
│   ├── landmark_model.py        # LandmarkClassifier (MLP) architecture
│   ├── cnn_model.py             # CNN model architecture (fallback)
│   └── hand_landmarker.task     # MediaPipe HandLandmarker model file
│
├── training/                    # Model training pipeline
│   ├── train.py                 # CNN training script
│   ├── train_landmarks.py       # Landmark MLP training script
│   ├── dataset.py               # Dataset loader and augmentation
│   ├── evaluate_model.py        # Model evaluation + confusion matrix
│   └── validate_dataset.py      # Dataset integrity validation
│
├── Dockerfile                   # Docker build (Render deployment)
├── render.yaml                  # Render Blueprint configuration
├── .github/workflows/
│   └── deploy.yml               # GitHub Actions → GitHub Pages
├── landmark_classifier.pt       # Trained MLP model weights (24 classes)
└── sign_language_cnn_trained.pt # Trained CNN model weights (fallback)
```

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Frontend** | React 19, Vite 7, Tailwind CSS | Single Page Application |
| **UI Library** | Framer Motion, Lucide Icons, Recharts | Animations, icons, charts |
| **Backend** | FastAPI, Uvicorn | REST API + ASGI server |
| **Real-Time** | Socket.IO (python-socketio) | WebSocket frame streaming |
| **ML Framework** | PyTorch, MediaPipe | Hand detection + classification |
| **Hand Tracking** | MediaPipe HandLandmarker | 21-point hand skeleton extraction |
| **Hosting** | GitHub Pages (frontend), Render Docker (backend) | Production deployment |
| **CI/CD** | GitHub Actions | Auto-deploy on push to `main` |

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam

### 1. Clone the Repository
```bash
git clone https://github.com/SGP23/Signlens.git
cd Signlens
```

### 2. Start the Backend
```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt

# Start the server
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

The backend starts at **http://localhost:8000**. You should see:
```
INFO  LANDMARK model loaded: 24 classes on cpu
INFO  MediaPipe HandLandmarker initialized successfully at startup
INFO  Uvicorn running on http://0.0.0.0:8000
```

### 3. Start the Frontend
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

The frontend starts at **http://localhost:5173/Signlens/**. Open it in your browser, allow camera access, and start signing!

---

## 🌍 Deployment

### Frontend → GitHub Pages
Automated via GitHub Actions (`.github/workflows/deploy.yml`):
- Triggers on push to `main`
- Builds with `VITE_API_URL` pointing to the Render backend
- Deploys static files to GitHub Pages

### Backend → Render (Docker)
Configured via `render.yaml` Blueprint:
- Uses `Dockerfile` with `python:3.11-slim` base
- Installs OpenGL/GLES system libraries (required by MediaPipe)
- Builds and deploys automatically on push to `main`

**Why Docker?** MediaPipe requires `libGLESv2.so.2` (OpenGL ES) at runtime. Render's native Python runtime cannot install system packages, so Docker is required.

---

## 📡 API Reference

### REST Endpoints
| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check — returns `{"status": "ok", "model_loaded": true}` |
| `GET` | `/model-status` | Model configuration, classes, device, smoothing settings |
| `GET` | `/dataset-info` | Training dataset metadata |
| `GET` | `/logs?level=all` | Server log buffer (last 500 entries) |
| `POST` | `/predict` | Single-frame prediction (upload image) |
| `POST` | `/suggest-words` | Dictionary word suggestions for partial input |
| `POST` | `/speak` | Text-to-speech synthesis |
| `GET` | `/docs` | Interactive Swagger UI |

### Socket.IO Events
| Event | Direction | Description |
|:---|:---|:---|
| `predict_frame` | Client → Server | Send base64-encoded webcam frame |
| `prediction` | Server → Client | Returns `{letter, confidence, landmarks, is_stable}` |

---

## 🧠 ML Pipeline

### Model Architecture
**LandmarkClassifier (MLP)** — Primary model
- Input: 63 features (21 landmarks × 3 coordinates, normalized)
- Architecture: `63 → 256 → BatchNorm → ReLU → Dropout(0.3) → 128 → BatchNorm → ReLU → Dropout(0.3) → 24`
- Output: 24 ASL letter classes

### Inference Pipeline
1. **Frame Capture** — Browser captures webcam at ~10 FPS, Base64-encodes frames
2. **Hand Detection** — MediaPipe HandLandmarker extracts 21 hand landmarks
3. **Feature Extraction** — Landmarks normalized for translation/scale invariance
4. **Classification** — MLP predicts letter with confidence score
5. **Disambiguation** — Geometric rules resolve confusable pairs (M↔N, U↔V, etc.)
6. **Temporal Smoothing** — Sliding window (10 frames) with confidence-weighted voting
7. **Word Building** — Hold-for-8-frames + 1.8s cooldown for letter acceptance

### Recognized Classes (24)
```
A  B  C  D  E  F  G  H  I  K  L  M
N  O  P  Q  R  S  T  U  V  W  X  Y
```
> J and Z require motion gestures and are not supported in the current static-frame model.

---

## ⚙️ Environment Variables

### Frontend
| Variable | Description | Default |
|:---|:---|:---|
| `VITE_API_URL` | Backend server URL | `http://localhost:8000` |

### Backend
| Variable | Description | Default |
|:---|:---|:---|
| `MEDIAPIPE_DISABLE_GPU` | Force CPU for MediaPipe | `1` |
| `LIBGL_ALWAYS_SOFTWARE` | Software OpenGL rendering | `1` |
| `PYOPENGL_PLATFORM` | OpenGL platform backend | `egl` |
| `PYTHONUNBUFFERED` | Unbuffered Python output | `1` |

---

## ⚠️ Known Limitations

- **Cold Start Delay** — Render free tier sleeps after 15 min inactivity; first request takes ~30s
- **CPU-Only Inference** — No GPU acceleration on free tier; ~100-150ms per frame
- **Static Gestures Only** — Letters J and Z (motion-based) are not supported
- **Lighting Sensitivity** — Best results with even front-lighting on a plain background
- **Single Hand** — Only detects one hand at a time

---

## 🔮 Future Improvements

- [ ] Dynamic gesture recognition (J, Z) using temporal sequences
- [ ] Word-level and sentence-level ASL recognition
- [ ] WebGL/WASM inference for client-side prediction (no backend needed)
- [ ] Mobile-responsive UI with touch controls
- [ ] GPU backend deployment for lower latency
- [ ] Multi-hand support and two-hand gesture vocabulary
- [ ] Continuous model retraining with user feedback

---

## 🤝 Contributors

- **Sundari Pathak** — Backend architecture, ML model training, inference pipeline, and deployment

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.