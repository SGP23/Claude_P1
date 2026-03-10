# 🤟 SignLens — Real-Time ASL Fingerspelling Recognition

A real-time American Sign Language (ASL) fingerspelling recognition system built with PyTorch, MediaPipe, FastAPI, and React. SignLens detects hand landmarks from a webcam feed, classifies them into **24 ASL letters** using a trained MLP, applies geometric disambiguation rules for confusing letter pairs, and builds words with intelligent suggestions — all at sub-millisecond inference speed.

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Disambiguation System](#-disambiguation-system)
- [Word Prediction](#-word-prediction)
- [API Reference](#-api-reference)
- [Frontend Pages](#-frontend-pages)
- [Testing](#-testing)
- [Training](#-training)
- [Performance](#-performance)
- [License](#-license)

---

## ✨ Features

- **Real-time recognition** — Webcam-to-letter prediction via Socket.IO WebSocket streaming
- **24 ASL letter classes** — A–Y excluding J and Z (which require motion)
- **Geometric disambiguation** — 8 rule groups using finger geometry to resolve confusing letter pairs (A/E/M/N/S/T, B/D/F/I/K/R/U/V/W, C/O, G/H, etc.)
- **Temporal smoothing** — Confidence-weighted voting over a 10-frame sliding window for stable predictions
- **Word prediction** — Context-aware autocomplete suggestions powered by PyEnchant with common-word fallback
- **Text-to-speech** — Speak built sentences aloud via pyttsx3
- **Modern dashboard** — React + TailwindCSS frontend with live charts, prediction history, and system logs
- **Dual model support** — Landmark MLP (primary, sub-millisecond) with CNN fallback

---

## 🏗 Architecture

```
Webcam Frame
    │
    ▼
MediaPipe HandLandmarker (21 landmarks × 3 coords)
    │
    ▼
Landmark Normalization (wrist-origin, unit-sphere scaling)
    │
    ▼
LandmarkClassifier MLP (63 → 256 → 128 → 64 → 24)
    │
    ▼
Geometric Disambiguation (8 rule groups, finger geometry)
    │
    ▼
Temporal Smoothing (10-frame confidence-weighted voting)
    │
    ▼
Letter → Word Builder → Word Prediction → Sentence → TTS
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, python-socketio, Uvicorn |
| **Frontend** | React 19, Vite 7, TailwindCSS 4, Framer Motion, Recharts |
| **ML Model** | PyTorch (LandmarkClassifier MLP + SignLanguageCNN fallback) |
| **Hand Detection** | MediaPipe HandLandmarker |
| **Word Prediction** | PyEnchant + built-in common word list |
| **Text-to-Speech** | pyttsx3 |
| **Real-time Comm** | Socket.IO (WebSocket + polling fallback) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Webcam

### 1. Clone & Set Up Backend

```bash
git clone https://github.com/YOUR_USERNAME/SignLens.git
cd SignLens

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Start the Backend

```bash
# From the project root
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

The server loads the trained landmark model on startup and begins listening for WebSocket connections.

### 3. Set Up & Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Open the App

Navigate to **http://localhost:5173** in your browser. Go to **Live Recognition** and allow webcam access.

---

## 📁 Project Structure

```
SignLens/
├── backend/
│   ├── server.py                 # FastAPI + Socket.IO server (entry point)
│   ├── prediction_engine.py      # Full prediction pipeline orchestration
│   ├── preprocessing.py          # Skeleton extraction & rendering
│   ├── word_prediction.py        # Word autocomplete engine
│   ├── requirements.txt          # Python dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # REST API route definitions
│   ├── models/
│   │   └── disambiguation.py     # Geometric disambiguation rules
│   └── tests/
│       ├── test_api.py           # API endpoint tests (10 tests)
│       ├── test_disambiguation.py # Disambiguation logic tests (14 tests)
│       ├── test_word_prediction.py # Word prediction tests (10 tests)
│       ├── test_preprocessing.py  # Preprocessing tests (6 tests)
│       ├── test_accuracy.py       # Model accuracy & sanity tests
│       ├── test_model_manual.py   # Manual model loading validation
│       ├── test_pipeline_manual.py # Manual live pipeline test
│       ├── test_preprocessing_manual.py # Manual skeleton visual test
│       ├── test_server_manual.py  # Manual HTTP + Socket.IO tests
│       └── benchmark.py          # Performance benchmarking
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Router setup
│   │   ├── main.jsx              # Entry point
│   │   ├── index.css             # Global styles
│   │   ├── components/
│   │   │   ├── GradientCard.jsx      # Animated card wrapper
│   │   │   ├── LiveChart.jsx         # Recharts bar chart
│   │   │   ├── PredictionDisplay.jsx # Letter + confidence display
│   │   │   ├── StatCard.jsx          # Stat card with icon
│   │   │   ├── WebcamPreview.jsx     # Webcam + landmark overlay
│   │   │   └── WordSuggestions.jsx   # 2×2 suggestion grid
│   │   ├── layouts/
│   │   │   └── DashboardLayout.jsx   # Sidebar + header layout
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx         # Overview & stats
│   │   │   ├── LiveRecognition.jsx   # Real-time recognition
│   │   │   ├── DatasetManager.jsx    # Class distribution viewer
│   │   │   ├── Logs.jsx             # System log console
│   │   │   └── Settings.jsx         # Camera & threshold config
│   │   └── services/
│   │       ├── api.js            # Axios HTTP client
│   │       └── websocket.js      # Socket.IO client
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── cnn_model.py              # SignLanguageCNN architecture
│   ├── landmark_model.py         # LandmarkClassifier MLP architecture
│   └── hand_landmarker.task      # MediaPipe hand model
├── training/
│   ├── train.py                  # CNN training pipeline
│   ├── train_landmarks.py        # Landmark MLP training pipeline
│   ├── dataset.py                # Dataset loading & augmentation
│   ├── evaluate_model.py         # Model evaluation & confusion matrix
│   ├── validate_dataset.py       # Dataset quality validation
│   └── ml_improvements.py        # Improved training pipeline (v2)
├── class_labels.txt              # 24 class labels (A–Y, no J/Z)
├── landmark_class_labels.txt     # Landmark model class labels
├── logs/                         # Training & evaluation logs (JSON)
└── README.md
```

---

## 🧠 ML Pipeline

### Primary Model: LandmarkClassifier (MLP)

The preferred model takes 21 MediaPipe hand landmarks (63 input features) and classifies them through a 4-layer MLP:

```
Input(63) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
          → Linear(24)
```

**Preprocessing**: Landmarks are normalized to wrist-origin coordinates and scaled to a unit sphere.

### Fallback Model: SignLanguageCNN

If the landmark model is unavailable, the system falls back to a 5-block CNN operating on 224×224 RGB images:

```
Conv2d(3→32) → BN → ReLU → MaxPool
Conv2d(32→64) → BN → ReLU → MaxPool
Conv2d(64→128) → BN → ReLU → MaxPool
Conv2d(128→256) → BN → ReLU → MaxPool
Conv2d(256→512) → BN → ReLU → MaxPool
AdaptiveAvgPool → FC(512→256→128→24)
```

### Prediction Flow

1. **Frame received** via Socket.IO (base64 JPEG)
2. **MediaPipe** detects hand landmarks (21 points, confidence ≥ 0.5)
3. **Landmark normalization** — translate to wrist origin, scale to unit sphere → 63 features
4. **Model inference** — MLP forward pass → softmax → predicted letter + confidence
5. **Disambiguation** — geometric rules refine confused pairs (skipped if confidence > 0.92)
6. **Temporal smoothing** — 10-frame sliding window, confidence-weighted majority vote, 60% consistency threshold
7. **Result emitted** back to client with letter, confidence, stability flag, and landmarks

---

## 🔍 Disambiguation System

Eight geometric rule groups resolve letter pairs that the model frequently confuses:

| Group | Letters | Strategy |
|---|---|---|
| **Fist variations** | A, E, M, N, S, T | Thumb position (side/over/under), finger curl state, thumb-to-MCP distances |
| **Open palm** | B, D, F, I, K, R, U, V, W | Finger count, spread distance, index-middle gap (V vs U vs R), pinky isolation (I) |
| **Curved shapes** | C, O | Thumb-index tip distance (far → C, close → O) |
| **Pointing sideways** | G, H | Extended finger count (1 → G, 2 → H) |
| **L-shape** | L | Single letter — no disambiguation needed |
| **Hook shape** | X | Single letter — no disambiguation needed |
| **Y-shape** | Y | Single letter — no disambiguation needed |
| **Downward pointing** | P, Q | Index extension angle / pointing direction |

**Excluded**: J, Z (require hand motion — not suitable for static classification)

Disambiguation uses helper functions: `finger_is_up()`, `finger_is_curled()`, `count_fingers_up()`, `distance()`, `distance_3d()`.

---

## 💬 Word Prediction

The `WordPredictor` module provides autocomplete suggestions as users fingerspell:

1. Extracts the last incomplete word from the sentence (e.g., `"I AM H"` → prefix `"h"`)
2. Searches for matches using prefix matching across three data sources (in priority order):
   - Custom word list file (`backend/wordlist.txt`)
   - PyEnchant dictionary (`en_US`)
   - Built-in common word list (~160 words)
3. Returns up to 4 suggestions, sorted by length then alphabetically

**Example**: Spelling `"HEL"` → suggestions: `["HELD", "HELL", "HELM", "HELLO"]`

---

## 📡 API Reference

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info, version, list of all endpoints |
| `GET` | `/health` | Health check — model status, device, timestamp |
| `GET` | `/model-status` | Model details — type, classes, class names, temporal smoothing config |
| `GET` | `/dataset-info` | Class names, count, model type |
| `GET` | `/confidence-settings` | Confidence threshold info |
| `GET` | `/training-status` | Training state (is_training, progress, epoch) |
| `GET` | `/logs?level=all` | System logs (last 500 entries, filterable: INFO/ERROR/WARNING) |
| `GET` | `/groups` | Disambiguation groups with letter members |
| `POST` | `/predict` | Upload image for single prediction |
| `POST` | `/speak` | Text-to-speech via pyttsx3 |
| `POST` | `/suggest-words` | Get word suggestions `{ sentence, max_suggestions }` |
| `POST` | `/complete-word` | Complete word in sentence `{ sentence, suggestion }` |

### Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `predict_frame` | Client → Server | `{ frame: "<base64 JPEG>", use_smoothing: true }` |
| `prediction` | Server → Client | `{ letter, confidence, hand_detected, is_stable, landmarks, timestamp }` |
| `log_message` | Server → Client | `{ timestamp, level, message }` (broadcast) |

---

## 🖥 Frontend Pages

| Page | Description |
|---|---|
| **Dashboard** | Model status cards, class distribution chart, quick navigation |
| **Live Recognition** | Webcam feed with landmark overlay, real-time letter/word prediction, word suggestions (2×2 grid), sentence builder, TTS speak button, prediction history |
| **Dataset Manager** | Read-only view of trained classes, stats, class distribution chart |
| **Logs** | Terminal-styled real-time log console with level filtering, auto-scroll, live Socket.IO updates |
| **Settings** | Webcam device selector, confidence threshold slider, saved to localStorage |

---

## 🧪 Testing

### Automated Tests (46 tests)

```bash
# Run all tests from project root
python -m pytest backend/tests/ -v

# Run specific test suites
python -m pytest backend/tests/test_api.py -v           # 10 API endpoint tests
python -m pytest backend/tests/test_disambiguation.py -v # 14 disambiguation tests
python -m pytest backend/tests/test_word_prediction.py -v # 10 word prediction tests
python -m pytest backend/tests/test_preprocessing.py -v   # 6 preprocessing tests
python -m pytest backend/tests/test_accuracy.py -v        # 3 accuracy/sanity tests
```

### Manual Tests (require running server or webcam)

```bash
# Start server first, then:
python backend/tests/test_server_manual.py    # HTTP + Socket.IO tests
python backend/tests/test_model_manual.py     # Model loading validation
python backend/tests/test_pipeline_manual.py  # Live webcam pipeline
python backend/tests/test_preprocessing_manual.py # Skeleton visualization
```

### Benchmarks

```bash
python backend/tests/benchmark.py
```

Typical results: model inference ~0.14ms, disambiguation ~0.005ms, word prediction ~0.009ms.

---

## 🎓 Training

### Train the Landmark Model (Recommended)

```bash
# Extract landmarks from image dataset
python training/train_landmarks.py --extract --data_dirs path/to/images

# Train the MLP
python training/train_landmarks.py --train
```

### Train the CNN Model

```bash
python training/train.py --data_dirs path/to/images --epochs 100 --batch_size 64
```

### Evaluate

```bash
python training/evaluate_model.py
```

Generates confusion matrix, per-class accuracy, and misclassification analysis in `logs/evaluation_report.json`.

### Validate Dataset

```bash
python training/validate_dataset.py --data_dirs path/to/images
```

Detects corrupted images, missing hands, blurry images, and class imbalance.

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Model inference | ~0.14ms per frame |
| Disambiguation | ~0.005ms |
| Word prediction | ~0.009ms |
| Supported classes | 24 (A–Y, excluding J, Z) |
| Temporal smoothing | 10-frame window, 60% consistency |
| Frontend bundle | ~818KB |

---

## 📄 License

MIT
