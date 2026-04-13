# SignLens Frontend

> React-based Single Page Application for real-time ASL sign language recognition. Provides webcam capture, hand landmark visualization, and live prediction display.

## 🌐 Live URL

**[https://sgp23.github.io/Signlens/](https://sgp23.github.io/Signlens/)**

---

## Tech Stack

| Technology | Version | Purpose |
|:---|:---|:---|
| React | 19.x | UI framework |
| Vite | 7.x | Build tool and dev server |
| Tailwind CSS | 4.x | Utility-first styling |
| Framer Motion | 12.x | Animations and transitions |
| Socket.IO Client | 4.x | Real-time WebSocket communication |
| Axios | 1.x | REST API calls |
| Recharts | 2.x | Dashboard charts |
| Lucide React | — | Icon library |
| React Router DOM | 7.x | Client-side routing |

---

## Pages

| Page | Route | Description |
|:---|:---|:---|
| **Dashboard** | `/Signlens/` | System overview — model status, class distribution, quick info |
| **Live Recognition** | `/Signlens/live` | Main interface — webcam, hand tracking, letter detection, word builder |
| **Dataset Manager** | `/Signlens/dataset` | View training dataset info and class statistics |
| **Logs** | `/Signlens/logs` | Real-time server log viewer with level filtering |
| **Settings** | `/Signlens/settings` | Confidence thresholds, camera device selection |

---

## Key Components

| Component | File | Purpose |
|:---|:---|:---|
| `WebcamPreview` | `components/WebcamPreview.jsx` | Camera capture, frame emission, hand landmark canvas overlay |
| `PredictionDisplay` | `components/PredictionDisplay.jsx` | Detected letter, confidence bar, current word display |
| `WordSuggestions` | `components/WordSuggestions.jsx` | Dictionary auto-complete with click-to-select |
| `GradientCard` | `components/GradientCard.jsx` | Styled glassmorphism card wrapper |

---

## Services

| Service | File | Purpose |
|:---|:---|:---|
| `socketClient.js` | `services/socketClient.js` | Socket.IO connection management — connect, emit frames, receive predictions |
| `apiClient.js` | `services/apiClient.js` | REST API client — model status, dataset info, word suggestions, TTS |
| `logger.js` | `services/logger.js` | Client-side error logging |

---

## Local Development

```bash
# Install dependencies
npm install

# Start dev server (connects to local backend at http://localhost:8000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The dev server starts at **http://localhost:5173/Signlens/**.

> **Prerequisite:** The backend server must be running at `http://localhost:8000` for predictions to work during local development.

---

## Environment Variables

| Variable | File | Description |
|:---|:---|:---|
| `VITE_API_URL` | `.env.production` | Backend API URL for production builds |

**Development** (default): Falls back to `http://localhost:8000` when `VITE_API_URL` is not set.

**Production**: Set to `https://signlens-api.onrender.com` in `.env.production` and in the GitHub Actions workflow.

---

## Deployment

Deployed automatically via **GitHub Actions** on every push to `main`:

1. Checks out the repo
2. Installs Node.js 22 + npm dependencies
3. Builds with `VITE_API_URL=https://signlens-api.onrender.com`
4. Deploys `frontend/dist/` to GitHub Pages

The workflow is defined in `.github/workflows/deploy.yml`.

---

## Live Recognition Flow

```
User shows hand gesture
       │
       ▼
┌──────────────┐     Base64 frame      ┌──────────────────┐
│  WebcamPreview│────────────────────→  │  Backend (Render) │
│  (10 FPS)     │                       │  predict_frame    │
└──────────────┘                       └────────┬─────────┘
                                                │
       ┌────────────────────────────────────────┘
       │  prediction event
       ▼
┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐
│ PredictionDisplay │  │ Landmark Canvas  │  │ Word Builder   │
│ Letter: "B"       │  │ 21-point skeleton│  │ Current: "BA"  │
│ Confidence: 88%   │  │                  │  │ [Suggestions]  │
└──────────────────┘  └─────────────────┘  └────────────────┘
```

### Letter Acceptance Logic
- A letter must be detected **8 consecutive frames** before acceptance
- After acceptance, a **1.8-second cooldown** prevents double-entries
- These thresholds are configurable in `LiveRecognition.jsx` (`HOLD_FRAMES_REQUIRED`, `COOLDOWN_MS`)
