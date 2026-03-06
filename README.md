# 🤟 SignLens — Sign Language Recognition

Real-time American Sign Language fingerspelling recognition using a trained CNN model, MediaPipe hand detection, and a modern React dashboard.

---

## Architecture

```
Webcam → MediaPipe Hand Detection → Crop Hand Region
       → CNN Model (24 classes) → Confidence Score
       → Letter → Word Builder → Sentence → TTS
```

| Component | Technology |
|---|---|
| Backend | FastAPI + Socket.IO (Python) |
| Frontend | React (Vite) + TailwindCSS |
| AI Model | PyTorch CNN (5-block, pre-trained) |
| Hand Detection | MediaPipe HandLandmarker |
| Real-time | Socket.IO WebSocket |

---

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server will automatically load the pre-trained CNN model on startup.

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Start the Frontend

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Pages

| Page | Description |
|---|---|
| **Dashboard** | Model status, class overview, quick stats |
| **Live Recognition** | Webcam feed + real-time letter/word prediction |
| **Dataset Manager** | Read-only view of trained classes |
| **Logs** | Live system log console |
| **Settings** | Camera selector, confidence threshold, dark mode |

---

## Project Structure

```
├── backend/
│   ├── server.py              # FastAPI + Socket.IO backend
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Dashboard, Live, Dataset, Logs, Settings
│   │   ├── layouts/           # DashboardLayout with sidebar
│   │   └── services/          # API + WebSocket clients
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── cnn_model.py           # CNN architecture definition
│   └── hand_landmarker.task   # MediaPipe hand model
├── assets/
│   └── word_dict.txt          # Word suggestion dictionary
├── sign_language_cnn_trained.pt  # Pre-trained model weights
├── class_labels.txt              # Class label mapping (24 letters)
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/model-status` | Model loaded status, class count, device |
| GET | `/dataset-info` | Class names, model file info |
| GET | `/logs?level=all` | System log entries (all/ERROR/INFO) |
| POST | `/predict` | Upload image for prediction |
| POST | `/speak` | Text-to-speech |

### Socket.IO Events

| Event | Direction | Description |
|---|---|---|
| `predict_frame` | Client → Server | Send base64 JPEG frame |
| `prediction` | Server → Client | Letter + confidence result |
| `log_message` | Server → Client | Live log entry broadcast |

---

## Model Architecture

```
Input (63) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
           → Dense(64)  → ReLU
           → Dense(26)  → Softmax
```

Optimizer: Adam | Loss: Sparse Categorical Crossentropy

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.60 | Minimum prediction confidence |
| `SMOOTHING_WINDOW` | 10 | Frames for majority vote |
| `AUGMENTATION_FACTOR` | 5 | Dataset multiplication |
| `EPOCHS` | 100 | Max training epochs |
| `BATCH_SIZE` | 64 | Training batch size |

---

## Performance Tips

- **GPU**: Install `tensorflow[and-cuda]` for GPU training
- **ONNX**: The system automatically prefers ONNX at inference for ~2-5× speedup
- **Samples**: More data per letter = better accuracy (100+ recommended)
- **Lighting**: Consistent lighting during data collection improves generalization

---

## License

MIT
