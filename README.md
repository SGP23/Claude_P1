# SignLens

## Live Demo Links
- **Frontend App**: [https://sgp23.github.io/Signlens/](https://sgp23.github.io/Signlens/)
- **Backend API Docs**: [https://signlens-jt9a.onrender.com/docs](https://signlens-jt9a.onrender.com/docs)
- **Backend Health Check**: [https://signlens-jt9a.onrender.com/health](https://signlens-jt9a.onrender.com/health)
- **Repository**: [https://github.com/SGP23/Signlens](https://github.com/SGP23/Signlens)

## Overview
SignLens is a real-time American Sign Language (ASL) alphabet recognition system. It utilizes MediaPipe for hand landmark extraction and pairs it with neural network classification to provide live webcam inference. The application seamlessly translates ASL gestures into text and offers speech synthesis for enhanced accessibility.

## Features
- Real-time ASL alphabet recognition via webcam.
- Robust hand landmark extraction using MediaPipe.
- Hybrid neural network classification (MLP and CNN layer fallback).
- Temporal smoothing logic to stabilize live predictions.
- Dictionary-based word formation and suggestions.
- Text-to-speech (TTS) synthesis of formed sentences.
- Low-latency data streaming using Socket.IO.

## System Architecture

The pipeline processes live video frames sequentially:

```text
Webcam → MediaPipe Landmark Extraction → Normalization → MLP Classifier 
   ↳ (Fallback) CNN Disambiguation → Smoothing Buffer → Word Builder 
         ↳ Dictionary Suggestions → TTS Output → Frontend Visualization
```

## Tech Stack
- **Frontend**: React, Vite, Tailwind CSS, hosted on GitHub Pages.
- **Backend**: FastAPI, PyTorch, MediaPipe, Socket.IO, hosted on Render.

## Repository Structure
- `/frontend/`: Contains the React + Vite frontend application.
- `/backend/`: FastAPI application, prediction endpoints, and Socket.IO server.
- `/training/`: Scripts and utilities for training the neural network models.
- `/models/`: Directory housing the trained model weights.
- `README.md`: Project documentation.

## Model Pipeline Summary
1. **Landmark Extraction**: Isolates hand structures from video frames using MediaPipe.
2. **Feature Normalization**: Standardizes landmark coordinates for distance and scale invariance.
3. **MLP Classification**: A Multi-Layer Perceptron efficiently predicts base ASL characters from the normalized landmarks.
4. **CNN Fallback Logic**: Resolves geometrically similar or ambiguous gestures structurally.
5. **Temporal Smoothing**: Analyzes frame sequences over sliding windows to prevent prediction flickering.
6. **Dictionary Suggestions**: Suggests completions based on the predicted character sequences.
7. **Speech Synthesis**: Converts the final text output into audible speech.

## API Endpoints
- `GET /health`: Basic health check to verify backend operational status.
- `GET /model-status`: Returns the current load status and configuration of the ML models.
- `GET /docs`: Interactive Swagger UI for exploring the REST API.
- `WS /socket.io`: Real-time WebSocket endpoint for low-latency frame submission and prediction retrieval.

## Local Setup Instructions

Clone the repository:
```bash
git clone https://github.com/SGP23/Signlens.git
cd Signlens
```

### Backend
Start a virtual environment, install dependencies, and run the FastAPI server:
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

### Frontend
Install dependencies and run the Vite development server:
```bash
cd frontend
npm install
npm run dev
```

## Deployment Architecture
- **Frontend Environment**: Built as a static Single Page Application (SPA), deployed to **GitHub Pages**.
- **Backend Environment**: Containerized/serverless environment deployed to **Render**.
- **Communication Bridge**: Frame data and predictions flow over a persistent **Socket.IO streaming connection** mapped to the backend.

## Environment Variables

### Frontend (`frontend/.env.production`)
- `VITE_API_URL`: Points the frontend to the deployed backend server during production builds.
  ```env
  VITE_API_URL=https://signlens-jt9a.onrender.com
  ```

## Known Limitations
- Variable CPU inference latency depending on device hardware.
- High initial load times / cold-start delays on Render's free tier.
- Limited scope to static single-hand ASL alphabet characters (ignores motion).
- Recognition performance may degrade under poor lighting or highly cluttered backgrounds.

## Future Improvements
- Implement word-level and dynamic gesture ASL recognition.
- Expand to full sentence-level prediction logic.
- Model pruning and WebGL integration for mobile-browser optimization.
- Scalable GPU backend deployment for unthrottled concurrent inference.
- Continuous dataset expansion to integrate edge cases and distinct skin tones.

## Contributors
- **Sundari Pathak** — Backend architecture, ML model training, and prediction pipeline design.

## License
MIT License