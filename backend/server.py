"""
FastAPI + Socket.IO Backend — Sign Language Recognition
───────────────────────────────────────────────────────
REST endpoints + Socket.IO real-time prediction.

Features:
- Always returns the best predicted ASL letter (no rejection)
- Temporal smoothing for stable predictions across frames
- Hand landmark detection and validation
- Confidence level metadata (high/medium/low)

Run:
    cd backend && uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import base64
import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio

# ─── Project root ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.landmark_model import LandmarkClassifier, extract_landmark_features
from backend.models.disambiguation import GeometricDisambiguator
from backend.prediction.word_predictor import WordPredictor

# ─── Disambiguation & Word Prediction ────────────────────
DISAMBIGUATOR = GeometricDisambiguator()
WORD_PREDICTOR = WordPredictor()

# ─── Socket.IO Server (created early so logger can reference it) ──
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

# ─── Logging ─────────────────────────────────────────────
LOG_BUFFER: deque = deque(maxlen=500)


class BufferHandler(logging.Handler):
    """Store log entries in a ring buffer and broadcast via Socket.IO."""

    def emit(self, record):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": self.format(record),
        }
        LOG_BUFFER.append(entry)
        # Broadcast to all connected Socket.IO clients (fire-and-forget)
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(sio.emit("log_message", entry))
        except Exception:
            pass


logger = logging.getLogger("slr")
logger.setLevel(logging.INFO)
bh = BufferHandler()
bh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(bh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(sh)

# ─── Global State ────────────────────────────────────────
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES: list[str] = []
MODEL_LOAD_TIME: float = 0.0
MODEL_TYPE = "landmark"  # "landmark" or "cnn"
CNN_IMAGE_SIZE = 224  # Legacy, kept for compatibility

# ─── Confidence Settings (Informational Only) ────────────
# NOTE: We no longer reject predictions. The model always returns its best guess.
# These thresholds are kept for metadata/debugging purposes only.

# Confidence thresholds (informational - not used for rejection)
CONFIDENCE_THRESHOLD_DEFAULT = 0.30   # Low threshold - for metadata only
CONFIDENCE_THRESHOLD_HIGH = 0.60      # High confidence marker
ENTROPY_THRESHOLD = 2.5               # For metadata analysis only
TOP2_RATIO_THRESHOLD = 1.2            # For metadata analysis only

# Hand detection quality thresholds
MIN_HAND_DETECTION_CONF = 0.5         # MediaPipe hand detection confidence
MIN_HAND_PRESENCE_CONF = 0.5          # MediaPipe hand presence confidence
MIN_HAND_SIZE_RATIO = 0.03            # Minimum hand size relative to frame

# ─── Temporal Smoothing Settings (Improved) ──────────────
TEMPORAL_WINDOW_SIZE = 10             # Larger window for stability
TEMPORAL_MIN_FRAMES = 4               # Minimum frames before outputting
TEMPORAL_CONSISTENCY_THRESHOLD = 0.6  # 60% agreement required

TRANSFORM = T.Compose(
    [
        T.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# MediaPipe hand detector (loaded lazily)
HAND_DETECTOR = None

# ─── Temporal Smoothing Buffer ───────────────────────────
# Per-session prediction history for smoothing
# Format: {session_id: deque([(prediction, confidence, timestamp), ...])}
PREDICTION_BUFFERS: dict = {}


def get_smoothed_prediction(
    session_id: str, 
    current_prediction: str, 
    current_confidence: float,
    timestamp: float
) -> Tuple[str, float, bool]:
    """
    Apply improved temporal smoothing with confidence-weighted voting.
    
    Algorithm:
    1. Maintain sliding window of last N predictions
    2. Filter out low-confidence predictions
    3. Use confidence-weighted voting
    4. Require minimum agreement for stable output
    
    Returns:
        (smoothed_prediction, smoothed_confidence, is_stable)
    """
    global PREDICTION_BUFFERS
    
    # Initialize buffer for new sessions
    if session_id not in PREDICTION_BUFFERS:
        PREDICTION_BUFFERS[session_id] = deque(maxlen=TEMPORAL_WINDOW_SIZE)
    
    buffer = PREDICTION_BUFFERS[session_id]
    
    # Add current prediction to buffer
    buffer.append((current_prediction, current_confidence, timestamp))
    
    # Need minimum frames before outputting stable prediction
    if len(buffer) < TEMPORAL_MIN_FRAMES:
        return current_prediction, current_confidence, False
    
    # Confidence-weighted voting
    weighted_votes = {}
    
    for pred, conf, _ in buffer:
        # Weight predictions by their confidence
        weight = conf ** 2  # Square confidence for stronger weighting
        weighted_votes[pred] = weighted_votes.get(pred, 0.0) + weight
    
    # Find best prediction
    best_pred = max(weighted_votes.keys(), key=lambda p: weighted_votes[p])
    
    # Count raw occurrences for stability check
    raw_counts = {}
    for pred, _, _ in buffer:
        raw_counts[pred] = raw_counts.get(pred, 0) + 1
    
    consistency_ratio = raw_counts.get(best_pred, 0) / len(buffer)
    is_stable = consistency_ratio >= TEMPORAL_CONSISTENCY_THRESHOLD
    
    # Calculate smoothed confidence
    matching_confs = [conf for pred, conf, _ in buffer if pred == best_pred]
    avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else current_confidence
    
    # Blend current and average confidence
    smoothed_conf = 0.6 * avg_conf + 0.4 * current_confidence if current_prediction == best_pred else avg_conf
    
    return best_pred, smoothed_conf, is_stable


def clear_prediction_buffer(session_id: str):
    """Clear the prediction buffer for a session."""
    global PREDICTION_BUFFERS
    if session_id in PREDICTION_BUFFERS:
        del PREDICTION_BUFFERS[session_id]


def get_hand_detector():
    global HAND_DETECTOR
    if HAND_DETECTOR is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        model_path = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            logger.error(f"Hand landmarker model not found: {model_path}")
            return None
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONF,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONF,
        )
        HAND_DETECTOR = HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker loaded")
    return HAND_DETECTOR


# ─── Prediction Quality Utilities ────────────────────────

def calculate_entropy(probs: torch.Tensor) -> float:
    """
    Calculate Shannon entropy of probability distribution.
    Lower entropy = more confident prediction.
    """
    # Add small epsilon to avoid log(0)
    probs_safe = probs.clamp(min=1e-10)
    entropy = -torch.sum(probs_safe * torch.log(probs_safe))
    return float(entropy)


def calculate_prediction_quality(probs: torch.Tensor) -> dict:
    """
    Analyze prediction quality metrics (informational only - no rejection).
    
    Returns dict with:
        - top1_conf: highest probability
        - top2_conf: second highest probability
        - top1_idx: index of top prediction
        - top2_idx: index of second prediction
        - entropy: Shannon entropy
        - top2_ratio: ratio of top1/top2 (higher = more decisive)
        - confidence_level: "high", "medium", or "low" (informational)
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    top1_conf = float(sorted_probs[0])
    top2_conf = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
    
    entropy = calculate_entropy(probs)
    top2_ratio = top1_conf / max(top2_conf, 1e-10)
    
    # Informational confidence level (not used for rejection)
    if top1_conf >= CONFIDENCE_THRESHOLD_HIGH:
        confidence_level = "high"
    elif top1_conf >= CONFIDENCE_THRESHOLD_DEFAULT:
        confidence_level = "medium" 
    else:
        confidence_level = "low"
    
    return {
        "top1_conf": top1_conf,
        "top2_conf": top2_conf,
        "top1_idx": top1_idx,
        "top2_idx": top2_idx,
        "entropy": entropy,
        "top2_ratio": top2_ratio,
        "confidence_level": confidence_level,
    }


def validate_hand_landmarks(landmarks, img_width: int, img_height: int) -> Tuple[bool, list]:
    """
    Validate detected hand landmarks for quality.
    
    Returns (is_valid, list_of_issues)
    
    NOTE: This is now more permissive - only rejects if hand is completely invalid.
    """
    issues = []
    
    if not landmarks:
        return False, ["NO_LANDMARKS"]
    
    # Extract normalized coordinates
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    
    # Note edge cases but don't reject
    if min(xs) < 0.02 or max(xs) > 0.98:
        issues.append("HAND_NEAR_EDGE_X")
    if min(ys) < 0.02 or max(ys) > 0.98:
        issues.append("HAND_NEAR_EDGE_Y")
    
    # Check hand size - only reject if extremely small
    hand_width = max(xs) - min(xs)
    hand_height = max(ys) - min(ys)
    hand_area = hand_width * hand_height
    
    if hand_area < MIN_HAND_SIZE_RATIO:
        issues.append("HAND_TOO_SMALL")
        # Still try to process, only fail if area is tiny
        if hand_area < 0.01:
            return False, issues
    
    # Note unusual proportions but don't reject
    aspect_ratio = hand_width / max(hand_height, 0.01)
    if aspect_ratio < 0.2 or aspect_ratio > 4.0:
        issues.append("UNUSUAL_HAND_PROPORTIONS")
    
    # Only fail if landmarks are truly invalid
    return True, issues


def load_model() -> bool:
    """Load the landmark-based classifier model."""
    global MODEL, CLASS_NAMES, MODEL_LOAD_TIME, MODEL_TYPE
    start = time.time()

    # Try landmark model first
    landmark_model_path = os.path.join(PROJECT_ROOT, "landmark_classifier.pt")
    landmark_labels_path = os.path.join(PROJECT_ROOT, "landmark_class_labels.txt")
    
    # Fallback to CNN model if landmark model not found
    cnn_model_path = os.path.join(PROJECT_ROOT, "sign_language_cnn_trained.pt")
    cnn_labels_path = os.path.join(PROJECT_ROOT, "class_labels.txt")

    if os.path.exists(landmark_model_path):
        model_path = landmark_model_path
        labels_path = landmark_labels_path if os.path.exists(landmark_labels_path) else cnn_labels_path
        MODEL_TYPE = "landmark"
    elif os.path.exists(cnn_model_path):
        logger.warning("Landmark model not found, falling back to CNN model")
        model_path = cnn_model_path
        labels_path = cnn_labels_path
        MODEL_TYPE = "cnn"
    else:
        logger.error(f"No model file found. Expected: {landmark_model_path}")
        return False

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Handle both checkpoint dict and raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 24)
            if "class_names" in checkpoint:
                CLASS_NAMES = checkpoint["class_names"]
        else:
            state_dict = checkpoint
            # Detect number of classes from last linear layer
            final_keys = [k for k in state_dict if k.endswith(".weight")]
            if not final_keys:
                logger.error("Model state dict has no weight keys")
                return False
            num_classes = state_dict[final_keys[-1]].shape[0]
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return False

    # Load class labels if not already loaded from checkpoint
    if not CLASS_NAMES:
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                CLASS_NAMES = [line.strip() for line in f if line.strip()]
            if len(CLASS_NAMES) != num_classes:
                logger.warning(
                    f"Label file has {len(CLASS_NAMES)} classes but model has {num_classes}. Using defaults."
                )
                CLASS_NAMES = [chr(65 + i) for i in range(num_classes)]
        else:
            CLASS_NAMES = [chr(65 + i) for i in range(num_classes)]

    # Create model based on type
    if MODEL_TYPE == "landmark":
        MODEL = LandmarkClassifier(num_classes)
    else:
        from models.cnn_model import SignLanguageCNN
        MODEL = SignLanguageCNN(num_classes)
    
    MODEL.load_state_dict(state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()

    MODEL_LOAD_TIME = time.time() - start
    logger.info(
        f"{MODEL_TYPE.upper()} model loaded: {num_classes} classes on {DEVICE} in {MODEL_LOAD_TIME:.2f}s"
    )
    return True


def predict_from_frame(
    frame_rgb: np.ndarray, confidence_threshold: float = None
) -> Tuple[Optional[str], float, dict]:
    """
    Detect hand in frame -> extract landmarks -> classify.
    
    Uses landmark-based classification (63 features from 21 hand landmarks)
    for more robust recognition across lighting/background conditions.
    
    ALWAYS returns the best predicted letter (no rejection based on confidence).
    Only returns None if no hand is detected at all.
    
    Returns:
        (predicted_letter, confidence, metadata_dict)
        
    The metadata dict contains:
        - hand_detected: bool
        - hand_confidence: float
        - prediction_quality: dict (entropy, top2_ratio, etc.)
        - info: Optional[str] - informational messages (not rejection)
        - model_type: str - "landmark" or "cnn"
    """
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD_DEFAULT
    
    metadata = {
        "hand_detected": False,
        "hand_confidence": 0.0,
        "hand_issues": [],
        "prediction_quality": None,
        "info": None,
        "model_type": MODEL_TYPE,
        "landmarks": None,  # Will contain normalized landmark coordinates for visualization
    }
    
    if MODEL is None:
        metadata["info"] = "MODEL_NOT_LOADED"
        return None, 0.0, metadata

    import mediapipe as mp

    detector = get_hand_detector()
    if detector is None:
        metadata["info"] = "DETECTOR_NOT_LOADED"
        return None, 0.0, metadata

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        metadata["info"] = "NO_HAND_DETECTED"
        return None, 0.0, metadata

    metadata["hand_detected"] = True
    
    # Get hand detection confidence
    if result.handedness and len(result.handedness) > 0:
        metadata["hand_confidence"] = result.handedness[0][0].score
    
    h, w, _ = frame_rgb.shape
    hand = result.hand_landmarks[0]
    
    # Validate hand landmarks (permissive - rarely fails)
    is_valid_hand, hand_issues = validate_hand_landmarks(hand, w, h)
    metadata["hand_issues"] = hand_issues
    
    # Extract landmark coordinates for visualization (normalized 0-1)
    metadata["landmarks"] = [
        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand
    ]
    
    if not is_valid_hand:
        metadata["info"] = f"HAND_QUALITY_ISSUE: {', '.join(hand_issues)}"
        return None, 0.0, metadata
    
    # Use landmark-based or CNN-based prediction
    if MODEL_TYPE == "landmark":
        # Extract and normalize landmarks (63 features)
        features = extract_landmark_features(hand)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.inference_mode():
            logits = MODEL(tensor)
            probs = TF.softmax(logits, dim=1)[0]
    else:
        # Fallback: CNN-based prediction (crop hand region)
        xs = [lm.x for lm in hand]
        ys = [lm.y for lm in hand]
        pad = 0.15
        x_min = max(0, int((min(xs) - pad) * w))
        x_max = min(w, int((max(xs) + pad) * w))
        y_min = max(0, int((min(ys) - pad) * h))
        y_max = min(h, int((max(ys) + pad) * h))

        if x_max - x_min < 10 or y_max - y_min < 10:
            metadata["info"] = "CROP_TOO_SMALL"
            return None, 0.0, metadata

        crop = frame_rgb[y_min:y_max, x_min:x_max]
        pil_img = Image.fromarray(crop)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = MODEL(tensor)
            probs = TF.softmax(logits, dim=1)[0]

    # Analyze prediction quality (informational only)
    quality = calculate_prediction_quality(probs)
    metadata["prediction_quality"] = quality
    
    # ALWAYS return the best prediction - no rejection
    predicted_idx = quality["top1_idx"]
    predicted_label = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else CLASS_NAMES[0]
    
    # Apply geometric disambiguation to refine confused letters
    landmarks_np = np.array([(lm.x, lm.y, lm.z) for lm in hand])
    refined_label, was_corrected = DISAMBIGUATOR.disambiguate(
        predicted_label, quality["top1_conf"], landmarks_np
    )
    if was_corrected:
        metadata["info"] = f"DISAMBIGUATED: {predicted_label} → {refined_label}"
        predicted_label = refined_label
    
    # Add informational note about confidence level
    if quality["confidence_level"] == "low" and not was_corrected:
        metadata["info"] = f"LOW_CONFIDENCE: {quality['top1_conf']:.3f}"
    
    return predicted_label, quality["top1_conf"], metadata


# ─── Socket.IO Events ───────────────────────────────────


@sio.event
async def connect(sid, environ):
    logger.info(f"Socket.IO client connected: {sid}")


@sio.event
async def disconnect(sid):
    logger.info(f"Socket.IO client disconnected: {sid}")
    # Clean up prediction buffer for this session
    clear_prediction_buffer(sid)


@sio.event
async def predict_frame(sid, data):
    """
    Receive a base64-encoded JPEG frame and return prediction.
    data: { "frame": "<base64>", "use_smoothing": true }
    
    Uses temporal smoothing for stable predictions across frames.
    Always returns the best predicted letter (never UNKNOWN_GESTURE for valid hands).
    """
    try:
        if not data or "frame" not in data:
            return

        frame_b64 = data["frame"]
        use_smoothing = data.get("use_smoothing", True)

        # Strip data URL prefix if present
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letter, confidence, metadata = predict_from_frame(frame_rgb)
        
        timestamp = time.time()
        is_stable = False
        
        # Apply temporal smoothing if enabled and we got a valid prediction
        if use_smoothing and letter is not None:
            letter, confidence, is_stable = get_smoothed_prediction(
                sid, letter, confidence, timestamp
            )

        response = {
            "letter": letter,
            "confidence": round(confidence, 4),
            "timestamp": timestamp,
            "hand_detected": metadata.get("hand_detected", False),
            "is_stable": is_stable,
            "landmarks": metadata.get("landmarks"),  # For hand skeleton visualization
        }
        
        # Include metadata for debugging (optional)
        if data.get("include_metadata", False):
            response["metadata"] = {
                "prediction_quality": metadata.get("prediction_quality"),
                "hand_issues": metadata.get("hand_issues", []),
                "info": metadata.get("info"),
            }

        await sio.emit("prediction", response, to=sid)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        await sio.emit(
            "prediction",
            {
                "letter": None,
                "confidence": 0.0,
                "error": "Prediction failed",
                "hand_detected": False,
                "is_stable": False,
            },
            to=sid,
        )


# ─── FastAPI App ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Starting Sign Language Recognition API...")
    success = load_model()
    if not success:
        logger.warning("Model not loaded — predictions will be unavailable")
    yield
    logger.info("Shutting down API")


fastapi_app = FastAPI(title="Sign Language Recognition API", lifespan=lifespan)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include additional API routes ───────────────────────
from backend.api.routes import router as api_router

fastapi_app.include_router(api_router)


# ─── REST Endpoints ──────────────────────────────────────
@fastapi_app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SignLens API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/model-status",
            "/dataset-info",
            "/training-status",
            "/confidence-settings",
            "/logs",
            "/predict",
            "/speak",
            "/suggest-words",
            "/complete-word",
            "/groups",
        ],
    }


@fastapi_app.get("/model-status")
async def model_status():
    return {
        "loaded": MODEL is not None,
        "model_type": MODEL_TYPE,
        "classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "device": str(DEVICE),
        "load_time_seconds": round(MODEL_LOAD_TIME, 3),
        "prediction_mode": "landmark_based" if MODEL_TYPE == "landmark" else "cnn_based",
        "temporal_smoothing": {
            "enabled": True,
            "window_size": TEMPORAL_WINDOW_SIZE,
            "min_frames": TEMPORAL_MIN_FRAMES,
            "consistency_threshold": TEMPORAL_CONSISTENCY_THRESHOLD,
        },
    }


@fastapi_app.get("/confidence-settings")
async def get_confidence_settings():
    """Get current confidence settings (informational - no longer used for rejection)."""
    return {
        "mode": "always_predict",
        "note": "Model always returns best prediction. These thresholds are informational only.",
        "confidence_threshold_low": CONFIDENCE_THRESHOLD_DEFAULT,
        "confidence_threshold_high": CONFIDENCE_THRESHOLD_HIGH,
        "min_hand_detection_conf": MIN_HAND_DETECTION_CONF,
        "min_hand_size_ratio": MIN_HAND_SIZE_RATIO,
        "temporal_smoothing": {
            "window_size": TEMPORAL_WINDOW_SIZE,
            "min_frames": TEMPORAL_MIN_FRAMES,
            "consistency_threshold": TEMPORAL_CONSISTENCY_THRESHOLD,
        },
    }


@fastapi_app.get("/dataset-info")
async def dataset_info():
    return {
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_type": MODEL_TYPE,
        "model_file": "landmark_classifier.pt" if MODEL_TYPE == "landmark" else "sign_language_cnn_trained.pt",
    }


@fastapi_app.get("/logs")
async def get_logs(level: str = "all"):
    logs = list(LOG_BUFFER)
    if level.lower() != "all":
        logs = [l for l in logs if l["level"].lower() == level.lower()]
    return {"logs": logs}


@fastapi_app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict ASL letter from uploaded image.
    Always returns the best predicted letter when a hand is detected.
    Returns letter=null only if no hand is found.
    """
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."},
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(
                status_code=400, content={"error": "Invalid image format"}
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letter, confidence, metadata = predict_from_frame(frame_rgb)

        return {
            "letter": letter,
            "confidence": round(confidence, 4),
            "timestamp": time.time(),
            "hand_detected": metadata.get("hand_detected", False),
            "confidence_level": metadata.get("prediction_quality", {}).get("confidence_level", "unknown"),
            "info": metadata.get("info"),
        }
    except Exception as e:
        logger.error(f"Predict endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred during prediction. Please try again."
            },
        )


@fastapi_app.post("/speak")
async def speak_text(data: dict):
    text = data.get("text", "")
    if not text.strip():
        return {"status": "empty"}
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        logger.info(f"TTS: '{text}'")
        return {"status": "spoken", "text": text}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Text-to-speech is unavailable on the server."},
        )


# ─── Health & Training Endpoints ─────────────────────────
@fastapi_app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "timestamp": time.time(),
    }


# Training state (in-memory for now)
TRAINING_STATE = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "status": "idle",
    "message": "",
    "start_time": None,
}


@fastapi_app.get("/training-status")
async def training_status():
    """Get current training status."""
    return TRAINING_STATE.copy()


@fastapi_app.post("/start-training")
async def start_training(data: dict = None):
    """
    Start model training (placeholder).
    
    Note: For actual training, use the training/train.py script from command line.
    This endpoint is for frontend compatibility.
    """
    if TRAINING_STATE["is_training"]:
        return JSONResponse(
            status_code=409,
            content={"error": "Training already in progress"},
        )
    
    # For now, just return a message explaining how to train
    return {
        "status": "info",
        "message": "Training should be run from command line for GPU support.",
        "instruction": "Run: python -m training.train --exclude J Z --num-workers 0",
        "model_loaded": MODEL is not None,
    }


@fastapi_app.post("/stop-training")
async def stop_training():
    """Stop current training session."""
    if not TRAINING_STATE["is_training"]:
        return {"status": "idle", "message": "No training in progress"}
    
    TRAINING_STATE["is_training"] = False
    TRAINING_STATE["status"] = "stopped"
    TRAINING_STATE["message"] = "Training stopped by user"
    logger.info("Training stopped by user request")
    
    return {"status": "stopped", "message": "Training stop requested"}


@fastapi_app.post("/collect-data")
async def collect_data(data: dict):
    """
    Placeholder for data collection endpoint.
    
    Expected data:
    - letter: the ASL letter being collected (A-Z)
    - frame: base64 encoded image
    """
    letter = data.get("letter", "").upper()
    if not letter or len(letter) != 1 or not letter.isalpha():
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid letter. Must be A-Z."},
        )
    
    # Placeholder - in a real implementation, save the frame
    return {
        "status": "info",
        "message": f"Data collection for '{letter}' is not implemented in this version.",
        "letter": letter,
    }


# ─── Mount Socket.IO on the ASGI app ────────────────────
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
