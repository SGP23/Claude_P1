"""
Prediction Engine
─────────────────
Combines preprocessing, model inference, geometric disambiguation,
and temporal smoothing into a complete prediction pipeline.

WORKFLOW:
1. Receive frame from webcam
2. Detect hand & extract landmarks (MediaPipe)
3. Classify letter (landmark MLP or CNN)
4. Apply geometric disambiguation rules
5. Apply temporal smoothing (confidence-weighted voting)
6. Optionally generate skeleton visualization
7. Return final prediction with metadata
"""

import os
import sys
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image as PILImage
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.landmark_model import LandmarkClassifier, extract_landmark_features
from backend.models.disambiguation import GeometricDisambiguator
from backend.tracking.hand_skeleton_preprocessor import SkeletonPreprocessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PredictionEngine:
    """
    End-to-end prediction engine that orchestrates all pipeline stages.
    """

    def __init__(
        self,
        model_path=None,
        labels_path=None,
        hand_model_path=None,
        smoothing_window=10,
        consistency_threshold=0.6,
        min_frames=4,
        use_disambiguation=True,
        generate_skeleton=True,
    ):
        self.device = DEVICE
        self.model = None
        self.model_type = None
        self.class_names = []
        self.disambiguator = GeometricDisambiguator() if use_disambiguation else None
        self.use_disambiguation = use_disambiguation
        self.generate_skeleton = generate_skeleton
        self.skeleton_preprocessor = None
        self.last_error = None

        # Temporal smoothing settings
        self.smoothing_window = smoothing_window
        self.consistency_threshold = consistency_threshold
        self.min_frames = min_frames

        # Per-session smoothing buffers
        self._buffers = {}

        # MediaPipe detector
        self._detector = None
        self._hand_model_path = hand_model_path or os.path.join(
            PROJECT_ROOT, "models", "hand_landmarker.task"
        )

        # Load model
        self._load_model(model_path, labels_path)

        # Initialize skeleton preprocessor lazily
        if self.generate_skeleton:
            try:
                self.skeleton_preprocessor = SkeletonPreprocessor(self._hand_model_path)
            except FileNotFoundError:
                self.skeleton_preprocessor = None

    @property
    def detector(self):
        if self._detector is None:
            if not os.path.exists(self._hand_model_path):
                self.last_error = f"Hand model not found: {self._hand_model_path}"
                return None
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._hand_model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            try:
                self._detector = HandLandmarker.create_from_options(options)
            except Exception as exc:
                self.last_error = f"Failed to initialize hand detector: {exc}"
                self._detector = None
        return self._detector

    def _load_model(self, model_path=None, labels_path=None):
        """Load a landmark or CNN model."""
        landmark_path = model_path or os.path.join(PROJECT_ROOT, "landmark_classifier.pt")
        cnn_path = os.path.join(PROJECT_ROOT, "sign_language_cnn_trained.pt")
        lbl_landmark = labels_path or os.path.join(PROJECT_ROOT, "landmark_class_labels.txt")
        lbl_cnn = os.path.join(PROJECT_ROOT, "class_labels.txt")

        # Prefer landmark model
        if os.path.exists(landmark_path):
            chosen_model = landmark_path
            chosen_labels = lbl_landmark if os.path.exists(lbl_landmark) else lbl_cnn
            self.model_type = "landmark"
        elif os.path.exists(cnn_path):
            chosen_model = cnn_path
            chosen_labels = lbl_cnn
            self.model_type = "cnn"
        else:
            self.last_error = "No model file found (landmark or cnn)."
            return

        try:
            checkpoint = torch.load(chosen_model, map_location=self.device, weights_only=False)
        except Exception as exc:
            self.last_error = f"Failed to load model '{chosen_model}': {exc}"
            self.model = None
            return
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 24)
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]
        else:
            state_dict = checkpoint
            final_keys = [k for k in state_dict if k.endswith(".weight")]
            num_classes = state_dict[final_keys[-1]].shape[0] if final_keys else 24

        if not self.class_names and os.path.exists(chosen_labels):
            with open(chosen_labels, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]

        if not self.class_names:
            self.class_names = [chr(65 + i) for i in range(num_classes)]

        if self.model_type == "landmark":
            self.model = LandmarkClassifier(num_classes)
        else:
            from models.cnn_model import SignLanguageCNN
            self.model = SignLanguageCNN(num_classes)

        try:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            self.last_error = f"Failed to initialize model runtime: {exc}"
            self.model = None

    def predict_frame(self, frame_rgb, session_id="default", use_smoothing=True):
        """
        Predict letter from a single webcam frame.

        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format
            session_id: string identifier for temporal smoothing buffer
            use_smoothing: whether to apply temporal smoothing

        Returns:
            dict with keys:
                letter: predicted letter or None
                confidence: float 0-1
                landmarks: list of {x, y, z} dicts or None
                skeleton_b64: base64-encoded skeleton JPEG or None
                is_stable: bool - whether prediction is stable
                was_corrected: bool - whether disambiguation changed prediction
                hand_detected: bool
                timestamp: float
        """
        result = {
            "letter": None,
            "confidence": 0.0,
            "landmarks": None,
            "skeleton_b64": None,
            "is_stable": False,
            "was_corrected": False,
            "hand_detected": False,
            "timestamp": time.time(),
        }

        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
            return result
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            return result

        if self.model is None:
            return result

        detector = self.detector
        if detector is None:
            return result

        # Step 1: Detect hand
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection = detector.detect(mp_image)
        except Exception:
            return result

        if not detection.hand_landmarks:
            return result

        result["hand_detected"] = True
        hand = detection.hand_landmarks[0]

        # Extract landmarks as list of dicts (for frontend visualization)
        landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
        result["landmarks"] = landmarks_list

        landmarks_np = np.array([(lm.x, lm.y, lm.z) for lm in hand])

        # Step 2: Generate skeleton image if enabled
        if self.generate_skeleton and self.skeleton_preprocessor:
            try:
                skeleton, _, _ = self.skeleton_preprocessor.extract_skeleton(frame_rgb)
                if skeleton is not None:
                    import base64
                    _, buf = cv2.imencode(".jpg", skeleton)
                    result["skeleton_b64"] = (
                        "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
                    )
            except Exception:
                pass

        # Step 3: Model inference
        if self.model_type == "landmark":
            features = extract_landmark_features(hand)
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            # CNN path - crop hand region
            h, w = frame_rgb.shape[:2]
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            pad = 0.15
            x_min = max(0, int((min(xs) - pad) * w))
            x_max = min(w, int((max(xs) + pad) * w))
            y_min = max(0, int((min(ys) - pad) * h))
            y_max = min(h, int((max(ys) + pad) * h))
            if x_max - x_min < 10 or y_max - y_min < 10:
                return result
            crop = frame_rgb[y_min:y_max, x_min:x_max]
            pil_img = PILImage.fromarray(crop)
            img_tensor: torch.Tensor = CNN_TRANSFORM(pil_img)  # type: ignore[assignment]
            tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_conf, top_idx = probs.max(dim=0)
        predicted_label = (
            self.class_names[int(top_idx)]
            if int(top_idx) < len(self.class_names)
            else self.class_names[0]
        )
        confidence = float(top_conf)

        # Step 4: Geometric disambiguation
        was_corrected = False
        if self.use_disambiguation and self.disambiguator:
            predicted_label, was_corrected = self.disambiguator.disambiguate(
                predicted_label, confidence, landmarks_np
            )

        result["confidence"] = confidence
        result["was_corrected"] = was_corrected

        # Step 5: Temporal smoothing
        if use_smoothing:
            predicted_label, confidence, is_stable = self._smooth(
                session_id, predicted_label, confidence
            )
            result["confidence"] = confidence
            result["is_stable"] = is_stable

        result["letter"] = predicted_label
        return result

    def _smooth(self, session_id, prediction, confidence):
        """Apply temporal smoothing with confidence-weighted voting."""
        if session_id not in self._buffers:
            self._buffers[session_id] = deque(maxlen=self.smoothing_window)

        buf = self._buffers[session_id]
        buf.append((prediction, confidence, time.time()))

        if len(buf) < self.min_frames:
            return prediction, confidence, False

        # Confidence-weighted voting
        weighted = {}
        for pred, conf, _ in buf:
            weighted[pred] = weighted.get(pred, 0.0) + conf ** 2

        best = max(weighted, key=lambda k: weighted[k])

        # Raw consistency check
        raw_count = sum(1 for p, _, _ in buf if p == best)
        is_stable = raw_count / len(buf) >= self.consistency_threshold

        # Average confidence for best prediction
        matching = [c for p, c, _ in buf if p == best]
        avg_conf = sum(matching) / len(matching)

        blended = 0.6 * avg_conf + 0.4 * confidence if prediction == best else avg_conf
        return best, blended, is_stable

    def clear_buffer(self, session_id="default"):
        """Clear temporal smoothing buffer for a session."""
        self._buffers.pop(session_id, None)

    def reset(self):
        """Reset all session buffers."""
        self._buffers.clear()
