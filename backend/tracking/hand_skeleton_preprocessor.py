"""
Skeleton Preprocessing Module
─────────────────────────────
Extracts hand landmarks from webcam frames using MediaPipe and renders
a normalized skeleton on a white 400×400 canvas.

This skeleton-based approach is the key innovation that makes the system
robust to varying lighting, backgrounds, and skin tones.

INPUT:  Raw webcam frame (H×W×3 RGB)
OUTPUT: Skeleton image (400×400×3 RGB, white background)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import os

# MediaPipe hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20),
    # Palm connections
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

CANVAS_SIZE = 400
SKELETON_LINE_COLOR = (0, 255, 0)   # Green
SKELETON_LINE_THICKNESS = 3
LANDMARK_COLOR = (0, 0, 255)        # Red
LANDMARK_RADIUS = 2
CANVAS_PADDING = 30


class SkeletonPreprocessor:
    """
    Extracts hand skeleton from webcam frames and renders it
    on a normalized white canvas.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "models", "hand_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        try:
            self.detector = HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize hand landmarker: {exc}") from exc

    def extract_skeleton(self, frame_rgb):
        """
        Extract skeleton from frame.

        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format

        Returns:
            skeleton_image: 400×400×3 numpy array (white background with green skeleton), or None
            landmarks: 21×3 numpy array of (x, y, z) normalized coordinates, or None
            success: bool indicating if hand was detected
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, None, False

        hand = result.hand_landmarks[0]
        h, w, _ = frame_rgb.shape

        # Extract landmarks as numpy array (normalized 0-1)
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand])

        # Convert to pixel coordinates
        pixel_coords = np.zeros((21, 2), dtype=np.float64)
        pixel_coords[:, 0] = landmarks[:, 0] * w
        pixel_coords[:, 1] = landmarks[:, 1] * h

        # Compute bounding box
        x_min = pixel_coords[:, 0].min()
        x_max = pixel_coords[:, 0].max()
        y_min = pixel_coords[:, 1].min()
        y_max = pixel_coords[:, 1].max()
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Draw skeleton on white canvas
        skeleton = self.draw_skeleton(pixel_coords, bbox)

        return skeleton, landmarks, True

    def draw_skeleton(self, pixel_coords, bbox):
        """
        Draw skeleton on white 400×400 background, centered and scaled.

        Args:
            pixel_coords: 21×2 array of (x, y) pixel coordinates
            bbox: (x, y, w, h) bounding box of hand

        Returns:
            skeleton: 400×400×3 numpy array
        """
        canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255

        bx, by, bw, bh = bbox
        # Scale hand to fit canvas with padding
        usable = CANVAS_SIZE - 2 * CANVAS_PADDING
        scale = usable / max(bw, bh, 1)

        # Center hand on canvas
        cx = bx + bw / 2
        cy = by + bh / 2

        mapped = np.zeros_like(pixel_coords)
        mapped[:, 0] = (pixel_coords[:, 0] - cx) * scale + CANVAS_SIZE / 2
        mapped[:, 1] = (pixel_coords[:, 1] - cy) * scale + CANVAS_SIZE / 2
        mapped = mapped.astype(np.int32)

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            pt1 = tuple(mapped[start])
            pt2 = tuple(mapped[end])
            cv2.line(canvas, pt1, pt2, SKELETON_LINE_COLOR, SKELETON_LINE_THICKNESS)

        # Draw landmark points
        for pt in mapped:
            cv2.circle(canvas, tuple(pt), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        return canvas
