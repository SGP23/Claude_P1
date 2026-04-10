"""
Unit tests for the skeleton preprocessing module.
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from backend.tracking.hand_skeleton_preprocessor import (
    SkeletonPreprocessor,
    CANVAS_SIZE,
    HAND_CONNECTIONS,
)


@pytest.fixture
def preprocessor():
    """Create a SkeletonPreprocessor instance (requires hand_landmarker.task)."""
    model_path = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        pytest.skip("hand_landmarker.task not found")
    return SkeletonPreprocessor(model_path)


class TestSkeletonDrawing:
    """Test the skeleton drawing logic (no MediaPipe needed)."""

    def test_draw_skeleton_shape(self):
        """Skeleton image should be 400x400x3."""
        model_path = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            pytest.skip("hand_landmarker.task not found")
        proc = SkeletonPreprocessor(model_path)

        # Create fake pixel coordinates (21 landmarks)
        fake_coords = np.random.uniform(100, 500, size=(21, 2))
        bbox = (
            fake_coords[:, 0].min(),
            fake_coords[:, 1].min(),
            fake_coords[:, 0].max() - fake_coords[:, 0].min(),
            fake_coords[:, 1].max() - fake_coords[:, 1].min(),
        )
        skeleton = proc.draw_skeleton(fake_coords, bbox)

        assert skeleton.shape == (CANVAS_SIZE, CANVAS_SIZE, 3)
        assert skeleton.dtype == np.uint8

    def test_draw_skeleton_white_background(self):
        """Most pixels should be white on the skeleton canvas."""
        model_path = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            pytest.skip("hand_landmarker.task not found")
        proc = SkeletonPreprocessor(model_path)

        fake_coords = np.array([[200 + i * 10, 200 + i * 5] for i in range(21)], dtype=np.float64)
        bbox = (200, 200, 200, 100)
        skeleton = proc.draw_skeleton(fake_coords, bbox)

        # Majority of pixels should be white (255)
        white_pixels = np.sum(np.all(skeleton == 255, axis=2))
        total_pixels = CANVAS_SIZE * CANVAS_SIZE
        assert white_pixels / total_pixels > 0.8, "Background should be mostly white"


class TestSkeletonExtraction:
    """Test full extraction pipeline (requires MediaPipe model)."""

    def test_no_hand_frame(self, preprocessor):
        """Black frame should return no hand detected."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        skeleton, landmarks, success = preprocessor.extract_skeleton(black_frame)

        assert success is False
        assert skeleton is None
        assert landmarks is None

    def test_random_noise_frame(self, preprocessor):
        """Random noise frame should likely not detect a hand."""
        noise_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        skeleton, landmarks, success = preprocessor.extract_skeleton(noise_frame)

        # Most random noise won't look like a hand
        if success:
            assert skeleton.shape == (CANVAS_SIZE, CANVAS_SIZE, 3)
            assert landmarks.shape == (21, 3)


class TestHandConnections:
    """Validate hand connection definitions."""

    def test_connection_count(self):
        """Should have correct number of connections."""
        assert len(HAND_CONNECTIONS) == 21  # 4+3+3+3+3+5 = 21

    def test_connection_indices(self):
        """All connection indices should be valid (0-20)."""
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start <= 20, f"Invalid start index: {start}"
            assert 0 <= end <= 20, f"Invalid end index: {end}"
