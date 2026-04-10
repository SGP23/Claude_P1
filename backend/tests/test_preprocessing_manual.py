"""
Manual Skeleton Preprocessing Test
───────────────────────────────────
Shows webcam feed alongside extracted skeleton for visual verification.

Run:
    cd backend && python tests/test_preprocessing_manual.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from backend.tracking.hand_skeleton_preprocessor import (
    SkeletonPreprocessor,
    HAND_CONNECTIONS,
    CANVAS_SIZE,
)


def test_skeleton_visual():
    """Visual test - display skeleton to verify correctness."""

    preprocessor = SkeletonPreprocessor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("=" * 50)
    print("SKELETON PREPROCESSING TEST")
    print("=" * 50)
    print("Show your hand to the camera")
    print("Press 'q' to quit")
    print("=" * 50)

    frame_count = 0
    detect_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        skeleton, landmarks, success = preprocessor.extract_skeleton(frame_rgb)

        cv2.imshow("Original", frame)

        frame_count += 1

        if success:
            detect_count += 1
            assert skeleton is not None
            assert landmarks is not None
            # Convert RGB skeleton to BGR for OpenCV display
            skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR)
            cv2.imshow("Skeleton", skeleton_bgr)

            # Validate skeleton
            assert skeleton.shape == (CANVAS_SIZE, CANVAS_SIZE, 3), (
                f"Wrong shape: {skeleton.shape}"
            )
            assert landmarks.shape == (21, 3), (
                f"Wrong landmarks shape: {landmarks.shape}"
            )

            # Check for green pixels (lines) and red pixels (dots)
            green_pixels = np.sum(
                (skeleton[:, :, 1] > 200)
                & (skeleton[:, :, 0] < 50)
                & (skeleton[:, :, 2] < 50)
            )
            red_pixels = np.sum(
                (skeleton[:, :, 0] < 50)
                & (skeleton[:, :, 1] < 50)
                & (skeleton[:, :, 2] > 200)
            )

            if frame_count % 30 == 0:
                print(
                    f"[OK] Frame {frame_count}: Hand detected | "
                    f"Green px: {green_pixels} | Red px: {red_pixels} | "
                    f"Landmarks: {landmarks.shape}"
                )
        else:
            no_hand_img = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
            cv2.putText(
                no_hand_img,
                "No hand detected",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
            cv2.imshow("Skeleton", no_hand_img)

            if frame_count % 30 == 0:
                print(f"[--] Frame {frame_count}: No hand detected")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nResults: {detect_count}/{frame_count} frames with hand detected")
    print(f"Connection count: {len(HAND_CONNECTIONS)} (expected 20)")
    assert len(HAND_CONNECTIONS) == 20, f"Expected 20 connections, got {len(HAND_CONNECTIONS)}"
    print("[PASS] Preprocessing test complete!")


if __name__ == "__main__":
    test_skeleton_visual()
