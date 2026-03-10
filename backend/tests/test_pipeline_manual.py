"""
Manual Pipeline Test
────────────────────
Tests the complete prediction pipeline with live webcam input.

Run:
    cd backend && python tests/test_pipeline_manual.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import time
from backend.prediction_engine import PredictionEngine


def test_pipeline_live():
    print("=" * 50)
    print("COMPLETE PIPELINE TEST")
    print("=" * 50)
    print("1. Show ASL letters to camera")
    print("2. The predicted letter + confidence will display")
    print("3. Press 'q' to quit")
    print("=" * 50)

    engine = PredictionEngine()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    frame_count = 0
    total_latency = 0.0
    latency_samples = 0
    detect_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.time()
        result = engine.predict_frame(frame_rgb)
        latency_ms = (time.time() - start) * 1000

        total_latency += latency_ms
        latency_samples += 1
        frame_count += 1

        # Display info on frame
        letter = result.get("letter") or "--"
        confidence = result.get("confidence", 0)
        is_stable = result.get("is_stable", False)
        hand_detected = result.get("hand_detected", False)

        if hand_detected:
            detect_count += 1

        color = (0, 255, 0) if is_stable else (0, 165, 255)
        cv2.putText(frame, f"Letter: {letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Conf: {confidence:.1%}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Stable: {is_stable}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow("Pipeline Test", frame)

        if frame_count % 30 == 0:
            avg_latency = total_latency / latency_samples
            print(
                f"Frame {frame_count}: "
                f"Letter={letter} Conf={confidence:.1%} "
                f"Stable={is_stable} "
                f"Latency={latency_ms:.1f}ms (avg={avg_latency:.1f}ms)"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_latency = total_latency / max(latency_samples, 1)
    print(f"\n{'=' * 50}")
    print(f"RESULTS")
    print(f"{'=' * 50}")
    print(f"Frames processed: {frame_count}")
    print(f"Hands detected:   {detect_count}/{frame_count} ({detect_count/max(frame_count,1)*100:.1f}%)")
    print(f"Average latency:  {avg_latency:.1f}ms")
    print(f"Target latency:   <50ms")

    if avg_latency < 50:
        print("[PASS] Latency target met!")
    else:
        print(f"[WARN] Latency {avg_latency:.1f}ms exceeds 50ms target")


if __name__ == "__main__":
    test_pipeline_live()
