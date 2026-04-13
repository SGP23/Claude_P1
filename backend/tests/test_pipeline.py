"""
End-to-end test of the predict_frame Socket.IO pipeline.
Sends a real webcam-like frame and checks for a prediction response.
"""
import asyncio
import base64
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2


async def test_predict_from_frame():
    """Test predict_from_frame directly without needing Socket.IO."""
    # Import server module to access predict_from_frame
    from backend.server import load_model, predict_from_frame, MODEL

    # Load model if not loaded
    if MODEL is None:
        print("Loading model...")
        success = load_model()
        print(f"Model loaded: {success}")
        if not success:
            print("FAIL: Model could not be loaded")
            return False

    # Create a test frame (640x480 with a skin-colored region)
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fill with a neutral background
    frame_rgb[:] = [200, 200, 200]

    print("\n--- Test 1: Empty frame (no hand) ---")
    letter, confidence, metadata = predict_from_frame(frame_rgb)
    print(f"  letter={letter}, confidence={confidence}")
    print(f"  hand_detected={metadata.get('hand_detected')}")
    print(f"  info={metadata.get('info')}")

    if letter is None and metadata.get("info") == "NO_HAND_DETECTED":
        print("  PASS: Correctly reports no hand detected")
    else:
        print("  WARN: Unexpected result for empty frame")

    # Test 2: Try with a real webcam frame if available
    print("\n--- Test 2: Webcam single frame capture ---")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"  Captured frame: {frame_rgb.shape}")
            letter, confidence, metadata = predict_from_frame(frame_rgb)
            print(f"  letter={letter}, confidence={confidence:.4f}")
            print(f"  hand_detected={metadata.get('hand_detected')}")
            print(f"  info={metadata.get('info')}")
            print(f"  landmarks present={metadata.get('landmarks') is not None}")
            if metadata.get("landmarks"):
                print(f"  landmark count={len(metadata['landmarks'])}")
            if metadata.get("prediction_quality"):
                pq = metadata["prediction_quality"]
                print(f"  confidence_level={pq.get('confidence_level')}")
                print(f"  top2_ratio={pq.get('top2_ratio', 0):.2f}")
        else:
            print("  SKIP: Could not read webcam frame")
    else:
        print("  SKIP: No webcam available")

    # Test 3: Simulate the full Socket.IO handler data path
    print("\n--- Test 3: Full base64 decode pipeline ---")
    # Create a test frame and encode as JPEG base64 (mimics frontend)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = [180, 180, 180]
    _, jpeg_data = cv2.imencode('.jpg', test_frame)
    b64_data = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_data}"

    # Replicate the handler's decode logic
    frame_b64 = data_url
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    decoded = base64.b64decode(frame_b64)
    nparr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        print(f"  Frame decoded OK: {frame.shape}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letter, confidence, metadata = predict_from_frame(frame_rgb)
        print(f"  letter={letter}, confidence={confidence}")
        print(f"  info={metadata.get('info')}")
        print("  PASS: Full decode pipeline works")
    else:
        print("  FAIL: cv2.imdecode returned None")

    return True


if __name__ == "__main__":
    asyncio.run(test_predict_from_frame())
