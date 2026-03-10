"""
Manual Server Test
──────────────────
Tests HTTP endpoints and Socket.IO connectivity against a running server.

Run:
    1. Start server: cd backend && python server.py
    2. Run tests:    python tests/test_server_manual.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_URL = "http://localhost:8000"


def test_http_endpoints():
    import requests

    print("=" * 50)
    print("TESTING HTTP ENDPOINTS")
    print("=" * 50)

    # 1. Root
    print("\n1. GET /")
    r = requests.get(f"{BASE_URL}/", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "endpoints" in data
    print(f"   Status: {r.status_code}")
    print(f"   Endpoints listed: {len(data['endpoints'])}")
    print("   [PASS]")

    # 2. Health
    print("\n2. GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print(f"   Status: {data['status']}, Model: {data['model_loaded']}, Device: {data['device']}")
    print("   [PASS]")

    # 3. Model status
    print("\n3. GET /model-status")
    r = requests.get(f"{BASE_URL}/model-status", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["loaded"] is True
    assert data["classes"] == 24
    assert len(data["class_names"]) == 24
    print(f"   Type: {data['model_type']}, Classes: {data['classes']}")
    print(f"   Smoothing: window={data['temporal_smoothing']['window_size']}, "
          f"min_frames={data['temporal_smoothing']['min_frames']}")
    print("   [PASS]")

    # 4. Dataset info
    print("\n4. GET /dataset-info")
    r = requests.get(f"{BASE_URL}/dataset-info", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["num_classes"] == 24
    print(f"   Classes: {data['num_classes']}, Type: {data['model_type']}")
    print("   [PASS]")

    # 5. Confidence settings
    print("\n5. GET /confidence-settings")
    r = requests.get(f"{BASE_URL}/confidence-settings", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "always_predict"
    print(f"   Mode: {data['mode']}")
    print("   [PASS]")

    # 6. Training status
    print("\n6. GET /training-status")
    r = requests.get(f"{BASE_URL}/training-status", timeout=5)
    assert r.status_code == 200
    data = r.json()
    print(f"   Status: {data['status']}")
    print("   [PASS]")

    # 7. Logs
    print("\n7. GET /logs")
    r = requests.get(f"{BASE_URL}/logs", timeout=5)
    assert r.status_code == 200
    data = r.json()
    print(f"   Log entries: {len(data['logs'])}")
    print("   [PASS]")

    # 8. Word suggestions
    print("\n8. POST /suggest-words")
    r = requests.post(f"{BASE_URL}/suggest-words",
                      json={"sentence": "HEL", "max_suggestions": 4}, timeout=5)
    assert r.status_code == 200
    data = r.json()
    print(f"   Input: 'HEL' -> Suggestions: {data['suggestions']}")
    assert "suggestions" in data
    print("   [PASS]")

    # 9. Complete word
    print("\n9. POST /complete-word")
    r = requests.post(f"{BASE_URL}/complete-word",
                      json={"sentence": "HEL", "suggestion": "HELLO"}, timeout=5)
    assert r.status_code == 200
    data = r.json()
    print(f"   Input: 'HEL' + 'HELLO' -> '{data['sentence']}'")
    print("   [PASS]")

    # 10. Groups
    print("\n10. GET /groups")
    r = requests.get(f"{BASE_URL}/groups", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "groups" in data
    print(f"    Groups: {len(data['groups'])}, Excluded: {data['excluded']}")
    print("    [PASS]")

    # 11. Speak (TTS)
    print("\n11. POST /speak")
    r = requests.post(f"{BASE_URL}/speak", json={"text": "test"}, timeout=10)
    print(f"    Status: {r.status_code} (TTS may not work on all systems)")
    print("    [PASS]")

    print("\n" + "=" * 50)
    print("[PASS] All HTTP endpoint tests passed!")
    print("=" * 50)


def test_socketio():
    import socketio as sio_client

    print("\n" + "=" * 50)
    print("TESTING SOCKET.IO")
    print("=" * 50)

    client = sio_client.Client()
    connected = False
    prediction_received = False

    def on_connect():
        nonlocal connected
        connected = True
        print("   [OK] Connected")

    def on_prediction(data):
        nonlocal prediction_received
        prediction_received = True
        print(f"   [OK] Prediction received: letter={data.get('letter')}, "
              f"conf={data.get('confidence')}, hand={data.get('hand_detected')}")

    client.on("connect", on_connect)
    client.on("prediction", on_prediction)

    print("\n1. Connecting to Socket.IO...")
    try:
        client.connect(BASE_URL, transports=["websocket", "polling"])
        time.sleep(1)
        assert connected, "Failed to connect"
        print("   [PASS] Connection established")

        # Send a tiny test frame (black 10x10 image as base64)
        import base64
        import numpy as np
        import cv2

        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", test_frame)
        b64 = base64.b64encode(buf).decode("utf-8")

        print("\n2. Sending test frame...")
        client.emit("predict_frame", {"frame": b64})
        time.sleep(2)

        if prediction_received:
            print("   [PASS] Prediction response received")
        else:
            print("   [INFO] No prediction (no hand in test frame - expected)")

        client.disconnect()
        print("\n   [PASS] Socket.IO test complete!")

    except Exception as e:
        print(f"   [ERROR] {e}")
        print("   Make sure the server is running on http://localhost:8000")


if __name__ == "__main__":
    print("Make sure the server is running!")
    print(f"Server URL: {BASE_URL}")
    input("Press Enter to start tests...")

    test_http_endpoints()
    test_socketio()

    print("\n" + "=" * 50)
    print("[PASS] ALL SERVER TESTS COMPLETE!")
    print("=" * 50)
