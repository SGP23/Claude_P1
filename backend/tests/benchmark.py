"""
Performance benchmark for the prediction pipeline.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def benchmark_model_inference():
    """Benchmark raw model inference speed (without hand detection)."""
    import torch
    import torch.nn.functional as F

    # Load model
    landmark_path = os.path.join(PROJECT_ROOT, "landmark_classifier.pt")
    if not os.path.exists(landmark_path):
        print("SKIP: landmark_classifier.pt not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(landmark_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", 24)
    else:
        state_dict = checkpoint
        final_keys = [k for k in state_dict if k.endswith(".weight")]
        num_classes = state_dict[final_keys[-1]].shape[0]

    from models.landmark_model import LandmarkClassifier

    model = LandmarkClassifier(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create random landmark features (63 features)
    fake_input = torch.randn(1, 63, device=device)

    # Warmup
    for _ in range(50):
        with torch.inference_mode():
            model(fake_input)

    # Benchmark
    times = []
    for _ in range(500):
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(fake_input)
            _ = F.softmax(logits, dim=1)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    p99_ms = np.percentile(times, 99) * 1000

    print(f"Model inference benchmark ({device}):")
    print(f"  Average: {avg_ms:.3f} ms")
    print(f"  P95:     {p95_ms:.3f} ms")
    print(f"  P99:     {p99_ms:.3f} ms")
    print(f"  Target:  < 5 ms (model only)")

    assert avg_ms < 50, f"Model inference too slow: {avg_ms:.2f}ms average"


def benchmark_disambiguation():
    """Benchmark geometric disambiguation speed."""
    from backend.models.disambiguation import GeometricDisambiguator

    disambiguator = GeometricDisambiguator()
    fake_landmarks = np.random.rand(21, 3).astype(np.float64)

    # Warmup
    for _ in range(100):
        disambiguator.disambiguate("A", 0.7, fake_landmarks)

    # Benchmark
    times = []
    letters = list("ABCDEFGHIKLMNOPQRSTUVWXY")
    for _ in range(1000):
        letter = letters[np.random.randint(0, len(letters))]
        start = time.perf_counter()
        disambiguator.disambiguate(letter, 0.7, fake_landmarks)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    print(f"\nDisambiguation benchmark:")
    print(f"  Average: {avg_ms:.4f} ms")
    print(f"  Target:  < 1 ms")

    assert avg_ms < 5, f"Disambiguation too slow: {avg_ms:.4f}ms average"


def benchmark_word_prediction():
    """Benchmark word prediction speed."""
    from backend.word_prediction import WordPredictor

    predictor = WordPredictor()

    prefixes = ["H", "HE", "HEL", "HELL", "HELLO", "TH", "THE", "A", "AB"]

    # Warmup
    for p in prefixes:
        predictor.get_suggestions(p)

    # Benchmark
    times = []
    for _ in range(200):
        for p in prefixes:
            start = time.perf_counter()
            predictor.get_suggestions(f"I AM {p}")
            times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    print(f"\nWord prediction benchmark:")
    print(f"  Average: {avg_ms:.3f} ms")
    print(f"  Target:  < 10 ms")

    assert avg_ms < 50, f"Word prediction too slow: {avg_ms:.3f}ms average"


if __name__ == "__main__":
    benchmark_model_inference()
    benchmark_disambiguation()
    benchmark_word_prediction()
    print("\n[PASS] All benchmarks passed!")
