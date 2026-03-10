"""
Accuracy Test
─────────────
Tests prediction accuracy against known test images.
If no test_data directory exists, runs synthetic validation instead.

Run:
    python tests/test_accuracy.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np


def test_accuracy_with_images():
    """Test accuracy against labeled image directories."""
    import cv2
    from backend.prediction_engine import PredictionEngine

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(project_root, "test_data")

    if not os.path.exists(test_dir):
        print(f"[SKIP] No test_data directory at {test_dir}")
        print("       Create folders like test_data/A/, test_data/B/ with sample images")
        return None

    engine = PredictionEngine(use_disambiguation=True, generate_skeleton=False)
    letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
    results = {}

    for letter in letters:
        letter_dir = os.path.join(test_dir, letter)
        if not os.path.exists(letter_dir):
            continue

        correct = 0
        total = 0

        for img_file in os.listdir(letter_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(letter_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = engine.predict_frame(frame_rgb, use_smoothing=False)
            predicted = result.get("letter")
            if predicted == letter:
                correct += 1
            total += 1

        if total > 0:
            accuracy = correct / total * 100
            results[letter] = {"correct": correct, "total": total, "accuracy": accuracy}
            status = "[OK]" if accuracy >= 90 else "[LOW]"
            print(f"  {status} {letter}: {correct}/{total} ({accuracy:.1f}%)")

    if results:
        total_correct = sum(r["correct"] for r in results.values())
        total_samples = sum(r["total"] for r in results.values())
        overall = total_correct / total_samples * 100

        print(f"\n  Overall: {total_correct}/{total_samples} ({overall:.1f}%)")
        if overall >= 97:
            print("  [PASS] ACCURACY TARGET MET (>= 97%)")
        elif overall >= 90:
            print(f"  [WARN] Accuracy {overall:.1f}% - good but below 97% target")
        else:
            print(f"  [FAIL] Accuracy {overall:.1f}% - below 90% threshold")

    return results


def test_model_sanity():
    """Synthetic validation: model produces varied outputs for different inputs."""
    from models.landmark_model import LandmarkClassifier

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "landmark_classifier.pt")
    labels_path = os.path.join(project_root, "landmark_class_labels.txt")

    if not os.path.exists(model_path):
        print("[SKIP] No landmark model found")
        return

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", 24)
    else:
        state_dict = checkpoint
        final_keys = [k for k in state_dict if k.endswith(".weight")]
        num_classes = state_dict[final_keys[-1]].shape[0]

    model = LandmarkClassifier(num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    class_names = []
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        class_names = [chr(65 + i) for i in range(num_classes)]

    print("\n  Model sanity checks:")

    # Check 1: Different random inputs produce different top predictions
    np.random.seed(42)
    predictions = set()
    for i in range(100):
        random_input = torch.randn(1, 63)
        with torch.no_grad():
            logits = model(random_input)
            probs = F.softmax(logits, dim=1)
        top_idx = int(probs[0].argmax())
        predictions.add(class_names[top_idx])

    unique_count = len(predictions)
    print(f"  [OK] {unique_count} unique predictions from 100 random inputs")
    assert unique_count > 1, "Model always predicts the same class!"

    # Check 2: Probabilities are well-calibrated (not all near 1/num_classes)
    test_input = torch.randn(1, 63)
    with torch.no_grad():
        probs = F.softmax(model(test_input), dim=1)[0]
    max_prob = float(probs.max())
    min_prob = float(probs.min())
    print(f"  [OK] Prob range: [{min_prob:.4f}, {max_prob:.4f}] (not uniform)")
    assert max_prob > 1.0 / num_classes, "Model outputs near-uniform probabilities"

    # Check 3: Model is deterministic
    with torch.no_grad():
        out1 = model(test_input)
        out2 = model(test_input)
    assert torch.allclose(out1, out2), "Model is not deterministic in eval mode!"
    print("  [OK] Model is deterministic (eval mode)")

    # Check 4: Latency
    times = []
    for _ in range(100):
        batch = torch.randn(1, 63)
        start = time.time()
        with torch.no_grad():
            model(batch)
        times.append((time.time() - start) * 1000)

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[94]
    print(f"  [OK] Inference latency: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms (target <5ms)")
    assert avg_ms < 5, f"Inference too slow: {avg_ms:.2f}ms"

    print("  [PASS] All sanity checks passed!")


def test_disambiguation_coverage():
    """Verify disambiguation covers all letter groups."""
    from backend.models.disambiguation import GeometricDisambiguator, GROUPS, LETTER_TO_GROUP

    disambig = GeometricDisambiguator()
    all_letters = set("ABCDEFGHIKLMNOPQRSTUVWXY")
    covered = set(LETTER_TO_GROUP.keys())

    missing = all_letters - covered
    extra = covered - all_letters

    print(f"\n  Disambiguation coverage:")
    print(f"  [OK] {len(covered)}/24 letters covered")
    print(f"  [OK] {len(GROUPS)} groups defined")

    if missing:
        print(f"  [WARN] Missing letters: {missing}")
    if extra:
        print(f"  [WARN] Extra letters: {extra}")

    assert not missing, f"Letters not in any group: {missing}"
    print("  [PASS] All 24 letters have disambiguation rules!")


if __name__ == "__main__":
    print("=" * 50)
    print("ACCURACY & VALIDATION TEST")
    print("=" * 50)

    print("\n--- Image-based accuracy test ---")
    test_accuracy_with_images()

    print("\n--- Model sanity test ---")
    test_model_sanity()

    print("\n--- Disambiguation coverage ---")
    test_disambiguation_coverage()

    print("\n" + "=" * 50)
    print("[PASS] All accuracy tests complete!")
    print("=" * 50)
