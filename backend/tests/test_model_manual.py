"""
Manual Model Test
─────────────────
Tests model loading and inference with synthetic input.
Adapted for the 24-class landmark MLP architecture.

Run:
    cd backend && python tests/test_model_manual.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np


def test_model_loading():
    print("=" * 50)
    print("MODEL LOADING & INFERENCE TEST")
    print("=" * 50)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── Test 1: Landmark Model (primary) ──
    landmark_path = os.path.join(project_root, "landmark_classifier.pt")
    labels_path = os.path.join(project_root, "landmark_class_labels.txt")

    if os.path.exists(landmark_path):
        print("\n1. Loading Landmark MLP model...")
        from models.landmark_model import LandmarkClassifier

        checkpoint = torch.load(landmark_path, map_location="cpu", weights_only=False)
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
        print(f"   [OK] Loaded: {num_classes} classes")

        # Load class labels
        class_names = []
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
        if not class_names:
            class_names = [chr(65 + i) for i in range(num_classes)]
        print(f"   [OK] Labels: {class_names}")

        # Synthetic 63-feature input (21 landmarks x 3 coords)
        test_input = torch.randn(1, 63)
        print(f"   Input shape: {test_input.shape}")

        with torch.no_grad():
            logits = model(test_input)
            probs = F.softmax(logits, dim=1)

        print(f"   Output shape: {logits.shape}")
        assert logits.shape == (1, num_classes), f"Expected (1, {num_classes}), got {logits.shape}"

        prob_sum = probs.sum().item()
        print(f"   Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 0.01, f"Probs don't sum to 1: {prob_sum}"

        top_conf, top_idx = probs[0].max(dim=0)
        print(f"   Top prediction: {class_names[top_idx]} ({top_conf:.4f})")

        print("\n   Per-class probabilities:")
        for i, (name, p) in enumerate(zip(class_names, probs[0].numpy())):
            bar = "#" * int(p * 50)
            print(f"     {name}: {p:.4f} {bar}")

        print("\n   [PASS] Landmark model test passed!")
    else:
        print(f"\n   [SKIP] Landmark model not found: {landmark_path}")

    # ── Test 2: CNN Model (fallback) ──
    cnn_path = os.path.join(project_root, "sign_language_cnn_trained.pt")

    if os.path.exists(cnn_path):
        print("\n2. Loading CNN model...")
        from models.cnn_model import SignLanguageCNN

        checkpoint = torch.load(cnn_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 24)
        else:
            state_dict = checkpoint
            final_keys = [k for k in state_dict if k.endswith(".weight")]
            num_classes = state_dict[final_keys[-1]].shape[0]

        model = SignLanguageCNN(num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"   [OK] Loaded: {num_classes} classes")

        # Synthetic 3x224x224 image input
        test_input = torch.randn(1, 3, 224, 224)
        print(f"   Input shape: {test_input.shape}")

        with torch.no_grad():
            logits = model(test_input)
            probs = F.softmax(logits, dim=1)

        print(f"   Output shape: {logits.shape}")
        assert logits.shape == (1, num_classes), f"Expected (1, {num_classes}), got {logits.shape}"

        prob_sum = probs.sum().item()
        print(f"   Probability sum: {prob_sum:.6f}")
        assert abs(prob_sum - 1.0) < 0.01, f"Probs don't sum to 1: {prob_sum}"

        top_conf, top_idx = probs[0].max(dim=0)
        print(f"   Top prediction: index {top_idx} ({top_conf:.4f})")
        print("   [PASS] CNN model test passed!")
    else:
        print(f"\n   [SKIP] CNN model not found: {cnn_path}")

    print("\n" + "=" * 50)
    print("[PASS] All model tests passed!")


if __name__ == "__main__":
    test_model_loading()
