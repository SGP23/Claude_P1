"""
Model Validation Report — Sign Language Recognition
────────────────────────────────────────────────────
Evaluates trained model on validation set and generates:
- Confusion matrix
- Per-class accuracy
- Rejection analysis
- Misclassification analysis
- Confusing letter pairs

Usage:
    python -m training.evaluate_model
    python -m training.evaluate_model --model path/to/model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

# ─── Project root setup ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import SignLanguageCNN
from training.dataset import create_dataloaders

# ─── Configuration ───────────────────────────────────────
# Rejection thresholds (same as server.py)
CONFIDENCE_THRESHOLD = 0.60
ENTROPY_THRESHOLD = 1.5
TOP2_RATIO_THRESHOLD = 1.5


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate Shannon entropy."""
    probs_safe = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs_safe * np.log(probs_safe))


def calculate_prediction_quality(probs: np.ndarray) -> dict:
    """Analyze prediction quality metrics."""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    top1_conf = float(sorted_probs[0])
    top2_conf = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
    
    entropy = calculate_entropy(probs)
    top2_ratio = top1_conf / max(top2_conf, 1e-10)
    
    is_rejected = (
        top1_conf < CONFIDENCE_THRESHOLD or
        entropy > ENTROPY_THRESHOLD or
        top2_ratio < TOP2_RATIO_THRESHOLD
    )
    
    return {
        "top1_conf": top1_conf,
        "top2_conf": top2_conf,
        "top1_idx": top1_idx,
        "top2_idx": top2_idx,
        "entropy": entropy,
        "top2_ratio": top2_ratio,
        "is_rejected": is_rejected,
    }


def evaluate_model(
    model_path: str,
    labels_path: str,
    dataset_paths: list[str],
    exclude_classes: set[str] = None,
    output_path: str = None,
    device: str = None,
) -> dict:
    """
    Evaluate model on validation set and generate comprehensive report.
    """
    exclude_classes = exclude_classes or {"J", "Z"}
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print("=" * 70)
    print("  MODEL VALIDATION REPORT")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print("=" * 70)
    
    # Load class labels
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    num_classes = len(class_names)
    print(f"\n  Classes: {num_classes} ({', '.join(class_names)})")
    
    # Load model
    model = SignLanguageCNN(num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  Model loaded successfully")
    
    # Create validation dataloader
    print("\n  Loading validation data...")
    _, val_loader, dataset_class_names, stats = create_dataloaders(
        dataset_paths=dataset_paths,
        batch_size=32,
        val_split=0.15,
        num_workers=0,
        exclude_classes=exclude_classes,
        balance_classes=False,
    )
    
    # Ensure class order matches
    assert dataset_class_names == class_names, "Class name mismatch between model and dataset"
    
    # Initialize tracking
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_rejected = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_rejected = defaultdict(int)
    class_confidences = defaultdict(list)
    
    # Confusion pairs tracking
    confusion_pairs = defaultdict(int)
    
    print(f"\n  Evaluating on {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            for i in range(len(labels)):
                true_label = int(labels[i])
                prob_vector = probs[i].cpu().numpy()
                
                quality = calculate_prediction_quality(prob_vector)
                pred_label = quality["top1_idx"]
                confidence = quality["top1_conf"]
                is_rejected = quality["is_rejected"]
                
                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_confidences.append(confidence)
                all_rejected.append(is_rejected)
                
                class_total[true_label] += 1
                class_confidences[true_label].append(confidence)
                
                if is_rejected:
                    class_rejected[true_label] += 1
                else:
                    confusion_matrix[true_label, pred_label] += 1
                    
                    if pred_label == true_label:
                        class_correct[true_label] += 1
                    else:
                        # Track confusion pairs
                        true_name = class_names[true_label]
                        pred_name = class_names[pred_label]
                        pair = tuple(sorted([true_name, pred_name]))
                        confusion_pairs[pair] += 1
            
            if (batch_idx + 1) % 20 == 0:
                print(f"    Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    # Calculate metrics
    total_samples = len(all_labels)
    total_rejected = sum(all_rejected)
    total_accepted = total_samples - total_rejected
    
    # Overall accuracy (excluding rejected)
    correct = sum(1 for p, l, r in zip(all_predictions, all_labels, all_rejected) 
                  if not r and p == l)
    overall_accuracy = correct / max(total_accepted, 1) * 100
    
    # Per-class accuracy
    per_class_accuracy = {}
    for idx, name in enumerate(class_names):
        total = class_total[idx]
        rejected = class_rejected[idx]
        accepted = total - rejected
        correct = class_correct[idx]
        
        if accepted > 0:
            accuracy = correct / accepted * 100
        else:
            accuracy = 0.0
        
        avg_conf = np.mean(class_confidences[idx]) if class_confidences[idx] else 0.0
        
        per_class_accuracy[name] = {
            "total": int(total),
            "accepted": int(accepted),
            "rejected": int(rejected),
            "correct": int(correct),
            "accuracy": round(float(accuracy), 2),
            "rejection_rate": round(float(rejected / max(total, 1) * 100), 2),
            "avg_confidence": round(float(avg_conf), 4),
        }
    
    # Identify most confused pairs
    top_confusion_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]
    
    # Identify hardest classes (lowest accuracy)
    hardest_classes = sorted(
        per_class_accuracy.items(),
        key=lambda x: x[1]["accuracy"]
    )[:5]
    
    # Identify most rejected classes
    most_rejected = sorted(
        per_class_accuracy.items(),
        key=lambda x: -x[1]["rejection_rate"]
    )[:5]
    
    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "device": str(device),
        "thresholds": {
            "confidence": CONFIDENCE_THRESHOLD,
            "entropy": ENTROPY_THRESHOLD,
            "top2_ratio": TOP2_RATIO_THRESHOLD,
        },
        "summary": {
            "total_samples": int(total_samples),
            "total_accepted": int(total_accepted),
            "total_rejected": int(total_rejected),
            "rejection_rate": round(total_rejected / max(total_samples, 1) * 100, 2),
            "overall_accuracy": round(overall_accuracy, 2),
            "num_classes": int(num_classes),
        },
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": [[int(c) for c in row] for row in confusion_matrix.tolist()],
        "class_names": class_names,
        "top_confusion_pairs": [
            {"pair": list(pair), "count": int(count)}
            for pair, count in top_confusion_pairs
        ],
        "hardest_classes": [
            {"class": name, "accuracy": float(data["accuracy"])}
            for name, data in hardest_classes
        ],
        "most_rejected_classes": [
            {"class": name, "rejection_rate": float(data["rejection_rate"])}
            for name, data in most_rejected
        ],
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total samples:      {total_samples}")
    print(f"  Accepted:           {total_accepted} ({100 - report['summary']['rejection_rate']:.1f}%)")
    print(f"  Rejected:           {total_rejected} ({report['summary']['rejection_rate']:.1f}%)")
    print(f"  Overall accuracy:   {overall_accuracy:.2f}%")
    
    print("\n  Per-class accuracy:")
    for name in class_names:
        data = per_class_accuracy[name]
        bar = "█" * int(data["accuracy"] / 5) + "░" * (20 - int(data["accuracy"] / 5))
        print(f"    {name}: {bar} {data['accuracy']:5.1f}% ({data['correct']}/{data['accepted']})")
    
    if top_confusion_pairs:
        print("\n  Most confused pairs:")
        for pair, count in top_confusion_pairs[:10]:
            print(f"    {pair[0]} ↔ {pair[1]}: {count} errors")
    
    if hardest_classes:
        print("\n  Hardest classes (lowest accuracy):")
        for name, data in hardest_classes:
            print(f"    {name}: {data['accuracy']:.1f}%")
    
    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {output_path}")
    
    # Save confusion matrix visualization
    try:
        save_confusion_matrix_image(
            confusion_matrix, 
            class_names,
            output_path.parent / "confusion_matrix.png" if output_path else None
        )
    except Exception as e:
        print(f"  Could not save confusion matrix image: {e}")
    
    print("=" * 70)
    
    return report


def save_confusion_matrix_image(cm: np.ndarray, class_names: list, output_path: Path = None):
    """Save confusion matrix as an image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not installed, skipping confusion matrix image")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Confusion matrix saved to: {output_path}")
    
    plt.close()


def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Evaluate trained sign language model")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "sign_language_cnn_trained.pt"),
        help="Path to model .pt file",
    )
    parser.add_argument(
        "--labels",
        default=str(PROJECT_ROOT / "class_labels.txt"),
        help="Path to class labels file",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="dataset_paths",
        help="Dataset root directory (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["J", "Z"],
        help="Classes to exclude",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "evaluation_report.json"),
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Default dataset paths
    if not args.dataset_paths:
        args.dataset_paths = [
            r"D:\dataset5\AtoZ_3.1",
            r"D:\dataset5\B",
        ]
    
    exclude = {c.strip().upper() for c in args.exclude if c.strip()}
    
    evaluate_model(
        model_path=args.model,
        labels_path=args.labels,
        dataset_paths=args.dataset_paths,
        exclude_classes=exclude,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
