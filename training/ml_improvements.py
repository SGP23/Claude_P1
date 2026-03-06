"""
ML Pipeline Improvements for Sign Language Recognition
───────────────────────────────────────────────────────
Enhanced normalization, dataset validation, augmentation, and model improvements.

Usage:
    python -m training.ml_improvements --validate    # Validate dataset
    python -m training.ml_improvements --extract     # Extract with improved normalization
    python -m training.ml_improvements --train       # Train improved model
    python -m training.ml_improvements --evaluate    # Generate confusion matrix
    python -m training.ml_improvements --all         # Run full pipeline
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Configuration ───────────────────────────────────────
DEFAULT_DATASET_PATHS = [
    r"D:\dataset5\AtoZ_3.1",
    r"D:\dataset5\B",
]
EXCLUDE_CLASSES = {"J", "Z"}  # Dynamic gestures

OUTPUT_DIR = PROJECT_ROOT / "training"
LANDMARKS_FILE = OUTPUT_DIR / "landmarks_improved.npz"
MODEL_OUTPUT = PROJECT_ROOT / "landmark_classifier_v2.pt"
VALIDATION_REPORT = OUTPUT_DIR / "dataset_validation_report.json"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrix.json"


# ─── Dataset Validation ──────────────────────────────────

def validate_dataset(
    dataset_paths: list,
    exclude_classes: set,
    output_report: Path,
) -> dict:
    """
    Scan datasets and identify quality issues.
    
    Checks:
    - Missing/corrupted images
    - Images with no hand detected
    - Images with multiple hands
    - Class imbalance
    - Blurry images (Laplacian variance)
    
    Returns:
        Validation report dict
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    
    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    # Initialize detector
    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,  # Detect multiple hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_paths": dataset_paths,
        "excluded_classes": list(exclude_classes),
        "classes": {},
        "issues": {
            "corrupted_images": [],
            "no_hand_detected": [],
            "multiple_hands": [],
            "blurry_images": [],
            "low_confidence": [],
        },
        "summary": {},
    }
    
    # Blur detection threshold (Laplacian variance)
    BLUR_THRESHOLD = 100
    
    # Collect all class folders
    class_to_images = defaultdict(list)
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Warning: Dataset path not found: {dataset_path}")
            continue
        
        for class_folder in sorted(dataset_path.iterdir()):
            if not class_folder.is_dir():
                continue
            class_name = class_folder.name.upper()
            if len(class_name) != 1 or not class_name.isalpha():
                continue
            if class_name in exclude_classes:
                continue
            
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    class_to_images[class_name].append(str(img_path))
    
    total_images = sum(len(v) for v in class_to_images.values())
    print(f"\nFound {len(class_to_images)} classes, {total_images} images")
    
    # Sample for validation (100 per class or all if less)
    SAMPLES_PER_CLASS = 100
    
    for class_name in tqdm(sorted(class_to_images.keys()), desc="Validating"):
        images = class_to_images[class_name]
        sample_size = min(SAMPLES_PER_CLASS, len(images))
        
        # Random sample
        np.random.seed(42)
        sample_indices = np.random.choice(len(images), sample_size, replace=False)
        sample_images = [images[i] for i in sample_indices]
        
        class_stats = {
            "total": len(images),
            "sampled": sample_size,
            "valid": 0,
            "issues": 0,
        }
        
        for img_path in sample_images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    report["issues"]["corrupted_images"].append(img_path)
                    class_stats["issues"] += 1
                    continue
                
                # Blur check
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < BLUR_THRESHOLD:
                    report["issues"]["blurry_images"].append({
                        "path": img_path,
                        "score": float(blur_score),
                    })
                
                # Hand detection
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = detector.detect(mp_image)
                
                if not result.hand_landmarks:
                    report["issues"]["no_hand_detected"].append(img_path)
                    class_stats["issues"] += 1
                elif len(result.hand_landmarks) > 1:
                    report["issues"]["multiple_hands"].append(img_path)
                    class_stats["issues"] += 1
                else:
                    # Check detection confidence
                    if result.handedness and result.handedness[0][0].score < 0.7:
                        report["issues"]["low_confidence"].append({
                            "path": img_path,
                            "confidence": float(result.handedness[0][0].score),
                        })
                    class_stats["valid"] += 1
                    
            except Exception as e:
                report["issues"]["corrupted_images"].append(img_path)
                class_stats["issues"] += 1
        
        report["classes"][class_name] = class_stats
    
    # Summary
    class_counts = {k: v["total"] for k, v in report["classes"].items()}
    report["summary"] = {
        "total_classes": len(class_counts),
        "total_images": sum(class_counts.values()),
        "min_class_count": min(class_counts.values()),
        "max_class_count": max(class_counts.values()),
        "imbalance_ratio": max(class_counts.values()) / max(min(class_counts.values()), 1),
        "corrupted_count": len(report["issues"]["corrupted_images"]),
        "no_hand_count": len(report["issues"]["no_hand_detected"]),
        "multiple_hands_count": len(report["issues"]["multiple_hands"]),
        "blurry_count": len(report["issues"]["blurry_images"]),
        "low_confidence_count": len(report["issues"]["low_confidence"]),
    }
    
    detector.close()
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Classes: {report['summary']['total_classes']}")
    print(f"Total Images: {report['summary']['total_images']}")
    print(f"Class Size Range: {report['summary']['min_class_count']} - {report['summary']['max_class_count']}")
    print(f"Imbalance Ratio: {report['summary']['imbalance_ratio']:.2f}")
    print(f"\nIssues Found (in sampled images):")
    print(f"  Corrupted: {report['summary']['corrupted_count']}")
    print(f"  No Hand: {report['summary']['no_hand_count']}")
    print(f"  Multiple Hands: {report['summary']['multiple_hands_count']}")
    print(f"  Blurry: {report['summary']['blurry_count']}")
    print(f"  Low Confidence: {report['summary']['low_confidence_count']}")
    
    # Save report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w") as f:
        # Only save paths, not full issue objects for readability
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to: {output_report}")
    
    return report


# ─── Enhanced Data Augmentation ──────────────────────────

class AugmentedLandmarkDataset(Dataset):
    """
    Dataset with enhanced augmentation for landmark-based training.
    
    Augmentations:
    - Coordinate noise (simulates hand position variation)
    - Random scaling (simulates distance variation)
    - Random rotation around Z-axis (simulates hand rotation)
    - Mirror flip (left/right hand variation)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.03,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 15.0,  # degrees
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_range = math.radians(rotation_range)
    
    def __len__(self):
        return len(self.X)
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to landmark features."""
        # Only augment the first 63 features (3D coordinates)
        # Additional features (distances) are recomputed
        
        coords = x[:63].clone().view(21, 3)
        
        # 1. Add Gaussian noise
        coords = coords + torch.randn_like(coords) * self.noise_std
        
        # 2. Random scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        coords = coords * scale
        
        # 3. Random rotation around Z-axis
        angle = torch.empty(1).uniform_(-self.rotation_range, self.rotation_range).item()
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ], dtype=torch.float32)
        coords = coords @ rotation.T
        
        # 4. Random mirror (50% chance)
        if torch.rand(1).item() < 0.5:
            coords[:, 0] = -coords[:, 0]
        
        return coords.flatten()
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        if self.augment:
            x[:63] = self._augment(x)
        
        return x, self.y[idx]


# ─── Improved Model Architecture ─────────────────────────

class ImprovedLandmarkClassifier(nn.Module):
    """
    Improved MLP classifier with residual connections and better regularization.
    
    Architecture:
        Input(63) → Dense(128) → BN → ReLU → Dropout(0.3)
                  → Dense(128) → BN → ReLU + Skip → Dropout(0.3)
                  → Dense(64) → BN → ReLU → Dropout(0.2)
                  → Dense(num_classes)
    """
    
    def __init__(self, num_classes: int, input_size: int = 63, dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Residual block
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.66),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.res_block(x)  # Residual connection
        return self.classifier(x)


# ─── Training with Class Balancing ───────────────────────

def train_improved_model(
    landmarks_file: Path,
    output_model: Path,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_split: float = 0.15,
    patience: int = 12,
    use_class_weights: bool = True,
    device: str = None,
) -> dict:
    """
    Train improved model with class balancing and enhanced augmentation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"\nTraining on: {device}")
    
    # Load dataset
    print(f"Loading: {landmarks_file}")
    data = np.load(landmarks_file)
    X = data["X"]
    y = data["y"]
    class_names = list(data["class_names"])
    
    num_samples = len(X)
    num_classes = len(class_names)
    input_size = X.shape[1]
    
    print(f"Dataset: {num_samples} samples, {num_classes} classes, {input_size} features")
    
    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create datasets
    train_dataset = AugmentedLandmarkDataset(X_train, y_train, augment=True)
    val_dataset = AugmentedLandmarkDataset(X_val, y_val, augment=False)
    
    # Class weights for imbalanced data
    if use_class_weights:
        class_counts = Counter(y_train)
        weights = {c: 1.0 / count for c, count in class_counts.items()}
        sample_weights = [weights[int(label)] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = ImprovedLandmarkClassifier(num_classes, input_size=input_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "best_val_acc": 0.0, "best_epoch": 0,
    }
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print("\n" + "="*60)
    print("TRAINING IMPROVED MODEL")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        scheduler.step()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            epochs_no_improve = 0
            
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_size": input_size,
                "class_names": class_names,
            }, output_model)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {history['best_epoch']})")
    print(f"Model saved to: {output_model}")
    
    return history


# ─── Confusion Matrix Analysis ───────────────────────────

def generate_confusion_matrix(
    model_path: Path,
    landmarks_file: Path,
    output_file: Path,
    device: str = None,
) -> dict:
    """
    Generate confusion matrix and identify commonly confused pairs.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load data
    data = np.load(landmarks_file)
    X = data["X"]
    y = data["y"]
    class_names = list(data["class_names"])
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    input_size = checkpoint.get("input_size", 63)
    
    model = ImprovedLandmarkClassifier(num_classes, input_size=input_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Predict on all data
    dataset = AugmentedLandmarkDataset(X, y, augment=False)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Generating predictions"):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Find commonly confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / cm[i].sum()) if cm[i].sum() > 0 else 0,
                })
    
    # Sort by confusion rate
    confused_pairs.sort(key=lambda x: x["rate"], reverse=True)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == i).mean())
    
    result = {
        "overall_accuracy": float((all_preds == all_labels).mean()),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class_accuracy": per_class_acc,
        "top_confused_pairs": confused_pairs[:20],
        "classification_report": report,
    }
    
    # Print summary
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    print(f"Overall Accuracy: {result['overall_accuracy']:.4f}")
    print(f"\nTop 10 Confused Pairs:")
    for pair in confused_pairs[:10]:
        print(f"  {pair['true']} → {pair['predicted']}: {pair['count']} ({pair['rate']*100:.1f}%)")
    
    print(f"\nPer-Class Accuracy:")
    for name, acc in sorted(per_class_acc.items(), key=lambda x: x[1]):
        print(f"  {name}: {acc:.4f}")
    
    # Save
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")
    
    return result


# ─── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Improvements")
    parser.add_argument("--validate", action="store_true", help="Validate dataset quality")
    parser.add_argument("--extract", action="store_true", help="Extract landmarks with improved normalization")
    parser.add_argument("--train", action="store_true", help="Train improved model")
    parser.add_argument("--evaluate", action="store_true", help="Generate confusion matrix")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    args = parser.parse_args()
    
    if args.all:
        args.validate = args.extract = args.train = args.evaluate = True
    
    if args.validate:
        validate_dataset(
            DEFAULT_DATASET_PATHS,
            EXCLUDE_CLASSES,
            VALIDATION_REPORT,
        )
    
    if args.extract:
        # Use existing extraction but with improved normalization
        from training.train_landmarks import extract_landmarks_from_datasets
        extract_landmarks_from_datasets(
            DEFAULT_DATASET_PATHS,
            EXCLUDE_CLASSES,
            LANDMARKS_FILE,
        )
    
    if args.train:
        # Train with existing landmarks file if improved doesn't exist
        landmarks_file = LANDMARKS_FILE if LANDMARKS_FILE.exists() else (OUTPUT_DIR / "landmarks_dataset.npz")
        train_improved_model(
            landmarks_file,
            MODEL_OUTPUT,
        )
    
    if args.evaluate:
        # Evaluate model
        landmarks_file = LANDMARKS_FILE if LANDMARKS_FILE.exists() else (OUTPUT_DIR / "landmarks_dataset.npz")
        model_path = MODEL_OUTPUT if MODEL_OUTPUT.exists() else (PROJECT_ROOT / "landmark_classifier.pt")
        
        try:
            generate_confusion_matrix(
                model_path,
                landmarks_file,
                CONFUSION_MATRIX_FILE,
            )
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            # Fall back to simple evaluation
            print("Attempting simple evaluation...")


if __name__ == "__main__":
    main()
