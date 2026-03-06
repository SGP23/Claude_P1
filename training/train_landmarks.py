"""
Landmark-based Training Pipeline
─────────────────────────────────
Train the LandmarkClassifier using hand landmarks extracted from dataset images.

Usage:
    python -m training.train_landmarks --extract   # First: extract landmarks from images
    python -m training.train_landmarks --train     # Then: train the model
    python -m training.train_landmarks --both      # Do both in sequence
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.landmark_model import LandmarkClassifier, normalize_landmarks


# ─── Dataset Paths ───────────────────────────────────────
DEFAULT_DATASET_PATHS = [
    r"D:\dataset5\AtoZ_3.1",
    r"D:\dataset5\B",
]
EXCLUDE_CLASSES = {"J", "Z"}  # Dynamic gestures, not suitable for static landmarks

# ─── Output Paths ────────────────────────────────────────
LANDMARKS_FILE = PROJECT_ROOT / "training" / "landmarks_dataset.npz"
MODEL_OUTPUT = PROJECT_ROOT / "landmark_classifier.pt"
LABELS_OUTPUT = PROJECT_ROOT / "landmark_class_labels.txt"
LOG_FILE = PROJECT_ROOT / "training" / "landmark_training_log.json"


def extract_landmarks_from_datasets(
    dataset_paths: list,
    exclude_classes: set,
    output_file: Path,
    min_samples_per_class: int = 50,
) -> dict:
    """
    Extract hand landmarks from all dataset images using MediaPipe.
    
    Args:
        dataset_paths: List of dataset root directories
        exclude_classes: Set of class names to skip
        output_file: Path to save the .npz file
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        Statistics dict with extraction results
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    
    # Initialize MediaPipe
    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    if not model_path.exists():
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    
    print(f"MediaPipe HandLandmarker loaded from {model_path}")
    print(f"Excluding classes: {exclude_classes}")
    print(f"Dataset paths: {dataset_paths}")
    
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
            
            # Collect image files
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    class_to_images[class_name].append(str(img_path))
    
    print(f"\nFound {len(class_to_images)} classes")
    for cls, imgs in sorted(class_to_images.items()):
        print(f"  {cls}: {len(imgs)} images")
    
    # Extract landmarks
    all_landmarks = []
    all_labels = []
    class_names = sorted(class_to_images.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    stats = {
        "total_images": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "no_hand_detected": 0,
        "per_class": {},
    }
    
    print("\nExtracting landmarks...")
    for class_name in tqdm(class_names, desc="Classes"):
        images = class_to_images[class_name]
        class_landmarks = []
        
        for img_path in tqdm(images, desc=f"  {class_name}", leave=False):
            stats["total_images"] += 1
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    stats["failed_extractions"] += 1
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                
                # Detect hand
                result = detector.detect(mp_image)
                
                if not result.hand_landmarks:
                    stats["no_hand_detected"] += 1
                    continue
                
                # Extract and normalize landmarks
                hand = result.hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
                normalized = normalize_landmarks(landmarks)
                
                class_landmarks.append(normalized)
                stats["successful_extractions"] += 1
                
            except Exception as e:
                stats["failed_extractions"] += 1
                continue
        
        # Add class samples
        for lm in class_landmarks:
            all_landmarks.append(lm)
            all_labels.append(class_to_idx[class_name])
        
        stats["per_class"][class_name] = len(class_landmarks)
    
    # Convert to numpy arrays
    X = np.array(all_landmarks, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\nExtraction complete:")
    print(f"  Total images scanned: {stats['total_images']}")
    print(f"  Successful extractions: {stats['successful_extractions']}")
    print(f"  No hand detected: {stats['no_hand_detected']}")
    print(f"  Failed/corrupted: {stats['failed_extractions']}")
    print(f"  Final dataset shape: X={X.shape}, y={y.shape}")
    
    # Save to npz file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        class_names=np.array(class_names),
    )
    print(f"\nSaved landmarks to: {output_file}")
    
    # Save class labels
    with open(LABELS_OUTPUT, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class labels to: {LABELS_OUTPUT}")
    
    detector.close()
    return stats


class LandmarkDataset(Dataset):
    """Dataset for landmark-based classification."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.augment:
            # Add small noise for augmentation
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        
        return x, self.y[idx]


def train_landmark_model(
    landmarks_file: Path,
    output_model: Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_split: float = 0.15,
    patience: int = 10,
    device: str = None,
) -> dict:
    """
    Train the LandmarkClassifier model.
    
    Args:
        landmarks_file: Path to the .npz file with extracted landmarks
        output_model: Path to save the trained model
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        val_split: Validation split ratio
        patience: Early stopping patience
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Training history dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on device: {device}")
    
    # Load dataset
    print(f"Loading landmarks from: {landmarks_file}")
    data = np.load(landmarks_file)
    X = data["X"]
    y = data["y"]
    class_names = list(data["class_names"])
    
    print(f"Dataset: {len(X)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Create datasets
    full_dataset = LandmarkDataset(X, y, augment=False)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Enable augmentation for training
    train_dataset.dataset.augment = True
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    num_classes = len(class_names)
    model = LandmarkClassifier(num_classes).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print("Training Landmark Classifier")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(y_batch).sum().item()
            train_total += y_batch.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model
            torch.save(model.state_dict(), output_model)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f} (epoch {history['best_epoch']})")
    print(f"Model saved to: {output_model}")
    print("="*60)
    
    # Save training log
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": total_time,
        "epochs_completed": epoch + 1,
        "best_epoch": history["best_epoch"],
        "best_val_accuracy": history["best_val_acc"],
        "final_train_accuracy": history["train_acc"][-1],
        "num_classes": num_classes,
        "class_names": class_names,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "model_params": num_params,
        "device": str(device),
    }
    
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Landmark-based model training")
    parser.add_argument("--extract", action="store_true", help="Extract landmarks from dataset images")
    parser.add_argument("--train", action="store_true", help="Train the landmark classifier")
    parser.add_argument("--both", action="store_true", help="Extract landmarks and then train")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    if args.both or args.extract:
        print("\n" + "="*60)
        print("STEP 1: Extracting landmarks from dataset images")
        print("="*60 + "\n")
        
        extract_landmarks_from_datasets(
            dataset_paths=DEFAULT_DATASET_PATHS,
            exclude_classes=EXCLUDE_CLASSES,
            output_file=LANDMARKS_FILE,
        )
    
    if args.both or args.train:
        print("\n" + "="*60)
        print("STEP 2: Training landmark classifier")
        print("="*60 + "\n")
        
        if not LANDMARKS_FILE.exists():
            print(f"Error: Landmarks file not found: {LANDMARKS_FILE}")
            print("Run with --extract first to generate landmarks from dataset images")
            sys.exit(1)
        
        train_landmark_model(
            landmarks_file=LANDMARKS_FILE,
            output_model=MODEL_OUTPUT,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
        )
    
    if not (args.extract or args.train or args.both):
        parser.print_help()
        print("\nExample usage:")
        print("  python -m training.train_landmarks --both        # Extract + Train")
        print("  python -m training.train_landmarks --extract     # Extract only")
        print("  python -m training.train_landmarks --train       # Train only (requires extracted landmarks)")


if __name__ == "__main__":
    main()
