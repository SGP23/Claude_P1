"""
Training Pipeline — Sign Language CNN
──────────────────────────────────────
Trains the SignLanguageCNN model on combined local datasets with
data augmentation, class balancing, and learning rate scheduling.

Outputs:
    - sign_language_cnn_trained.pt   (model state_dict)
    - class_labels.txt               (one label per line)

Both files are saved to the project root and are directly compatible
with the existing inference backend — no changes needed in server.py.

Usage:
    cd CLAUDE_P1
    python -m training.train
    # or: python training/train.py
"""

import os
import sys
import time
import json
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ─── Project root setup ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn_model import SignLanguageCNN
from training.dataset import create_dataloaders

# ══════════════════════════════════════════════════════════
# CONFIGURATION — Edit these values as needed
# ══════════════════════════════════════════════════════════

CONFIG = {
    # ── Dataset paths ────────────────────────────────────
    "dataset_paths": [
        r"D:\dataset5\AtoZ_3.1",
        r"D:\dataset5\B",
    ],

    # ── Classes to exclude (default keeps current production behavior) ────
    # Your current shipped model + UI expects 24 static letters (no J, Z).
    # J and Z are typically motion-based in ASL.
    "exclude_classes": {"J", "Z"},

    # ── Training hyperparameters ─────────────────────────
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "num_workers": 4,

    # ── Scheduler ────────────────────────────────────────
    "scheduler_T0": 10,          # CosineAnnealingWarmRestarts T_0
    "scheduler_T_mult": 2,       # T_mult

    # ── Early stopping ───────────────────────────────────
    "patience": 3,               # Stop if no val improvement for N epochs

    # ── Class balancing ──────────────────────────────────
    "balance_classes": True,     # WeightedRandomSampler for imbalanced data

    # ── Output ───────────────────────────────────────────
    "model_save_path": str(PROJECT_ROOT / "sign_language_cnn_trained.pt"),
    "labels_save_path": str(PROJECT_ROOT / "class_labels.txt"),
    "training_log_path": str(SCRIPT_DIR / "training_log.json"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SignLanguageCNN on local datasets")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="dataset_paths",
        help="Dataset root directory (repeatable). Defaults to the two D:\\dataset5 paths.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--val-split", type=float, default=None, help="Validation split (0-1)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Class folder names to exclude (e.g. --exclude J Z). Default excludes J and Z.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable weighted sampling for class imbalance",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Output path for state_dict (.pt). Default: project root sign_language_cnn_trained.pt",
    )
    parser.add_argument(
        "--labels-out",
        default=None,
        help="Output path for class labels (.txt). Default: project root class_labels.txt",
    )
    parser.add_argument(
        "--log-out",
        default=None,
        help="Output path for training log (.json). Default: training/training_log.json",
    )
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> dict:
    cfg = dict(CONFIG)
    if args.dataset_paths:
        cfg["dataset_paths"] = args.dataset_paths
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.val_split is not None:
        cfg["val_split"] = args.val_split
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.exclude is not None:
        cfg["exclude_classes"] = {c.strip().upper() for c in args.exclude if c.strip()}
    if args.no_balance:
        cfg["balance_classes"] = False
    if args.model_out is not None:
        cfg["model_save_path"] = args.model_out
    if args.labels_out is not None:
        cfg["labels_save_path"] = args.labels_out
    if args.log_out is not None:
        cfg["training_log_path"] = args.log_out
    return cfg


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)
    print(f"    Starting batch loop (total batches: {num_batches})...")
    sys.stdout.flush()

    loader_iter = iter(loader)
    print(f"    Iterator created, getting first batch...")
    sys.stdout.flush()

    for batch_idx in range(num_batches):
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            break
        
        if batch_idx < 3 or batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}: loaded {images.shape}")
            sys.stdout.flush()

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress logging
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            batch_acc = 100.0 * correct / total
            print(
                f"    Batch [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {loss.item():.4f}  "
                f"Running Acc: {batch_acc:.1f}%"
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(cfg: dict):
    """Main training function."""
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE CNN — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── Device setup ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print()

    # ── Load datasets ────────────────────────────────────
    train_loader, val_loader, class_names, stats = create_dataloaders(
        dataset_paths=cfg["dataset_paths"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        num_workers=cfg["num_workers"],
        exclude_classes=cfg["exclude_classes"],
        balance_classes=cfg["balance_classes"],
    )

    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {', '.join(class_names)}")

    # ── Create model ─────────────────────────────────────
    model = SignLanguageCNN(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SignLanguageCNN")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Loss, optimizer, scheduler ───────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["scheduler_T0"],
        T_mult=CONFIG["scheduler_T_mult"],
    )

    print(f"\nOptimizer: AdamW (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T0={CONFIG['scheduler_T0']})")
    print(f"Loss: CrossEntropyLoss (label_smoothing=0.1)")
    print(f"Epochs: {CONFIG['epochs']}  |  Patience: {CONFIG['patience']}")

    # ── Training loop ────────────────────────────────────
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = []

    print("\n" + "─" * 60)
    print("TRAINING STARTED")
    print("─" * 60)

    total_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch}/{cfg['epochs']}]  LR: {current_lr:.6f}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": round(current_lr, 8),
            "time_seconds": round(epoch_time, 1),
        })

        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ★ New best val accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")

        # Early stopping
        if patience_counter >= cfg["patience"]:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - total_start

    print("\n" + "─" * 60)
    print("TRAINING COMPLETE")
    print("─" * 60)
    print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")

    # ── Save model ───────────────────────────────────────
    if best_model_state is not None:
        # Back up existing model if present
        model_path = cfg["model_save_path"]
        backup_path = model_path.replace(".pt", "_backup.pt")
        if os.path.exists(model_path):
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"\n  Backed up previous model to: {backup_path}")

        torch.save(best_model_state, model_path)
        print(f"  Model saved to: {model_path}")

        # Save class labels
        labels_path = cfg["labels_save_path"]
        labels_backup = labels_path.replace(".txt", "_backup.txt")
        if os.path.exists(labels_path):
            import shutil
            shutil.copy2(labels_path, labels_backup)

        with open(labels_path, "w", encoding="utf-8") as f:
            for name in class_names:
                f.write(name + "\n")
        print(f"  Labels saved to: {labels_path}")
    else:
        print("\n  [WARN] No model was saved (training may have failed).")

    # ── Save training log ────────────────────────────────
    log_data = {
        "config": {k: v if not isinstance(v, set) else list(v) for k, v in cfg.items()},
        "stats": stats,
        "device": str(device),
        "total_time_seconds": round(total_time, 1),
        "best_val_accuracy": round(best_val_acc, 2),
        "history": history,
    }
    with open(cfg["training_log_path"], "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"  Training log saved to: {cfg['training_log_path']}")

    # ── Verify saved model loads correctly ───────────────
    print("\n" + "─" * 60)
    print("VERIFICATION")
    print("─" * 60)
    try:
        state = torch.load(cfg["model_save_path"], map_location="cpu", weights_only=True)
        verify_model = SignLanguageCNN(num_classes)
        verify_model.load_state_dict(state)
        verify_model.eval()

        # Quick forward pass test
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = verify_model(dummy_input)
        assert out.shape == (1, num_classes), f"Unexpected output shape: {out.shape}"

        print(f"  ✓ Model loads correctly")
        print(f"  ✓ Forward pass produces shape {out.shape}")
        print(f"  ✓ Compatible with inference pipeline")
    except Exception as e:
        print(f"  ✗ Verification FAILED: {e}")

    print("\n" + "=" * 60)
    print("  DONE — Restart the backend to use the new model.")
    print("=" * 60 + "\n")

    return best_val_acc


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Required for Windows CUDA multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    
    args = _parse_args()
    cfg = _apply_overrides(args)
    train(cfg)
