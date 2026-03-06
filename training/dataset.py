"""
Dataset Loading & Augmentation
──────────────────────────────
Scans multiple dataset directories, merges images by class label,
applies augmentation, and produces PyTorch DataLoaders ready for training.

Compatible with the existing SignLanguageCNN pipeline.
"""

import os
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np


# ─── Constants (match inference pipeline) ────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ─── Custom Augmentation Transforms ─────────────────────
class AddGaussianNoise:
    """Add small Gaussian noise to the tensor."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomGaussianBlur:
    """Randomly apply Gaussian blur to a PIL image."""

    def __init__(self, p: float = 0.1, radius: float = 1.0):
        self.p = p
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img


# ─── Dataset Scanner ─────────────────────────────────────
def scan_dataset_dirs(
    dataset_paths: list[str],
    exclude_classes: Optional[set[str]] = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Scan one or more dataset directories and collect (image_path, class_label) pairs.

    Each dataset directory should have the structure:
        dataset_root/
            ClassA/
                img1.jpg
                img2.jpg
            ClassB/
                ...

    Folder names are normalized to UPPERCASE.
    Returns (samples, sorted_class_names).
    """
    exclude_classes = exclude_classes or set()
    class_samples: dict[str, list[str]] = {}

    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"  [WARN] Dataset path not found, skipping: {dataset_path}")
            continue

        print(f"  Scanning: {dataset_path}")
        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            # Normalize folder name to uppercase (e.g., 'a' -> 'A')
            class_label = class_dir.name.strip().upper()

            if class_label in exclude_classes:
                continue

            if class_label not in class_samples:
                class_samples[class_label] = []

            # Collect valid images
            count = 0
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in VALID_EXTENSIONS and img_file.is_file():
                    class_samples[class_label].append(str(img_file))
                    count += 1

            if count > 0:
                print(f"    {class_label}: +{count} images (from {class_dir.name}/)")

    # Build sorted class list and flat sample list
    sorted_classes = sorted(class_samples.keys())
    samples = []
    for cls_name in sorted_classes:
        for img_path in class_samples[cls_name]:
            samples.append((img_path, cls_name))

    # Print summary
    print(f"\n  Total classes: {len(sorted_classes)}")
    print(f"  Total images:  {len(samples)}")
    for cls_name in sorted_classes:
        print(f"    {cls_name}: {len(class_samples[cls_name])} images")

    return samples, sorted_classes


# ─── PyTorch Dataset ─────────────────────────────────────
class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset that loads images on-the-fly with transforms.
    Handles corrupted files gracefully by returning a random valid sample.
    """

    def __init__(
        self,
        samples: list[tuple[str, str]],
        class_names: list[str],
        transform: Optional[T.Compose] = None,
    ):
        self.samples = samples
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.num_classes = len(class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_name = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.class_to_idx[class_name]
            return img, label
        except Exception:
            # If image is corrupted, return a random other sample
            alt_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(alt_idx)


# ─── Transform Factories ────────────────────────────────
def get_train_transform() -> T.Compose:
    """
    Training transform with augmentation.
    Augmentations: rotation, color jitter, perspective, blur, noise, flips.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # Slightly larger for random crop
        T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),    # Horizontal/vertical shift
            scale=(0.85, 1.15),      # Zoom in/out
        ),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        RandomGaussianBlur(p=0.1, radius=1.0),
        T.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.015),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> T.Compose:
    """Validation transform — resize + normalize only (matches inference)."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── DataLoader Factory ─────────────────────────────────
def create_dataloaders(
    dataset_paths: list[str],
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
    exclude_classes: Optional[set[str]] = None,
    balance_classes: bool = True,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    """
    Create training and validation DataLoaders from the given dataset paths.

    Returns:
        (train_loader, val_loader, class_names, stats_dict)
    """
    print("=" * 60)
    print("DATASET LOADING")
    print("=" * 60)

    # 1. Scan all datasets
    samples, class_names = scan_dataset_dirs(dataset_paths, exclude_classes)

    if len(samples) == 0:
        raise RuntimeError("No images found in any dataset directory!")

    # 2. Shuffle and split into train/val
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n  Train: {len(train_samples)} images")
    print(f"  Val:   {len(val_samples)} images")
    print("=" * 60)

    # 3. Create datasets with transforms
    train_dataset = SignLanguageDataset(
        train_samples, class_names, transform=get_train_transform()
    )
    val_dataset = SignLanguageDataset(
        val_samples, class_names, transform=get_val_transform()
    )

    # 4. Optional: weighted sampler to handle class imbalance
    sampler = None
    if balance_classes and len(train_samples) > 0:
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        label_list = [class_to_idx[s[1]] for s in train_samples]
        class_counts = np.bincount(label_list, minlength=len(class_names))
        # Avoid division by zero
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
        sample_weights = [class_weights[label] for label in label_list]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )

    # 5. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    stats = {
        "total_images": len(samples),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    return train_loader, val_loader, class_names, stats
