"""
Dataset Validation Pipeline — Sign Language Recognition
────────────────────────────────────────────────────────
Validates training datasets for quality issues:
- Corrupted images
- Missing/incomplete hands
- Multiple hands in frame
- Blurry images
- Incorrect hand orientation
- Class distribution analysis

Generates a comprehensive quality report.

Usage:
    python -m training.validate_dataset
    python -m training.validate_dataset --dataset "D:\dataset5\AtoZ_3.1"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import multiprocessing

import cv2
import numpy as np

# ─── Project root setup ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# MediaPipe imports (lazy loaded)
mp = None
mp_hands = None
hands_detector = None


def get_mediapipe_hands():
    """Lazy load MediaPipe hands detector."""
    global mp, mp_hands, hands_detector
    if hands_detector is None:
        try:
            import mediapipe
            mp = mediapipe
            # Try solutions API first (older mediapipe)
            if hasattr(mp, 'solutions'):
                mp_hands = mp.solutions.hands
                hands_detector = mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            else:
                # Newer mediapipe uses different API
                mp_hands = None
                hands_detector = "TASK_API"  # Flag to use task API
        except ImportError:
            print("  [ERROR] MediaPipe not installed. Run: pip install mediapipe")
            return None
    return hands_detector


# ─── Image Quality Metrics ───────────────────────────────

def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher = sharper, lower = blurrier.
    Typical threshold: < 100 is blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate average brightness (0-255)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast as standard deviation of pixel values."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.std(gray))


# ─── Hand Detection and Validation ──────────────────────

class HandValidator:
    """Validates hand presence and quality using MediaPipe."""
    
    def __init__(self):
        self.hands = get_mediapipe_hands()
        self.use_task_api = (self.hands == "TASK_API")
        
        if self.use_task_api:
            # Use MediaPipe Tasks API for newer versions
            self._init_task_api()
        
        # Expected landmark structure for validation
        # Finger tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcp = [1, 5, 9, 13, 17]  # Base joints
    
    def _init_task_api(self):
        """Initialize MediaPipe Tasks API (newer mediapipe versions)."""
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
            
            # Look for hand_landmarker.task model file
            model_path = SCRIPT_DIR.parent / "models" / "hand_landmarker.task"
            if not model_path.exists():
                print(f"  [WARN] Hand landmarker model not found: {model_path}")
                self.task_detector = None
                return
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
            self.task_detector = HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"  [WARN] Could not initialize Task API: {e}")
            self.task_detector = None
        
    def validate_image(self, image_path: str) -> dict:
        """
        Validate a single image for hand presence and quality.
        
        Returns dict with:
            - valid: bool
            - issues: list of issue strings
            - num_hands: int
            - hand_confidence: float
            - blur_score: float
            - brightness: float
            - landmarks: Optional[list] (21 landmarks if detected)
        """
        result = {
            "valid": True,
            "issues": [],
            "num_hands": 0,
            "hand_confidence": 0.0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "landmarks": None,
            "image_size": (0, 0),
        }
        
        # Try to load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                result["valid"] = False
                result["issues"].append("CORRUPTED_OR_UNREADABLE")
                return result
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"LOAD_ERROR: {str(e)}")
            return result
        
        h, w = image.shape[:2]
        result["image_size"] = (w, h)
        
        # Image quality metrics
        result["blur_score"] = calculate_blur_score(image)
        result["brightness"] = calculate_brightness(image)
        result["contrast"] = calculate_contrast(image)
        
        # Check for very blurry images
        if result["blur_score"] < 50:
            result["issues"].append("VERY_BLURRY")
        elif result["blur_score"] < 100:
            result["issues"].append("MODERATELY_BLURRY")
        
        # Check brightness extremes
        if result["brightness"] < 30:
            result["issues"].append("TOO_DARK")
        elif result["brightness"] > 225:
            result["issues"].append("TOO_BRIGHT")
        
        # Check contrast
        if result["contrast"] < 20:
            result["issues"].append("LOW_CONTRAST")
        
        # Check image size
        if w < 50 or h < 50:
            result["issues"].append("IMAGE_TOO_SMALL")
        
        # Hand detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            if self.use_task_api and self.task_detector:
                # Use Tasks API
                import mediapipe as mp_module
                mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=image_rgb)
                detection = self.task_detector.detect(mp_image)
                
                if not detection.hand_landmarks:
                    result["valid"] = False
                    result["issues"].append("NO_HAND_DETECTED")
                    return result
                
                num_hands = len(detection.hand_landmarks)
                result["num_hands"] = num_hands
                
                if num_hands > 1:
                    result["issues"].append("MULTIPLE_HANDS")
                
                hand_landmarks = detection.hand_landmarks[0]
                
                if detection.handedness:
                    result["hand_confidence"] = detection.handedness[0][0].score
                
                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                
            elif self.hands and self.hands != "TASK_API":
                # Use Solutions API
                detection = self.hands.process(image_rgb)
                
                if not detection.multi_hand_landmarks:
                    result["valid"] = False
                    result["issues"].append("NO_HAND_DETECTED")
                    return result
                
                num_hands = len(detection.multi_hand_landmarks)
                result["num_hands"] = num_hands
                
                if num_hands > 1:
                    result["issues"].append("MULTIPLE_HANDS")
                
                hand_landmarks = detection.multi_hand_landmarks[0]
                
                if detection.multi_handedness:
                    result["hand_confidence"] = detection.multi_handedness[0].classification[0].score
                
                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            else:
                # No detector available, skip hand detection
                result["issues"].append("DETECTOR_UNAVAILABLE")
                return result
                
        except Exception as e:
            result["issues"].append(f"MEDIAPIPE_ERROR: {str(e)}")
            return result
        
        result["landmarks"] = landmarks
        
        # Validate hand structure
        structure_issues = self._validate_hand_structure(landmarks, w, h)
        result["issues"].extend(structure_issues)
        
        # Final validity check
        critical_issues = {"NO_HAND_DETECTED", "CORRUPTED_OR_UNREADABLE", "IMAGE_TOO_SMALL"}
        if any(issue in critical_issues for issue in result["issues"]):
            result["valid"] = False
        
        return result
    
    def _validate_hand_structure(self, landmarks: list, img_w: int, img_h: int) -> list:
        """Validate that the hand has reasonable structure."""
        issues = []
        
        # Check if hand is mostly within frame
        xs = [lm["x"] for lm in landmarks]
        ys = [lm["y"] for lm in landmarks]
        
        if min(xs) < 0.02 or max(xs) > 0.98:
            issues.append("HAND_PARTIALLY_OUT_OF_FRAME_X")
        if min(ys) < 0.02 or max(ys) > 0.98:
            issues.append("HAND_PARTIALLY_OUT_OF_FRAME_Y")
        
        # Check hand size relative to image
        hand_width = (max(xs) - min(xs)) * img_w
        hand_height = (max(ys) - min(ys)) * img_h
        hand_area_ratio = (hand_width * hand_height) / (img_w * img_h)
        
        if hand_area_ratio < 0.05:
            issues.append("HAND_TOO_SMALL_IN_FRAME")
        elif hand_area_ratio > 0.95:
            issues.append("HAND_TOO_ZOOMED_IN")
        
        # Check finger tip visibility (should be distinct from palm)
        wrist = landmarks[0]
        palm_center = landmarks[9]  # Middle finger MCP
        
        # Verify basic hand orientation (wrist should be generally below or to side of fingers)
        # This is a simplified check
        finger_tips_y = np.mean([landmarks[i]["y"] for i in self.finger_tips])
        if wrist["y"] < finger_tips_y - 0.15:
            issues.append("UNUSUAL_HAND_ORIENTATION")
        
        return issues
    
    def close(self):
        """Release resources."""
        if self.use_task_api and hasattr(self, 'task_detector') and self.task_detector:
            self.task_detector.close()
        elif self.hands and self.hands != "TASK_API" and hasattr(self.hands, 'close'):
            self.hands.close()


# ─── Dataset Scanner with Validation ────────────────────

def validate_dataset(
    dataset_paths: list[str],
    exclude_classes: Optional[set[str]] = None,
    output_path: Optional[str] = None,
    max_samples_per_class: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Validate all images in the given dataset directories.
    
    Returns a comprehensive report dict.
    """
    exclude_classes = exclude_classes or set()
    
    print("=" * 70)
    print("  DATASET VALIDATION PIPELINE")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Datasets: {dataset_paths}")
    print(f"  Excluded classes: {exclude_classes or 'None'}")
    print("=" * 70)
    
    # Collect all images by class
    class_images: dict[str, list[str]] = defaultdict(list)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"\n  [WARN] Dataset path not found: {dataset_path}")
            continue
        
        print(f"\n  Scanning: {dataset_path}")
        
        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_label = class_dir.name.strip().upper()
            if class_label in exclude_classes:
                continue
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in valid_extensions and img_file.is_file():
                    class_images[class_label].append(str(img_file))
    
    # Prepare report structure
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_paths": dataset_paths,
        "excluded_classes": list(exclude_classes),
        "summary": {
            "total_classes": 0,
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "images_with_issues": 0,
        },
        "class_reports": {},
        "issue_counts": defaultdict(int),
        "recommendations": [],
    }
    
    sorted_classes = sorted(class_images.keys())
    report["summary"]["total_classes"] = len(sorted_classes)
    
    # Create validator
    validator = HandValidator()
    
    try:
        for class_label in sorted_classes:
            images = class_images[class_label]
            
            # Optionally limit samples per class for faster scanning
            if max_samples_per_class and len(images) > max_samples_per_class:
                import random
                random.seed(42)  # Reproducible
                images = random.sample(images, max_samples_per_class)
            
            class_report = {
                "total": len(images),
                "valid": 0,
                "invalid": 0,
                "with_issues": 0,
                "issue_breakdown": defaultdict(int),
                "invalid_samples": [],
                "samples_with_issues": [],
                "avg_blur_score": 0.0,
                "avg_hand_confidence": 0.0,
            }
            
            blur_scores = []
            hand_confidences = []
            
            if verbose:
                print(f"\n  Validating class '{class_label}' ({len(images)} images)...")
            
            for i, img_path in enumerate(images):
                if verbose and (i + 1) % 50 == 0:
                    print(f"    Progress: {i + 1}/{len(images)}")
                
                result = validator.validate_image(img_path)
                
                report["summary"]["total_images"] += 1
                
                blur_scores.append(result["blur_score"])
                if result["hand_confidence"] > 0:
                    hand_confidences.append(result["hand_confidence"])
                
                if result["valid"]:
                    class_report["valid"] += 1
                    report["summary"]["valid_images"] += 1
                else:
                    class_report["invalid"] += 1
                    report["summary"]["invalid_images"] += 1
                    class_report["invalid_samples"].append({
                        "path": img_path,
                        "issues": result["issues"],
                    })
                
                if result["issues"]:
                    class_report["with_issues"] += 1
                    if result["valid"]:  # Valid but with warnings
                        class_report["samples_with_issues"].append({
                            "path": img_path,
                            "issues": result["issues"],
                        })
                    
                    for issue in result["issues"]:
                        class_report["issue_breakdown"][issue] += 1
                        report["issue_counts"][issue] += 1
            
            # Compute averages
            class_report["avg_blur_score"] = float(np.mean(blur_scores)) if blur_scores else 0.0
            class_report["avg_hand_confidence"] = float(np.mean(hand_confidences)) if hand_confidences else 0.0
            
            # Convert defaultdicts to regular dicts for JSON
            class_report["issue_breakdown"] = dict(class_report["issue_breakdown"])
            
            # Limit stored samples to first 20
            class_report["invalid_samples"] = class_report["invalid_samples"][:20]
            class_report["samples_with_issues"] = class_report["samples_with_issues"][:20]
            
            report["class_reports"][class_label] = class_report
            report["summary"]["images_with_issues"] += class_report["with_issues"]
            
            if verbose:
                valid_pct = (class_report["valid"] / class_report["total"] * 100) if class_report["total"] > 0 else 0
                print(f"    {class_label}: {class_report['valid']}/{class_report['total']} valid ({valid_pct:.1f}%)")
    
    finally:
        validator.close()
    
    # Convert defaultdict to dict
    report["issue_counts"] = dict(report["issue_counts"])
    
    # Generate recommendations
    report["recommendations"] = generate_recommendations(report)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total classes:        {report['summary']['total_classes']}")
    print(f"  Total images:         {report['summary']['total_images']}")
    print(f"  Valid images:         {report['summary']['valid_images']} ({report['summary']['valid_images']/max(1,report['summary']['total_images'])*100:.1f}%)")
    print(f"  Invalid images:       {report['summary']['invalid_images']}")
    print(f"  Images with issues:   {report['summary']['images_with_issues']}")
    
    if report["issue_counts"]:
        print("\n  Issue breakdown:")
        for issue, count in sorted(report["issue_counts"].items(), key=lambda x: -x[1]):
            print(f"    {issue}: {count}")
    
    if report["recommendations"]:
        print("\n  Recommendations:")
        for rec in report["recommendations"]:
            print(f"    • {rec}")
    
    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {output_path}")
    
    print("=" * 70)
    
    return report


def generate_recommendations(report: dict) -> list[str]:
    """Generate actionable recommendations based on validation results."""
    recs = []
    
    total = report["summary"]["total_images"]
    invalid = report["summary"]["invalid_images"]
    
    if total == 0:
        recs.append("No images found in dataset. Check dataset paths.")
        return recs
    
    invalid_pct = invalid / total * 100
    
    if invalid_pct > 20:
        recs.append(f"High invalid rate ({invalid_pct:.1f}%). Review and clean dataset before training.")
    elif invalid_pct > 10:
        recs.append(f"Moderate invalid rate ({invalid_pct:.1f}%). Consider cleaning problematic samples.")
    
    issue_counts = report["issue_counts"]
    
    if issue_counts.get("NO_HAND_DETECTED", 0) > total * 0.05:
        recs.append("Many images have no detectable hand. Check image quality and hand visibility.")
    
    if issue_counts.get("VERY_BLURRY", 0) > total * 0.05:
        recs.append("Many blurry images detected. Improve capture quality or remove blurry samples.")
    
    if issue_counts.get("MULTIPLE_HANDS", 0) > total * 0.1:
        recs.append("Many images contain multiple hands. Ensure single-hand dataset for better training.")
    
    if issue_counts.get("HAND_TOO_SMALL_IN_FRAME", 0) > total * 0.1:
        recs.append("Many images have hands that are too small. Crop closer to hand region.")
    
    # Check class balance
    class_sizes = [cr["total"] for cr in report["class_reports"].values()]
    if class_sizes:
        max_size = max(class_sizes)
        min_size = min(class_sizes)
        if max_size > min_size * 3:
            recs.append(f"Significant class imbalance detected (max={max_size}, min={min_size}). Use balanced sampling.")
    
    # Check per-class validity
    for class_label, cr in report["class_reports"].items():
        if cr["total"] > 0:
            valid_pct = cr["valid"] / cr["total"] * 100
            if valid_pct < 70:
                recs.append(f"Class '{class_label}' has low validity ({valid_pct:.1f}%). Review samples.")
    
    return recs


# ─── CLI Entry Point ─────────────────────────────────────

def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Validate sign language dataset quality")
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
        help="Classes to exclude (default: J Z)",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "validation_report.json"),
        help="Output path for validation report JSON",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max samples to validate per class (for faster scanning)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Default dataset paths
    if not args.dataset_paths:
        args.dataset_paths = [
            r"D:\dataset5\AtoZ_3.1",
            r"D:\dataset5\B",
        ]
    
    exclude = {c.strip().upper() for c in args.exclude if c.strip()}
    
    validate_dataset(
        dataset_paths=args.dataset_paths,
        exclude_classes=exclude,
        output_path=args.output,
        max_samples_per_class=args.max_per_class,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
