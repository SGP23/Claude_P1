"""
Landmark-based Hand Gesture Classifier
───────────────────────────────────────
Uses 21 hand landmarks (63 features) from MediaPipe for ASL letter classification.

This approach is more robust than image-based CNN because:
- Invariant to lighting conditions
- Invariant to skin color
- Invariant to background
- Lower computational cost
- More interpretable features
"""

import torch
import torch.nn as nn


class LandmarkClassifier(nn.Module):
    """
    MLP classifier for hand landmarks.
    
    Input: 63 features (21 landmarks × 3 coordinates: x, y, z)
    Output: num_classes (e.g., 24 for A-Y excluding J, Z)
    
    Architecture:
        Input(63) → Dense(256) → BN → ReLU → Dropout(0.3)
                  → Dense(128) → BN → ReLU → Dropout(0.3)
                  → Dense(64) → BN → ReLU → Dropout(0.2)
                  → Dense(num_classes)
    """
    
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3
    INPUT_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
    
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            # Layer 1: 63 → 256
            nn.Linear(self.INPUT_SIZE, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.66),
            
            # Output layer
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, 63) containing normalized landmarks
            
        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        return self.classifier(x)


def normalize_landmarks(landmarks: list) -> list:
    """
    Normalize landmarks to be translation and scale invariant.
    
    Normalization:
    1. Translate so wrist (landmark 0) is at origin
    2. Scale so max distance from wrist is 1.0
    
    Args:
        landmarks: List of 21 (x, y, z) tuples
        
    Returns:
        Flattened list of 63 normalized values
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    
    # Convert to list of lists for easier manipulation
    coords = [[lm[0], lm[1], lm[2]] for lm in landmarks]
    
    # Get wrist position (landmark 0)
    wrist = coords[0]
    
    # Translate all landmarks relative to wrist
    translated = [[c[i] - wrist[i] for i in range(3)] for c in coords]
    
    # Find max distance from wrist for scaling
    max_dist = 0.0
    for c in translated:
        dist = (c[0]**2 + c[1]**2 + c[2]**2) ** 0.5
        if dist > max_dist:
            max_dist = dist
    
    # Scale to unit sphere (avoid division by zero)
    if max_dist > 1e-6:
        normalized = [[c[i] / max_dist for i in range(3)] for c in translated]
    else:
        normalized = translated
    
    # Flatten to 63 values
    return [coord for lm in normalized for coord in lm]


def extract_landmark_features(hand_landmarks) -> list:
    """
    Extract normalized feature vector from MediaPipe hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe HandLandmarks object
        
    Returns:
        List of 63 normalized features
    """
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    return normalize_landmarks(landmarks)
