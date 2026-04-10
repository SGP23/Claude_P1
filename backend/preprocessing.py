"""Compatibility shim for legacy imports.

Use backend.tracking.hand_skeleton_preprocessor instead.
"""

from backend.tracking.hand_skeleton_preprocessor import (
    SkeletonPreprocessor,
    HAND_CONNECTIONS,
    CANVAS_SIZE,
)

__all__ = ["SkeletonPreprocessor", "HAND_CONNECTIONS", "CANVAS_SIZE"]
