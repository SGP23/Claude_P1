"""
Unit tests for the geometric disambiguation module.
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from backend.models.disambiguation import GeometricDisambiguator, GROUPS, LETTER_TO_GROUP


@pytest.fixture
def disambiguator():
    return GeometricDisambiguator()


class TestGroupDefinitions:
    """Test group and letter mapping definitions."""

    def test_groups_cover_all_letters(self):
        """All 24 class letters (A-Y minus J,Z) should have a group."""
        expected = set("ABCDEFGHIKLMNOPQRSTUVWXY")
        mapped = set(LETTER_TO_GROUP.keys())
        assert mapped == expected, f"Missing: {expected - mapped}, Extra: {mapped - expected}"

    def test_reverse_mapping_consistent(self):
        """LETTER_TO_GROUP should be consistent with GROUPS."""
        for gid, letters in GROUPS.items():
            for letter in letters:
                assert LETTER_TO_GROUP[letter] == gid


class TestDistanceFunctions:
    """Test distance calculation functions."""

    def test_distance_zero(self, disambiguator):
        assert disambiguator.distance([0, 0], [0, 0]) == 0.0

    def test_distance_unit(self, disambiguator):
        assert abs(disambiguator.distance([0, 0], [3, 4]) - 5.0) < 1e-6

    def test_distance_3d(self, disambiguator):
        dist = disambiguator.distance_3d([0, 0, 0], [1, 2, 2])
        assert abs(dist - 3.0) < 1e-6


class TestFingerStates:
    """Test finger state detection."""

    def test_all_fingers_up(self, disambiguator):
        """Create landmarks where all fingers are extended (tip above PIP)."""
        pts = np.zeros((21, 3))
        # Set PIP joints at y=0.5, tips at y=0.2 (above in image coords)
        # Index: PIP=6, TIP=8
        pts[6] = [0.3, 0.5, 0]
        pts[8] = [0.3, 0.2, 0]
        # Middle: PIP=10, TIP=12
        pts[10] = [0.4, 0.5, 0]
        pts[12] = [0.4, 0.2, 0]
        # Ring: PIP=14, TIP=16
        pts[14] = [0.5, 0.5, 0]
        pts[16] = [0.5, 0.2, 0]
        # Pinky: PIP=18, TIP=20
        pts[18] = [0.6, 0.5, 0]
        pts[20] = [0.6, 0.2, 0]
        # Thumb: IP=3, TIP=4
        pts[3] = [0.2, 0.4, 0]
        pts[4] = [0.1, 0.3, 0]

        states = disambiguator.get_finger_states(pts)
        assert states["index"] == True
        assert states["middle"] == True
        assert states["ring"] == True
        assert states["pinky"] == True

    def test_all_fingers_curled(self, disambiguator):
        """Create landmarks where all fingers are curled (tip below PIP)."""
        pts = np.zeros((21, 3))
        # PIP joints at y=0.3, tips at y=0.6 (below in image coords)
        pts[6] = [0.3, 0.3, 0]
        pts[8] = [0.3, 0.6, 0]
        pts[10] = [0.4, 0.3, 0]
        pts[12] = [0.4, 0.6, 0]
        pts[14] = [0.5, 0.3, 0]
        pts[16] = [0.5, 0.6, 0]
        pts[18] = [0.6, 0.3, 0]
        pts[20] = [0.6, 0.6, 0]
        pts[3] = [0.2, 0.3, 0]
        pts[4] = [0.3, 0.4, 0]

        states = disambiguator.get_finger_states(pts)
        assert states["index"] == False
        assert states["middle"] == False
        assert states["ring"] == False
        assert states["pinky"] == False


class TestDisambiguation:
    """Test disambiguation logic for specific letter groups."""

    def test_high_confidence_skipped(self, disambiguator):
        """High-confidence predictions should not be modified."""
        pts = np.zeros((21, 3))
        result, corrected = disambiguator.disambiguate("A", 0.95, pts)
        assert result == "A"
        assert corrected is False

    def test_none_landmarks(self, disambiguator):
        """None landmarks should return original prediction."""
        result, corrected = disambiguator.disambiguate("B", 0.7, None)
        assert result == "B"
        assert corrected is False

    def test_unknown_letter(self, disambiguator):
        """Unknown letters should return original prediction."""
        pts = np.zeros((21, 3))
        result, corrected = disambiguator.disambiguate("Z", 0.5, pts)
        assert result == "Z"
        assert corrected is False

    def test_single_letter_groups_unchanged(self, disambiguator):
        """Single-letter groups (L, X, Y) should not be changed."""
        pts = np.zeros((21, 3))
        for letter in ["L", "X", "Y"]:
            result, corrected = disambiguator.disambiguate(letter, 0.5, pts)
            assert result == letter
            assert corrected is False

    def test_c_vs_o_distant(self, disambiguator):
        """C should be returned when thumb and index are far apart."""
        pts = np.zeros((21, 3))
        pts[4] = [0.2, 0.3, 0]  # thumb tip
        pts[8] = [0.5, 0.3, 0]  # index tip - far from thumb
        result, _ = disambiguator.disambiguate("C", 0.6, pts)
        assert result == "C"

    def test_c_vs_o_close(self, disambiguator):
        """O should be returned when thumb and index are very close."""
        pts = np.zeros((21, 3))
        pts[4] = [0.3, 0.3, 0]  # thumb tip
        pts[8] = [0.32, 0.31, 0]  # index tip - very close to thumb
        result, _ = disambiguator.disambiguate("C", 0.6, pts)
        assert result == "O"

    def test_g_vs_h_one_finger(self, disambiguator):
        """G when only index extended."""
        pts = np.zeros((21, 3))
        # Index up (tip above PIP)
        pts[6] = [0.3, 0.5, 0]
        pts[8] = [0.3, 0.2, 0]
        # Middle down (tip below PIP)
        pts[10] = [0.4, 0.3, 0]
        pts[12] = [0.4, 0.6, 0]

        result, _ = disambiguator.disambiguate("G", 0.6, pts)
        assert result == "G"

    def test_g_vs_h_two_fingers(self, disambiguator):
        """H when index and middle both extended."""
        pts = np.zeros((21, 3))
        # Index up
        pts[6] = [0.3, 0.5, 0]
        pts[8] = [0.3, 0.2, 0]
        # Middle up
        pts[10] = [0.4, 0.5, 0]
        pts[12] = [0.4, 0.2, 0]

        result, _ = disambiguator.disambiguate("G", 0.6, pts)
        assert result == "H"


class TestRefineFromDicts:
    """Test the convenience method that takes list-of-dicts landmarks."""

    def test_refine_prediction_none(self, disambiguator):
        result, corrected = disambiguator.refine_prediction("A", 0.9, None)
        assert result == "A"
        assert corrected is False

    def test_refine_prediction_with_dicts(self, disambiguator):
        landmarks = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
        result, _ = disambiguator.refine_prediction("L", 0.5, landmarks)
        assert result == "L"  # L is single-letter group, no change
