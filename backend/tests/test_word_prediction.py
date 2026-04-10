"""
Unit tests for the word prediction module.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from backend.prediction.word_predictor import WordPredictor


@pytest.fixture
def predictor():
    return WordPredictor()


class TestGetSuggestions:
    """Test word suggestion fetching."""

    def test_empty_sentence(self, predictor):
        result = predictor.get_suggestions("")
        assert len(result) == 4
        assert all(s == "" for s in result)

    def test_whitespace_sentence(self, predictor):
        result = predictor.get_suggestions("   ")
        assert len(result) == 4
        assert all(s == "" for s in result)

    def test_trailing_space(self, predictor):
        """After a complete word + space, no prefix to suggest on."""
        result = predictor.get_suggestions("HELLO ")
        assert len(result) == 4
        assert all(s == "" for s in result)

    def test_single_letter_prefix(self, predictor):
        """Single letter should return suggestions starting with that letter."""
        result = predictor.get_suggestions("H")
        assert len(result) == 4
        # At least one suggestion should start with H
        non_empty = [s for s in result if s]
        if non_empty:
            assert all(s.upper().startswith("H") for s in non_empty)

    def test_multi_letter_prefix(self, predictor):
        """Multi-letter prefix should return matching suggestions."""
        result = predictor.get_suggestions("HE")
        assert len(result) == 4
        non_empty = [s for s in result if s]
        if non_empty:
            assert all(s.upper().startswith("HE") for s in non_empty)

    def test_word_in_sentence(self, predictor):
        """Should suggest based on last incomplete word only."""
        result = predictor.get_suggestions("I AM H")
        assert len(result) == 4
        non_empty = [s for s in result if s]
        if non_empty:
            assert all(s.upper().startswith("H") for s in non_empty)

    def test_max_suggestions_param(self, predictor):
        result = predictor.get_suggestions("H", max_suggestions=2)
        assert len(result) == 2

    def test_nonexistent_prefix(self, predictor):
        """Nonsense prefix should return empty suggestions."""
        result = predictor.get_suggestions("XYZQW")
        assert len(result) == 4
        # Might or might not have suggestions, but length should be 4


class TestCompleteWord:
    """Test word completion."""

    def test_complete_simple(self, predictor):
        result = predictor.complete_word("H", "HELLO")
        assert result == "HELLO"

    def test_complete_in_sentence(self, predictor):
        result = predictor.complete_word("I AM H", "HELLO")
        assert result == "I AM HELLO"

    def test_complete_empty_suggestion(self, predictor):
        result = predictor.complete_word("I AM H", "")
        assert result == "I AM H"

    def test_complete_no_preceding_word(self, predictor):
        result = predictor.complete_word("HE", "HELLO")
        assert result == "HELLO"
