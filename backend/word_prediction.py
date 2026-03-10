"""
Word Prediction Module
──────────────────────
Provides real-time word suggestions as the user spells words
using ASL fingerspelling.

Uses a dictionary-based approach: given a partially-typed word prefix,
returns the best matching word completions.
"""

import os
import re

# Path to a simple word list bundled with the project
_WORD_LIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "wordlist.txt"
)

# Fallback common English words (used if no dictionary file is found)
_COMMON_WORDS = [
    "a", "about", "after", "again", "all", "also", "an", "and", "any", "are",
    "as", "at", "back", "be", "because", "been", "before", "big", "but", "by",
    "call", "came", "can", "come", "could", "day", "did", "do", "down", "each",
    "end", "even", "every", "few", "find", "first", "for", "from", "get", "give",
    "go", "going", "good", "got", "great", "had", "has", "have", "he", "help",
    "her", "here", "him", "his", "home", "house", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "know", "last", "left", "let", "life", "like",
    "line", "little", "long", "look", "made", "make", "man", "many", "may", "me",
    "might", "more", "most", "much", "must", "my", "name", "need", "never", "new",
    "next", "no", "not", "now", "number", "of", "off", "old", "on", "one", "only",
    "or", "other", "our", "out", "over", "own", "part", "people", "place", "point",
    "put", "read", "right", "run", "said", "same", "say", "see", "she", "should",
    "show", "side", "small", "so", "some", "something", "still", "story", "such",
    "take", "tell", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "thing", "think", "this", "those", "through", "time", "to",
    "too", "turn", "two", "under", "up", "us", "use", "very", "want", "was",
    "water", "way", "we", "well", "went", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "word", "work", "world", "would",
    "write", "year", "you", "your",
    # Additional common words
    "hello", "please", "thank", "thanks", "sorry", "yes", "okay", "bye",
    "food", "drink", "eat", "love", "happy", "sad", "school", "friend",
    "family", "mother", "father", "brother", "sister", "baby", "child",
    "hand", "sign", "language", "learn", "teach", "speak", "listen",
]


class WordPredictor:
    """Dictionary-based word prediction for real-time suggestions."""

    def __init__(self, word_list_path=None):
        self.words = set()
        self._load_words(word_list_path or _WORD_LIST_PATH)

    def _load_words(self, path):
        """Load word list from file, falling back to built-in list."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w and w.isalpha():
                        self.words.add(w)
        if not self.words:
            # Try enchant if available
            try:
                import enchant  # type: ignore[import-not-found]
                self._enchant_dict = enchant.Dict("en_US")
            except Exception:
                self._enchant_dict = None
            self.words = set(_COMMON_WORDS)
        else:
            self._enchant_dict = None

    def get_suggestions(self, sentence, max_suggestions=4):
        """
        Get word suggestions for the current incomplete word.

        Args:
            sentence: str like "I AM H"
            max_suggestions: maximum number of suggestions to return

        Returns:
            List of suggestion strings (padded to max_suggestions with "")
        """
        if not sentence or not sentence.strip():
            return [""] * max_suggestions

        # Extract last incomplete word
        last_space = sentence.rfind(" ")
        prefix = sentence[last_space + 1:].strip().lower()

        if not prefix:
            return [""] * max_suggestions

        suggestions = self._find_matches(prefix, max_suggestions)

        # Pad to desired length
        while len(suggestions) < max_suggestions:
            suggestions.append("")

        return suggestions[:max_suggestions]

    def _find_matches(self, prefix, limit):
        """Find words starting with prefix, sorted by length then alphabetically."""
        # Try enchant first if available
        if hasattr(self, "_enchant_dict") and self._enchant_dict:
            try:
                raw = self._enchant_dict.suggest(prefix)
                # Filter to only those starting with the prefix
                matches = [w for w in raw if w.lower().startswith(prefix.lower())]
                if matches:
                    return [w.upper() for w in matches[:limit]]
            except Exception:
                pass

        # Fall back to word list prefix matching
        matches = sorted(
            (w for w in self.words if w.startswith(prefix)),
            key=lambda w: (len(w), w),
        )
        return [w.upper() for w in matches[:limit]]

    def complete_word(self, sentence, suggestion):
        """
        Replace the incomplete word with the suggestion.

        Args:
            sentence: "I AM H"
            suggestion: "HELLO"

        Returns:
            "I AM HELLO"
        """
        if not suggestion:
            return sentence

        last_space = sentence.rfind(" ")
        base = sentence[: last_space + 1] if last_space >= 0 else ""
        return base + suggestion.upper()
