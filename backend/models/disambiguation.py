"""
Geometric Disambiguation Module
────────────────────────────────
Uses hand landmark geometry to refine letter predictions and resolve
confusion between visually similar ASL signs.

The existing 24-class landmark model is accurate but sometimes confuses
similar letters (G↔H, S↔T, M↔N, V↔W, etc.). This module applies
geometric rules based on finger positions to correct those confusions.

Landmark indices (MediaPipe):
  0: Wrist
  1-4: Thumb (CMC, MCP, IP, TIP)
  5-8: Index (MCP, PIP, DIP, TIP)
  9-12: Middle (MCP, PIP, DIP, TIP)
  13-16: Ring (MCP, PIP, DIP, TIP)
  17-20: Pinky (MCP, PIP, DIP, TIP)
"""

import math
import numpy as np


# ASL letter groups for disambiguation
GROUPS = {
    0: ["A", "E", "M", "N", "S", "T"],     # Fist variations
    1: ["B", "D", "F", "I", "K", "R", "U", "V", "W"],  # Open palm variations
    2: ["C", "O"],                           # Curved shapes
    3: ["G", "H"],                           # Pointing sideways
    4: ["L"],                                # L-shape
    5: ["P", "Q"],                           # Downward pointing
    6: ["X"],                                # Hook shape
    7: ["Y"],                                # Y-shape
}

# Reverse mapping: letter → group
LETTER_TO_GROUP = {}
for group_id, letters in GROUPS.items():
    for letter in letters:
        LETTER_TO_GROUP[letter] = group_id


class GeometricDisambiguator:
    """
    Applies geometric rules to refine letter predictions based on
    hand landmark positions.
    """

    def __init__(self):
        pass

    @staticmethod
    def distance(pt1, pt2):
        """Euclidean distance between two landmark points."""
        return math.sqrt(
            (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2
        )

    @staticmethod
    def distance_3d(pt1, pt2):
        """3D Euclidean distance between two landmarks."""
        return math.sqrt(
            (pt1[0] - pt2[0]) ** 2 +
            (pt1[1] - pt2[1]) ** 2 +
            (pt1[2] - pt2[2]) ** 2
        )

    @staticmethod
    def finger_is_up(landmarks, finger_tip, finger_pip):
        """Check if a finger is extended (tip above PIP in image coords)."""
        # In image coordinates, y increases downward
        return landmarks[finger_tip][1] < landmarks[finger_pip][1]

    @staticmethod
    def finger_is_curled(landmarks, finger_tip, finger_mcp):
        """Check if a finger is curled (tip below MCP)."""
        return landmarks[finger_tip][1] > landmarks[finger_mcp][1]

    def get_finger_states(self, pts):
        """
        Determine the state of each finger.

        Returns:
            dict with keys: thumb, index, middle, ring, pinky
            values: True if extended, False if curled
        """
        # Thumb: compare tip x-position relative to IP joint
        # (works for right hand; left hand would be reversed)
        thumb_up = pts[4][0] < pts[3][0] or pts[4][1] < pts[3][1]

        index_up = self.finger_is_up(pts, 8, 6)
        middle_up = self.finger_is_up(pts, 12, 10)
        ring_up = self.finger_is_up(pts, 16, 14)
        pinky_up = self.finger_is_up(pts, 20, 18)

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        }

    def count_fingers_up(self, pts):
        """Count how many fingers are extended (excluding thumb)."""
        states = self.get_finger_states(pts)
        return sum([states["index"], states["middle"], states["ring"], states["pinky"]])

    def disambiguate(self, predicted_letter, confidence, landmarks):
        """
        Refine a letter prediction using geometric rules.

        Args:
            predicted_letter: str - the model's predicted letter (A-Y)
            confidence: float - model confidence (0-1)
            landmarks: 21×3 numpy array of (x, y, z) normalized landmarks

        Returns:
            refined_letter: str - the refined prediction
            was_corrected: bool - whether the prediction was changed
        """
        if landmarks is None or len(landmarks) != 21:
            return predicted_letter, False

        pts = landmarks
        group = LETTER_TO_GROUP.get(predicted_letter)
        if group is None:
            return predicted_letter, False

        original = predicted_letter
        refined = predicted_letter

        # Only apply disambiguation when confidence is not very high
        # High-confidence predictions are likely correct
        if confidence > 0.92:
            return predicted_letter, False

        # ==============================
        # Group 0: [A, E, M, N, S, T]
        # Fist variations
        # ==============================
        if group == 0:
            refined = self._disambiguate_fist(pts, predicted_letter)

        # ==============================
        # Group 1: [B, D, F, I, K, R, U, V, W]
        # Open palm / finger variations
        # ==============================
        elif group == 1:
            refined = self._disambiguate_open(pts, predicted_letter)

        # ==============================
        # Group 2: [C, O]
        # Curved shapes
        # ==============================
        elif group == 2:
            refined = self._disambiguate_curved(pts, predicted_letter)

        # ==============================
        # Group 3: [G, H]
        # Pointing sideways
        # ==============================
        elif group == 3:
            refined = self._disambiguate_pointing(pts, predicted_letter)

        # ==============================
        # Group 5: [P, Q]
        # Downward pointing
        # ==============================
        elif group == 5:
            refined = self._disambiguate_downward(pts, predicted_letter)

        # Groups 4, 6, 7 have single letters (L, X, Y) - no disambiguation needed

        return refined, refined != original

    def _disambiguate_fist(self, pts, predicted):
        """
        Disambiguate within fist group: A, E, M, N, S, T.

        Key distinguishing features:
        - A: Thumb alongside fist (thumb tip left of index MCP)
        - E: Fingertips curled onto palm, thumb crosses fingers
        - M: Thumb under 3 fingers (between ring and pinky)
        - N: Thumb under 2 fingers (between middle and ring)
        - S: Fist with thumb over fingers
        - T: Thumb between index and middle, peeking out
        """
        # All fingers should be curled for this group
        fingers_up = self.count_fingers_up(pts)
        if fingers_up >= 3:
            # This is probably not a fist - model may be wrong
            return predicted

        thumb_tip = pts[4]
        thumb_ip = pts[3]
        index_tip = pts[8]
        index_mcp = pts[5]
        middle_tip = pts[12]
        middle_mcp = pts[9]
        ring_mcp = pts[13]

        # A: Thumb is to the side of the fist
        # Thumb tip is to the left of (or at same x as) index MCP
        # and thumb tip is above the finger tips
        thumb_beside = abs(thumb_tip[0] - index_mcp[0]) > 0.03
        thumb_above_fingers = thumb_tip[1] < index_tip[1]

        if thumb_beside and thumb_above_fingers:
            return "A"

        # E: All fingertips visible, curled down, thumb crosses
        # Fingertips are close to palm
        tips_y = [pts[8][1], pts[12][1], pts[16][1], pts[20][1]]
        mcps_y = [pts[5][1], pts[9][1], pts[13][1], pts[17][1]]
        tips_below_mcps = all(t > m for t, m in zip(tips_y, mcps_y))

        if tips_below_mcps:
            # Fingers are curled - check thumb position
            thumb_over = thumb_tip[1] < index_tip[1]
            thumb_near_index = abs(thumb_tip[0] - index_tip[0]) < 0.08

            if thumb_over and not thumb_near_index:
                return "E"

        # M vs N: Count how many fingers the thumb is under
        # M: thumb between ring and pinky (under 3 fingers)
        # N: thumb between middle and ring (under 2 fingers)
        thumb_x = thumb_tip[0]
        if thumb_tip[1] > index_mcp[1]:
            # Thumb is below MCPs - could be M, N, or T
            if thumb_x > ring_mcp[0]:
                return "M"
            elif thumb_x > middle_mcp[0]:
                return "N"

        # T: Thumb tip between index and middle, visible
        thumb_between = (
            min(index_mcp[0], middle_mcp[0]) < thumb_tip[0] < max(index_mcp[0], middle_mcp[0])
        )
        if thumb_between and thumb_tip[1] < index_tip[1]:
            return "T"

        # S: Default fist - thumb over fingers
        if predicted in ("S", "A", "E", "M", "N", "T"):
            return "S"

        return predicted

    def _disambiguate_open(self, pts, predicted):
        """
        Disambiguate within open palm group: B, D, F, I, K, R, U, V, W.

        Key features:
        - B: 4 fingers up, thumb across palm
        - D: Index up, rest curled, thumb touches middle
        - F: Index-thumb circle (OK sign), 3 fingers up
        - I: Only pinky up
        - K: Index+middle up, thumb between (like V with thumb)
        - R: Index+middle crossed
        - U: Index+middle up, together
        - V: Index+middle up, spread apart
        - W: Index+middle+ring up, spread
        """
        states = self.get_finger_states(pts)
        n_up = sum([states["index"], states["middle"], states["ring"], states["pinky"]])

        index_tip = pts[8]
        middle_tip = pts[12]
        ring_tip = pts[16]
        pinky_tip = pts[20]

        # I: Only pinky up
        if states["pinky"] and not states["index"] and not states["middle"] and not states["ring"]:
            return "I"

        # D: Only index up, thumb touching middle
        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            thumb_mid_dist = self.distance(pts[4], pts[12])
            if thumb_mid_dist < 0.1:
                return "D"

        # B: All four fingers up, close together
        if n_up >= 4:
            # Check fingers are close together (not spread)
            spread = self.distance(index_tip, pinky_tip)
            if spread < 0.15:
                return "B"
            else:
                return "W" if n_up == 3 else "B"

        # W: Three fingers up (index, middle, ring), spread
        if states["index"] and states["middle"] and states["ring"] and not states["pinky"]:
            return "W"

        # V vs U vs R vs K: Two fingers up (index + middle)
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            im_dist = self.distance(index_tip, middle_tip)

            # R: Fingers crossed (very close)
            if im_dist < 0.04:
                return "R"

            # U: Fingers together (close but not crossed)
            if im_dist < 0.08:
                return "U"

            # K: Thumb between index and middle
            thumb_between_y = (
                min(index_tip[1], middle_tip[1]) < pts[4][1] < max(index_tip[1], middle_tip[1])
            )
            if thumb_between_y and pts[4][1] < pts[5][1]:
                return "K"

            # V: Fingers spread apart
            return "V"

        # F: Thumb and index touching (circle), other 3 fingers up
        if states["middle"] and states["ring"] and states["pinky"]:
            thumb_index_dist = self.distance(pts[4], pts[8])
            if thumb_index_dist < 0.06:
                return "F"

        return predicted

    def _disambiguate_curved(self, pts, predicted):
        """
        Disambiguate C vs O.

        - C: Hand curved like C, gap between thumb and fingers
        - O: Fingers touching thumb tip (closed circle)
        """
        thumb_tip = pts[4]
        index_tip = pts[8]

        # Distance between thumb tip and index tip
        gap = self.distance(thumb_tip, index_tip)

        # O: Fingertips close to thumb (circle shape)
        if gap < 0.06:
            return "O"
        # C: Wider gap (open curve)
        return "C"

    def _disambiguate_pointing(self, pts, predicted):
        """
        Disambiguate G vs H.

        - G: Index pointing sideways, other fingers curled
        - H: Index+middle pointing sideways
        """
        states = self.get_finger_states(pts)

        # H: Both index and middle extended
        if states["index"] and states["middle"]:
            return "H"

        # G: Only index extended
        if states["index"] and not states["middle"]:
            return "G"

        return predicted

    def _disambiguate_downward(self, pts, predicted):
        """
        Disambiguate P vs Q.

        - P: Like K but pointing down (index+middle down, thumb out)
        - Q: Like G but pointing down (index down, thumb down)
        """
        states = self.get_finger_states(pts)

        # P: Two fingers pointing down
        if not states["ring"] and not states["pinky"]:
            index_down = pts[8][1] > pts[6][1]
            middle_down = pts[12][1] > pts[10][1]
            if index_down and middle_down:
                return "P"
            if index_down and not middle_down:
                return "Q"

        return predicted

    def refine_prediction(self, predicted_letter, confidence, landmarks_list):
        """
        Convenience method that accepts landmarks in list-of-dicts format
        (as returned by the server's metadata).

        Args:
            predicted_letter: str
            confidence: float
            landmarks_list: list of 21 dicts with 'x', 'y', 'z' keys

        Returns:
            refined_letter: str
            was_corrected: bool
        """
        if landmarks_list is None:
            return predicted_letter, False

        landmarks = np.array(
            [(lm["x"], lm["y"], lm["z"]) for lm in landmarks_list]
        )
        return self.disambiguate(predicted_letter, confidence, landmarks)
