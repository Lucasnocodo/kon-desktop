"""Bounded recognition hysteresis; raw pose evidence remains authoritative.

Only near-boundary shape mismatches can borrow a recent actual match. Missing
data, a different hand/model/source, or a discontinuous camera timeline cannot.
Enrollment and the underlying classifier's thresholds are unchanged.
"""

from __future__ import annotations

import math

from .geometry import GestureResult

GESTURE_JITTER_GRACE = .20
GESTURE_HISTORY_MAX_GAP = .35
_NEAR_MISS_SCORES = {
    # Personal matches have score >= .60; generic matches score >= .50.
    "personal gesture mismatch": .52,
    "finger pattern": .40,
    "thumb contact or finger margin": .40,
}


class GestureStabilizer:
    """Keep a recent match through brief, validated threshold chatter.

    Misses never refresh the last-match time. Callers separately use raw pose
    evidence for release timing and voice eligibility; this output keeps the
    visible gesture, cursor and target dwell from snapping on camera noise.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._timestamp: float | None = None
        self._matched_at: float | None = None
        self._source: str | None = None
        self._profile = None

    def update(self, gesture: GestureResult, timestamp: float, *,
               shape_valid: bool = True, same_hand: bool = True, profile=None) -> bool:
        previous = self._timestamp
        if not math.isfinite(timestamp) or previous is not None and timestamp <= previous:
            self.reset()
            self._timestamp = previous
            return False
        if (not same_hand or self._source != gesture.source or self._profile is not profile
                or previous is not None and timestamp - previous > GESTURE_HISTORY_MAX_GAP + 1e-9):
            self.reset()
        self._timestamp = timestamp
        self._source, self._profile = gesture.source, profile
        if not shape_valid or not math.isfinite(gesture.score):
            self._matched_at = None
            return False
        if gesture.is_gao:
            self._matched_at = timestamp
            return True
        floor = _NEAR_MISS_SCORES.get(gesture.reason)
        if (floor is not None and gesture.score >= floor and self._matched_at is not None
                and timestamp - self._matched_at <= GESTURE_JITTER_GRACE + 1e-9):
            return True
        self._matched_at = None
        return False
