"""Determine aiming stability from fresh hand motion and measured grid cells."""

from __future__ import annotations

import math
from numbers import Real


class AimDwellWeight:
    """Boost future dwell after slow hand motion in one actual cell.

    Positions use normalized camera XY, independently of monitor resolution.
    Display interpolation never supplies observations. Missing or repeated
    observations break stability; the gate separately retains recent dwell.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._timestamp = None
        self._position = None
        self._cell = None
        self._stable_at = None
        self.speed = None
        self.moving = None
        self.stable = False
        self.weight = 1.0

    def observe(self, timestamp, hand_position, cell, *, stable_seconds=.3,
                multiplier=2.0, speed_limit=.04, max_gap=.3):
        try:
            valid = (isinstance(timestamp, Real) and not isinstance(timestamp, bool)
                     and math.isfinite(timestamp) and len(hand_position) == 2
                     and all(isinstance(axis, Real) and not isinstance(axis, bool)
                             and math.isfinite(axis) for axis in hand_position)
                     and cell is not None)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self.reset()
            return self.weight
        position = tuple(float(axis) for axis in hand_position)
        if (self._timestamp is None or timestamp <= self._timestamp
                or timestamp - self._timestamp > max_gap):
            self.reset()
            self._timestamp, self._position, self._cell = timestamp, position, cell
            self._stable_at = timestamp
            return self.weight
        elapsed = timestamp - self._timestamp
        self.speed = math.hypot(*(a - b for a, b in zip(position, self._position))) / elapsed
        self.moving = self.speed > speed_limit
        if self.moving:
            self._stable_at = None
        elif cell != self._cell or self._stable_at is None:
            self._stable_at = timestamp
        self.stable = (not self.moving and self._stable_at is not None
                       and timestamp - self._stable_at >= stable_seconds - 1e-9)
        self.weight = float(multiplier) if self.stable else 1.0
        self._timestamp, self._position, self._cell = timestamp, position, cell
        return self.weight
