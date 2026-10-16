"""Display-only interpolation between already measured cursor positions.

UI arrival time drives this animation, never camera freshness, gesture dwell
or audio eligibility. The caller retains those decisions on actual tracking
frames, and clears this helper when tracking becomes invalid. Sampling for a
60Hz repaint does not manufacture any new measured points or extend tracking.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Sequence

XY = tuple[float, float]


def _point(value: Sequence[float]) -> XY | None:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        if len(value) != 2:
            return None
        x, y = value[0], value[1]
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in (x, y)):
            return None
        result = float(x), float(y)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _time(value: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


class CursorMotion:
    """Chase the latest measured XY without overshoot, prediction or a queue.

    tau=.035 gives about 76% response in 50ms and 86% in 70ms. At the
    settle_duration endpoint the position becomes the exact latest target,
    allowing a repaint timer to stop. Repeated identical targets do not reset
    that endpoint. A new target interrupts from the current interpolated
    position, so the displayed cursor remains continuous.

    max_gap resets on a later push after a UI-arrival gap. It is an animation
    continuity rule, not a camera-validity timeout: sample never auto-expires
    a static point. The root's source-timestamp watchdog owns that decision.
    First points and snap (completed file lock) are immediate.
    """

    def __init__(self, tau: float = .035, max_gap: float = .35,
                 settle_duration: float = .12) -> None:
        values = tuple(_time(value) for value in (tau, max_gap, settle_duration))
        if any(value is None or value <= 0 for value in values):
            raise ValueError("Cursor motion durations must be finite and positive")
        self.tau, self.max_gap, self.settle_duration = values
        self.clear()

    @property
    def active(self) -> bool:
        return self._target is not None

    @property
    def is_animating(self) -> bool:
        return self._animating

    def clear(self) -> None:
        self._start: XY | None = None
        self._target: XY | None = None
        self._current: XY | None = None
        self._transition_started: float | None = None
        self._last_push: float | None = None
        self._last_event: float | None = None
        self._animating = False

    def _valid_time(self, now: float | None) -> bool:
        return now is not None and (self._last_event is None or now >= self._last_event)

    def _sample_position(self, now: float) -> XY | None:
        if self._target is None:
            return None
        if not self._animating:
            return self._current
        elapsed = now - self._transition_started
        if now >= self._transition_started + self.settle_duration:
            self._current = self._target
            self._animating = False
        else:
            remaining = math.exp(-elapsed / self.tau)
            # A convex combination also avoids overflow from subtracting two
            # individually finite but opposite, very large endpoint values.
            self._current = tuple(self._start[index] * remaining + self._target[index] * (1 - remaining)
                                  for index in (0, 1))
        return self._current

    def push(self, point: Sequence[float], arrival_time: float) -> bool:
        """Accept a measured point; invalid/backward UI inputs leave state alone."""
        point, now = _point(point), _time(arrival_time)
        if point is None or not self._valid_time(now):
            return False
        if not self.active or now - self._last_push > self.max_gap:
            return self.snap(point, now)
        current = self._sample_position(now)
        self._last_event, self._last_push = now, now
        if point == self._target:
            return True
        self._start, self._current, self._target = current, current, point
        self._transition_started = now
        self._animating = current != point
        return True

    def sample(self, now: float) -> XY | None:
        """Render at UI time only; never touch or refresh measured-arrival time."""
        now = _time(now)
        if not self._valid_time(now) or not self.active:
            return None
        result = self._sample_position(now)
        self._last_event = now
        return result

    def snap(self, point: Sequence[float], now: float) -> bool:
        """Immediately anchor a completed lock or a new tracking session."""
        point, now = _point(point), _time(now)
        if point is None or not self._valid_time(now):
            return False
        self._start = self._target = self._current = point
        self._transition_started = self._last_push = self._last_event = now
        self._animating = False
        return True
