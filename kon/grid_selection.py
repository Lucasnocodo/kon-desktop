"""Unique-file grid hit areas and bounded candidate hysteresis, in pixels.

This helper never moves desktop icons or operates on files. Target centers
assign files to cells; the existing TriggerGate owns completed sticky locks.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .desktop import DesktopTarget

Cell = tuple[int, int]
Bounds = tuple[float, float, float, float]


def _xy(value) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        if len(value) != 2:
            return None
        x, y = value[0], value[1]
        if (isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, Real)
                or not isinstance(y, Real)):
            return None
        point = float(x), float(y)
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _size(value) -> tuple[float, float] | None:
    size = _xy(value)
    return size if size is not None and all(item > 0 for item in size) else None


class GridSelector:
    """Select the sole file in a 30x12 cell, with an 8px boundary buffer.

    Cells use zero-based (column, row). Coordinates are local screen pixels;
    the exact far right/bottom edge belongs to the last cell. A cursor may
    enter a neighboring occupied or empty cell by up to hysteresis_pixels
    while retaining the previous unique candidate. A genuinely pointed-to
    collision cell always returns None immediately, even inside that buffer.

    Invalid/outside-screen points clear the candidate. A resize, removal,
    move or collision of the candidate resets its old ownership. Duplicate
    identical IDs/centers describe one file; contradictory duplicate IDs or
    malformed target metadata invalidate the selection rather than picking
    an arbitrary record. Off-screen target centers do not own a screen cell.
    """

    def __init__(self, columns: int = 30, rows: int = 12,
                 hysteresis_pixels: float = 8) -> None:
        if (isinstance(columns, bool) or isinstance(rows, bool)
                or not isinstance(columns, Integral) or not isinstance(rows, Integral)
                or columns < 1 or rows < 1):
            raise ValueError("Grid columns and rows must be positive integers")
        if (isinstance(hysteresis_pixels, bool) or not isinstance(hysteresis_pixels, Real)
                or not math.isfinite(hysteresis_pixels) or hysteresis_pixels < 0):
            raise ValueError("Grid hysteresis must be finite and nonnegative")
        self._columns, self._rows = int(columns), int(rows)
        self._hysteresis_pixels = float(hysteresis_pixels)
        self.reset()

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def hysteresis_pixels(self) -> float:
        return self._hysteresis_pixels

    @property
    def active_cell(self) -> Cell | None:
        """The current unique candidate's cell, including boundary buffering."""
        return self._active_cell

    @property
    def ambiguous_cell(self) -> Cell | None:
        """The actual cursor cell when it contains more than one file."""
        return self._ambiguous_cell

    @property
    def sample_valid(self) -> bool:
        """Whether the latest pick is a valid unique-file or empty cell."""
        return self._sample_valid

    @property
    def layout_signature(self):
        """Canonical ownership snapshot for invalidating accumulated dwell."""
        return self._layout_signature

    def reset(self) -> None:
        self._active_cell: Cell | None = None
        self._ambiguous_cell: Cell | None = None
        self._candidate_id: str | None = None
        self._screen_size: tuple[float, float] | None = None
        self._sample_valid = False
        self._layout_signature = None

    def _clear_candidate(self) -> None:
        self._active_cell, self._candidate_id = None, None

    def cell_for(self, point: Sequence[float], screen_size: Sequence[float]) -> Cell | None:
        """Pure geometry lookup; utility calls never change candidate state."""
        point, size = _xy(point), _size(screen_size)
        if point is None or size is None or any(point[index] < 0 or point[index] > size[index] for index in (0, 1)):
            return None
        indices = []
        for coordinate, extent, count in zip(point, size, (self.columns, self.rows)):
            scaled = coordinate / extent * count
            boundary = round(scaled)
            # Equivalent calibration arithmetic can put the same pixel a few
            # floating-point ULPs below its exact grid boundary. Snap only a
            # negligible 1e-10 cell-unit roundoff, independently of hysteresis.
            if abs(scaled - boundary) <= 1e-10:
                scaled = boundary
            indices.append(min(count - 1, math.floor(scaled)))
        return tuple(indices)

    def cell_bounds(self, cell: Cell, screen_size: Sequence[float]) -> Bounds | None:
        """Return (x,y,width,height), or None for an invalid cell/display."""
        size = _size(screen_size)
        if size is None or isinstance(cell, (str, bytes, Mapping)):
            return None
        try:
            if len(cell) != 2:
                return None
            column, row = cell[0], cell[1]
            if (isinstance(column, bool) or isinstance(row, bool)
                    or not isinstance(column, Integral) or not isinstance(row, Integral)
                    or not 0 <= column < self.columns or not 0 <= row < self.rows):
                return None
        except (TypeError, IndexError, KeyError):
            return None
        width, height = size[0] / self.columns, size[1] / self.rows
        return float(column * width), float(row * height), width, height

    def pick(self, targets: Sequence[DesktopTarget], point: Sequence[float],
             screen_size: Sequence[float]) -> DesktopTarget | None:
        """Return a current target object, never a cached/removed file record."""
        self._sample_valid = False
        point, size = _xy(point), _size(screen_size)
        cell = self.cell_for(point, size)
        if point is None or size is None or cell is None:
            self.reset()
            return None
        if size != self._screen_size:
            self.reset()
            self._screen_size = size
        self._ambiguous_cell = None
        buckets: dict[Cell, list[DesktopTarget]] = {}
        identities: dict[str, tuple[float, float]] = {}
        try:
            for target in targets:
                identity, center = getattr(target, "id", None), _xy(getattr(target, "center", None))
                if not isinstance(identity, str) or not identity or center is None:
                    raise ValueError("Invalid grid target metadata")
                if identity in identities:
                    if identities[identity] != center:
                        raise ValueError("Contradictory target positions")
                    continue
                identities[identity] = center
                target_cell = self.cell_for(center, size)
                if target_cell is not None:
                    buckets.setdefault(target_cell, []).append(target)
        except (TypeError, ValueError):
            self.reset()
            return None

        self._layout_signature = (size, tuple(sorted(identities.items())))

        occupants = buckets.get(cell, [])
        # A collision under the actual cursor cannot be hidden by hysteresis.
        if len(occupants) > 1:
            self._clear_candidate()
            self._ambiguous_cell = cell
            return None

        self._sample_valid = True

        old = buckets.get(self._active_cell, [])
        if len(old) != 1 or old[0].id != self._candidate_id:
            self._clear_candidate()
        elif cell != self._active_cell:
            x, y, width, height = self.cell_bounds(self._active_cell, size)
            margin = self.hysteresis_pixels
            if (margin > 0 and x - margin <= point[0] <= x + width + margin
                    and y - margin <= point[1] <= y + height + margin):
                return old[0]
            self._clear_candidate()

        if len(occupants) == 1:
            self._active_cell, self._candidate_id = cell, occupants[0].id
            return occupants[0]
        self._clear_candidate()
        return None
