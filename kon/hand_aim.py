"""Mirrored 2D hand-circle aiming and pixel-aware cursor smoothing.

The input cursor is the camera worker's hand-circle center in mirrored image
coordinates, not gaze, eye depth or a 3D screen ray. A comfortable rectangle
maps directly onto the screen. Optional two-corner personalization changes
that rectangle; it does not claim measured screen-aim accuracy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import ClassVar, Sequence

import numpy as np


def _xy(point: Sequence[float]) -> tuple[float, float] | None:
    try:
        value = np.asarray(point, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if value.shape != (2,) or not np.isfinite(value).all():
        return None
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class HandAimCalibration:
    """Map a mirrored image rectangle to normalized screen XY, with clamping."""

    top_left: tuple[float, float] = (.20, .20)
    bottom_right: tuple[float, float] = (.80, .80)
    personalized: bool = False
    calibration_model: ClassVar[str] = "hand_2d_rectangle"
    source: ClassVar[str] = "mirrored_image_xy"
    minimum_span: ClassVar[float] = .15

    def __post_init__(self):
        left, right = _xy(self.top_left), _xy(self.bottom_right)
        if left is None or right is None:
            raise ValueError("手圈範圍需要有效的左上與右下 XY 座標")
        if any(value < 0 or value > 1 for value in (*left, *right)):
            raise ValueError("手圈記錄位置需要位於鏡頭畫面內")
        if any(right[index] <= left[index] for index in (0, 1)):
            raise ValueError("左上記錄需位於右下記錄的左上方，請依順序重新記錄")
        if any(right[index] - left[index] < self.minimum_span - 1e-9 for index in (0, 1)):
            raise ValueError("手圈活動範圍太小，請將左上與右下位置拉開一些")
        if type(self.personalized) is not bool:
            raise ValueError("手圈範圍的校正狀態無效")
        object.__setattr__(self, "top_left", left)
        object.__setattr__(self, "bottom_right", right)

    @classmethod
    def from_corners(cls, top_left: Sequence[float], bottom_right: Sequence[float]) -> "HandAimCalibration":
        """Use two fresh held corner centers collected by the caller's UI."""
        return cls(top_left, bottom_right, True)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.top_left, self.bottom_right

    def predict(self, hand_cursor: Sequence[float]) -> tuple[float, float] | None:
        point = _xy(hand_cursor)
        if point is None:
            return None
        return tuple(float(np.clip((point[index] - self.top_left[index])
                                   / (self.bottom_right[index] - self.top_left[index]), 0, 1))
                     for index in (0, 1))

    def save(self, path: str | Path) -> None:
        payload = {"version": 1, "calibration_model": self.calibration_model, "source": self.source,
                   "top_left": list(self.top_left), "bottom_right": list(self.bottom_right),
                   "personalized": self.personalized}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "HandAimCalibration":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if (not isinstance(payload, dict) or type(payload.get("version")) is not int
                    or payload.get("version") != 1 or payload.get("calibration_model") != cls.calibration_model
                    or payload.get("source") != cls.source or type(payload.get("personalized")) is not bool):
                raise ValueError("手圈範圍校正檔格式無效")
            for key in ("top_left", "bottom_right"):
                if (not isinstance(payload.get(key), list) or len(payload[key]) != 2
                        or any(type(value) not in (int, float) for value in payload[key])):
                    raise ValueError("手圈範圍座標無效")
            return cls(payload["top_left"], payload["bottom_right"], payload["personalized"])
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError("手圈範圍校正檔無效，請重新記錄兩個角落") from exc


class HandCursorSmoother:
    """Median-of-three, adaptive pixel-space low-pass and resting dead zone.

    min_cutoff is in Hz; beta is Hz per pixel/second, unlike a normalized-unit
    One Euro beta. Intentional fast movement raises the cutoff, while small
    stationary jitter uses stronger smoothing. The 3px radial dead zone is
    applied only to the reported position: the internal filter keeps moving,
    so slow intentional motion accumulates rather than getting stuck.

    Input is already mapped normalized screen XY; the same filter can smooth
    either a hand-position mapping or an eye-through-circle ray mapping.
    The app gates actual frame freshness and calls reset on confirmed release,
    stale tracking or session changes, not on every unmatched gesture frame.
    A long source gap or a screen-size change also resets this helper.
    """

    def __init__(self, min_cutoff: float = .70, beta: float = .006,
                 derivative_cutoff: float = 1.0, dead_zone_pixels: float = 3.0,
                 max_gap: float = .50) -> None:
        if (any(not math.isfinite(value) or value <= 0 for value in (min_cutoff, derivative_cutoff, max_gap))
                or not math.isfinite(beta) or beta < 0
                or not math.isfinite(dead_zone_pixels) or dead_zone_pixels < 0):
            raise ValueError("手圈游標平滑參數無效")
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.dead_zone_pixels, self.max_gap = dead_zone_pixels, max_gap
        self.reset()

    def reset(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=3)
        self._timestamp: float | None = None
        self._screen_size: tuple[float, float] | None = None
        self._median: np.ndarray | None = None
        self._velocity = np.zeros(2)
        self._filtered: np.ndarray | None = None
        self._reported: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def update(self, timestamp: float, normalized_xy: Sequence[float],
               screen_size: Sequence[float]) -> tuple[float, float] | None:
        """Consume a unique source timestamp; invalid/old samples return None."""
        point, size = _xy(normalized_xy), _xy(screen_size)
        if (point is None or size is None or any(value <= 0 for value in size)
                or not math.isfinite(timestamp)
                or self._timestamp is not None and timestamp <= self._timestamp):
            return None
        raw = np.clip(np.array(point), 0, 1) * size
        if (self._timestamp is None or size != self._screen_size
                or timestamp - self._timestamp > self.max_gap):
            self.reset()
            self._history.extend((raw.copy(), raw.copy(), raw.copy()))
            self._timestamp, self._screen_size = timestamp, size
            self._median, self._filtered, self._reported = raw.copy(), raw.copy(), raw.copy()
            return tuple(float(value) for value in raw / size)

        dt = timestamp - self._timestamp
        self._history.append(raw)
        median = np.median(np.array(self._history), axis=0)
        velocity = (median - self._median) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self._velocity += derivative_alpha * (velocity - self._velocity)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._velocity))
        self._filtered += self._alpha(cutoff, dt) * (median - self._filtered)
        if float(np.linalg.norm(self._filtered - self._reported)) > self.dead_zone_pixels:
            self._reported = self._filtered.copy()
        self._median, self._timestamp = median, timestamp
        return tuple(float(value) for value in np.clip(self._reported / size, 0, 1))
