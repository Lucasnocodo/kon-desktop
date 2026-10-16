"""Recent, robust XY samples for one calibration position.

The caller selects either metric eye-through-aperture rays or normalized
hand-circle positions, keeping their buffers and saved models separate.
Source timestamps, rather than GUI arrival times, define stability/freshness.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CaptureState:
    ready: bool
    point: tuple[float, float] | None
    count: int
    required: int
    reason: str


class CalibrationBuffer:
    """Keep a brief valid history while missing frames temporarily block capture.

    Call reset with the monotonic time when each screen marker is displayed.
    Old, duplicate and out-of-order source frames are ignored. An invalid
    new observation blocks capture without discarding previous good rays.
    evaluate returns actionable Chinese feedback; the app owns capture
    requests, their timeout and persistence of error messages.
    """

    def __init__(self, min_samples: int = 6, min_span: float = .25,
                 window: float = 2.0, max_age: float = .35,
                 max_jitter: float = .012, *, hand_only: bool = False) -> None:
        if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples < 3:
            raise ValueError("min_samples must be an integer >= 3")
        if any(not math.isfinite(value) or value <= 0 for value in (min_span, window, max_age, max_jitter)):
            raise ValueError("Calibration durations and jitter must be finite and positive")
        if window < min_span or max_age > window:
            raise ValueError("Calibration window must cover span and freshness")
        self.min_samples, self.min_span = min_samples, min_span
        self.window, self.max_age, self.max_jitter = window, max_age, max_jitter
        self.hand_only = hand_only
        # Six distinct frames at 3 FPS need 1.67 seconds of source history.
        # Older history never relaxes the latest-frame freshness requirement.
        self._samples: deque[tuple[float, tuple[float, float]]] = deque(maxlen=max(256, min_samples * 6))
        self._step_started_at = -math.inf
        self._latest_timestamp: float | None = None
        self._latest_valid = False
        self._latest_reason = ""

    def reset(self, step_started_at: float) -> None:
        if not math.isfinite(step_started_at):
            raise ValueError("Marker source time must be finite")
        self._samples.clear()
        self._step_started_at = step_started_at
        self._latest_timestamp = None
        self._latest_valid = False
        self._latest_reason = ""

    def observe(self, timestamp: float, hit: Sequence[float] | None, reason: str = "") -> bool:
        """Accept one new source observation; return whether its time was accepted."""
        if (not math.isfinite(timestamp) or timestamp < self._step_started_at
                or self._latest_timestamp is not None and timestamp <= self._latest_timestamp):
            return False
        self._latest_timestamp = timestamp
        try:
            point = None if hit is None else np.asarray(hit, dtype=float)
        except (TypeError, ValueError):
            point = None
        self._latest_valid = bool(point is not None and point.shape == (2,) and np.isfinite(point).all())
        self._latest_reason = str(reason) if reason else ""
        if self._latest_valid:
            self._samples.append((timestamp, (float(point[0]), float(point[1]))))
        self._prune(timestamp)
        return True

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window + 1e-9:
            self._samples.popleft()

    def evaluate(self, now: float) -> CaptureState:
        def state(reason: str, point: tuple[float, float] | None = None) -> CaptureState:
            return CaptureState(point is not None, point, len(self._samples), self.min_samples, reason)

        if not math.isfinite(now):
            return state("鏡頭時間資料無效，請重新啟動鏡頭。")
        # Window length describes source-frame history. Do not subtract GUI
        # delivery latency from that history while the latest frame is still
        # fresh: six 3 FPS samples span 1.67s, plus up to .35s of delivery age.
        # Once observations stop, expired history still drains normally.
        self._prune(now - self.max_age)
        if self._latest_timestamp is None:
            return state("把手圈移到預覽中舒適的位置；整隻手保持入鏡。" if self.hand_only else
                         "請讓慣用眼與手圈保持在鏡頭中，穿過手圈瞄準目前的圓點。")
        age = now - self._latest_timestamp
        if age < -1e-9:
            return state("鏡頭時間尚未同步，請稍候新的畫面。")
        if age > self.max_age + 1e-9:
            return state("鏡頭畫面已過期，請等待追蹤恢復後再記錄。")
        if not self._latest_valid:
            return state(self._latest_reason or ("暫時無法定位手圈；請讓整隻手清楚入鏡。" if self.hand_only else
                                                "暫時無法計算瞄準線；請讓慣用眼與整隻手清楚入鏡。"))
        if len(self._samples) < self.min_samples:
            return state(f"已取得 {len(self._samples)} / {self.min_samples} 筆位置，請保持手圈不動。" if self.hand_only else
                         f"已取得 {len(self._samples)} / {self.min_samples} 筆瞄準線，請保持手圈與頭部不動。")

        times = np.array([timestamp for timestamp, _ in self._samples])
        points = np.array([point for _, point in self._samples])
        median = np.median(points, axis=0)
        deviations = np.abs(points - median)
        sigma = 1.4826 * np.median(deviations, axis=0)
        # An isolated tracking spike should not spoil an otherwise steady hold.
        # Preserve the caller's jitter budget in its chosen coordinate units.
        if float(np.max(sigma)) > self.max_jitter:
            return state("手圈仍在移動，請穩定片刻再記錄。" if self.hand_only else
                         "瞄準仍在移動，請固定坐姿與手圈，穩定片刻再記錄。")
        limits = np.maximum(3.0 * sigma, min(.006, self.max_jitter))
        inliers = np.all(deviations <= limits + 1e-12, axis=1)
        count = int(np.count_nonzero(inliers))
        if count < self.min_samples or count / len(points) < .75:
            return state(f"手圈位置有跳動（{count} 筆穩定），請保持手圈不動。" if self.hand_only else
                         f"瞄準線有跳動（{count} 筆穩定），請保持手圈與頭部不動。")
        # Do not capture an older majority while the newest aim has moved.
        # An isolated newest spike recovers on the next stable camera sample.
        if not inliers[-1]:
            return state("最新手圈位置有跳動，請等待下一筆穩定畫面。" if self.hand_only else
                         "最新瞄準線有跳動，請保持姿勢，等待下一筆穩定畫面。")
        stable_times = times[inliers]
        if stable_times[-1] - stable_times[0] < self.min_span - 1e-9:
            if self.hand_only:
                return state("請再保持手圈片刻，記錄需要約半秒的穩定畫面。")
            if self.min_span >= 1:
                return state(f"穩定時間 {stable_times[-1] - stable_times[0]:.1f} / {self.min_span:.1f} 秒；繼續穿過手圈看圓點。")
            return state("樣本尚未涵蓋足夠時間，請再保持瞄準片刻。")
        stable_points = points[inliers]
        # Robust per-axis sigma can hide a brief two-cluster move. Reject
        # a large displacement between the beginning and end of the hold.
        half = len(stable_points) // 2
        shift = np.abs(np.median(stable_points[-half:], axis=0) - np.median(stable_points[:half], axis=0))
        if float(np.max(shift)) > self.max_jitter * 2:
            return state("手圈位置仍在改變，請穩定後再記錄。" if self.hand_only else
                         "瞄準位置仍在改變，請穩定手圈與頭部後再記錄。")
        point = tuple(float(value) for value in np.median(stable_points, axis=0))
        return state(f"已取得 {count} 筆穩定位置，可以記錄目前的手圈位置。" if self.hand_only else
                     f"已取得 {count} 筆穩定瞄準線，可以記錄目前的圓點。", point)
