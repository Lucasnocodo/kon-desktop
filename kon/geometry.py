"""KON pose, aiming, calibration and trigger math; no camera or file actions.

Hand landmarks use MediaPipe's 21-point index order. Palm scale and the
wrist-distance folding cue derive from Tactyle Touch's Apache-2.0
``touch.geometry`` (copyright 2026 gokulmc). KON additionally measures joint
curl and thumb contact; it does not use ILY's wrist-distance test as a veto.

Metric aiming points MUST share the same camera coordinate frame: x points
right in the original image, y down, z away from the camera, in meters. Never
combine normalized hand z or a hand-relative origin with a metric eye point.
The camera's z=0 plane approximates the display plane. Affine calibration
learns its screen offset, scale and mirror; it cannot correct arbitrary
display tilt or inaccurate monocular depth across every head position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from numbers import Real
from typing import Sequence

import numpy as np
from touch.geometry import hand_scale

APERTURE_INDICES = (4, 10, 11, 13, 14)
_GAO_MIN_SCORE = 0.50


@dataclass(frozen=True)
class GestureResult:
    score: float
    is_gao: bool
    aperture_indices: tuple[int, ...] = APERTURE_INDICES
    thumb_contact_ratio: float | None = None
    reason: str = ""
    source: str = "normalized"
    hint: str = ""
    finger_metrics: tuple[tuple[str, float], ...] = ()


def _hand_array(points: Sequence[Sequence[float]]) -> np.ndarray | None:
    try:
        result = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.shape != (21, 3) or not np.isfinite(result).all():
        return None
    return result


def _unit_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    t = float(np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1))
    return float(np.linalg.norm(point - (start + t * direction)))


def classify_gao(
    points: Sequence[Sequence[float]], image_aspect_ratio: float = 1.0,
    *, world_points: Sequence[Sequence[float]] | None = None,
) -> GestureResult:
    """Recognize straight index/pinky, curled middle/ring, opposed thumb.

    Prefer finite, nondegenerate 21x3 MediaPipe world landmarks. They describe
    joint curl in 3D without the image projection's foreshortening. Fall back
    to image landmarks only when world data is unavailable or invalid, never
    because a valid world hand fails the pose. For image landmarks pass frame
    width/height (normalized z uses image width); world data needs no aspect
    correction. This heuristic cannot prove physical contact from occluded
    or noisy monocular landmarks; ``score`` is not a probability.

    Contact is measured to thumb MP/IP/tip segments, not only the tip. The
    relaxed score cutoff allows contact within half a palm length. Both
    fingertips must satisfy it and the thumb must move toward them, rejecting
    the outward thumb of an ordinary ILY sign.
    """
    world = _hand_array(world_points)
    if world is not None and hand_scale([tuple(p) for p in world]) >= 1e-6:
        pts, source = world, "world"
    else:
        pts, source = _hand_array(points), "normalized"
        try:
            aspect = float(image_aspect_ratio)
        except (TypeError, ValueError, OverflowError):
            aspect = math.nan
        if pts is None or not math.isfinite(aspect) or aspect <= 0:
            return GestureResult(0.0, False, reason="invalid landmarks", source="none",
                                 hint="手部關節資料不完整，請讓整隻手清楚入鏡")
        pts = pts * np.array([aspect, 1.0, aspect])

    scale = hand_scale([tuple(point) for point in pts])
    if scale < 1e-6 or not math.isfinite(scale):
        return GestureResult(0.0, False, reason="degenerate palm", source=source,
                             hint="手掌追蹤不完整，請讓手腕與掌心清楚入鏡")
    radius = np.linalg.norm(pts - pts[0], axis=1)
    if not np.isfinite(radius).all():
        return GestureResult(0.0, False, reason="invalid landmarks", source=source,
                             hint="手部關節資料異常，請讓整隻手清楚入鏡")
    scores, failures, metrics = [], [], []
    for name, label, indices, extended in (
        ("index", "食指", (5, 6, 7, 8), True),
        ("pinky", "小指", (17, 18, 19, 20), True),
        ("middle", "中指", (9, 10, 11, 12), False),
        ("ring", "無名指", (13, 14, 15, 16), False),
    ):
        chain = pts[list(indices)]
        bones = np.diff(chain, axis=0)
        lengths = np.linalg.norm(bones, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths < max(scale * 0.001, 1e-8)):
            failures.append(f"{label}關節追蹤不完整，請讓手指清楚入鏡")
            scores.append(0.0)
            continue
        straightness = float(np.linalg.norm(chain[-1] - chain[0]) / sum(lengths))
        turns = np.degrees(np.arccos(np.clip(
            np.sum(bones[:-1] * bones[1:], axis=1) / (lengths[:-1] * lengths[1:]), -1, 1)))
        bend = float(max(turns))
        radial_margin = float((radius[indices[-1]] - radius[indices[1]]) / scale)
        metrics.extend(((name + "_straightness", straightness),
                        (name + "_bend_degrees", bend),
                        (name + "_radial_margin", radial_margin)))
        if extended:
            # Wrist progress alone can incorrectly accept a bent finger.
            straight = straightness >= 0.80 and bend <= 65 and radial_margin >= 0
            radial_score = _unit_score((radial_margin - 0.02) / 0.28)
            chain_score = min(_unit_score((straightness - 0.80) / 0.12),
                              _unit_score((65 - bend) / 25))
            score = max(radial_score, chain_score) if straight else 0.0
            if score < _GAO_MIN_SCORE:
                failures.append(f"請伸直{label}，並與彎曲的中指、無名指分開")
        else:
            # A forward curl may leave the tip farther from the wrist than
            # PIP. Measure the joints rather than vetoing that ordering.
            radial_score = _unit_score((-radial_margin - 0.02) / 0.28)
            curl_score = min(_unit_score((bend - 25) / 45),
                             _unit_score((0.99 - straightness) / 0.10))
            score = max(radial_score, curl_score)
            if score < _GAO_MIN_SCORE:
                failures.append(f"{label}尚未彎曲，請壓向拇指")
        scores.append(score)

    if min(scores) < _GAO_MIN_SCORE:
        return GestureResult(min(scores), False, reason="finger pattern", source=source,
                             hint="；".join(failures), finger_metrics=tuple(metrics))
    if any(np.linalg.norm(pts[b] - pts[a]) < max(scale * 0.001, 1e-8)
           for a, b in ((2, 3), (3, 4))):
        return GestureResult(0.0, False, reason="thumb contact or finger margin", source=source,
                             hint="拇指關節追蹤不完整，請讓拇指清楚入鏡",
                             finger_metrics=tuple(metrics))

    contacts = [min(_segment_distance(pts[tip], pts[a], pts[b])
                    for a, b in ((2, 3), (3, 4))) / scale for tip in (12, 16)]
    contact = float(max(contacts))
    contact_score = _unit_score((0.70 - contact) / 0.40)
    folded_center = (pts[12] + pts[16]) / 2
    opposition = float((np.linalg.norm(pts[2] - folded_center)
                        - np.linalg.norm(pts[4] - folded_center)) / scale)
    metrics.extend((("middle_thumb_contact", float(contacts[0])),
                    ("ring_thumb_contact", float(contacts[1])),
                    ("thumb_opposition", opposition)))
    score = min(min(scores), contact_score) if opposition >= 0.02 else 0.0
    is_gao = score >= _GAO_MIN_SCORE
    if opposition < 0.02:
        hint = "拇指朝外伸出，請收向中指與無名指下方"
    elif not is_gao:
        labels = [label for label, distance in zip(("中指", "無名指"), contacts)
                  if _unit_score((0.70 - distance) / 0.40) < _GAO_MIN_SCORE]
        hint = "、".join(labels) + "未靠到拇指指腹，請壓在拇指上"
    else:
        hint = "指令手勢已辨識：食指、小指伸直，中指、無名指壓住拇指"
    return GestureResult(score, is_gao, thumb_contact_ratio=contact,
                         reason="recognized" if is_gao else "thumb contact or finger margin",
                         source=source, hint=hint, finger_metrics=tuple(metrics))


def ring_center(points: Sequence[Sequence[float]]) -> tuple[float, float, float] | None:
    """Heuristic aperture center, averaging thumb4/middle10,11/ring13,14.

    This estimates the interior of the folded-finger loop; it does not
    segment an actual visible hole. Apply it to the transformed hand's
    common-camera metric landmarks for aiming, not hand-relative landmarks.
    """
    pts = _hand_array(points)
    if pts is None:
        return None
    boundary = pts[list(APERTURE_INDICES)]
    if float(np.linalg.norm(np.ptp(boundary, axis=0))) < 1e-8:
        return None
    return tuple(float(value) for value in boundary.mean(axis=0))


def ray_screen_intersection(
    eye_xyz: Sequence[float], aperture_xyz: Sequence[float]
) -> tuple[float, float] | None:
    """Continue the eye-to-aperture ray to camera plane z=0; return meters.

    Require an aperture in front of the eye and screen in front of both.
    Conservative plausibility bounds reject inverted/noisy monocular depth.
    No mirroring or screen clamping is done here.
    """
    try:
        eye, aperture = np.asarray(eye_xyz, dtype=float), np.asarray(aperture_xyz, dtype=float)
    except (TypeError, ValueError):
        return None
    if eye.shape != (3,) or aperture.shape != (3,) or not np.isfinite([eye, aperture]).all():
        return None
    if not (0.18 <= eye[2] <= 2.5 and 0.06 <= aperture[2] <= eye[2] - 0.035):
        return None
    direction = aperture - eye
    if not 0.04 <= np.linalg.norm(direction) <= 1.6:
        return None
    if abs(eye[0]) > eye[2] * 1.5 or abs(eye[1]) > eye[2] * 1.5:
        return None
    if abs(aperture[0]) > aperture[2] * 2.0 or abs(aperture[1]) > aperture[2] * 2.0:
        return None
    t = -eye[2] / direction[2]
    if t <= 1.0 or t > 30.0:
        return None
    hit = eye + t * direction
    return (float(hit[0]), float(hit[1]))


@dataclass(frozen=True)
class ScreenCalibration:
    """Affine fit from ray-plane meters to normalized screen coordinates."""

    coefficients: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    input_bounds: np.ndarray
    rms_error: float
    max_error: float
    condition_number: float
    sample_count: int

    @classmethod
    def fit(
        cls, raw_points: Sequence[Sequence[float]],
        screen_points: Sequence[Sequence[float]],
        *, max_rms_error: float = 0.10, max_point_error: float = 0.22,
    ) -> "ScreenCalibration":
        """Fit >=5 distributed samples; reject collinear or inaccurate data.

        Residuals are Euclidean distances in normalized screen units. Use
        four corners and center, or a 3x3 grid, with the same dominant eye.
        These are calibration-sample residuals, not a guarantee about a
        later head position, iris diameter or visible hole location.
        """
        if (not math.isfinite(max_rms_error) or not math.isfinite(max_point_error)
                or max_rms_error <= 0 or max_point_error <= 0):
            raise ValueError("Calibration error limits must be finite and positive")
        raw, screen = np.asarray(raw_points, dtype=float), np.asarray(screen_points, dtype=float)
        if raw.ndim != 2 or raw.shape[1:] != (2,) or screen.shape != raw.shape or len(raw) < 5:
            raise ValueError("Calibration needs at least five paired XY samples")
        if not np.isfinite(raw).all() or not np.isfinite(screen).all():
            raise ValueError("Calibration samples must be finite")
        if np.any(screen < 0) or np.any(screen > 1) or np.any(np.ptp(screen, axis=0) < 0.35):
            raise ValueError("Screen samples must cover both screen axes in [0, 1]")
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        if np.any(scale < 1e-5):
            raise ValueError("Ray samples have insufficient spread")
        design = np.column_stack(((raw - mean) / scale, np.ones(len(raw))))
        condition = float(np.linalg.cond(design))
        if not math.isfinite(condition) or condition > 1000:
            raise ValueError("Ray samples are collinear or poorly conditioned")
        coefficients, _, rank, _ = np.linalg.lstsq(design, screen, rcond=None)
        if rank != 3 or abs(np.linalg.det(coefficients[:2])) < 1e-6:
            raise ValueError("Calibration mapping is degenerate")
        errors = np.linalg.norm(design @ coefficients - screen, axis=1)
        rms, worst = float(np.sqrt(np.mean(errors ** 2))), float(errors.max())
        if rms > max_rms_error or worst > max_point_error:
            raise ValueError(f"Calibration residual is too high (RMS {rms:.3f}, max {worst:.3f})")
        return cls(coefficients, mean, scale, np.array([raw.min(axis=0), raw.max(axis=0)]),
                   rms, worst, condition, len(raw))

    def predict(self, point: Sequence[float]) -> tuple[float, float] | None:
        """Predict without clamping; reject distant extrapolation/off-screen."""
        try:
            raw = np.asarray(point, dtype=float)
        except (TypeError, ValueError):
            return None
        if raw.shape != (2,) or not np.isfinite(raw).all():
            return None
        low, high = self.input_bounds
        margin = (high - low) * 0.35
        if np.any(raw < low - margin) or np.any(raw > high + margin):
            return None
        result = np.r_[((raw - self.input_mean) / self.input_scale), 1.0] @ self.coefficients
        if not np.isfinite(result).all() or np.any(result < -0.10) or np.any(result > 1.10):
            return None
        return (float(result[0]), float(result[1]))

    def save(self, path: str | Path) -> None:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "coefficients": self.coefficients.tolist(),
                   "input_mean": self.input_mean.tolist(), "input_scale": self.input_scale.tolist(),
                   "input_bounds": self.input_bounds.tolist(), "rms_error": self.rms_error,
                   "max_error": self.max_error, "condition_number": self.condition_number,
                   "sample_count": self.sample_count}
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ScreenCalibration":
        try:
            payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
            if payload["version"] != 1:
                raise ValueError("Unsupported calibration version")
            coeff = np.asarray(payload["coefficients"], dtype=float)
            mean = np.asarray(payload["input_mean"], dtype=float)
            scale = np.asarray(payload["input_scale"], dtype=float)
            bounds = np.asarray(payload["input_bounds"], dtype=float)
            quality = np.array([payload["rms_error"], payload["max_error"], payload["condition_number"]], dtype=float)
            count = int(payload["sample_count"])
            if coeff.shape != (3, 2) or mean.shape != (2,) or scale.shape != (2,) or bounds.shape != (2, 2):
                raise ValueError("Invalid calibration dimensions")
            if not all(np.isfinite(value).all() for value in [coeff, mean, scale, bounds, quality]):
                raise ValueError("Calibration must be finite")
            if (np.any(scale < 1e-5) or np.any(bounds[1] <= bounds[0]) or count < 5
                    or abs(np.linalg.det(coeff[:2])) < 1e-6
                    or np.any(quality < 0) or quality[0] > 0.10 or quality[1] > 0.22
                    or not 1 <= quality[2] <= 1000):
                raise ValueError("Invalid calibration quality")
            return cls(coeff, mean, scale, bounds, float(quality[0]), float(quality[1]), float(quality[2]), count)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid calibration file") from error


class TriggerGate:
    """Lock a cumulative dwell winner within a rolling window, trigger once.

    ``update`` returns readiness, not a trigger. ``hear_keyword`` returns an
    immediate target. ``take_trigger`` consumes any queued trigger. Both
    paths consume the same latch, so a practice animation cannot repeat as a live deletion while
    the hand remains held. A completed lock ignores later aim targets until
    observed release for release_seconds or explicit clear_lock. Missing or
    uncertain tracking pauses speech/actions while retaining that file. The
    consumed latch survives disarm/arm and clear_lock. A caller may explicitly
    complete a successful action to select a different file while holding the
    gesture; the same file remains consumed until an observed non-KON pose for
    release_seconds starts a fresh gesture.
    """

    def __init__(self, dwell_seconds: float = 3.0, keyword_max_age: float = 1.5,
                 pose_max_age: float = 0.3, release_seconds: float = 0.25,
                 minimum_dwell_seconds: float | None = None,
                 tracking_grace_seconds: float = .5) -> None:
        for value in (dwell_seconds, keyword_max_age, pose_max_age, release_seconds, tracking_grace_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Trigger durations must be finite and positive")
        self.dwell_seconds, self.keyword_max_age = dwell_seconds, keyword_max_age
        minimum = dwell_seconds * .5 if minimum_dwell_seconds is None else minimum_dwell_seconds
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("Minimum weighted dwell must be finite and positive")
        self.minimum_dwell_seconds = minimum
        self.tracking_grace_seconds = tracking_grace_seconds
        self.pose_max_age, self.release_seconds = pose_max_age, release_seconds
        self._armed = False
        self._armed_at: float | None = None
        self._latched = False
        self._triggered_target: str | None = None
        self._consumed_targets: set[str] = set()
        self._last_event = -math.inf
        self._last_pose: float | None = None
        self._last_observation: float | None = None
        self._pose_ok = False
        self._target: str | None = None
        self._locked_target: str | None = None
        self._locked_at: float | None = None
        self._dwell_start: float | None = None
        self._dwell_history: deque[tuple[float, float, str | None, float]] = deque()
        self._dwell_times: dict[str | None, float] = {}
        self._sample_target: str | None = None
        self._sample_time: float | None = None
        self._sample_weight = 1.0
        self._last_valid_sample: float | None = None
        self._leading_target: str | None = None
        self._leading_dwell_seconds = 0.0
        self._release_start: float | None = None
        self._keyword_time: float | None = None
        self._pending: str | None = None
        self.ready = False
        self.progress = 0.0

    @property
    def locked_target(self) -> str | None:
        """Completed target lock, retained while tracking/action is paused."""
        return self._locked_target

    @property
    def locked_at(self) -> float | None:
        """Original acquisition time; pauses do not change this timestamp."""
        return self._locked_at

    @property
    def leading_target(self) -> str | None:
        """The unique file with the most accumulated recent dwell."""
        return self._leading_target

    @property
    def leading_dwell_seconds(self) -> float:
        return self._leading_dwell_seconds

    def _accept_time(self, now: float) -> bool:
        if not math.isfinite(now) or now < self._last_event:
            return False
        self._last_event = now
        return True

    def _clear_candidate(self) -> None:
        self._target, self._dwell_start = None, None
        self._sample_time, self._sample_target, self._last_valid_sample = None, None, None
        self._dwell_history.clear()
        self._dwell_times.clear()
        self._leading_target, self._leading_dwell_seconds = None, 0.0
        self.ready, self.progress = False, 0.0

    def _refresh_dwell(self, sample_time: float) -> None:
        cutoff = sample_time - self.dwell_seconds
        while self._dwell_history and self._dwell_history[0][1] <= cutoff:
            self._dwell_history.popleft()
        if self._dwell_history and self._dwell_history[0][0] < cutoff:
            _, end, target, weight = self._dwell_history[0]
            self._dwell_history[0] = (cutoff, end, target, weight)
        self._dwell_times.clear()
        for start, end, target, weight in self._dwell_history:
            self._dwell_times[target] = self._dwell_times.get(target, 0.0) + (end - start) * weight
        self._leading_dwell_seconds = max((duration for target, duration in self._dwell_times.items()
                                          if target is not None), default=0.0)
        leaders = [target for target, duration in self._dwell_times.items()
                   if target is not None and abs(duration - self._leading_dwell_seconds) <= 1e-9]
        self._leading_target = leaders[0] if len(leaders) == 1 else None
        self.progress = min(1.0, max(0.0, self._leading_dwell_seconds / self.minimum_dwell_seconds))

    def _record_dwell(self, sample_time: float, dwell_weight: float) -> None:
        if self._sample_time is None or sample_time <= self._sample_time:
            return
        weight = min(self._sample_weight, dwell_weight) if self._sample_target is not None else 1.0
        if (self._dwell_history and self._dwell_history[-1][1] == self._sample_time
                and self._dwell_history[-1][2:] == (self._sample_target, weight)):
            start, _, target, weight = self._dwell_history[-1]
            self._dwell_history[-1] = (start, sample_time, target, weight)
        else:
            self._dwell_history.append((self._sample_time, sample_time, self._sample_target, weight))

    def clear_lock(self, now: float) -> None:
        """Cancel a UI session/removed target without rearming consumption.

        Unlike camera invalidation, this explicitly discards a completed
        target lock. Call when leaving a target session or removing its file.
        It does not count as physical gesture release.
        """
        if not self._accept_time(now):
            return
        self._locked_target, self._locked_at = None, None
        self._clear_candidate()
        self._pose_ok = False
        self._release_start = None
        self._keyword_time, self._pending = None, None

    def _expire_pose(self, now: float) -> None:
        if self._last_pose is None or now - self._last_pose > self.pose_max_age + 1e-9:
            self.invalidate_tracking(now)

    def complete_action(self, target_id: str, now: float) -> bool:
        """Continue a held gesture only after its matching action succeeded.

        The caller must confirm completion, rather than cancellation or a
        tracking interruption. The consumed target remains unavailable until
        physical release; a different file needs entirely new dwell and audio.
        The trigger identity survives clear_lock/disarm and physical release
        so an animation may finish after its file or held gesture is gone.
        A newer trigger replaces that identity, rejecting older completions.
        """
        if (self._triggered_target is None or target_id != self._triggered_target
                or not self._accept_time(now)):
            return False
        self.clear_lock(now)
        self._latched = False
        self._triggered_target = None
        return True

    def _try_trigger(self, now: float) -> str | None:
        if (not self.ready or self._latched or self._keyword_time is None
                or now - self._keyword_time > self.keyword_max_age + 1e-9
                or self._target is None or self._target in self._consumed_targets):
            return None
        target = self._target
        self._latched, self._keyword_time = True, None
        self._triggered_target = target
        self._consumed_targets.add(target)
        self.ready = False
        return target

    def update(self, now: float, pose_ok: bool, target_id: str | None, armed: bool,
               *, release_observed: bool | None = None, movement_released: bool = False,
               observed_at: float | None = None, raw_pose_ok: bool | None = None,
               dwell_weight: float = 1.0) -> bool:
        """Observe an actual frame; return whether its retained file is ready.

        Accumulate observed elapsed time, including empty cells (None), in
        the latest dwell_seconds. Lock a unique winner immediately when it
        reaches minimum_dwell_seconds; separate visits to a cell add up.
        Brief missing observations pause counting and preserve recent dwell.
        A completed lock ignores ordinary aim changes; movement_released
        explicitly cancels it without rearming a consumed gesture. A
        false pose pauses readiness immediately and releases the lock only
        after release_seconds of observed non-command pose. Set
        release_observed=False for uncertain landmarks or synthetic frames;
        those invalidate tracking without accumulating physical release.
        None preserves the existing API's observed-false-pose behavior.
        A stabilized pose may preserve cursor/dwell through a near miss, but
        raw_pose_ok still controls release timing and voice/action readiness.
        """
        if (isinstance(dwell_weight, bool) or not isinstance(dwell_weight, Real)
                or not math.isfinite(dwell_weight) or dwell_weight < 1):
            raise ValueError("Dwell weight must be finite and at least one")
        if not self._accept_time(now):
            return False
        sample_time = now if observed_at is None else observed_at
        if (not math.isfinite(sample_time) or not 0 <= now - sample_time <= self.pose_max_age + 1e-9):
            self.invalidate_tracking(now)
            return False
        if self._last_observation is not None and sample_time <= self._last_observation:
            self.invalidate_tracking(now)
            return False
        if self._last_observation is not None and sample_time - self._last_observation > self.pose_max_age + 1e-9:
            self.invalidate_tracking(now)
        if (self._locked_target is None and self._sample_time is None
                and self._last_valid_sample is not None
                and sample_time - self._last_valid_sample > self.tracking_grace_seconds + 1e-9):
            self._clear_candidate()
        self._last_observation = sample_time
        if movement_released:
            self.clear_lock(now)
        if bool(armed) != self._armed:
            self._clear_candidate()
            self._keyword_time, self._pending = None, None
            self._armed = bool(armed)
            self._armed_at = now if armed else None
        observed_pose = bool(pose_ok) if raw_pose_ok is None else bool(raw_pose_ok)
        if not observed_pose and release_observed is False:
            self.invalidate_tracking(now)
            return False
        # Classification grace never delays observed release or permits a
        # command on a frame whose actual hand shape failed the classifier.
        command_pose = bool(pose_ok) and observed_pose
        if not observed_pose:
            self._keyword_time, self._pending = None, None
        # A gap in observations cannot prove a deliberate release.
        if not observed_pose:
            if self._release_start is None:
                self._release_start = sample_time
            if sample_time - self._release_start >= self.release_seconds - 1e-9:
                self._latched = False
                self._consumed_targets.clear()
                self._keyword_time = None
                self._locked_target, self._locked_at = None, None
                self._clear_candidate()
        else:
            # A positive endpoint does not prove that the unobserved time
            # since one negative frame was continuously released. Only an
            # actual negative observation can complete the release interval.
            self._release_start = None
        self._pose_ok, self._last_pose = command_pose, sample_time
        if not armed:
            self._clear_candidate()
            self._keyword_time, self._pending = None, None
            return False
        if not pose_ok:
            # A single missed gesture must not erase earlier valid visits.
            # Keep the observed-release interval, but break dwell continuity.
            self._target, self._dwell_start = None, None
            self._sample_time, self._sample_target = None, None
            self.ready = False
            self._keyword_time, self._pending = None, None
            self._refresh_dwell(sample_time)
            return False
        if self._locked_target is None and self._latched:
            # Once its file is removed/cancelled, the same consumed hold
            # cannot visually acquire another file before observed release.
            self._clear_candidate()
            self._keyword_time, self._pending = None, None
            return False
        if self._locked_target is not None:
            if self._target is None:
                self._target, self._dwell_start = self._locked_target, now
            self.progress, self.ready = 1.0, not self._latched and command_pose
        else:
            if target_id in self._consumed_targets:
                target_id = None
            self._record_dwell(sample_time, dwell_weight)
            if target_id != self._target:
                self._target, self._dwell_start = target_id, now
                self._keyword_time, self._pending = None, None
            self._sample_target, self._sample_time = target_id, sample_time
            self._sample_weight = dwell_weight
            self._refresh_dwell(sample_time)
            longest = max(self._dwell_times.values(), default=0.0)
            winners = [target for target, duration in self._dwell_times.items()
                       if abs(duration - longest) <= 1e-9]
            if len(winners) == 1 and winners[0] is not None and longest >= self.minimum_dwell_seconds - 1e-9:
                self._locked_target, self._locked_at = winners[0], now
                self._target, self._dwell_start = self._locked_target, now
                self._keyword_time, self._pending = None, None
                self.ready, self.progress = not self._latched and command_pose, 1.0
                self._dwell_history.clear()
                self._sample_time, self._sample_target = None, None
        # Visual dwell may continue through a near miss, while audio captured
        # before or during that pause must not survive the first actual match.
        if not command_pose:
            self._dwell_start = None
        elif self._dwell_start is None:
            self._dwell_start = now
        self._last_valid_sample = sample_time
        triggered = self._try_trigger(now)
        if triggered is not None:
            self._pending = triggered
        return self.ready

    def hear_keyword(self, now: float, captured_at: float | None = None) -> str | None:
        """Use dispatch time for pose freshness, audio time for keyword age.

        Offline speech recognition may return after newer camera frames.
        An audio token is eligible only if it was captured during this same
        target's uninterrupted fresh tracking and arming interval. A delayed token
        cannot transfer to a newly aimed file or survive disarm/rearm.
        A retained lock resumes immediately after a pause, but requires audio
        captured after its first fresh resumed frame.
        """
        if not self._accept_time(now):
            return None
        self._expire_pose(now)
        if not self._armed or self._latched or not self._pose_ok or self._target is None:
            return None
        captured = now if captured_at is None else captured_at
        if (not math.isfinite(captured) or captured > now
                or now - captured > self.keyword_max_age + 1e-9
                or self._dwell_start is None or captured < self._dwell_start
                or self._armed_at is None or captured < self._armed_at):
            return None
        self._keyword_time = captured
        return self._try_trigger(now)

    def invalidate_tracking(self, now: float, *, discard_dwell: bool = False) -> None:
        """Invalidate missing/stale tracking without inventing a pose release.

        Short interruptions break counting continuity but retain recent
        valid dwell. A long interruption or discard_dwell clears that history.
        Audio and queued actions always stop; a completed lock and its
        consumption survive. Watchdogs cannot count as observed release.
        """
        if not self._accept_time(now):
            return
        if (discard_dwell or self._last_valid_sample is not None
                and now - self._last_valid_sample > self.tracking_grace_seconds + 1e-9):
            self._clear_candidate()
        else:
            self._target, self._dwell_start = None, None
            self._sample_time, self._sample_target = None, None
            self.ready = False
            self._refresh_dwell(now)
        self._pose_ok = False
        self._release_start = None
        self._keyword_time = None
        self._pending = None

    def take_trigger(self, now: float | None = None) -> str | None:
        """Consume a queued trigger; optionally recheck freshness at dispatch.

        A UI processing a queued event later should pass its current
        monotonic time, especially before dispatching a live file action.
        """
        if now is not None:
            if not self._accept_time(now):
                return None
            self._expire_pose(now)
        pending, self._pending = self._pending, None
        return pending
