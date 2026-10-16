"""Personal hand-shape templates; no camera, GUI, speech or file actions.

Enrollment uses joint angles and palm-normalized distances, independent of
the built-in KON classifier. The caller records two fresh holds with
GestureEnrollmentBuffer; fit/confirm cannot infer freshness from bare arrays.
Only a confirmed numeric template can be persisted. An installed world
template never falls back to an image pose or the built-in classifier.
Recorded angles remain separate confirmed templates: centers and matching
bandwidths are never averaged or widened when an angle is appended.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from .geometry import GestureResult

FEATURE_NAMES = tuple(
    name + "_" + feature
    for name in ("index", "middle", "ring", "pinky")
    for feature in ("pip_cosine", "dip_cosine", "straightness", "radial_margin")
) + ("middle_thumb_contact", "ring_thumb_contact", "middle_thumb_tip",
     "ring_thumb_tip", "index_thumb_tip", "pinky_thumb_tip", "thumb_opposition",
     "thumb_straightness")
# Dimensionless matching bandwidths. Stable enrollment may adapt each by at
# most 20%; an unsteady hold cannot train an arbitrarily permissive template.
_BASE_WIDTH = np.array((.30, .35, .12, .18) * 4 + (.16, .16, .20, .20, .22, .22, .20, .15))
# Apply the same 35% extra tolerance to matching and second-hold confirmation,
# including existing v1/v2 profiles. Stored widths and enrollment stability
# remain unchanged so camera noise cannot compound this allowance.
_MATCH_TOLERANCE = 1.35
_GROUPS = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 24))
_GROUP_LABELS = ("食指", "中指", "無名指", "小指", "拇指接觸位置")
_MIN_SAMPLES = 12
STANDARD_GESTURE_ANGLES = ("中間", "上", "下", "左", "右")
_MAX_ANGLES = len(STANDARD_GESTURE_ANGLES)


def _side(value: str) -> str:
    value = str(value).strip().lower()
    if value not in ("left", "right", "unknown"):
        raise ValueError("左右手資料無效")
    return value


def _source(value: str) -> str:
    if value not in ("world", "normalized"):
        raise ValueError("手勢資料來源無效")
    return value


def _points(points: Sequence[Sequence[float]]) -> np.ndarray:
    try:
        result = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("手部關節資料不完整") from exc
    if result.shape != (21, 3) or not np.isfinite(result).all():
        raise ValueError("需要完整且有效的 21 個手部關節")
    return result


def _contact(point: np.ndarray, thumb: np.ndarray) -> float:
    distances = []
    for start, end in zip(thumb[:-1], thumb[1:]):
        direction = end - start
        position = float(np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1))
        distances.append(float(np.linalg.norm(point - start - position * direction)))
    return min(distances)


def extract_gesture_features(points: Sequence[Sequence[float]], *, source: str = "world",
                             aspect_ratio: float = 1.0) -> tuple[float, ...]:
    """Return 24 rotation/translation/uniform-scale invariant numeric features.

    World coordinates use meters or any consistent uniform unit. Normalized
    image coordinates require width/height correction for x and z. Invalid
    or collapsed finger bones raise ValueError instead of making a template.
    """
    source = _source(source)
    pts = _points(points)
    if source == "normalized":
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("鏡頭畫面比例無效")
        pts = pts * [aspect_ratio, 1, aspect_ratio]
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.linalg.norm(pts[9] - pts[0]))
        radius = np.linalg.norm(pts - pts[0], axis=1)
    if not math.isfinite(scale) or scale < 1e-6 or not np.isfinite(radius).all():
        raise ValueError("手掌追蹤不完整，請讓手腕與掌心清楚入鏡")
    features = []
    for indices, label in zip(((5, 6, 7, 8), (9, 10, 11, 12),
                               (13, 14, 15, 16), (17, 18, 19, 20)), _GROUP_LABELS):
        chain = pts[list(indices)]
        bones = np.diff(chain, axis=0)
        lengths = np.linalg.norm(bones, axis=1)
        if (not np.isfinite(lengths).all() or np.any(lengths < max(scale * .001, 1e-8))
                or np.any(lengths > scale * 3)):
            raise ValueError(f"{label}關節追蹤不完整，請讓整隻手清楚入鏡")
        cosines = np.clip(np.sum(bones[:-1] * bones[1:], axis=1)
                          / (lengths[:-1] * lengths[1:]), -1, 1)
        features.extend((float(cosines[0]), float(cosines[1]),
                         float(np.linalg.norm(chain[-1] - chain[0]) / sum(lengths)),
                         float((radius[indices[-1]] - radius[indices[1]]) / scale)))
    thumb = pts[2:5]
    lengths = np.linalg.norm(np.diff(thumb, axis=0), axis=1)
    if (not np.isfinite(lengths).all() or np.any(lengths < max(scale * .001, 1e-8))
            or np.any(lengths > scale * 3)):
        raise ValueError("拇指關節追蹤不完整，請讓拇指清楚入鏡")
    folded_center = (pts[12] + pts[16]) / 2
    features.extend((_contact(pts[12], thumb) / scale, _contact(pts[16], thumb) / scale,
                     *(float(np.linalg.norm(pts[tip] - pts[4]) / scale) for tip in (12, 16, 8, 20)),
                     float((np.linalg.norm(pts[2] - folded_center)
                            - np.linalg.norm(pts[4] - folded_center)) / scale),
                     float(np.linalg.norm(thumb[-1] - thumb[0]) / sum(lengths))))
    result = tuple(float(value) for value in features)
    _check_center(np.array(result))
    return result


def _check_center(center: np.ndarray) -> None:
    if center.shape != (len(FEATURE_NAMES),) or not np.isfinite(center).all():
        raise ValueError("個人手勢特徵資料無效")
    for start, _ in _GROUPS[:4]:
        if (np.any(np.abs(center[start:start + 2]) > 1 + 1e-9)
                or not 0 <= center[start + 2] <= 1 + 1e-9
                or abs(center[start + 3]) > 8):
            raise ValueError("個人手勢關節特徵超出有效範圍")
    if (np.any(center[16:22] < 0) or np.any(center[16:22] > 8)
            or abs(center[22]) > 8 or not 0 <= center[23] <= 1 + 1e-9):
        raise ValueError("個人手勢拇指特徵超出有效範圍")


def _distances(features: np.ndarray, center: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.abs(features - center) / width
    groups = np.stack([np.sqrt(np.mean(delta[..., start:end] ** 2, axis=-1))
                       for start, end in _GROUPS], axis=-1)
    return groups, np.max(delta, axis=-1)


def _match_distances(features: np.ndarray, center: np.ndarray,
                     width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _distances(features, center, width * _MATCH_TOLERANCE)


def _stable(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust center plus bounded noise; an isolated spike is not training."""
    if len(features) < _MIN_SAMPLES:
        raise ValueError("需要至少 12 筆穩定手勢，請持續保持姿勢")
    center = np.median(features, axis=0)
    sigma = 1.4826 * np.median(np.abs(features - center), axis=0)
    if np.any(sigma > _BASE_WIDTH * .60 + 1e-12):
        raise ValueError("手指姿勢仍在變動，請保持同一個召喚手勢")
    groups, peak = _distances(features, center, _BASE_WIDTH)
    inliers = (np.max(groups, axis=1) <= 1.25) & (peak <= 2.5)
    if np.count_nonzero(inliers) < _MIN_SAMPLES or np.mean(inliers) < .85:
        raise ValueError("手勢追蹤有跳動，請讓整隻手清楚入鏡並保持不動")
    stable = features[inliers]
    center = np.median(stable, axis=0)
    sigma = 1.4826 * np.median(np.abs(stable - center), axis=0)
    half = len(stable) // 2
    shift, peak_shift = _distances(np.median(stable[-half:], axis=0),
                                  np.median(stable[:half], axis=0), _BASE_WIDTH)
    if max(shift) > .80 or peak_shift > 1.5:
        raise ValueError("手勢在記錄期間改變，請穩定後重新記錄")
    width = np.clip(np.maximum(_BASE_WIDTH, sigma * 2), _BASE_WIDTH, _BASE_WIDTH * 1.2)
    return center, width, inliers


def _features_for_samples(samples: Sequence[Sequence[Sequence[float]]], source: str,
                          aspect_ratio: float) -> np.ndarray:
    try:
        frames = list(samples)
    except TypeError as exc:
        raise ValueError("個人手勢樣本資料無效") from exc
    if len(frames) < _MIN_SAMPLES or len(frames) > 2048:
        raise ValueError("每次需要至少 12 筆、至多 2048 筆手勢樣本")
    return np.array([extract_gesture_features(frame, source=source, aspect_ratio=aspect_ratio)
                     for frame in frames])


def _atomic_json_save(path: str | Path, payload: dict) -> None:
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


@dataclass(frozen=True)
class GestureCalibration:
    """Immutable personal template, confirmed by a separate fresh hold."""

    center: tuple[float, ...]
    bandwidth: tuple[float, ...]
    source: str
    handedness: str
    sample_count: int
    confirmation_sample_count: int = 0
    confirmed: bool = False

    @property
    def angle_count(self) -> int:
        return 1

    @classmethod
    def fit(cls, samples: Sequence[Sequence[Sequence[float]]], *, source: str = "world",
            handedness: str = "unknown", aspect_ratio: float = 1.0) -> "GestureCalibration":
        source, handedness = _source(source), _side(handedness)
        features = _features_for_samples(samples, source, aspect_ratio)
        center, width, inliers = _stable(features)
        return cls(tuple(center), tuple(width), source, handedness, int(np.count_nonzero(inliers)))

    def confirm(self, fresh_samples: Sequence[Sequence[Sequence[float]]], *, source: str = "world",
                handedness: str = "unknown", aspect_ratio: float = 1.0) -> "GestureCalibration":
        """Return a confirmed copy; the buffer must enforce capture freshness."""
        if self.confirmed:
            raise ValueError("個人手勢已完成確認；重新校正請重新記錄兩次")
        if _source(source) != self.source or _side(handedness) != self.handedness:
            raise ValueError("第二次確認請使用相同的手與相同的追蹤資料")
        features = _features_for_samples(fresh_samples, source, aspect_ratio)
        center, width, inliers = _stable(features)
        groups, peak = _match_distances(features[inliers], np.array(self.center), np.array(self.bandwidth))
        accepted = (np.max(groups, axis=1) <= 1) & (peak <= 2)
        center_groups, center_peak = _match_distances(center, np.array(self.center), np.array(self.bandwidth))
        if max(center_groups) > 1 or center_peak > 2 or np.mean(accepted) < .90:
            worst = _GROUP_LABELS[int(np.argmax(center_groups))]
            raise ValueError(f"第二次手勢與第一次不同（{worst}），請重做原來的召喚手勢")
        count = int(np.count_nonzero(inliers))
        combined = (np.array(self.center) * self.sample_count + center * count) / (self.sample_count + count)
        return replace(self, center=tuple(float(value) for value in combined),
                       bandwidth=tuple(float(value) for value in np.maximum(width, self.bandwidth)),
                       sample_count=self.sample_count + count,
                       confirmation_sample_count=count, confirmed=True)

    def match(self, points: Sequence[Sequence[float]], *, world_points: Sequence[Sequence[float]] | None = None,
              image_aspect_ratio: float = 1.0, source: str = "normalized",
              handedness: str = "unknown") -> GestureResult:
        """Match this template only; source/side failure never invokes a fallback.

        A provisional template can be previewed during enrollment. The app
        installs only confirmed models. Known enrolled handedness must match;
        an explicitly unknown template matches either side.
        """
        selected_source = "world" if world_points is not None else source
        selected = world_points if world_points is not None else points
        def failure(reason: str, hint: str) -> GestureResult:
            return GestureResult(0, False, reason=reason, source=selected_source, hint=hint)
        try:
            if _source(selected_source) != self.source:
                return failure("personal gesture source", "個人手勢需要原本的 3D 手部追蹤資料，請讓整隻手清楚入鏡")
            side = _side(handedness)
            if self.handedness != "unknown" and side != self.handedness:
                label = "左手" if self.handedness == "left" else "右手"
                return failure("personal gesture handedness", f"請使用校正時的{label}做出召喚手勢")
            features = np.array(extract_gesture_features(selected, source=selected_source,
                                                         aspect_ratio=image_aspect_ratio))
        except (TypeError, ValueError, OverflowError) as exc:
            return failure("personal gesture landmarks", str(exc))
        return self._match_features(features, selected_source)

    def _match_features(self, features: np.ndarray, selected_source: str) -> GestureResult:
        groups, peak = _match_distances(features, np.array(self.center), np.array(self.bandwidth))
        worst = int(np.argmax(groups))
        score = float(np.clip(min(1 - .4 * max(groups), 1 - .2 * peak), 0, 1))
        matched = max(groups) <= 1 + 1e-9 and peak <= 2 + 1e-9
        return GestureResult(score, bool(matched), thumb_contact_ratio=float(max(features[16:18])),
                             reason="personal gesture recognized" if matched else "personal gesture mismatch",
                             source=selected_source,
                             hint="個人召喚手勢已辨識" if matched else f"{_GROUP_LABELS[worst]}與校正姿勢不同，請重做你的召喚手勢",
                             finger_metrics=tuple(zip(FEATURE_NAMES, (float(value) for value in features))))

    def save(self, path: str | Path) -> None:
        if not self.confirmed:
            raise ValueError("請先完成第二次手勢確認，再儲存個人手勢")
        payload = self._payload()
        self._from_payload(payload)
        _atomic_json_save(path, payload)

    def _payload(self) -> dict:
        return {"version": 1, "feature_names": list(FEATURE_NAMES), "center": list(self.center),
                "bandwidth": list(self.bandwidth), "source": self.source, "handedness": self.handedness,
                "sample_count": self.sample_count, "confirmation_sample_count": self.confirmation_sample_count,
                "confirmed": self.confirmed}

    @classmethod
    def load(cls, path: str | Path) -> "GestureCalibration | GestureAngleCalibration":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and type(payload.get("version")) is int and payload["version"] == 2:
                return GestureAngleCalibration._from_payload(payload)
            return cls._from_payload(payload)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError("個人手勢校正檔無效，請重新校正手勢") from exc

    @classmethod
    def _from_payload(cls, payload: dict) -> "GestureCalibration":
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int or payload.get("version") != 1
                or payload.get("feature_names") != list(FEATURE_NAMES) or payload.get("confirmed") is not True):
            raise ValueError("個人手勢校正檔格式無效")
        for key in ("center", "bandwidth"):
            if (not isinstance(payload.get(key), list)
                    or any(type(value) not in (int, float) for value in payload[key])):
                raise ValueError("個人手勢數值資料無效")
        center, width = np.array(payload["center"], dtype=float), np.array(payload["bandwidth"], dtype=float)
        _check_center(center)
        if (width.shape != _BASE_WIDTH.shape or not np.isfinite(width).all()
                or np.any(width < _BASE_WIDTH - 1e-12) or np.any(width > _BASE_WIDTH * 1.2 + 1e-12)):
            raise ValueError("個人手勢容許範圍無效")
        count, confirmation = payload["sample_count"], payload["confirmation_sample_count"]
        if (type(count) is not int or type(confirmation) is not int
                or not _MIN_SAMPLES <= confirmation <= 2048 or not confirmation + _MIN_SAMPLES <= count <= 4096):
            raise ValueError("個人手勢確認樣本不足")
        return cls(tuple(float(value) for value in center), tuple(float(value) for value in width),
                   _source(payload["source"]), _side(payload["handedness"]), count, confirmation, True)


@dataclass(frozen=True)
class GestureAngleCalibration:
    """Up to five independent confirmed views; match any unchanged template."""

    models: tuple[GestureCalibration, ...]

    def __post_init__(self):
        try:
            models = tuple(self.models)
        except TypeError as exc:
            raise ValueError("個人手勢角度資料無效") from exc
        if not 2 <= len(models) <= _MAX_ANGLES:
            raise ValueError("個人手勢需有 2 至 5 個獨立角度")
        if any(not isinstance(model, GestureCalibration) or not model.confirmed for model in models):
            raise ValueError("每個角度都必須完成兩次手勢確認")
        for model in models:
            if not isinstance(model.center, tuple) or not isinstance(model.bandwidth, tuple):
                raise ValueError("手勢角度的數值模板必須不可變")
            GestureCalibration._from_payload(model._payload())
            if model.handedness != _side(model.handedness):
                raise ValueError("個人手勢左右手資料無效")
        if any(model.source != models[0].source or model.handedness != models[0].handedness for model in models):
            raise ValueError("新增角度必須使用相同的手與相同的追蹤資料")
        object.__setattr__(self, "models", models)

    @property
    def angle_count(self) -> int:
        return len(self.models)

    @property
    def source(self) -> str:
        return self.models[0].source

    @property
    def handedness(self) -> str:
        return self.models[0].handedness

    @property
    def confirmed(self) -> bool:
        return True

    @property
    def sample_count(self) -> int:
        return sum(model.sample_count for model in self.models)

    @property
    def confirmation_sample_count(self) -> int:
        return sum(model.confirmation_sample_count for model in self.models)

    def match(self, points: Sequence[Sequence[float]], *, world_points: Sequence[Sequence[float]] | None = None,
              image_aspect_ratio: float = 1.0, source: str = "normalized",
              handedness: str = "unknown") -> GestureResult:
        first = self.models[0].match(points, world_points=world_points, image_aspect_ratio=image_aspect_ratio,
                                     source=source, handedness=handedness)
        if first.reason not in ("personal gesture recognized", "personal gesture mismatch"):
            return first  # Invalid source, side or landmarks cannot try another view.
        features = np.array([value for _, value in first.finger_metrics])
        results = [first] + [model._match_features(features, first.source) for model in self.models[1:]]
        distances = []
        for model in self.models:
            groups, peak = _match_distances(features, np.array(model.center), np.array(model.bandwidth))
            # This is the unclipped equivalent of score: failures whose score
            # is already zero still get the genuinely nearest template's hint.
            distances.append(max(float(max(groups)), float(peak) / 2))
        eligible = [index for index, result in enumerate(results) if result.is_gao]
        candidates = eligible or list(range(len(results)))
        return results[min(candidates, key=lambda index: distances[index])]

    def save(self, path: str | Path) -> None:
        payload = {"version": 2, "model": "personal_gesture_angles", "confirmed": True,
                   "source": self.source, "handedness": self.handedness,
                   "angle_count": self.angle_count, "models": [model._payload() for model in self.models]}
        self._from_payload(payload)
        _atomic_json_save(path, payload)

    @classmethod
    def _from_payload(cls, payload: dict) -> "GestureAngleCalibration":
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 2
                or payload.get("model") != "personal_gesture_angles" or payload.get("confirmed") is not True
                or not isinstance(payload.get("models"), list)):
            raise ValueError("多角度個人手勢校正檔格式無效")
        models = payload["models"]
        if not 2 <= len(models) <= _MAX_ANGLES:
            raise ValueError("多角度手勢的角度數量無效")
        if type(payload.get("angle_count")) is not int or payload["angle_count"] != len(models):
            raise ValueError("多角度手勢的角度數量不一致")
        result = cls(tuple(GestureCalibration._from_payload(model) for model in models))
        if _source(payload["source"]) != result.source or _side(payload["handedness"]) != result.handedness:
            raise ValueError("多角度手勢的資料來源或左右手不一致")
        return result


def add_gesture_angle(existing_profile: GestureCalibration | GestureAngleCalibration,
                      confirmed_new: GestureCalibration) -> GestureAngleCalibration:
    """Append one two-hold-confirmed angle without changing old templates/files."""
    if not isinstance(confirmed_new, GestureCalibration) or not confirmed_new.confirmed:
        raise ValueError("新增角度必須先完成兩次獨立的手勢確認")
    if isinstance(existing_profile, GestureCalibration):
        models = (existing_profile,)
    elif isinstance(existing_profile, GestureAngleCalibration):
        models = existing_profile.models
    else:
        raise ValueError("請先完成第一個召喚手勢角度的校正")
    if len(models) >= _MAX_ANGLES:
        raise ValueError("最多保存 5 個召喚手勢角度，請重新校正以替換")
    return GestureAngleCalibration((*models, confirmed_new))


@dataclass(frozen=True)
class GestureCaptureState:
    ready: bool
    samples: tuple[tuple[tuple[float, float, float], ...], ...]
    count: int
    required: int
    span: float
    reason: str
    source: str = "world"
    handedness: str = "unknown"


class GestureEnrollmentBuffer:
    """One fresh held pose, used independently for each enrollment capture."""

    def __init__(self, min_samples: int = 12, min_span: float = 1.0, window: float = 4.5,
                 max_age: float = .35) -> None:
        if type(min_samples) is not int or min_samples < 12:
            raise ValueError("手勢校正需要至少 12 個畫面")
        if (any(not math.isfinite(value) or value <= 0 for value in (min_span, window, max_age))
                or min_span < 1 or window < min_span or max_age > window):
            raise ValueError("手勢校正時間設定無效")
        self.min_samples, self.min_span, self.window, self.max_age = min_samples, min_span, window, max_age
        self._samples: deque = deque(maxlen=max(256, min_samples * 8))
        self.reset(-1e100)

    def reset(self, step_started_at: float) -> None:
        if not math.isfinite(step_started_at):
            raise ValueError("手勢記錄時間無效")
        self._samples.clear()
        self._started = step_started_at
        self._latest: float | None = None
        self._latest_valid = False
        self._reason = ""
        self._handedness = "unknown"

    @property
    def has_observation(self) -> bool:
        """Whether tracking has supplied any fresh frame since reset."""
        return self._latest is not None

    def observe(self, timestamp: float, points: Sequence[Sequence[float]] | None,
                *, source: str = "world", handedness: str = "unknown", aspect_ratio: float = 1.0,
                unavailable_reason: str = "") -> bool:
        if (not math.isfinite(timestamp) or timestamp < self._started
                or self._latest is not None and timestamp <= self._latest):
            return False
        if self._latest is not None and timestamp - self._latest > .50:
            self._samples.clear()
        self._latest, self._latest_valid = timestamp, False
        try:
            if unavailable_reason:
                raise ValueError(str(unavailable_reason))
            if _source(source) != "world":
                raise ValueError("手勢校正需要 3D 關節，請讓整隻手清楚入鏡")
            side = _side(handedness)
            if side == "unknown":
                raise ValueError("尚未辨識左右手，請讓整隻手清楚入鏡")
            features = extract_gesture_features(points, source=source, aspect_ratio=aspect_ratio)
            raw = tuple(tuple(float(value) for value in point) for point in _points(points))
            if self._handedness not in ("unknown", side):
                self._samples.clear()
            self._handedness = side
            self._samples.append((timestamp, raw, features))
            self._latest_valid = True
            self._reason = ""
        except (TypeError, ValueError, OverflowError) as exc:
            self._reason = str(exc)
        self._prune(timestamp)
        return True

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window + 1e-9:
            self._samples.popleft()

    def evaluate(self, now: float) -> GestureCaptureState:
        def state(reason: str, samples=(), span=0., count=None):
            return GestureCaptureState(bool(samples), samples,
                                       len(self._samples) if count is None else count,
                                       self.min_samples, span, reason, "world", self._handedness)
        if not math.isfinite(now):
            return state("鏡頭時間無效，請重新啟動鏡頭")
        self._prune(now - self.max_age)
        if self._latest is None:
            return state("請做出你的召喚手勢，讓整隻手清楚入鏡並保持至少一秒")
        age = now - self._latest
        if age < -1e-9 or age > self.max_age + 1e-9:
            return state("鏡頭畫面已過期，請等待新的手部追蹤畫面")
        if not self._latest_valid:
            return state(self._reason or "尚未取得完整的 3D 手部關節")
        if len(self._samples) < self.min_samples:
            return state(f"已取得 {len(self._samples)} / {self.min_samples} 筆手勢，請繼續保持")
        features = np.array([item[2] for item in self._samples])
        try:
            _, _, inliers = _stable(features)
        except ValueError as exc:
            return state(str(exc))
        stable = [item for item, valid in zip(self._samples, inliers) if valid]
        if not inliers[-1]:
            return state("最新手勢有跳動，請保持姿勢並等待下一個畫面")
        span = float(stable[-1][0] - stable[0][0])
        if len(stable) < self.min_samples:
            return state("穩定手勢畫面還不足，請繼續保持", span=span, count=len(stable))
        if span < self.min_span - 1e-9:
            return state("已取得手勢，請保持至少一秒再記錄", span=span, count=len(stable))
        return state(f"已取得 {len(stable)} 筆穩定手勢，可以記錄", tuple(item[1] for item in stable),
                     span, len(stable))
