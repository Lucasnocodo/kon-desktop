"""Local hand-cursor tracking, with optional legacy eye-through-hand aiming.

Image landmarks remain UNMIRRORED. Only the preview is mirrored. Every metric
point uses OpenCV optical axes: x image-right, y image-down, z away, in meters.
Hand mode uses the mirrored image-circle center without eye/depth estimation.
Legacy mode's aiming line is eye -> hand aperture, not an eyeball gaze estimate.
Iris diameter (11.7 mm), predicted hand size and an assumed camera FOV supply
approximate scale; screen calibration is still required. In particular, the
face's normalized landmark z is never mixed with hand-world coordinates.
"""

from __future__ import annotations

from .paths import resource_root

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
from typing import Sequence

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .geometry import classify_gao, ray_screen_intersection, ring_center
from .gesture_calibration import extract_gesture_features
from .gesture_stability import GestureStabilizer

XYZ = tuple[float, float, float]
XY = tuple[float, float]
MODEL_DIR = resource_root() / "models"
SMOOTHING_HISTORY_MAX_GAP = 0.75
MAX_PROCESSING_FPS = 30.0
MAX_PREVIEW_FPS = 15.0


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    timestamp: float
    hand_points: tuple[XYZ, ...]
    face_points: tuple[XYZ, ...]
    gesture_score: float
    pose_ok: bool
    eye_xyz: XYZ | None
    aperture_xyz: XYZ | None
    raw_hit: XY | None
    message: str
    hand_detected: bool = False
    eye_detected: bool = False
    depth_status: str = ""
    hand_depth_error: str = ""
    eye_error: str = ""
    gesture_reason: str = ""
    source_fps: float | None = None
    source_size: tuple[int, int] | None = None
    inference_size: tuple[int, int] | None = None
    gesture_hint: str = ""
    gesture_source: str = "none"
    # Raw hand-centered 3D shape for enrollment, never camera-frame depth.
    hand_world_points: tuple[XYZ, ...] = ()
    handedness: str = "unknown"
    hand_cursor: XY | None = None
    aiming_mode: str = "eye_ray"
    # None preserves legacy/synthetic frames; camera frames always expose the
    # actual classifier evidence separately from the brief recognition grace.
    raw_pose_ok: bool | None = None


def camera_matrix(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Pinhole intrinsics from a horizontal FOV, not a measured calibration."""
    if width <= 0 or height <= 0 or not 20.0 <= fov_degrees <= 120.0:
        raise ValueError("鏡頭尺寸或水平視角無效")
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resize_for_inference(bgr: np.ndarray, max_width: int = 960, max_height: int = 540) -> np.ndarray:
    """Downsize the whole source image, preserving its aspect and field of view.

    A capture-size request is advisory. Never stretch or crop a delivered 4:3
    image into 16:9, and never upscale a smaller camera mode. Dimension rounding
    is at most one pixel; intrinsics use the actual resulting dimensions.
    """
    if bgr.ndim < 2 or min(bgr.shape[:2]) <= 0 or max_width <= 0 or max_height <= 0:
        raise ValueError("影像尺寸或推論尺寸無效")
    height, width = bgr.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return bgr
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _landmark_tuples(landmarks: Sequence) -> tuple[XYZ, ...]:
    return tuple((float(p.x), float(p.y), float(p.z)) for p in landmarks)


def hand_circle_cursor(points: Sequence[XYZ], *, diagnostics: dict[str, str] | None = None) -> XY | None:
    """Mirror the image aperture center; never interpret image z as depth."""
    def fail(reason):
        if diagnostics is not None:
            diagnostics["hand_cursor_error"] = reason
        return None

    try:
        hand = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return fail("手部追蹤點無效；請讓整隻手清楚入鏡")
    if hand.shape != (21, 3) or not np.isfinite(hand).all():
        return fail("手部追蹤點不完整；請讓整隻手清楚入鏡")
    if np.any(hand[:, :2] < 0) or np.any(hand[:, :2] > 1):
        return fail("手部部分超出畫面；請讓整隻手完整入鏡")
    if np.linalg.norm(hand[0, :2] - hand[9, :2]) < 1e-6:
        return fail("手掌追蹤不完整；請讓手腕與掌心清楚入鏡")
    center = ring_center(hand)
    if center is None or not 0 <= center[0] <= 1 or not 0 <= center[1] <= 1:
        return fail("手圈中心無法估計；請讓手圈清楚入鏡")
    return (float(1 - center[0]), float(center[1]))


def estimate_eye_xyz(
    face_points: Sequence[XYZ], width: int, height: int,
    intrinsics: np.ndarray, dominant_eye: str,
    *, diagnostics: dict[str, str] | None = None,
) -> XYZ | None:
    """Back-project the chosen iris using its approximate physical diameter.

    MediaPipe right iris: center 468, perimeter 469..472; left: 473, 474..477.
    These are anatomical sides on an unmirrored image, not preview sides.
    Closed eyelids, weak pixel resolution and inconsistent two-eye scale fail
    closed. We do not substitute the other eye when the chosen eye is hidden.
    """
    if diagnostics is not None:
        diagnostics.pop("eye_error", None)

    def fail(reason: str):
        if diagnostics is not None:
            diagnostics["eye_error"] = reason
        return None

    if len(face_points) < 478:
        return fail("尚未取得虹膜點；請讓臉部清楚入鏡")
    if dominant_eye not in ("right", "left"):
        return fail("慣用眼設定無效")
    points = np.asarray(face_points, dtype=np.float64)
    if not np.isfinite(points).all():
        return fail("眼睛追蹤點無效，請改善光線並面向鏡頭")
    pixels = points[:, :2] * np.array([width, height], dtype=np.float64)
    definitions = {
        "right": (468, (469, 470, 471, 472), (33, 133), (159, 145)),
        "left": (473, (474, 475, 476, 477), (263, 362), (386, 374)),
    }

    def iris_measure(side: str) -> tuple[tuple[np.ndarray, float] | None, str]:
        side_name = "右" if side == "right" else "左"
        center_index, ring_indices, corners, lids = definitions[side]
        center = pixels[center_index]
        ring = pixels[list(ring_indices)]
        eye_width = float(np.linalg.norm(pixels[corners[0]] - pixels[corners[1]]))
        opening = float(np.linalg.norm(pixels[lids[0]] - pixels[lids[1]]))
        if eye_width < 9.0:
            return None, f"{side_name}眼在影像中太小；請靠近鏡頭並面向鏡頭"
        if opening / eye_width < 0.12:
            return None, f"{side_name}眼可能閉合或被遮擋；請睜眼並避開手部遮擋"
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            return None, f"{side_name}眼虹膜在畫面外，請調整坐姿"
        # Opposite ring points give two projected diameters. Taking the major
        # axis reduces the depth bias from a foreshortened iris ellipse.
        diameter = max(float(np.linalg.norm(ring[0] - ring[2])),
                       float(np.linalg.norm(ring[1] - ring[3])))
        if diameter < 3.5:
            return None, f"{side_name}眼虹膜太小（{diameter:.1f} px）；請靠近鏡頭並改善光線"
        if diameter > eye_width * 0.82:
            return None, f"{side_name}眼虹膜輪廓不一致；請面向鏡頭並改善光線"
        if np.linalg.norm(center - np.mean(ring, axis=0)) > diameter * 0.3:
            return None, f"{side_name}眼虹膜中心不穩；請避免遮擋並改善光線"
        depth = float(intrinsics[0, 0] * 0.0117 / diameter)
        if not 0.18 <= depth <= 2.5:
            return None, f"{side_name}眼估計距離超出可用範圍；請調整與鏡頭的距離"
        return (center, depth), ""

    chosen, reason = iris_measure(dominant_eye)
    if chosen is None:
        return fail(reason)
    other, _ = iris_measure("left" if dominant_eye == "right" else "right")
    if other is not None and max(chosen[1], other[1]) / min(chosen[1], other[1]) > 1.4:
        return fail("左右眼估計距離差異過大；請正面看向鏡頭並改善光線")
    center, depth = chosen
    x = (center[0] - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (center[1] - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    return float(x), float(y), depth


def estimate_hand_camera_points(
    hand_points: Sequence[XYZ], world_points: Sequence[XYZ],
    width: int, height: int, intrinsics: np.ndarray,
    *, diagnostics: dict[str, str] | None = None,
) -> tuple[XYZ, ...] | None:
    """Fit metric hand-world landmarks to image pixels through a rigid PnP.

    World landmarks are centered on the hand, so they cannot be interpreted
    directly as camera positions. R/t from PnP explicitly transform them into
    the camera coordinate frame. Reprojection and depth gates reject bad fits.
    """
    if diagnostics is not None:
        diagnostics.pop("hand_depth_error", None)

    def fail(reason: str):
        if diagnostics is not None:
            diagnostics["hand_depth_error"] = reason
        return None

    if len(hand_points) != 21 or len(world_points) != 21:
        return fail("手部追蹤點或公尺座標不完整；請讓整隻手清楚入鏡")
    objects = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(hand_points, dtype=np.float64)[:, :2] * [width, height]
    if not np.isfinite(objects).all() or not np.isfinite(image).all():
        return fail("手部追蹤點無效；請改善光線並避免手指遮擋")
    if np.any(image < -0.025 * max(width, height)) or np.any(image > [width * 1.025, height * 1.025]):
        return fail("手部部分追蹤點超出畫面；請讓整隻手入鏡")
    if not 0.035 <= np.linalg.norm(objects[0] - objects[9]) <= 0.18:
        return fail("估計手掌尺寸不合理；請讓手掌與手指清楚入鏡")
    distortion = np.zeros((4, 1), dtype=np.float64)
    threshold = max(3.0, width * 0.008)
    try:
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            objects, image, intrinsics, distortion,
            iterationsCount=80, reprojectionError=threshold, confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 15:
            return fail("手部投影對應不足，無法估計深度；請讓整隻手清楚入鏡")
        indices = inliers.ravel()
        ok, rotation, translation = cv2.solvePnP(
            objects[indices], image[indices], intrinsics, distortion,
            rvec=rotation, tvec=translation, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return fail("手部深度擬合失敗；請保持手部穩定並避免遮擋")
        projected, _ = cv2.projectPoints(objects, rotation, translation, intrinsics, distortion)
        residuals = np.linalg.norm(projected[:, 0, :] - image, axis=1)
        if float(np.median(residuals)) > threshold or float(np.percentile(residuals, 90)) > threshold * 1.8:
            return fail("手部投影誤差過大；請避免遮擋並保持整隻手清楚")
        matrix, _ = cv2.Rodrigues(rotation)
        camera_points = (matrix @ objects.T).T + translation.reshape(1, 3)
        if not np.isfinite(camera_points).all():
            return fail("手部深度計算無效，請重新調整手部位置")
        if np.min(camera_points[:, 2]) < 0.06 or np.max(camera_points[:, 2]) > 1.7:
            return fail("手部估計距離超出可用範圍；請調整與鏡頭的距離")
        return tuple(tuple(float(v) for v in p) for p in camera_points)
    except cv2.error:
        return fail("手部深度求解失敗；請讓整隻手清楚入鏡")


class _StablePoint:
    """Warm up across fresh low-rate observations without reusing missing data.

    History continuity is separate from source freshness: a fresh frame every
    333 ms should reach three samples. The worker/UI still reject old frames at
    their original 0.35/0.3 s gates. None, jumps or a long gap reset immediately.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.point: np.ndarray | None = None
        self.timestamp = 0.0
        self.count = 0

    def update(self, sample: XYZ | None, timestamp: float) -> XYZ | None:
        if sample is None:
            self.reset()
            return None
        value = np.asarray(sample, dtype=np.float64)
        dt = timestamp - self.timestamp
        if (self.point is None or dt > SMOOTHING_HISTORY_MAX_GAP or dt <= 0 or
                np.linalg.norm(value - self.point) > 0.14 or abs(value[2] - self.point[2]) > 0.09):
            self.point = value
            self.count = 1
        else:
            alpha = 1.0 - math.exp(-dt / 0.075)
            self.point += alpha * (value - self.point)
            self.count += 1
        self.timestamp = timestamp
        return tuple(float(v) for v in self.point) if self.count >= 3 else None


class _LatestCapture(threading.Thread):
    """Camera-only producer with one replaceable frame, never a growing queue."""

    def __init__(self, capture: cv2.VideoCapture, stop_event: threading.Event) -> None:
        super().__init__(name="KON camera capture", daemon=True)
        self.capture = capture
        self.stop_event = stop_event
        self.condition = threading.Condition()
        self.latest: tuple[float, np.ndarray] | None = None
        self.failure: str | None = None

    def run(self) -> None:
        try:
            failures = 0
            while not self.stop_event.is_set():
                ok, pixels = self.capture.read()
                if not ok or pixels is None:
                    failures += 1
                    if failures >= 5:
                        self.failure = "鏡頭無法讀取影像；請檢查權限或是否被其他程式占用。"
                        break
                    self.stop_event.wait(0.05)
                    continue
                failures = 0
                with self.condition:
                    self.latest = (time.monotonic(), pixels)
                    self.condition.notify()
        except Exception as exc:
            self.failure = f"鏡頭讀取失敗：{exc}"
        finally:
            self.capture.release()
            with self.condition:
                self.condition.notify_all()

    def take(self) -> tuple[float, np.ndarray] | None:
        with self.condition:
            if self.latest is None and self.is_alive() and not self.stop_event.is_set():
                self.condition.wait(0.15)
            result, self.latest = self.latest, None
            return result


class CameraWorker(QThread):
    frame = Signal(QImage)
    tracking = Signal(object)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, camera_index: int = 0, fov_degrees: float = 60.0,
                 dominant_eye: str = "right", parent=None, *, gesture_profile=None,
                 aim_mode: str = "eye_ray") -> None:
        super().__init__(parent)
        if camera_index < 0:
            raise ValueError("請指定非負的鏡頭編號")
        if dominant_eye not in ("right", "left"):
            raise ValueError("dominant_eye 必須是 right 或 left")
        if not 20.0 <= fov_degrees <= 120.0:
            raise ValueError("fov_degrees 必須介於 20 與 120")
        if aim_mode not in ("eye_ray", "hand"):
            raise ValueError("aim_mode 必須是 eye_ray 或 hand")
        self.camera_index = int(camera_index)
        self.fov_degrees = float(fov_degrees)
        self.dominant_eye = dominant_eye
        self.aim_mode = aim_mode
        self._stop_event = threading.Event()
        self._eye_filter = _StablePoint()
        self._hand_filter = _StablePoint()
        self._gesture_stabilizer = GestureStabilizer()
        self._gesture_frame_size = None
        self._last_handedness: str | None = None
        self._last_hand_wrist: XY | None = None
        self._last_hand_timestamp = 0.0
        self._last_source_timestamp: float | None = None
        self._gesture_profile_lock = threading.Lock()
        self._gesture_profile = None
        self.set_gesture_profile(gesture_profile)
        self._preview_lock = threading.Lock()
        self._preview_enabled = True
        self._last_preview_timestamp: float | None = None

    def set_gesture_profile(self, profile) -> None:
        """Replace an immutable personal model for the next tracking frame.

        An installed profile is the command-pose authority. Its mismatch does
        not fall back to the generic pose. The short reference lock is never
        held during model matching or camera/geometry work.
        """
        if profile is not None and not callable(getattr(profile, "match", None)):
            raise TypeError("個人手勢設定必須提供 match 方法")
        with self._gesture_profile_lock:
            self._gesture_profile = profile

    def set_preview_enabled(self, enabled: bool) -> None:
        """Skip preview images when hidden without stopping source tracking.

        Calibration may use a preview while the main window is minimized;
        the app owns visibility policy. Re-enabling allows the next source
        frame immediately, and repeated enables do not bypass the rate cap.
        """
        value = bool(enabled)
        with self._preview_lock:
            if value != self._preview_enabled:
                self._preview_enabled = value
                self._last_preview_timestamp = None

    def _preview_due(self, source_timestamp: float) -> bool:
        if not math.isfinite(source_timestamp):
            return False
        with self._preview_lock:
            if not self._preview_enabled:
                return False
            previous = self._last_preview_timestamp
            if previous is not None and source_timestamp - previous < 1.0 / MAX_PREVIEW_FPS - 1e-9:
                return False
            self._last_preview_timestamp = source_timestamp
            return True

    def stop(self) -> None:
        """Nonblocking request; finished is emitted after camera/model cleanup."""
        self._stop_event.set()
        self.requestInterruption()

    def run(self) -> None:
        # Creating the worker does not request camera access; start() does.
        # Imports and both model instances stay off the Qt/UI thread.
        capture = None
        reader = None
        hand_model = face_model = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            hand_path = MODEL_DIR / "hand_landmarker.task"
            face_path = MODEL_DIR / "face_landmarker.task"
            hand_mode = self.aim_mode == "hand"
            for path in ((hand_path,) if hand_mode else (hand_path, face_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"缺少本機追蹤模型：{path.name}，請先完成安裝。")
            self.status.emit("正在載入本機手部模型…" if hand_mode else "正在載入本機手部與臉部模型…")
            base = python.BaseOptions
            hand_model = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base(model_asset_path=str(hand_path), delegate=base.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
            )
            if not hand_mode:
                face_model = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=base(model_asset_path=str(face_path), delegate=base.Delegate.CPU),
                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                )
            if self._stop_event.is_set():
                return
            self.status.emit(f"正在開啟鏡頭 {self.camera_index}…")
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.camera_index, backend)
            if not capture.isOpened():
                raise RuntimeError(f"無法開啟鏡頭 {self.camera_index}。請在 macOS 隱私權設定允許鏡頭存取。")
            # Prefer a standard 16:9 mode, then verify the delivered frame.
            # AVFoundation/driver can ignore or substitute these requests.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            reader = _LatestCapture(capture, self._stop_event)
            reader.start()
            self.status.emit("鏡頭已啟動；移動手圈來控制準星。" if hand_mode else
                             "鏡頭已啟動；眼睛與手孔深度為估計值，需先校正螢幕。")
            last_ms = -1
            last_source_size = None
            while not self._stop_event.is_set() and not self.isInterruptionRequested():
                started = time.monotonic()
                latest = reader.take()
                if latest is None:
                    if reader.failure:
                        raise RuntimeError(reader.failure)
                    if not reader.is_alive():
                        break
                    continue
                timestamp, bgr = latest
                if time.monotonic() - timestamp > 0.3:
                    continue
                source_size = (bgr.shape[1], bgr.shape[0])
                bgr = resize_for_inference(bgr)
                inference_size = (bgr.shape[1], bgr.shape[0])
                if source_size != last_source_size:
                    aspect = source_size[0] / source_size[1]
                    message = (f"鏡頭實際影像 {source_size[0]}×{source_size[1]}（比例 {aspect:.2f}:1）；"
                               f"推論 {inference_size[0]}×{inference_size[1]}，保留完整畫面")
                    if not math.isclose(aspect, 16 / 9, abs_tol=0.01):
                        message += "；非 16:9，視角可能因鏡頭裁切而不同"
                    self.status.emit(message)
                    last_source_size = source_size
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_ms + 1, int(timestamp * 1000))
                last_ms = timestamp_ms
                hand_result = hand_model.detect_for_video(image, timestamp_ms)
                face_result = face_model.detect_for_video(image, timestamp_ms) if face_model is not None else None
                if self._stop_event.is_set():
                    break
                observation = self._make_tracking(
                    hand_result, face_result, bgr.shape[1], bgr.shape[0], timestamp,
                    source_size=source_size,
                )
                # Do not publish a prediction whose source frame is already old.
                if time.monotonic() - timestamp <= 0.35:
                    self.tracking.emit(observation)
                    if self._preview_due(timestamp):
                        preview = self._preview(bgr, observation)
                        if not self._stop_event.is_set() and time.monotonic() - timestamp <= 0.35:
                            self.frame.emit(preview)
                else:
                    self._eye_filter.reset()
                    self._hand_filter.reset()
                    self._gesture_stabilizer.reset()
                    self.tracking.emit(TrackingFrame(time.monotonic(), (), (), 0.0, False, None, None, None,
                                                    "影像處理延遲，暫停瞄準；請降低其他程式的負載。",
                                                    aiming_mode=self.aim_mode))
                self._stop_event.wait(max(0.0, 1.0 / MAX_PROCESSING_FPS - (time.monotonic() - started)))
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error.emit(str(exc))
        finally:
            self._stop_event.set()
            camera_closed = True
            if reader is not None:
                reader.join(timeout=2.0)
                if reader.is_alive():
                    camera_closed = False
                    self.error.emit("鏡頭驅動尚未完成停止；請先關閉程式再重新啟動。")
            elif capture is not None:
                capture.release()
            for model in (hand_model, face_model):
                if model is not None:
                    try:
                        model.close()
                    except Exception:
                        pass
            stopped_message = "鏡頭已停止" if camera_closed else "已停止追蹤；鏡頭驅動仍在關閉。"
            self.tracking.emit(TrackingFrame(time.monotonic(), (), (), 0.0, False, None, None, None, stopped_message,
                                            aiming_mode=self.aim_mode))
            self.status.emit(stopped_message)

    def _make_tracking(
        self, hand_result, face_result, width: int, height: int, timestamp: float,
        *, source_size: tuple[int, int] | None = None,
    ) -> TrackingFrame:
        with self._gesture_profile_lock:
            gesture_profile = self._gesture_profile
        if (not math.isfinite(timestamp) or self._last_source_timestamp is not None
                and timestamp <= self._last_source_timestamp):
            self._gesture_stabilizer.reset()
            self._eye_filter.reset()
            self._hand_filter.reset()
            return TrackingFrame(
                timestamp, (), (), 0., False, None, None, None,
                "等待新的追蹤畫面；忽略重複、倒序或無效時間的影格",
                depth_status="invalid_timestamp", source_size=source_size or (width, height),
                inference_size=(width, height), aiming_mode=self.aim_mode, raw_pose_ok=False,
            )
        frame_size = (width, height, source_size)
        if frame_size != self._gesture_frame_size:
            self._gesture_stabilizer.reset()
            self._gesture_frame_size = frame_size
        source_gap = timestamp - self._last_source_timestamp if self._last_source_timestamp is not None else 0.0
        source_fps = 1.0 / source_gap if math.isfinite(source_gap) and source_gap > 0 else None
        self._last_source_timestamp = timestamp
        hand_mode = self.aim_mode == "hand"
        intrinsics = None if hand_mode else camera_matrix(width, height, self.fov_degrees)
        faces = () if hand_mode else face_result.face_landmarks
        face_points = _landmark_tuples(faces[0]) if faces else ()
        hands = hand_result.hand_landmarks
        hand_points: tuple[XYZ, ...] = ()
        hand_world_points: tuple[XYZ, ...] = ()
        handedness = "unknown"
        score = 0.0
        pose_ok = False
        raw_pose_ok = False
        aperture = None
        diagnostics: dict[str, str] = {}
        gesture_reason = ""
        gesture_hint = "沒有偵測到手部；請讓整隻手完整入鏡"
        gesture_source = "none"
        hand_stage = "hand_pnp_failed"
        if hands:
            candidates = []
            world = getattr(hand_result, "hand_world_landmarks", ()) or ()
            for index, points in enumerate(hands):
                normalized = _landmark_tuples(points)
                world_points = _landmark_tuples(world[index]) if index < len(world) else None
                identities = getattr(hand_result, "handedness", ())
                handedness = identities[index][0].category_name if index < len(identities) and identities[index] else "unknown"
                # Pose math prefers the hand's metric shape to perspective-
                # distorted image distances. The classifier validates world
                # points and falls back only if they are absent or unusable.
                # The aiming reconstruction below still uses the same PnP.
                if gesture_profile is not None:
                    gesture = gesture_profile.match(
                        normalized, image_aspect_ratio=width / height,
                        world_points=world_points, handedness=handedness,
                    )
                else:
                    gesture = classify_gao(
                        normalized, image_aspect_ratio=width / height,
                        world_points=world_points,
                    )
                pixels = np.asarray(normalized)[:, :2] * [width, height] if normalized else np.zeros((1, 2))
                area = float(np.prod(np.ptp(pixels, axis=0))) if np.isfinite(pixels).all() else 0.0
                candidates.append((float(gesture.score), area, index, normalized, gesture, handedness))
            # Gesture quality takes priority; image size breaks all-zero ties.
            # Keep a visible previous hand when classification is uncertain,
            # rather than jumping with MediaPipe's result-list ordering.
            selected = max(candidates, key=lambda item: (item[0], item[1]))
            if self._last_hand_wrist is not None and 0 < timestamp - self._last_hand_timestamp <= SMOOTHING_HISTORY_MAX_GAP:
                previous = [item for item in candidates if item[5] == self._last_handedness and item[3]
                            and math.dist(item[3][0][:2], self._last_hand_wrist) <= 0.22]
                if previous:
                    retained = min(previous, key=lambda item: math.dist(item[3][0][:2], self._last_hand_wrist))
                    retained_command = retained[4].is_gao
                    selected_command = selected[4].is_gao
                    if not (selected_command and (not retained_command or selected[0] > retained[0] + 0.20)):
                        selected = retained
            score, _, index, hand_points, gesture, handedness = selected
            hand_world_points = _landmark_tuples(world[index]) if index < len(world) else ()
            raw_pose_ok = bool(gesture.is_gao)
            gesture_reason = gesture.reason
            gesture_hint = getattr(gesture, "hint", "") or (
                "KON 手勢已辨識" if raw_pose_ok else "尚未辨識 KON 手勢；請讓手指清楚入鏡"
            )
            gesture_source = getattr(gesture, "source", "normalized")
            wrist = hand_points[0][:2] if hand_points else None
            same_hand = (handedness == self._last_handedness and self._last_hand_wrist is not None
                         and wrist is not None and math.dist(wrist, self._last_hand_wrist) <= 0.22)
            if not same_hand:
                self._hand_filter.reset()
            shape_valid = True
            if not raw_pose_ok:
                # Never turn malformed landmarks into a tolerated shape miss.
                try:
                    extract_gesture_features(
                        hand_world_points if gesture_source == "world" else hand_points,
                        source=gesture_source, aspect_ratio=width / height,
                    )
                except (TypeError, ValueError, OverflowError):
                    shape_valid = False
            pose_ok = self._gesture_stabilizer.update(
                gesture, timestamp, shape_valid=shape_valid, same_hand=same_hand,
                profile=gesture_profile,
            )
            self._last_handedness = handedness
            self._last_hand_wrist = wrist
            self._last_hand_timestamp = timestamp
            # Calibration is a geometry task. The strict command gesture is
            # still reported separately and remains required by the trigger.
            if hand_mode:
                metric = None
            elif index < len(world):
                metric = estimate_hand_camera_points(
                    hand_points, _landmark_tuples(world[index]), width, height, intrinsics,
                    diagnostics=diagnostics,
                )
            else:
                metric = None
                diagnostics["hand_depth_error"] = "尚未取得手部公尺座標，請讓整隻手清楚入鏡"
            aperture = ring_center(metric) if metric is not None else None
            if metric is not None and aperture is None:
                hand_stage = "hand_aperture_unusable"
                diagnostics["hand_depth_error"] = "手孔中心無法估計；請調整手部並讓手指清楚入鏡"
        else:
            self._gesture_stabilizer.reset()
            self._last_handedness = None
            self._last_hand_wrist = None
            diagnostics["hand_depth_error"] = "沒有偵測到手部；請讓整隻手完整入鏡"
        if hand_mode:
            cursor = hand_circle_cursor(hand_points, diagnostics=diagnostics) if hands else None
            if not hands:
                depth_status, message = "no_hand", diagnostics["hand_depth_error"]
            elif cursor is None:
                depth_status, message = "hand_cursor_unusable", diagnostics["hand_cursor_error"]
            else:
                depth_status = "ready"
                message = "手圈游標已建立；移動手圈控制準星" if pose_ok else f"手圈位置已取得；{gesture_hint}（校正可記錄）"
            return TrackingFrame(
                timestamp, hand_points, (), score, pose_ok, None, None, None, message,
                hand_detected=bool(hands), eye_detected=False, depth_status=depth_status,
                hand_depth_error=diagnostics.get("hand_depth_error", ""),
                gesture_reason=gesture_reason, gesture_hint=gesture_hint, gesture_source=gesture_source,
                source_fps=source_fps, source_size=source_size or (width, height), inference_size=(width, height),
                hand_world_points=hand_world_points, handedness=handedness,
                hand_cursor=cursor, aiming_mode=self.aim_mode,
                raw_pose_ok=raw_pose_ok,
            )
        eye_sample = estimate_eye_xyz(
            face_points, width, height, intrinsics, self.dominant_eye, diagnostics=diagnostics,
        )
        if not faces:
            diagnostics["eye_error"] = "沒有偵測到臉部；請讓臉與手同時入鏡"
        eye = self._eye_filter.update(eye_sample, timestamp)
        aperture_xyz = self._hand_filter.update(aperture, timestamp)
        raw_hit = None
        if eye is not None and aperture_xyz is not None:
            raw_hit = ray_screen_intersection(eye, aperture_xyz)
        if not hands:
            depth_status = "no_hand"
            message = diagnostics["hand_depth_error"]
        elif not faces:
            depth_status = "no_face"
            message = diagnostics["eye_error"]
        elif eye_sample is None:
            depth_status = "eye_unusable"
            message = diagnostics.get("eye_error", "慣用眼定位無效；請讓眼睛清楚入鏡")
        elif aperture is None:
            depth_status = hand_stage
            message = diagnostics.get("hand_depth_error", "手部深度無法估計，請讓整隻手清楚入鏡")
        elif eye is None or aperture_xyz is None:
            depth_status = "stabilizing"
            endpoint = "眼睛與手孔" if eye is None and aperture_xyz is None else ("眼睛" if eye is None else "手孔")
            message = f"已取得追蹤點；請保持{endpoint}位置片刻，等待深度穩定"
            if source_fps is not None and source_gap > SMOOTHING_HISTORY_MAX_GAP:
                message += f"（追蹤間隔過長，約 {source_fps:.1f} FPS；請減少其他程式負載）"
            elif source_fps is not None and source_fps < 5.0:
                message += f"（追蹤約 {source_fps:.1f} FPS，請多停留片刻）"
        elif raw_hit is None:
            if aperture_xyz[2] > eye[2] - 0.035:
                depth_status = "depth_order_failed"
                message = (f"眼距約 {eye[2] * 100:.0f} cm、手孔約 {aperture_xyz[2] * 100:.0f} cm；"
                           "請把手孔放在眼睛與螢幕之間，與眼睛拉開距離")
            else:
                depth_status = "ray_rejected"
                message = "眼睛與手孔的距離或射線不合理；請把手往螢幕移並面向鏡頭"
        else:
            depth_status = "ready"
            message = ("眼睛 → 手孔射線已建立；深度為估計值" if pose_ok else
                       f"瞄準射線已建立；{gesture_hint}（校正可記錄）")
        return TrackingFrame(
            timestamp, hand_points, face_points, score, pose_ok, eye, aperture_xyz, raw_hit, message,
            hand_detected=bool(hands), eye_detected=eye_sample is not None,
            depth_status=depth_status, hand_depth_error=diagnostics.get("hand_depth_error", ""),
            eye_error=diagnostics.get("eye_error", ""), gesture_reason=gesture_reason,
            source_fps=source_fps,
            source_size=source_size or (width, height), inference_size=(width, height),
            gesture_hint=gesture_hint, gesture_source=gesture_source,
            hand_world_points=hand_world_points, handedness=handedness,
            aiming_mode=self.aim_mode,
            raw_pose_ok=raw_pose_ok,
        )

    @staticmethod
    def _preview(bgr: np.ndarray, observation: TrackingFrame) -> QImage:
        pixels = bgr.copy()
        height, width = pixels.shape[:2]
        points = observation.hand_points
        if len(points) == 21 and np.isfinite(points).all():
            positions = [(round(p[0] * width), round(p[1] * height)) for p in points]
            connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                           (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                           (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
            color = (180, 233, 124) if observation.pose_ok else (235, 171, 188)
            for start, end in connections:
                cv2.line(pixels, positions[start], positions[end], color, 1, cv2.LINE_AA)
            for point in positions:
                cv2.circle(pixels, point, 2, color, -1, cv2.LINE_AA)
            aperture = ring_center(points)
            if aperture is not None:
                center = (round(aperture[0] * width), round(aperture[1] * height))
                cv2.circle(pixels, center, 7, color, 2, cv2.LINE_AA)
        if observation.aiming_mode == "eye_ray" and len(observation.face_points) >= 478 and np.isfinite(observation.face_points).all():
            for index in (468, 473):
                p = observation.face_points[index]
                cv2.circle(pixels, (round(p[0] * width), round(p[1] * height)), 4, (250, 205, 161), 1, cv2.LINE_AA)
        mirrored = cv2.flip(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), 1)
        return QImage(mirrored.data, width, height, mirrored.strides[0], QImage.Format.Format_RGB888).copy()
