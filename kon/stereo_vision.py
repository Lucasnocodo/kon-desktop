"""Two-view hand/iris reconstruction using calibrated image observations.

Camera timestamps are host receive times, not a promise of exposure sync.
No monocular depth is substituted when either view is unusable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPainter

from .stereo import StereoCalibration, ScreenPlane, triangulate_points
from .stereo_capture import CameraSource, DualCapture, list_camera_sources
from .stereo_live_diagnostics import LiveStereoDiagnostics
from .vision import (
    CameraWorker, TrackingFrame, MODEL_DIR, MAX_PROCESSING_FPS,
    _landmark_tuples, resize_for_inference,
)


def verified_sources(calibration):
    """Never apply saved extrinsics to a different enumerated device."""
    sources = {source.index: source for source in list_camera_sources()}
    result = []
    for index, expected in ((calibration.primary_source, calibration.primary_id),
                            (calibration.secondary_source, calibration.secondary_id)):
        actual = sources.get(index)
        if expected and (actual is None or actual.device_id != expected):
            raise ValueError("鏡頭編號或裝置身分已改變；請重新選擇兩台鏡頭並校正，勿沿用舊的雙鏡頭對應。")
        result.append(actual if actual is not None else CameraSource(index, f"鏡頭 {index}"))
    return tuple(result)


def iris_pixel_center(face, image_size, side):
    """Image-space quality gates only; no assumed iris diameter or depth."""
    definitions = {"right": (468, (469, 470, 471, 472), (33, 133), (159, 145)),
                   "left": (473, (474, 475, 476, 477), (263, 362), (386, 374))}
    if side not in definitions:
        raise ValueError("慣用眼設定無效")
    points = np.asarray(face, dtype=float)
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise ValueError("兩台鏡頭都需要清楚看見臉與慣用眼")
    width, height = image_size
    center_index, ring_indices, corners, lids = definitions[side]
    pixels = points[:, :2] * [width, height]
    relevant = pixels[[center_index, *ring_indices, *corners, *lids]]
    if np.any(relevant < 0) or np.any(relevant >= [width, height]):
        raise ValueError("慣用眼部分超出畫面；請讓眼睛完整入鏡")
    center = pixels[center_index]
    ring = pixels[list(ring_indices)]
    eye_width = float(np.linalg.norm(pixels[corners[0]] - pixels[corners[1]]))
    opening = float(np.linalg.norm(pixels[lids[0]] - pixels[lids[1]]))
    diameter = max(float(np.linalg.norm(ring[0] - ring[2])), float(np.linalg.norm(ring[1] - ring[3])))
    if eye_width < 9 or opening / eye_width < .12 or diameter < 3.5:
        raise ValueError("其中一台鏡頭的慣用眼太小、閉合或被遮擋；請改善光線與拍攝角度")
    if (np.any(center < 0) or np.any(center >= [width, height]) or diameter > eye_width * .82
            or np.linalg.norm(center - ring.mean(axis=0)) > diameter * .3):
        raise ValueError("慣用眼虹膜輪廓不穩；請避免遮擋並面向兩台鏡頭")
    return center


def _hand_label(result):
    identities = getattr(result, "handedness", ())
    return identities[0][0].category_name if identities and identities[0] else "unknown"


class StereoCameraWorker(CameraWorker):
    """Uses the existing gesture/aim gate after reconstructing metric geometry."""

    def __init__(self, calibration: StereoCalibration, screen_plane: ScreenPlane,
                 dominant_eye="right", parent=None, *, gesture_profile=None, max_skew_s=.020,
                 diagnostics_path=None):
        super().__init__(calibration.primary_source, dominant_eye=dominant_eye,
                         parent=parent, gesture_profile=gesture_profile, aim_mode="eye_ray")
        if not math.isfinite(max_skew_s) or not 0 < max_skew_s <= .05:
            raise ValueError("雙鏡頭時間差上限必須介於 0 與 50 毫秒之間")
        self.stereo_calibration = calibration
        self.screen_plane = screen_plane
        self.max_skew_s = max_skew_s
        self.live_diagnostics = (LiveStereoDiagnostics(diagnostics_path, calibration)
                                 if diagnostics_path is not None else None)

    def paused(self, timestamp, message, *, status="stereo_unusable", primary_hand=None,
               primary_face=None, source_size=None, secondary_size=None, skew_ms=None):
        self._eye_filter.reset()
        self._hand_filter.reset()
        self._gesture_stabilizer.reset()
        if math.isfinite(timestamp) and (self._last_source_timestamp is None or timestamp > self._last_source_timestamp):
            self._last_source_timestamp = timestamp
        hands = getattr(primary_hand, "hand_landmarks", ()) or ()
        faces = getattr(primary_face, "face_landmarks", ()) or ()
        return TrackingFrame(
            timestamp, _landmark_tuples(hands[0]) if len(hands) == 1 else (),
            _landmark_tuples(faces[0]) if len(faces) == 1 else (),
            0., False, None, None, None, message, hand_detected=bool(hands),
            depth_status=status, hand_depth_error=message, gesture_hint=message, raw_pose_ok=False,
            acquisition_mode="stereo", pair_skew_ms=skew_ms,
            source_size=source_size, secondary_source_size=secondary_size,
            handedness=_hand_label(primary_hand) if len(hands) == 1 else "unknown",
        )

    def make_stereo_tracking(self, hand0, face0, hand1, face1, timestamp,
                             primary_size, secondary_size, *, inference_size=None, skew_ms=0.):
        """Reconstruct corresponding physical joints, then form the 3D aperture.

        Averaging independently predicted world landmarks or triangulating two
        view-dependent 2D aperture centers would not identify the same 3D point.
        """
        try:
            if not math.isfinite(timestamp) or (self._last_source_timestamp is not None and timestamp <= self._last_source_timestamp):
                raise ValueError("等待新的雙鏡頭影格；忽略重複或倒序時間")
            self.stereo_calibration.validate_sizes(primary_size, secondary_size)
            if not math.isfinite(skew_ms) or not 0 <= skew_ms <= self.max_skew_s * 1000 + 1e-8:
                raise ValueError("雙鏡頭影格時間差過大；請確認有線連接與拍攝頻率")
            hands0, hands1 = hand0.hand_landmarks, hand1.hand_landmarks
            if len(hands0) != 1 or len(hands1) != 1:
                raise ValueError("兩台鏡頭都需看見同一隻完整的手；請將另一隻手移出畫面")
            if _hand_label(hand0) == "unknown" or _hand_label(hand0) != _hand_label(hand1):
                raise ValueError("兩台鏡頭的左右手對應不一致；請調整角度，避免鏡像或手指遮擋")
            p0, p1 = np.asarray(_landmark_tuples(hands0[0])), np.asarray(_landmark_tuples(hands1[0]))
            if p0.shape != (21, 3) or p1.shape != (21, 3):
                raise ValueError("手部關節不完整")
            if any(not np.isfinite(p).all() or np.any(p[:, :2] < 0) or np.any(p[:, :2] > 1)
                   for p in (p0, p1)):
                raise ValueError("手部部分超出畫面；兩台鏡頭都需要看見整隻手")
            faces0, faces1 = face0.face_landmarks, face1.face_landmarks
            if len(faces0) != 1 or len(faces1) != 1:
                raise ValueError("兩台鏡頭都需看見同一張臉與慣用眼")
            eye0 = iris_pixel_center(_landmark_tuples(faces0[0]), primary_size, self.dominant_eye)
            eye1 = iris_pixel_center(_landmark_tuples(faces1[0]), secondary_size, self.dominant_eye)
            hand_pixels = [p0[:, :2] * primary_size, p1[:, :2] * secondary_size]
            eye_pixels = [np.array([eye0]), np.array([eye1])]
            # Check both independently for diagnosis; either rejection still
            # cancels the complete observation, with no substitute geometry.
            reconstructed, failures = {}, {}
            for stage, pixels in (("hand", hand_pixels), ("iris", eye_pixels)):
                try:
                    reconstructed[stage] = triangulate_points(self.stereo_calibration, *pixels)
                except ValueError as exc:
                    failures[stage] = exc
            if failures:
                stage, exc = next(iter(failures.items()))
                detail = getattr(exc, "diagnostics", {})
                label = "手部" if stage == "hand" else "慣用眼"
                worst = detail.get("worst_reprojection_error_px")
                point = detail.get("worst_point_index")
                if worst is not None:
                    label += (f"關節 #{point}" if stage == "hand" else "")
                    label += f"重投影偏差 {worst:.1f} px（上限 3 px）"
                else:
                    label += "定位"
                if self.live_diagnostics is not None:
                    self.live_diagnostics.record(timestamp, {
                        "failure_stages": list(failures),
                        "failures": {key: {"message": str(error),
                                           "metrics": getattr(error, "diagnostics", {})}
                                     for key, error in failures.items()},
                        "primary_size": list(primary_size), "secondary_size": list(secondary_size),
                        "inference_size": list(inference_size or primary_size),
                        "skew_ms": skew_ms, "dominant_eye": self.dominant_eye,
                        "handedness": [_hand_label(hand0), _hand_label(hand1)],
                        "hand_primary_pixels": hand_pixels[0].tolist(),
                        "hand_secondary_pixels": hand_pixels[1].tolist(),
                        "iris_primary_pixels": eye_pixels[0].tolist(),
                        "iris_secondary_pixels": eye_pixels[1].tolist(),
                    })
                raise ValueError(f"{label}：{exc}") from exc
            xyz, eye = reconstructed["hand"], reconstructed["iris"][0]
            palm = float(np.linalg.norm(xyz[0] - xyz[9]))
            if not .035 <= palm <= .18 or np.max(np.linalg.norm(xyz - xyz[0], axis=1)) > .40:
                raise ValueError("重建的手掌尺寸不合理；請檢查校正板實際尺寸、遮擋及兩機時間差")
            # Enroll/match actual reconstructed shape with the existing rotation-
            # invariant gesture features, never mix camera translation into it.
            centered = xyz - xyz.mean(axis=0)
            metric_hand = SimpleNamespace(
                hand_landmarks=hands0,
                hand_world_landmarks=[[SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in centered]],
                handedness=hand0.handedness,
            )
            width, height = inference_size or primary_size
            return self._make_tracking(
                metric_hand, face0, width, height, timestamp, source_size=primary_size,
                geometry_override=True, metric_hand_points=xyz,
                metric_eye_point=tuple(float(v) for v in eye), screen_plane=self.screen_plane,
                pair_skew_ms=skew_ms, secondary_source_size=secondary_size,
            )
        except (ValueError, TypeError, cv2.error) as exc:
            no_hand = not getattr(hand0, "hand_landmarks", ()) and not getattr(hand1, "hand_landmarks", ())
            return self.paused(timestamp, str(exc), status="no_hand" if no_hand else "stereo_unusable",
                               primary_hand=hand0, primary_face=face0,
                               source_size=primary_size, secondary_size=secondary_size, skew_ms=skew_ms)

    @staticmethod
    def combined_preview(bgr0, bgr1, observation, hand1, face1):
        hands, faces = hand1.hand_landmarks, face1.face_landmarks
        second = replace(observation,
                         hand_points=_landmark_tuples(hands[0]) if len(hands) == 1 else (),
                         face_points=_landmark_tuples(faces[0]) if len(faces) == 1 else ())
        images = [CameraWorker._preview(bgr0, observation), CameraWorker._preview(bgr1, second)]
        tile_height = min(360, max(image.height() for image in images))
        images = [image.scaledToHeight(tile_height) for image in images]
        canvas = QImage(sum(image.width() for image in images), tile_height + 30, QImage.Format_RGB888)
        canvas.fill(QColor("#100e17"))
        painter = QPainter(canvas)
        painter.setFont(QFont("PingFang TC", 12))
        painter.setPen(QColor("#eee9fa"))
        x = 0
        for title, image in zip(("主鏡頭 · 外部攝影機", "副鏡頭"), images):
            painter.drawText(x + 10, 20, title)
            painter.drawImage(x, 30, image)
            x += image.width()
        painter.end()
        return canvas

    def run(self):
        capture = None
        models = []
        executor = None
        reported_failure = False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            primary, secondary = verified_sources(self.stereo_calibration)
            for name in ("hand_landmarker.task", "face_landmarker.task"):
                if not (MODEL_DIR / name).is_file():
                    raise FileNotFoundError(f"缺少本機追蹤模型：{name}")
            self.status.emit("正在載入雙鏡頭手部與臉部模型…")
            base = python.BaseOptions
            for _ in range(2):
                hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=base(model_asset_path=str(MODEL_DIR / "hand_landmarker.task"), delegate=base.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=.6, min_hand_presence_confidence=.6, min_tracking_confidence=.6))
                models.append(hand)
                face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=base(model_asset_path=str(MODEL_DIR / "face_landmarker.task"), delegate=base.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO, num_faces=2,
                    min_face_detection_confidence=.6, min_face_presence_confidence=.6, min_tracking_confidence=.6))
                models.append(face)
            if self._stop_event.is_set():
                return
            capture = DualCapture(primary, secondary, max_skew_s=self.max_skew_s, stop_event=self._stop_event)
            capture.start()
            executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="KON stereo inference")
            self.status.emit("雙鏡頭已啟動 · 軟體時間配對；請讓兩台都看見慣用眼與同一隻手")
            last_ms = [-1, -1]
            last_status = 0.

            def infer(index, bgr, timestamp):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                stamp = max(last_ms[index] + 1, int(timestamp * 1000))
                last_ms[index] = stamp
                return (models[index * 2].detect_for_video(image, stamp),
                        models[index * 2 + 1].detect_for_video(image, stamp))

            while not self._stop_event.is_set() and not self.isInterruptionRequested():
                started = time.monotonic()
                pair = capture.take(timeout=.05)
                if pair is None:
                    if capture.failure:
                        raise RuntimeError(capture.failure)
                    if started - last_status >= .15:
                        self.tracking.emit(self.paused(started, "等待新的雙鏡頭影格；時間差過大或其中一路暫停"))
                        last_status = started
                    continue
                source_sizes = [(f.pixels.shape[1], f.pixels.shape[0]) for f in (pair.primary, pair.secondary)]
                self.stereo_calibration.validate_sizes(*source_sizes)
                images = [resize_for_inference(f.pixels, 1280, 720) for f in (pair.primary, pair.secondary)]
                futures = [executor.submit(infer, i, image, frame.timestamp)
                           for i, (image, frame) in enumerate(zip(images, (pair.primary, pair.secondary)))]
                (hand0, face0), (hand1, face1) = [future.result() for future in futures]
                if self._stop_event.is_set():
                    break
                if time.monotonic() - pair.timestamp > .35:
                    observation = self.paused(time.monotonic(), "雙鏡頭處理延遲；已暫停瞄準與觸發，請減少其他程式負載")
                else:
                    observation = self.make_stereo_tracking(
                        hand0, face0, hand1, face1, pair.timestamp, *source_sizes,
                        inference_size=(images[0].shape[1], images[0].shape[0]), skew_ms=pair.skew_s * 1000)
                self.tracking.emit(observation)
                last_status = time.monotonic()
                if self._preview_due(pair.timestamp):
                    preview = self.combined_preview(*images, observation, hand1, face1)
                    if time.monotonic() - pair.timestamp <= .35:
                        self.frame.emit(preview)
                if started - getattr(self, "_last_metrics", 0.) >= 1.:
                    self._last_metrics = started
                    depth = (f" · 眼距 {observation.eye_xyz[2] * 100:.0f} cm"
                             f" · 手圈 {observation.aperture_xyz[2] * 100:.0f} cm"
                             if observation.eye_xyz is not None and observation.aperture_xyz is not None else "")
                    self.status.emit(f"雙鏡頭 · 讀取時間差 {pair.skew_s * 1000:.1f} ms{depth} · {observation.message}")
                self._stop_event.wait(max(0., 1. / MAX_PROCESSING_FPS - (time.monotonic() - started)))
        except Exception as exc:
            if not self.isInterruptionRequested():
                reported_failure = True
                self.error.emit(str(exc))
        finally:
            # Capture failure sets the shared stop event too. It must still
            # reach the control room, even if it occurred during inference
            # or immediately before the while-condition was reevaluated.
            if capture is not None and capture.failure and not reported_failure and not self.isInterruptionRequested():
                self.error.emit(capture.failure)
            self._stop_event.set()
            if capture is not None:
                warned = False
                while not capture.close():
                    if not warned:
                        self.error.emit("雙鏡頭驅動仍在關閉；等待鏡頭釋放後才能重新啟動")
                        warned = True
                    # Keep QThread alive so the control room cannot reopen a
                    # device whose owning capture thread is still inside read.
                    self.msleep(100)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for model in models:
                try:
                    model.close()
                except Exception:
                    pass
            message = "雙鏡頭已停止"
            self.tracking.emit(self.paused(time.monotonic(), message))
            self.status.emit(message)
