"""Metric reconstruction and camera lifecycle checks; never open a real camera."""

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from kon.vision import (
    CameraWorker, TrackingFrame, _StablePoint, _LatestCapture, camera_matrix,
    estimate_eye_xyz, estimate_hand_camera_points,
    resize_for_inference, hand_circle_cursor,
)
from kon.geometry import GestureResult, TriggerGate, ray_screen_intersection, ring_center
from kon.calibration import CalibrationBuffer


WIDTH, HEIGHT = 960, 540


def synthetic_hand():
    # Nonplanar meter landmarks, not normalized image z values.
    return np.array([
        [0, .035, .006], [-.022, .022, .004], [-.044, .005, .005],
        [-.053, -.009, .003], [-.046, -.024, .001],
        [-.028, -.018, 0], [-.029, -.043, -.005],
        [-.031, -.067, -.001], [-.032, -.089, .003],
        [0, -.03, 0], [0, -.054, -.017], [.008, -.036, -.027], [.011, -.023, -.018],
        [.025, -.022, .002], [.028, -.045, -.017], [.022, -.025, -.027], [.013, -.018, -.015],
        [.044, -.009, .005], [.05, -.035, .002], [.051, -.056, .004], [.054, -.072, .007],
    ])


def projected_hand(depth=.36):
    objects = synthetic_hand()
    intrinsics = camera_matrix(WIDTH, HEIGHT, 60)
    rotation = np.array([.12, -.15, .07])
    translation = np.array([.02, .025, depth])
    projected, _ = cv2.projectPoints(objects, rotation, translation, intrinsics, np.zeros(4))
    normalized = np.column_stack((projected[:, 0, :] / [WIDTH, HEIGHT], np.zeros(21)))
    matrix, _ = cv2.Rodrigues(rotation)
    expected = (matrix @ objects.T).T + translation
    return normalized, objects, intrinsics, expected


def synthetic_face(side="right", depth=.6):
    face = np.zeros((478, 3))
    intrinsics = camera_matrix(WIDTH, HEIGHT, 60)
    cx, cy = 455., 240.
    radius = intrinsics[0, 0] * .0117 / depth / 2
    if side == "right":
        iris, corners, lids = (468, 469, 470, 471, 472), (33, 133), (159, 145)
    else:
        iris, corners, lids = (473, 474, 475, 476, 477), (263, 362), (386, 374)
    for index, offset in zip(iris, [(0, 0), (radius, 0), (0, -radius), (-radius, 0), (0, radius)]):
        face[index, :2] = ((cx + offset[0]) / WIDTH, (cy + offset[1]) / HEIGHT)
    for index, offset in zip((*corners, *lids), [(-14, 0), (14, 0), (0, -5), (0, 5)]):
        face[index, :2] = ((cx + offset[0]) / WIDTH, (cy + offset[1]) / HEIGHT)
    return face, intrinsics


def test_pnp_recovers_common_camera_coordinates():
    normalized, world, intrinsics, expected = projected_hand()
    actual = estimate_hand_camera_points(normalized, world, WIDTH, HEIGHT, intrinsics)
    assert actual is not None
    assert np.max(np.abs(np.array(actual) - expected)) < 1e-5
    # Normalized MediaPipe z is never used as metric depth by the PnP solver.
    normalized[:, 2] = np.linspace(-20, 20, 21)
    assert np.asarray(estimate_hand_camera_points(normalized, world, WIDTH, HEIGHT, intrinsics)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("source, expected", [
    ((1280, 720), (960, 540)), ((1920, 1080), (960, 540)),
    ((1600, 1200), (720, 540)), ((640, 480), (640, 480)),
    ((320, 240), (320, 240)), ((960, 540), (960, 540)),
])
def test_inference_resize_caps_size_without_stretch_crop_or_upscale(source, expected):
    width, height = source
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :20, 0] = 255
    pixels[:, -20:, 2] = 255
    pixels[:20, :, 1] = 255
    pixels[-20:, :, 1] = 255
    resized = resize_for_inference(pixels)
    assert resized.shape[:2] == (expected[1], expected[0])
    assert expected[0] / expected[1] == pytest.approx(width / height)
    # All four source borders survive: a center crop would lose them.
    assert resized[expected[1] // 2, 0, 0] > 0
    assert resized[expected[1] // 2, -1, 2] > 0
    assert resized[0, expected[0] // 2, 1] > 0
    assert resized[-1, expected[0] // 2, 1] > 0
    if expected == source:
        np.testing.assert_array_equal(resized, pixels)


def test_uniform_720p_to_540p_resize_preserves_metric_eye_hand_and_ray():
    normalized, world, k540, _expected = projected_hand()
    face, _ = synthetic_face()
    k720 = camera_matrix(1280, 720, 60)
    eye540 = estimate_eye_xyz(face, 960, 540, k540, "right")
    eye720 = estimate_eye_xyz(face, 1280, 720, k720, "right")
    hand540 = estimate_hand_camera_points(normalized, world, 960, 540, k540)
    hand720 = estimate_hand_camera_points(normalized, world, 1280, 720, k720)
    assert eye720 == pytest.approx(eye540, abs=1e-8)
    np.testing.assert_allclose(hand720, hand540, atol=1e-5)
    assert ray_screen_intersection(eye720, ring_center(hand720)) == pytest.approx(
        ray_screen_intersection(eye540, ring_center(hand540)), abs=1e-5,
    )


def test_tracking_sizes_keep_delivered_source_separate_from_inference():
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    observation = CameraWorker()._make_tracking(hand, face, 960, 540, 1, source_size=(1280, 720))
    assert observation.source_size == (1280, 720)
    assert observation.inference_size == (960, 540)


def test_hand_cursor_mirrors_image_motion_without_metric_projection():
    normalized, _world, *_ = projected_hand()
    cursor = hand_circle_cursor(normalized)
    center = ring_center(normalized)
    assert cursor == pytest.approx((1 - center[0], center[1]))
    moved = normalized.copy()
    moved[:, :2] += [.01, -.02]
    assert hand_circle_cursor(moved) == pytest.approx((cursor[0] - .01, cursor[1] - .02))
    # Normalized depth values never create a camera-position ray here.
    moved[:, 2] = np.linspace(-20, 20, 21)
    assert hand_circle_cursor(moved) == pytest.approx((cursor[0] - .01, cursor[1] - .02))


@pytest.mark.parametrize("failure", ["nan", "incomplete", "off_left", "off_bottom", "degenerate"])
def test_hand_cursor_pauses_when_the_complete_visible_hand_is_unusable(failure):
    normalized, _world, *_ = projected_hand()
    if failure == "nan":
        normalized[0, 2] = np.nan
    elif failure == "incomplete":
        normalized = normalized[:20]
    elif failure == "off_left":
        normalized[8, 0] = -.001  # Cursor center itself is still inside.
    elif failure == "off_bottom":
        normalized[20, 1] = 1.001
    else:
        normalized[:] = [.5, .5, 0]
    diagnostics = {}
    assert hand_circle_cursor(normalized, diagnostics=diagnostics) is None
    assert diagnostics["hand_cursor_error"]


def test_hand_mode_cursor_is_available_without_face_or_strict_gesture(monkeypatch):
    normalized, world, *_ = projected_hand()
    hand, _face = _results(((normalized, world),))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hand mode must not perform face or metric projection work")

    for name in ("camera_matrix", "estimate_eye_xyz", "estimate_hand_camera_points", "ray_screen_intersection"):
        monkeypatch.setattr("kon.vision." + name, forbidden)
    worker = CameraWorker(aim_mode="hand")
    first = worker._make_tracking(hand, None, WIDTH, HEIGHT, 1, source_size=(1280, 720))
    assert first.aiming_mode == "hand" and first.hand_cursor == pytest.approx(hand_circle_cursor(normalized))
    assert first.hand_detected and not first.pose_ok
    assert first.depth_status == "ready" and "校正可記錄" in first.message
    assert first.face_points == () and not first.eye_detected
    assert first.eye_xyz is first.aperture_xyz is first.raw_hit is None
    assert first.source_size == (1280, 720) and first.inference_size == (WIDTH, HEIGHT)
    assert first.timestamp == 1
    second = worker._make_tracking(hand, None, WIDTH, HEIGHT, 1 + 1 / 30)
    assert second.source_fps == pytest.approx(30)
    assert second.timestamp == 1 + 1 / 30


def test_hand_mode_separates_observed_absence_from_invalid_cursor_data():
    worker = CameraWorker(aim_mode="hand")
    hand, _face = _results()
    absent = worker._make_tracking(hand, None, WIDTH, HEIGHT, 1)
    assert absent.depth_status == "no_hand" and not absent.hand_detected
    normalized, world, *_ = projected_hand()
    normalized[8, 0] = 1.01
    hand, _face = _results(((normalized, world),))
    uncertain = worker._make_tracking(hand, None, WIDTH, HEIGHT, 1.04)
    assert uncertain.hand_detected and uncertain.depth_status == "hand_cursor_unusable"
    assert uncertain.hand_cursor is None and uncertain.raw_hit is None
    assert "超出畫面" in uncertain.message


def test_pnp_rejects_inconsistent_correspondences():
    normalized, world, intrinsics, _ = projected_hand()
    normalized[:10, :2] += .15
    assert estimate_hand_camera_points(normalized, world, WIDTH, HEIGHT, intrinsics) is None


@pytest.mark.parametrize("depth", [.025, 2.0])
def test_pnp_rejects_implausible_depth(depth):
    normalized, world, intrinsics, _ = projected_hand(depth)
    assert estimate_hand_camera_points(normalized, world, WIDTH, HEIGHT, intrinsics) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_iris_depth_and_anatomical_side(side):
    face, intrinsics = synthetic_face(side)
    actual = estimate_eye_xyz(face, WIDTH, HEIGHT, intrinsics, side)
    assert actual is not None
    assert actual == pytest.approx(((455 - 480) * .6 / intrinsics[0, 0],
                                  (240 - 270) * .6 / intrinsics[1, 1], .6))
    other = "left" if side == "right" else "right"
    assert estimate_eye_xyz(face, WIDTH, HEIGHT, intrinsics, other) is None


def test_blink_clears_iris_instead_of_switching_eye():
    face, intrinsics = synthetic_face()
    face[159] = face[145]
    assert estimate_eye_xyz(face, WIDTH, HEIGHT, intrinsics, "right") is None


def _landmarks(points):
    return [SimpleNamespace(x=float(point[0]), y=float(point[1]), z=float(point[2])) for point in points]


def _results(hand_arrays=(), face_array=None, labels=()):
    hand = SimpleNamespace(
        hand_landmarks=[_landmarks(normalized) for normalized, _world in hand_arrays],
        hand_world_landmarks=[_landmarks(world) for _normalized, world in hand_arrays],
        handedness=[[SimpleNamespace(category_name=labels[index] if index < len(labels) else "Right")]
                    for index in range(len(hand_arrays))],
    )
    face = SimpleNamespace(face_landmarks=[_landmarks(face_array)] if face_array is not None else [])
    return hand, face


def _steady_observation(worker, hand, face):
    for timestamp in (1.0, 1.07, 1.14):
        observation = worker._make_tracking(hand, face, WIDTH, HEIGHT, timestamp)
    return observation


def test_visible_hand_can_calibrate_without_command_gesture(monkeypatch):
    # The calibration failure came from conditioning PnP on this classifier.
    # Keep it false while exercising the actual iris, PnP and ray pipeline.
    monkeypatch.setattr("kon.vision.classify_gao", lambda *_args, **_kwargs:
                        SimpleNamespace(score=0.0, is_gao=False, reason="finger pattern",
                                        hint="請伸直小指", source="world"))
    normalized, world, _intrinsics, _expected = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    observation = _steady_observation(CameraWorker(), hand, face)
    assert observation.hand_detected and observation.eye_detected
    assert not observation.pose_ok and observation.gesture_score == 0
    assert observation.eye_xyz is not None and observation.aperture_xyz is not None
    assert observation.raw_hit is not None and observation.depth_status == "ready"
    assert observation.gesture_reason == "finger pattern"
    assert observation.gesture_hint == "請伸直小指"
    assert observation.gesture_source == "world"
    assert "瞄準射線已建立" in observation.message
    assert "請伸直小指" in observation.message and "校正可記錄" in observation.message


def test_pose_classifier_gets_the_corresponding_unmirrored_world_hand(monkeypatch):
    calls = []

    def classify(points, image_aspect_ratio, *, world_points=None):
        calls.append((np.asarray(points), image_aspect_ratio, np.asarray(world_points)))
        return SimpleNamespace(score=0.0, is_gao=False, reason="finger pattern",
                               hint="請彎曲中指", source="world")

    monkeypatch.setattr("kon.vision.classify_gao", classify)
    near, world, *_ = projected_hand(depth=.30)
    far, _, *_ = projected_hand(depth=.55)
    other_world = world * 1.05
    # MediaPipe normalized z is unsuitable for metric aiming. Pose receives
    # the independent world shape rather than reusing these image z values.
    near[:, 2] = np.linspace(-2, 2, 21)
    face_points, _ = synthetic_face()
    hand, face = _results(((far, other_world), (near, world)), face_points, ("Left", "Right"))
    observation = CameraWorker()._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert len(calls) == 2
    for call, normalized, expected_world in zip(calls, (far, near), (other_world, world)):
        np.testing.assert_array_equal(call[0], normalized)
        assert call[1] == WIDTH / HEIGHT
        np.testing.assert_array_equal(call[2], expected_world)
    assert observation.hand_points[0] == pytest.approx(near[0])
    assert observation.gesture_hint == "請彎曲中指"
    np.testing.assert_array_equal(observation.hand_world_points, world)
    assert observation.handedness == "Right"


@pytest.mark.parametrize("world_value", [[], None])
def test_missing_world_pose_fallback_cannot_invent_an_aiming_depth(monkeypatch, world_value):
    calls = []

    def classify(_points, image_aspect_ratio, *, world_points=None):
        calls.append(world_points)
        return SimpleNamespace(score=1.0, is_gao=True, reason="recognized",
                               hint="KON 手勢已辨識", source="normalized")

    monkeypatch.setattr("kon.vision.classify_gao", classify)
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    hand.hand_world_landmarks = world_value
    observation = _steady_observation(CameraWorker(), hand, face)
    assert calls == [None, None, None]
    assert observation.pose_ok and observation.gesture_source == "normalized"
    assert observation.raw_hit is None and observation.aperture_xyz is None
    assert observation.hand_world_points == ()
    assert observation.depth_status == "hand_pnp_failed"
    assert "公尺座標" in observation.message


@pytest.mark.parametrize("matches", [True, False])
def test_personal_pose_is_primary_even_when_generic_pose_would_disagree(monkeypatch, matches):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an installed personal model must never use the generic pose")

    monkeypatch.setattr("kon.vision.classify_gao", forbidden)
    calls = []

    class Profile:
        def match(self, points, *, world_points, image_aspect_ratio, handedness):
            calls.append((points, world_points, image_aspect_ratio, handedness))
            return GestureResult(.95 if matches else .15, matches,
                                 reason="personal match" if matches else "personal mismatch",
                                 source="world", hint="個人手勢已辨識" if matches else "目前手勢與個人樣本不同")

    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points, ("Left",))
    observation = _steady_observation(CameraWorker(gesture_profile=Profile()), hand, face)
    assert observation.pose_ok is matches
    assert observation.raw_hit is not None
    assert observation.gesture_reason == ("personal match" if matches else "personal mismatch")
    assert observation.gesture_hint == ("個人手勢已辨識" if matches else "目前手勢與個人樣本不同")
    assert observation.handedness == "Left"
    np.testing.assert_array_equal(observation.hand_world_points, world)
    assert all(call[2:] == (WIDTH / HEIGHT, "Left") for call in calls)
    for points, world_points, *_ in calls:
        np.testing.assert_array_equal(points, normalized)
        np.testing.assert_array_equal(world_points, world)


def test_profile_change_during_matching_applies_only_to_the_next_complete_frame(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("profile mismatch must not fall back to the generic pose")

    monkeypatch.setattr("kon.vision.classify_gao", forbidden)
    calls = []
    worker = CameraWorker()

    class Replacement:
        def match(self, _points, **_kwargs):
            calls.append("replacement")
            return GestureResult(.1, False, reason="personal mismatch", hint="樣本不同")

    replacement = Replacement()

    class Original:
        def match(self, _points, **_kwargs):
            calls.append("original")
            # Model replacement happens while the worker is processing a
            # multi-hand frame. This must not deadlock or mix two authorities.
            worker.set_gesture_profile(replacement)
            return GestureResult(.9, True, reason="personal match", hint="個人手勢已辨識")

    worker.set_gesture_profile(Original())
    near, world, *_ = projected_hand(depth=.30)
    far, _, *_ = projected_hand(depth=.55)
    face_points, _ = synthetic_face()
    hand, face = _results(((near, world), (far, world)), face_points, ("Right", "Left"))
    first = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert first.pose_ok and calls == ["original", "original"]
    second = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.07)
    assert not second.pose_ok and calls == ["original", "original", "replacement", "replacement"]


def test_clearing_personal_profile_restores_generic_classifier(monkeypatch):
    calls = []

    class Profile:
        def match(self, _points, **_kwargs):
            calls.append("personal")
            return GestureResult(.1, False, hint="樣本不同")

    def generic(_points, **_kwargs):
        calls.append("generic")
        return GestureResult(.9, True, hint="KON 手勢已辨識")

    monkeypatch.setattr("kon.vision.classify_gao", generic)
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    worker = CameraWorker(gesture_profile=Profile())
    assert not worker._make_tracking(hand, face, WIDTH, HEIGHT, 1).pose_ok
    worker.set_gesture_profile(None)
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.07).pose_ok
    assert calls == ["personal", "generic"]


def test_invalid_profile_replacement_cannot_restore_generic_pose(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid configuration must not remove the personal pose authority")

    monkeypatch.setattr("kon.vision.classify_gao", forbidden)
    class Profile:
        def match(self, _points, **_kwargs):
            return GestureResult(.1, False, reason="personal mismatch")

    profile = Profile()
    worker = CameraWorker(gesture_profile=profile)
    with pytest.raises(TypeError, match="match"):
        worker.set_gesture_profile(object())
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    observation = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert not observation.pose_ok and observation.gesture_reason == "personal mismatch"


def test_two_fresh_holds_persist_a_personal_pose_rejected_by_the_generic_classifier(tmp_path, monkeypatch):
    from kon.gesture_calibration import GestureCalibration, GestureEnrollmentBuffer

    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points, ("Right",))
    worker = CameraWorker()
    buffer = GestureEnrollmentBuffer()

    def capture(start):
        buffer.reset(start)
        for offset in range(13):
            frame = worker._make_tracking(hand, face, WIDTH, HEIGHT, start + offset * .1)
            assert not frame.pose_ok  # Enrollment must not require the generic shape.
            buffer.observe(frame.timestamp, frame.hand_world_points, handedness=frame.handedness)
        state = buffer.evaluate(frame.timestamp + .05)
        assert state.ready
        return state, frame

    first, _frame = capture(1)
    provisional = GestureCalibration.fit(first.samples, source=first.source, handedness=first.handedness)
    second, before = capture(3)
    confirmed = provisional.confirm(second.samples, source=second.source, handedness=second.handedness)
    path = tmp_path / "personal-gesture.json"
    confirmed.save(path)
    restored = GestureCalibration.load(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("persisted personal pose must be the only command-pose authority")

    monkeypatch.setattr("kon.vision.classify_gao", forbidden)
    worker.set_gesture_profile(restored)
    after = worker._make_tracking(hand, face, WIDTH, HEIGHT, 4.3)
    assert after.pose_ok and after.gesture_reason == "personal gesture recognized"
    assert after.gesture_source == "world"
    assert before.raw_hit is not None and after.raw_hit == pytest.approx(before.raw_hit, abs=1e-8)
    # A wrong hand and missing original data must not invoke the generic pose.
    hand.handedness = [[SimpleNamespace(category_name="Left")]]
    wrong_hand = worker._make_tracking(hand, face, WIDTH, HEIGHT, 4.4)
    assert not wrong_hand.pose_ok and wrong_hand.gesture_reason == "personal gesture handedness"
    hand.handedness = [[SimpleNamespace(category_name="Right")]]
    hand.hand_world_landmarks = []
    missing_world = worker._make_tracking(hand, face, WIDTH, HEIGHT, 4.5)
    assert not missing_world.pose_ok and missing_world.gesture_reason == "personal gesture source"
    assert missing_world.raw_hit is None


def test_tracking_diagnostics_identify_missing_hand_and_face():
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results((), face_points)
    observation = CameraWorker()._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert observation.depth_status == "no_hand" and not observation.hand_detected
    assert "沒有偵測到手部" in observation.hand_depth_error
    assert "沒有偵測到手部" in observation.gesture_hint
    assert observation.gesture_source == "none"
    assert observation.hand_world_points == () and observation.handedness == "unknown"
    hand, face = _results(((normalized, world),))
    observation = CameraWorker()._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert observation.depth_status == "no_face" and not observation.eye_detected
    assert "沒有偵測到臉部" in observation.eye_error


@pytest.mark.parametrize("failure", ["closed", "tiny_iris"])
def test_face_landmarks_do_not_imply_usable_chosen_iris(failure):
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    if failure == "closed":
        face_points[159] = face_points[145]
    else:
        center = face_points[468].copy()
        for index in (469, 470, 471, 472):
            face_points[index] = center + (face_points[index] - center) * 0.1
    hand, face = _results(((normalized, world),), face_points)
    observation = _steady_observation(CameraWorker(), hand, face)
    assert len(observation.face_points) == 478
    assert not observation.eye_detected and observation.raw_hit is None
    assert observation.depth_status == "eye_unusable"
    assert ("閉合" if failure == "closed" else "虹膜太小") in observation.eye_error


def test_hand_projection_failure_is_distinct_from_gesture_failure():
    normalized, world, *_ = projected_hand()
    normalized[:10, :2] += 0.15
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    observation = _steady_observation(CameraWorker(), hand, face)
    assert observation.hand_detected and observation.eye_detected
    assert observation.aperture_xyz is None and observation.raw_hit is None
    assert observation.depth_status == "hand_pnp_failed"
    assert "投影" in observation.hand_depth_error
    assert observation.message == observation.hand_depth_error


def test_hand_behind_eye_still_fails_depth_order_with_actionable_message():
    normalized, world, *_ = projected_hand(depth=.85)
    face_points, _ = synthetic_face(depth=.6)
    hand, face = _results(((normalized, world),), face_points)
    observation = _steady_observation(CameraWorker(), hand, face)
    assert observation.eye_detected and observation.hand_detected
    assert observation.eye_xyz is not None and observation.aperture_xyz is not None
    assert observation.raw_hit is None and observation.depth_status == "depth_order_failed"
    assert "眼距約" in observation.message and "手孔約" in observation.message


def test_initial_zero_score_selection_uses_size_and_then_retains_visible_hand(monkeypatch):
    monkeypatch.setattr("kon.vision.classify_gao", lambda *_args, **_kwargs: GestureResult(0, False))
    near, world, *_ = projected_hand(depth=.30)
    far, _, *_ = projected_hand(depth=.55)
    far[:, 0] += .18
    face_points, _ = synthetic_face()
    hand, face = _results(((far, world), (near, world)), face_points, ("Left", "Right"))
    worker = CameraWorker()
    first = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1)
    assert first.hand_points[0] == pytest.approx(near[0])
    # Another hand becomes bigger and result order changes; an uncertain
    # classifier must not redirect calibration to it.
    bigger_other, _, *_ = projected_hand(depth=.25)
    bigger_other[:, 0] += .18
    hand, face = _results(((near, world), (bigger_other, world)), face_points, ("Right", "Left"))
    second = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.07)
    assert second.hand_points[0] == pytest.approx(near[0])


def test_eye_and_hand_estimator_diagnostics_preserve_rejection_thresholds():
    face, intrinsics = synthetic_face()
    face[159] = face[145]
    diagnostics = {}
    assert estimate_eye_xyz(face, WIDTH, HEIGHT, intrinsics, "right", diagnostics=diagnostics) is None
    assert "閉合" in diagnostics["eye_error"]
    normalized, world, intrinsics, _ = projected_hand(depth=2.0)
    assert estimate_hand_camera_points(normalized, world, WIDTH, HEIGHT, intrinsics, diagnostics=diagnostics) is None
    assert "距離超出" in diagnostics["hand_depth_error"]


def test_missing_or_jump_samples_cannot_reuse_stable_ray_endpoint():
    smoother = _StablePoint()
    assert smoother.update((0, 0, .6), 1.0) is None
    assert smoother.update((0, 0, .6), 1.07) is None
    assert smoother.update((0, 0, .6), 1.14) is not None
    assert smoother.update(None, 1.21) is None
    assert smoother.update((0, 0, .6), 1.28) is None
    assert smoother.update((0, 0, .6), 1.35) is None
    assert smoother.update((0, 0, .6), 1.42) is not None
    assert smoother.update((0, 0, .9), 1.49) is None


def test_fresh_three_fps_samples_can_finish_smoothing_warmup():
    smoother = _StablePoint()
    assert smoother.update((0, 0, .6), 1.0) is None
    assert smoother.update((0, 0, .6), 1.0 + 1 / 3) is None
    assert smoother.update((0, 0, .6), 1.0 + 2 / 3) == pytest.approx((0, 0, .6))
    assert smoother.timestamp == 1.0 + 2 / 3


def test_history_gap_boundary_resets_without_extending_sample_timestamp():
    smoother = _StablePoint()
    for timestamp in (1.0, 1.25, 1.5):
        smoother.update((0, 0, .6), timestamp)
    assert smoother.update((0, 0, .6), 2.25) is not None  # exactly .75 s
    assert smoother.timestamp == 2.25
    assert smoother.update((0, 0, .6), 3.01) is None  # .76 s resets warmup
    assert smoother.timestamp == 3.01
    assert smoother.update((0, 0, .6), 3.26) is None
    assert smoother.update((0, 0, .6), 3.51) is not None
    assert smoother.update((0, 0, .6), 3.51) is None  # duplicate source time


def test_low_rate_camera_geometry_keeps_actual_source_timestamps(monkeypatch):
    monkeypatch.setattr("kon.vision.classify_gao", lambda *_args, **_kwargs:
                        GestureResult(0, False, reason="finger pattern"))
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    worker = CameraWorker()
    observations = [worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.0 + index / 3)
                    for index in range(8)]
    assert observations[0].source_fps is None
    assert observations[1].source_fps == pytest.approx(3.0)
    assert "請多停留片刻" in observations[1].message
    latest = observations[-1]
    assert latest.raw_hit is not None and not latest.pose_ok
    assert latest.timestamp == 1.0 + 7 / 3
    buffer = CalibrationBuffer()
    buffer.reset(1.0)
    for observation in observations:
        buffer.observe(observation.timestamp, observation.raw_hit, reason=observation.message)
    # Six real rays now arrive after the two-frame warmup at 3 FPS, while the
    # existing latest-source .35 s calibration age limit still expires them.
    assert buffer.evaluate(latest.timestamp + .34).ready
    expired = buffer.evaluate(latest.timestamp + .36)
    assert not expired.ready and "過期" in expired.reason


@pytest.mark.parametrize("age, expected", [(0.30, "demo"), (0.31, None)])
def test_longer_smoothing_history_does_not_extend_command_pose_age(age, expected):
    gate = TriggerGate(dwell_seconds=.05)
    gate.update(1.0, True, "demo", True)
    assert gate.update(1.1, True, "demo", True)
    assert gate.hear_keyword(1.1 + age, captured_at=1.1) == expected


def test_excessive_source_gaps_explain_why_warmup_cannot_finish():
    normalized, world, *_ = projected_hand()
    face_points, _ = synthetic_face()
    hand, face = _results(((normalized, world),), face_points)
    worker = CameraWorker()
    for timestamp in (1.0, 2.0, 3.0):
        observation = worker._make_tracking(hand, face, WIDTH, HEIGHT, timestamp)
    assert observation.raw_hit is None and observation.depth_status == "stabilizing"
    assert observation.source_fps == 1.0
    assert "追蹤間隔過長" in observation.message


def test_preview_is_mirrored_and_owns_its_pixel_storage():
    pixels = np.zeros((12, 20, 3), dtype=np.uint8)
    pixels[:, :5] = (0, 0, 255)  # red BGR strip at camera-image left
    observation = TrackingFrame(0, (), (), 0, False, None, None, None, "")
    preview = CameraWorker._preview(pixels, observation)
    assert preview.width() == 20 and preview.height() == 12
    assert preview.pixelColor(19, 5).red() == 255
    assert preview.pixelColor(0, 5).red() == 0
    pixels[:] = 0
    assert preview.pixelColor(19, 5).red() == 255


def test_hand_preview_never_draws_eye_markers_and_tolerates_invalid_hand_points():
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    face, _intrinsics = synthetic_face()
    invalid = np.full((21, 3), np.nan)
    observation = TrackingFrame(0, tuple(map(tuple, invalid)), tuple(map(tuple, face)),
                                0, False, None, None, None, "", aiming_mode="hand")
    preview = CameraWorker._preview(pixels, observation)
    assert preview.pixelColor(WIDTH - 1 - 455, 240).getRgb()[:3] == (0, 0, 0)


def test_constructor_does_not_activate_camera(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("camera access before start")
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    worker = CameraWorker()
    worker.stop()


def test_preview_rate_uses_source_time_and_reenable_is_immediate():
    worker = CameraWorker()
    assert worker._preview_due(10.)
    worker.set_preview_enabled(True)  # Repeated visibility updates cannot bypass 15 FPS.
    assert not worker._preview_due(10 + 1 / 30)
    assert worker._preview_due(10 + 2 / 30)
    assert not worker._preview_due(10 + 3 / 30)
    worker.set_preview_enabled(False)
    assert not worker._preview_due(11.)
    worker.set_preview_enabled(True)
    assert worker._preview_due(11.001)
    assert not worker._preview_due(float("nan"))


def test_capture_producer_replaces_unconsumed_frames_instead_of_building_latency():
    stop = threading.Event()
    unblock = threading.Event()
    released = threading.Event()

    class Capture:
        count = 0
        def read(self):
            self.count += 1
            if self.count <= 3:
                return True, np.full((3, 4, 3), self.count, dtype=np.uint8)
            unblock.wait(2)
            return False, None
        def release(self):
            released.set()

    reader = _LatestCapture(Capture(), stop)
    reader.start()
    try:
        with reader.condition:
            assert reader.condition.wait_for(
                lambda: reader.latest is not None and reader.latest[1][0, 0, 0] == 3, timeout=2)
        timestamp, pixels = reader.take()
        assert pixels[0, 0, 0] == 3  # The two older sources were discarded.
        assert 0 <= time.monotonic() - timestamp < .3
    finally:
        stop.set()
        unblock.set()
        reader.join(timeout=3)
    assert not reader.is_alive() and released.is_set()
    assert reader.take() is None


@pytest.mark.parametrize("aim_mode", ["eye_ray", "hand"])
@pytest.mark.parametrize("preview_enabled", [True, False])
def test_worker_stop_releases_capture_and_only_loads_required_models(monkeypatch, tmp_path, aim_mode, preview_enabled):
    # Stub the hardware and inference engines while exercising the real QThread
    # and capture producer. No system camera is touched by this test.
    from mediapipe.tasks.python import vision

    opened = threading.Event()
    released = threading.Event()
    observed = threading.Event()
    previewed = threading.Event()
    configured = []
    observations = []
    messages = []
    images = []

    class FakeCapture:
        def __init__(self, *_args):
            opened.set()
        def isOpened(self):
            return True
        def set(self, *args):
            configured.append(args)
            return True
        def read(self):
            threading.Event().wait(.005)
            # Simulate a driver ignoring the 16:9 request and delivering 4:3.
            return True, np.zeros((600, 800, 3), dtype=np.uint8)
        def release(self):
            released.set()

    class FakeModel:
        def __init__(self, result, *, delay_first=False):
            self.result = result
            self.delay_first = delay_first
            self.closed = False
            self.inputs = []
        def detect_for_video(self, *_args):
            self.inputs.append(_args)
            if self.delay_first and len(self.inputs) == 1:
                # Startup work can outlast the publication age limit. Exercise
                # that case deliberately, then allow fresh frames to proceed.
                threading.Event().wait(.4)
            return self.result
        def close(self):
            self.closed = True

    hand = FakeModel(SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[]), delay_first=True)
    face = FakeModel(SimpleNamespace(face_landmarks=[]))
    (tmp_path / "hand_landmarker.task").write_bytes(b"fake asset for the stub")
    monkeypatch.setattr("kon.vision.MODEL_DIR", tmp_path)
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", lambda options: hand)
    if aim_mode == "hand":
        # A hand-only installation does not need the face model asset, and
        # must not construct options or the face model even if available.
        def forbidden_face(*_args, **_kwargs):
            raise AssertionError("hand mode must never load or construct a face tracker")
        monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", forbidden_face)
        monkeypatch.setattr(vision, "FaceLandmarkerOptions", forbidden_face)
    else:
        (tmp_path / "face_landmarker.task").write_bytes(b"fake asset for the stub")
        monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", lambda options: face)
    worker = CameraWorker(aim_mode=aim_mode)
    worker.set_preview_enabled(preview_enabled)
    if not preview_enabled:
        def forbidden_preview(*_args):
            raise AssertionError("disabled previews must not be generated")
        monkeypatch.setattr(worker, "_preview", forbidden_preview)
    worker.status.connect(messages.append, Qt.ConnectionType.DirectConnection)
    def received_tracking(frame):
        if frame.source_size is not None:
            observations.append(frame)
            observed.set()
    worker.tracking.connect(received_tracking, Qt.ConnectionType.DirectConnection)
    def received_preview(image):
        images.append(image)
        previewed.set()
    worker.frame.connect(received_preview, Qt.ConnectionType.DirectConnection)
    worker.start()
    try:
        assert opened.wait(5), "worker never reached fake capture"
        assert observed.wait(5), "worker never published the fake delivered mode"
        if preview_enabled:
            assert previewed.wait(5), "worker never published its enabled preview"
    finally:
        worker.stop()
        assert worker.wait(4000), "worker failed to stop"
    assert released.is_set() and hand.closed
    assert face.closed is (aim_mode == "eye_ray")
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in configured
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in configured
    assert observations[0].source_size == (800, 600)
    assert observations[0].inference_size == (720, 540)
    assert observations[0].aiming_mode == aim_mode
    assert bool(images) is preview_enabled
    input_ms = [arguments[1] for arguments in hand.inputs]
    observation_ms = [int(frame.timestamp * 1000) for frame in observations]
    assert input_ms == sorted(set(input_ms))
    assert observation_ms == sorted(set(observation_ms))
    # Stale inference is discarded, not relabeled with the time it completed.
    # Every published observation must still identify an actual source input.
    assert input_ms[0] not in observation_ms
    assert all(timestamp in input_ms for timestamp in observation_ms)
    if aim_mode == "eye_ray":
        assert len(hand.inputs) == len(face.inputs)
        for hand_input, face_input in zip(hand.inputs, face.inputs):
            assert hand_input[0] is face_input[0] and hand_input[1] == face_input[1]
    if aim_mode == "hand":
        assert observations[0].depth_status == "no_hand" and observations[0].face_points == ()
        assert any("移動手圈" in message for message in messages)
    assert any("800×600" in message and "視角可能因鏡頭裁切而不同" in message for message in messages)


def test_local_cpu_video_models_load_infer_and_close():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    root = Path(__file__).resolve().parent.parent
    if not all((root / "models" / name).is_file() for name in
               ("hand_landmarker.task", "face_landmarker.task")):
        pytest.skip("Download the public models to run local inference integration coverage")
    base = python.BaseOptions
    blank = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(root / "models/hand_landmarker.task"), delegate=base.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
    )) as hand:
        with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(root / "models/face_landmarker.task"), delegate=base.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
        )) as face:
            assert hand.detect_for_video(blank, 1).hand_landmarks == []
            assert face.detect_for_video(blank, 1).face_landmarks == []
