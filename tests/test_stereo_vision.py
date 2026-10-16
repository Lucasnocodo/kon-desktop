"""Synthetic two-view tracking tests; no model, camera or desktop actions."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from kon import stereo_vision as module
from kon.geometry import GestureResult, ring_center
from kon.stereo import CameraCalibration, ScreenPlane, StereoCalibration
from kon.stereo_capture import CameraSource, CapturedFrame, FramePair
from kon.stereo_vision import StereoCameraWorker, iris_pixel_center, verified_sources


def landmarks(points):
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]


def rig():
    primary = CameraCalibration((960, 720), [[760, 0, 480], [0, 765, 360], [0, 0, 1]],
                                [-.12, .04, .001, -.001, 0])
    secondary = CameraCalibration((1280, 720), [[940, 0, 640], [0, 930, 360], [0, 0, 1]],
                                  [-.08, .02, -.001, .001, 0])
    rotation = cv2.Rodrigues(np.array([.01, .12, -.015]))[0]
    calibration = StereoCalibration(primary, secondary, rotation, [-.18, .01, .005],
                                    primary_id="usb-camera", secondary_id="iphone-15-pro")
    plane = ScreenPlane.from_measurements(.53, .3, .265, -.025, .03, pitch_degrees=6, yaw_degrees=-4)
    return calibration, plane


def hand_shape():
    return np.array([
        [0, .035, .006], [-.022, .022, .004], [-.044, .005, .005],
        [-.053, -.009, .003], [-.046, -.024, .001],
        [-.028, -.018, 0], [-.029, -.043, -.005],
        [-.031, -.067, -.001], [-.032, -.089, .003],
        [0, -.03, 0], [0, -.054, -.017], [.008, -.036, -.027], [.011, -.023, -.018],
        [.025, -.022, .002], [.028, -.045, -.017], [.022, -.025, -.027], [.013, -.018, -.015],
        [.044, -.009, .005], [.05, -.035, .002], [.051, -.056, .004], [.054, -.072, .007],
    ])


def projected(calibration, points, camera):
    if camera == 0:
        intrinsics, rotation, translation = calibration.primary, np.zeros(3), np.zeros(3)
    else:
        intrinsics = calibration.secondary
        rotation = cv2.Rodrigues(calibration.rotation)[0]
        translation = calibration.translation
    pixels = cv2.projectPoints(np.asarray(points, float), rotation, translation,
                                intrinsics.matrix, intrinsics.distortion)[0].reshape(-1, 2)
    normalized = np.column_stack((pixels / intrinsics.image_size, np.zeros(len(pixels))))
    return normalized


def face_observation(calibration, eye, camera, side="right"):
    # A 3D iris patch plus eye corners/lids projected through the actual lenses.
    definitions = {"right": (468, (469, 470, 471, 472), (33, 133), (159, 145)),
                   "left": (473, (474, 475, 476, 477), (263, 362), (386, 374))}
    center, iris, corners, lids = definitions[side]
    points = np.repeat(np.asarray(eye, float)[None, :], 478, axis=0)
    for index, delta in zip(iris, ((.006, 0, 0), (0, -.006, 0), (-.006, 0, 0), (0, .006, 0))):
        points[index] += delta
    for index, delta in zip((*corners, *lids), ((-.016, 0, 0), (.016, 0, 0),
                                              (0, -.006, 0), (0, .006, 0))):
        points[index] += delta
    return projected(calibration, points, camera)


def observations(*, hand_scale=1., side="right"):
    calibration, plane = rig()
    xyz = hand_shape() * hand_scale + [.02, .025, .40]
    eye = np.array([-.02, -.04, .75])
    results = []
    for camera in (0, 1):
        normalized = projected(calibration, xyz, camera)
        # Deliberately unrelated monocular world coordinates cannot influence depth.
        hand = SimpleNamespace(hand_landmarks=[landmarks(normalized)],
                               hand_world_landmarks=[landmarks(hand_shape() * (8 + camera))],
                               handedness=[[SimpleNamespace(category_name="Right")]])
        face = SimpleNamespace(face_landmarks=[landmarks(face_observation(calibration, eye, camera, side))])
        results.extend((hand, face))
    return calibration, plane, xyz, eye, results


def observe(worker, results, timestamp, *, skew_ms=4., **kwargs):
    return worker.make_stereo_tracking(*results, timestamp, worker.stereo_calibration.primary.image_size,
                                       worker.stereo_calibration.secondary.image_size,
                                       skew_ms=skew_ms, **kwargs)


def steady(worker, results, start=1.):
    for offset in (0, .07, .14):
        result = observe(worker, results, start + offset)
    return result


@pytest.fixture
def forbid_monocular(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stereo observations must never call monocular depth or assumed FOV")
    for name in ("estimate_eye_xyz", "estimate_hand_camera_points", "camera_matrix", "ray_screen_intersection"):
        monkeypatch.setattr("kon.vision." + name, forbidden)


@pytest.mark.parametrize("side", ["left", "right"])
def test_distorted_unequal_resolution_stereo_recovers_same_physical_joints_and_iris(
        monkeypatch, forbid_monocular, side):
    calibration, plane, xyz, eye, results = observations(side=side)
    calls = []
    def classify(points, *, world_points, image_aspect_ratio):
        calls.append((np.asarray(points), np.asarray(world_points), image_aspect_ratio))
        return GestureResult(.95, True, reason="recognized", source="world")
    monkeypatch.setattr("kon.vision.classify_gao", classify)
    worker = StereoCameraWorker(calibration, plane, dominant_eye=side)
    result = steady(worker, results)
    assert result.acquisition_mode == "stereo" and result.aiming_mode == "eye_ray"
    assert result.depth_status == "ready" and result.pose_ok and result.raw_pose_ok
    assert result.eye_xyz == pytest.approx(eye, abs=2e-6)
    assert result.aperture_xyz == pytest.approx(ring_center(xyz), abs=2e-6)
    assert result.raw_hit == pytest.approx(plane.ray_hit(eye, ring_center(xyz)), abs=2e-6)
    assert result.source_size == (960, 720) and result.secondary_source_size == (1280, 720)
    assert result.pair_skew_ms == 4.
    for _normalized, shape, aspect in calls:
        assert shape == pytest.approx(xyz - xyz.mean(axis=0), abs=2e-6)
        assert aspect == 960 / 720
    assert result.hand_world_points == pytest.approx(xyz - xyz.mean(axis=0), abs=2e-6)


def test_stereo_geometry_keeps_source_pixel_sizes_when_inference_is_resized(forbid_monocular):
    calibration, plane, xyz, eye, results = observations()
    worker = StereoCameraWorker(calibration, plane)
    for stamp in (1., 1.07, 1.14):
        result = observe(worker, results, stamp, inference_size=(480, 360))
    assert result.eye_xyz == pytest.approx(eye, abs=2e-6)
    assert result.aperture_xyz == pytest.approx(ring_center(xyz), abs=2e-6)
    assert result.source_size == (960, 720)
    assert result.inference_size == (480, 360)
    assert result.raw_hit is not None


@pytest.mark.parametrize("bad", ["nohand0", "nohand1", "bothhandsabsent", "twohands", "noface0", "noface1", "twofaces",
                                  "wronglabel", "unknownlabel", "partialhand", "nan", "offscreen",
                                  "blink", "smalliris", "skew", "reprojection", "wrongsize"])
def test_missing_ambiguous_or_inconsistent_view_pauses_and_requires_fresh_stabilization(
        forbid_monocular, bad):
    calibration, plane, _xyz, _eye, results = observations()
    worker = StereoCameraWorker(calibration, plane)
    assert steady(worker, results).raw_hit is not None
    calibration2, _plane, _xyz, _eye, broken = observations()
    hand0, face0, hand1, face1 = broken
    kwargs = {}
    if bad == "nohand0":
        hand0.hand_landmarks = []
    elif bad == "nohand1":
        hand1.hand_landmarks = []
    elif bad == "bothhandsabsent":
        hand0.hand_landmarks = []
        hand1.hand_landmarks = []
    elif bad == "twohands":
        hand1.hand_landmarks *= 2
    elif bad == "noface0":
        face0.face_landmarks = []
    elif bad == "noface1":
        face1.face_landmarks = []
    elif bad == "twofaces":
        face1.face_landmarks *= 2
    elif bad == "wronglabel":
        hand1.handedness[0][0].category_name = "Left"
    elif bad == "unknownlabel":
        hand0.handedness = []
    elif bad == "partialhand":
        hand1.hand_landmarks[0].pop()
    elif bad == "nan":
        hand1.hand_landmarks[0][4].x = float("nan")
    elif bad == "offscreen":
        hand1.hand_landmarks[0][4].x = 1.01
    elif bad == "blink":
        face1.face_landmarks[0][159].y = face1.face_landmarks[0][145].y
    elif bad == "smalliris":
        for index in (469, 470, 471, 472):
            face1.face_landmarks[0][index] = face1.face_landmarks[0][468]
    elif bad == "skew":
        kwargs["skew_ms"] = 21.
    elif bad == "reprojection":
        hand1.hand_landmarks[0][4].y += .04
    else:
        worker.stereo_calibration = StereoCalibration(
            CameraCalibration((1280, 720), calibration2.primary.matrix, calibration2.primary.distortion),
            calibration2.secondary, calibration2.rotation, calibration2.translation)
        # Calling with the actual previous source dimensions must reject mode drift.
        result = worker.make_stereo_tracking(*broken, 1.21, (960, 720), (1280, 720))
    if bad != "wrongsize":
        result = observe(worker, broken, 1.21, **kwargs)
    assert result.raw_hit is None and result.eye_xyz is None and result.aperture_xyz is None
    assert not result.pose_ok and result.raw_pose_ok is False
    assert result.acquisition_mode == "stereo"
    assert result.depth_status == ("no_hand" if bad == "bothhandsabsent" else "stereo_unusable")
    if bad == "bothhandsabsent":
        assert not result.hand_detected
    worker.stereo_calibration = calibration
    recovering = observe(worker, results, 1.28)
    assert recovering.raw_hit is None and recovering.depth_status == "stabilizing"
    recovered = steady(worker, results, start=1.35)
    assert recovered.raw_hit is not None and recovered.depth_status == "ready"


def test_implausible_reconstructed_hand_scale_pauses(forbid_monocular):
    calibration, plane, _xyz, _eye, results = observations(hand_scale=.25)
    result = observe(StereoCameraWorker(calibration, plane), results, 1.)
    assert result.raw_hit is None and "手掌尺寸" in result.message


def test_rejected_hand_and_iris_are_reported_independently_without_metric_fallback(tmp_path, forbid_monocular):
    import json
    calibration, plane, _xyz, _eye, results = observations()
    results[2].hand_landmarks[0][4].y += .04
    for point in results[3].face_landmarks[0]:
        point.y += .04
    path = tmp_path / "live.json"
    worker = StereoCameraWorker(calibration, plane, diagnostics_path=path)
    result = observe(worker, results, 1.)
    assert result.raw_hit is None and result.hand_world_points == ()
    assert result.handedness == "Right"
    assert not result.pose_ok and result.hand_depth_error == result.message
    assert "手部" in result.message
    sample = json.loads(path.read_text())["samples"][0]
    assert sample["failure_stages"] == ["hand", "iris"]
    assert len(sample["hand_primary_pixels"]) == 21
    assert len(sample["iris_secondary_pixels"]) == 1
    assert sample["skew_ms"] == 4.


def test_ambiguous_primary_stereo_hand_does_not_publish_shape_or_identity(forbid_monocular):
    calibration, plane, _xyz, _eye, results = observations()
    results[0].hand_landmarks *= 2
    result = observe(StereoCameraWorker(calibration, plane), results, 1.)
    assert result.hand_points == () and result.hand_world_points == ()
    assert result.handedness == "unknown" and not result.raw_pose_ok


@pytest.mark.parametrize("side", ["left", "right"])
def test_iris_gate_uses_selected_anatomical_eye_and_ignores_normalized_z(side):
    calibration, _plane = rig()
    eye = np.array([-.02, -.04, .75])
    face = face_observation(calibration, eye, 0, side)
    expected = projected(calibration, [eye], 0)[0, :2] * calibration.primary.image_size
    face[:, 2] = np.linspace(-100, 100, 478)
    assert iris_pixel_center(face, calibration.primary.image_size, side) == pytest.approx(expected)
    with pytest.raises(ValueError):
        iris_pixel_center(face, calibration.primary.image_size, "left" if side == "right" else "right")


@pytest.mark.parametrize("bad", ["nan", "short", "blink", "tiny", "huge", "ring-center", "offscreen"])
def test_iris_quality_gates_fail_closed(bad):
    calibration, _plane = rig()
    face = face_observation(calibration, [-.02, -.04, .75], 0)
    if bad == "nan":
        face[468, 0] = float("nan")
    elif bad == "short":
        face = face[:477]
    elif bad == "blink":
        face[159] = face[145]
    elif bad == "tiny":
        face[469:473] = face[468]
    elif bad == "huge":
        face[469, 0] += .1
    elif bad == "ring-center":
        face[468, 0] += .02
    else:
        face[468, 0] = -1
    with pytest.raises(ValueError):
        iris_pixel_center(face, calibration.primary.image_size, "right")


def test_verified_source_ids_match_saved_device_order_without_opening_cameras(monkeypatch):
    calibration, _plane = rig()
    sources = [CameraSource(0, "External webcam", "usb-camera"), CameraSource(1, "iPhone 15 Pro", "iphone-15-pro")]
    monkeypatch.setattr(module, "list_camera_sources", lambda: sources)
    assert verified_sources(calibration) == tuple(sources)


@pytest.mark.parametrize("changed", ["reordered", "replaced", "missing", "unverified"])
def test_saved_source_id_refuses_different_or_unverified_device(monkeypatch, changed):
    calibration, _plane = rig()
    sources = [CameraSource(0, "External webcam", "usb-camera"), CameraSource(1, "iPhone 15 Pro", "iphone-15-pro")]
    if changed == "reordered":
        sources = [CameraSource(0, "iPhone 15 Pro", "iphone-15-pro"), CameraSource(1, "External webcam", "usb-camera")]
    elif changed == "replaced":
        sources[1] = CameraSource(1, "Other iPhone", "iphone-16")
    elif changed == "missing":
        sources.pop()
    else:
        sources[1] = CameraSource(1, "Unverified manual camera", "")
    monkeypatch.setattr(module, "list_camera_sources", lambda: sources)
    with pytest.raises(ValueError, match="裝置身分"):
        verified_sources(calibration)


@pytest.mark.parametrize("stamp", [1.14, 1.10, float("nan")])
def test_duplicate_backward_or_invalid_timestamp_never_reuses_metric_depth(stamp, forbid_monocular):
    calibration, plane, _xyz, _eye, results = observations()
    worker = StereoCameraWorker(calibration, plane)
    assert steady(worker, results).raw_hit is not None
    result = observe(worker, results, stamp)
    assert result.raw_hit is None and result.eye_xyz is None and result.aperture_xyz is None
    assert not result.pose_ok and result.raw_pose_ok is False
    assert result.acquisition_mode == "stereo"
    assert result.source_size == calibration.primary.image_size
    assert result.secondary_source_size == calibration.secondary.image_size


def test_paused_timestamp_cannot_allow_an_older_pair_to_resume_tracking(forbid_monocular):
    calibration, plane, _xyz, _eye, results = observations()
    worker = StereoCameraWorker(calibration, plane)
    assert steady(worker, results).raw_hit is not None
    worker.paused(1.4, "one source paused")
    for timestamp in (1.2, 1.27, 1.34):
        result = observe(worker, results, timestamp)
    assert result.raw_hit is None and result.eye_xyz is None and result.aperture_xyz is None
    assert not result.pose_ok
    assert result.acquisition_mode == "stereo"


def test_missing_monocular_world_data_and_arbitrary_image_z_do_not_change_stereo_depth(forbid_monocular):
    calibration, plane, xyz, eye, results = observations()
    for hand in results[::2]:
        hand.hand_world_landmarks = []
        for index, point in enumerate(hand.hand_landmarks[0]):
            point.z = index * 10 - 100
    result = steady(StereoCameraWorker(calibration, plane), results)
    assert result.raw_hit is not None
    assert result.eye_xyz == pytest.approx(eye, abs=2e-6)
    assert result.aperture_xyz == pytest.approx(ring_center(xyz), abs=2e-6)


def test_personal_gesture_model_receives_reconstructed_shape_not_independent_world_predictions(
        monkeypatch, forbid_monocular):
    calibration, plane, xyz, _eye, results = observations()
    calls = []
    class Profile:
        def match(self, _points, *, world_points, image_aspect_ratio, handedness):
            calls.append((np.asarray(world_points), handedness))
            return GestureResult(.95, True, reason="personal match", source="world")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Personal gesture must remain authoritative in stereo mode")
    monkeypatch.setattr("kon.vision.classify_gao", forbidden)
    worker = StereoCameraWorker(calibration, plane, gesture_profile=Profile())
    assert steady(worker, results).pose_ok
    assert len(calls) == 3
    for world, label in calls:
        assert world == pytest.approx(xyz - xyz.mean(axis=0), abs=2e-6)
        assert label == "Right"


@pytest.mark.parametrize("scenario", ["normal", "capture-start-failure", "capture-inference-failure", "close-retries"])
def test_worker_uses_recorded_sources_two_model_pairs_and_each_camera_timestamp(
        monkeypatch, tmp_path, forbid_monocular, scenario):
    from mediapipe.tasks.python import vision

    # Synthetic capture/inference has no wall-clock latency. Keep its source
    # timestamps and freshness clock in the same controlled time domain, so
    # first-use native imports cannot turn the first fake frame into a stale one.
    clock = [100.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calibration, plane, _xyz, _eye, results = observations()
    sources = [CameraSource(0, "External webcam", "usb-camera"), CameraSource(1, "iPhone", "iphone-15-pro")]
    monkeypatch.setattr(module, "list_camera_sources", lambda: sources)
    for name in ("hand_landmarker.task", "face_landmarker.task"):
        (tmp_path / name).write_bytes(b"stub asset: never loaded")
    monkeypatch.setattr(module, "MODEL_DIR", tmp_path)
    def forbidden_camera(*_args, **_kwargs):
        raise AssertionError("This test must never open a system camera")
    monkeypatch.setattr(cv2, "VideoCapture", forbidden_camera)

    models, hand_options, face_options = [], [], []
    class Model:
        def __init__(self, result):
            self.result, self.closed, self.timestamps = result, False, []
            models.append(self)
        def detect_for_video(self, _image, timestamp):
            self.timestamps.append(timestamp)
            if scenario == "capture-inference-failure" and self is models[0]:
                captures[0].failure = "mock camera disconnected during inference"
                worker._stop_event.set()
            return self.result
        def close(self):
            self.closed = True
    def hand_factory(options):
        hand_options.append(options)
        return Model(results[0 if len(hand_options) == 1 else 2])
    def face_factory(options):
        face_options.append(options)
        return Model(results[1 if len(face_options) == 1 else 3])
    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", hand_factory)
    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", face_factory)
    worker = StereoCameraWorker(calibration, plane)
    worker.set_preview_enabled(False)
    slept = []
    monkeypatch.setattr(worker, "msleep", slept.append)
    captures = []
    class Capture:
        failure = None
        def __init__(self, primary, secondary, **kwargs):
            self.sources, self.kwargs = (primary, secondary), kwargs
            self.pairs, self.closed, self.started, self.close_calls = [], False, False, 0
            captures.append(self)
        def start(self):
            self.started = True
            if scenario == "capture-start-failure":
                self.failure = "mock camera failed during startup"
                worker._stop_event.set()
        def take(self, timeout):
            if len(self.pairs) == 3:
                worker._stop_event.set()
                return None
            clock[0] += .1
            stamp = clock[0] - .01
            pair = FramePair(CapturedFrame(stamp, np.zeros((720, 960, 3), np.uint8)),
                             CapturedFrame(stamp + .004, np.zeros((720, 1280, 3), np.uint8)))
            self.pairs.append(pair)
            return pair
        def close(self):
            self.close_calls += 1
            if scenario == "close-retries" and self.close_calls <= 2:
                return False
            self.closed = True
            return True
    monkeypatch.setattr(module, "DualCapture", Capture)
    frames, errors = [], []
    worker.tracking.connect(frames.append, Qt.ConnectionType.DirectConnection)
    worker.error.connect(errors.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    if scenario in ("capture-start-failure", "capture-inference-failure"):
        assert len(errors) == 1 and "mock camera" in errors[0]
    elif scenario == "close-retries":
        assert len(errors) == 1 and "等待鏡頭釋放" in errors[0]
        assert slept == [100, 100] and captures[0].close_calls == 3
    else:
        assert errors == []
    assert len(captures) == 1 and captures[0].started and captures[0].closed
    assert captures[0].sources == tuple(sources)
    assert captures[0].kwargs["stop_event"] is worker._stop_event
    assert len(models) == 4 and all(model.closed for model in models)
    assert all(options.num_hands == 2 for options in hand_options)
    assert all(options.num_faces == 2 for options in face_options)
    for index, model in enumerate(models):
        expected = [int((pair.primary if index < 2 else pair.secondary).timestamp * 1000)
                    for pair in captures[0].pairs]
        assert model.timestamps == expected
    actual = [frame for frame in frames if frame.source_size is not None]
    assert len(actual) == (0 if "failure" in scenario else 3)
    assert all(frame.acquisition_mode == "stereo" for frame in frames)
    assert all(frame.pair_skew_ms == pytest.approx(4., abs=.001) for frame in actual)
    assert frames[-1].raw_hit is None and frames[-1].raw_pose_ok is False
