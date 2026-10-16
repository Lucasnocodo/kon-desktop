from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from kon.stereo import (
    CalibrationQualityError, CameraCalibration, ScreenPlane, StereoCalibration, checkerboard_object_points,
    detect_checkerboard, solve_stereo_calibration, triangulate_points,
)


def calibrated_pair():
    first = CameraCalibration((960, 720), [[760, 0, 480], [0, 765, 360], [0, 0, 1]],
                              [-.12, .04, .001, -.001, 0])
    second = CameraCalibration((1280, 720), [[940, 0, 640], [0, 930, 360], [0, 0, 1]],
                               [-.08, .02, -.001, .001, 0])
    rotation = cv2.Rodrigues(np.array([.01, .12, -.015]))[0]
    return StereoCalibration(first, second, rotation, np.array([-.18, .01, .005]),
                             primary_id="usb-camera", secondary_id="iphone-15-pro")


def project(calibration, points):
    first = cv2.projectPoints(np.asarray(points, dtype=float), np.zeros(3), np.zeros(3),
                              calibration.primary.matrix, calibration.primary.distortion)[0]
    second = cv2.projectPoints(np.asarray(points, dtype=float), cv2.Rodrigues(calibration.rotation)[0],
                               calibration.translation, calibration.secondary.matrix,
                               calibration.secondary.distortion)[0]
    return first.reshape(-1, 2), second.reshape(-1, 2)


def board_observations(*, noise=0., frontal=False):
    calibration = calibrated_pair()
    objects, first, second = [], [], []
    rng = np.random.default_rng(71)
    for index in range(20):
        board = checkerboard_object_points()
        rotation = np.zeros(3) if frontal else np.array([
            .35 * np.sin(index * 1.21), .4 * np.cos(index * .83), .18 * np.sin(index * .71)])
        translation = np.array([-.20 + .23 * (index % 5) / 4,
                                -.16 + .22 * (index // 5) / 3, .65 + .5 * (index % 4) / 3])
        points = board @ cv2.Rodrigues(rotation)[0].T + translation
        p1, p2 = project(calibration, points)
        objects.append(board)
        first.append(p1 + rng.normal(0, noise, p1.shape))
        second.append(p2 + rng.normal(0, noise, p2.shape))
    return calibration, objects, first, second


def test_distorted_unequal_resolution_triangulation_recovers_metric_coordinates():
    calibration = calibrated_pair()
    points = np.array([[-.07, -.04, .65], [.08, .12, .9], [.2, -.1, 1.3]])
    first, second = project(calibration, points)
    assert triangulate_points(calibration, first, second) == pytest.approx(points, abs=2e-6)


def test_calibration_round_trip_identity_and_capture_mode_validation(tmp_path):
    calibration = calibrated_pair()
    path = tmp_path / "stereo.json"
    calibration.save(path)
    loaded = StereoCalibration.load(path)
    assert loaded.calibration_id == calibration.calibration_id
    assert loaded.baseline_m == pytest.approx(np.linalg.norm(calibration.translation))
    assert loaded.rotation == pytest.approx(calibration.rotation)
    loaded.validate_sizes((960, 720), (1280, 720))
    with pytest.raises(ValueError, match="尺寸"):
        loaded.validate_sizes((1280, 720), (1280, 720))
    corrupt = json.loads(path.read_text())
    corrupt["translation"][0] -= .01
    with pytest.raises(ValueError, match="指紋"):
        StereoCalibration.from_dict(corrupt)
    assert not loaded.rotation.flags.writeable
    assert not loaded.primary.matrix.flags.writeable
    assert not list(tmp_path.glob("*.tmp"))


def test_calibration_id_includes_camera_order_and_physical_device_identity():
    calibration = calibrated_pair()
    data = calibration.to_dict()
    data["primary_id"] = "different-usb-camera"
    data.pop("calibration_id")
    with pytest.raises(ValueError, match="指紋"):
        StereoCalibration.from_dict(data)
    changed = StereoCalibration(calibration.primary, calibration.secondary, calibration.rotation,
                                calibration.translation, primary_id="different-usb-camera")
    assert changed.calibration_id != calibration.calibration_id


@pytest.mark.parametrize("change", ["rotation", "baseline", "rms", "same-camera", "intrinsics"])
def test_invalid_calibration_cannot_be_used(change):
    calibration = calibrated_pair()
    kwargs = dict(primary=calibration.primary, secondary=calibration.secondary,
                  rotation=calibration.rotation, translation=calibration.translation)
    if change == "rotation":
        kwargs["rotation"] = np.diag([1, 1, -1])
    elif change == "baseline":
        kwargs["translation"] = np.array([180., 0, 0])  # mm mistaken for m
    elif change == "rms":
        kwargs["rms"] = 2.1
    elif change == "same-camera":
        kwargs["secondary_source"] = 0
    else:
        with pytest.raises(ValueError):
            CameraCalibration((960, 720), np.zeros((3, 3)), np.zeros(5))
        return
    with pytest.raises(ValueError):
        StereoCalibration(**kwargs)


def test_feature_mismatch_fails_reprojection_gate_instead_of_returning_depth():
    calibration = calibrated_pair()
    first, second = project(calibration, [[.02, .03, .8], [.1, .02, .9]])
    second[1, 1] += 24
    with pytest.raises(ValueError, match="不一致或不同步"):
        triangulate_points(calibration, first, second)


@pytest.mark.parametrize("bad", ["missing", "nan", "offscreen", "count"])
def test_bad_triangulation_inputs_fail_closed(bad):
    calibration = calibrated_pair()
    first, second = project(calibration, [[.02, .03, .8], [.1, .02, .9]])
    if bad == "missing":
        first = []
    elif bad == "nan":
        first[0, 0] = np.nan
    elif bad == "offscreen":
        first[0, 0] = -1
    else:
        second = second[:1]
    with pytest.raises(ValueError):
        triangulate_points(calibration, first, second)


def test_negative_depth_and_negligible_parallax_are_rejected():
    calibration = calibrated_pair()
    first, second = project(calibration, [[.02, .03, -.8]])
    with pytest.raises(ValueError, match="鏡頭後方"):
        triangulate_points(calibration, first, second)
    first, second = project(calibration, [[.02, .03, 50.]])
    with pytest.raises(ValueError, match="夾角太小"):
        triangulate_points(calibration, first, second, max_depth_m=100)


def test_solver_recovers_unequal_resolution_camera_pair_with_subpixel_noise():
    expected, objects, first, second = board_observations(noise=.08)
    solved = solve_stereo_calibration(objects, first, second,
                                      expected.primary.image_size, expected.secondary.image_size,
                                      primary_id=expected.primary_id, secondary_id=expected.secondary_id)
    assert solved.sample_count == 20
    assert solved.rms < .3
    assert solved.baseline_m == pytest.approx(expected.baseline_m, abs=.003)
    points = np.array([[-.07, -.04, .65], [.08, .12, .9], [.2, -.1, 1.3]])
    p1, p2 = project(expected, points)
    assert triangulate_points(solved, p1, p2) == pytest.approx(points, abs=.008)


def test_solver_uses_joint_board_poses_when_small_far_boards_are_noisy(monkeypatch):
    # A low-resolution primary gives an uncertain planar pose at long range.
    # Applying that mono pose to the larger secondary image falsely exceeded
    # 3 px per-view RMS, even though the joint stereo fit is below 0.5 px.
    original_extended = cv2.stereoCalibrateExtended

    def require_joint_poses(*args, **kwargs):
        result = original_extended(*args, **kwargs)
        if len(result) != 12:
            pytest.skip("This OpenCV version does not expose optimized stereo board poses")
        return result

    monkeypatch.setattr(cv2, 'stereoCalibrateExtended', require_joint_poses)
    base = calibrated_pair()
    expected = StereoCalibration(
        CameraCalibration((640, 480), [[505, 0, 320], [0, 510, 240], [0, 0, 1]],
                          base.primary.distortion),
        CameraCalibration((1920, 1080), [[1410, 0, 960], [0, 1395, 540], [0, 0, 1]],
                          base.secondary.distortion),
        base.rotation, base.translation)
    objects, first, second = [], [], []
    rng = np.random.default_rng(1)
    for index in range(20):
        board = checkerboard_object_points(square_m=.023)
        rotation = np.array([.35 * np.sin(index * 1.21), .4 * np.cos(index * .83),
                             .18 * np.sin(index * .71)])
        translation = np.array([-.24 + .38 * (index % 5) / 4,
                                -.20 + .32 * (index // 5) / 3,
                                .60 + .85 * (index % 4) / 3])
        points = board @ cv2.Rodrigues(rotation)[0].T + translation
        p1, p2 = project(expected, points)
        objects.append(board)
        first.append(p1 + rng.normal(0, .65 if index % 4 == 3 else .08, p1.shape))
        second.append(p2 + rng.normal(0, .2, p2.shape))

    solved = solve_stereo_calibration(objects, first, second,
                                      expected.primary.image_size, expected.secondary.image_size)
    assert solved.rms < .5
    assert solved.baseline_m == pytest.approx(expected.baseline_m, abs=.002)
    points = np.array([[-.07, -.04, .65], [.08, .12, .9], [.2, -.1, 1.3]])
    p1, p2 = project(expected, points)
    assert triangulate_points(solved, p1, p2) == pytest.approx(points, abs=.02)


@pytest.mark.parametrize('camera', ['primary', 'secondary'])
@pytest.mark.parametrize('corruption', ['view', 'corner'])
def test_solver_rejects_a_bad_view_or_corner_hidden_by_overall_rms(camera, corruption):
    expected, objects, first, second = board_observations(noise=.08)
    observations = first if camera == 'primary' else second
    if corruption == 'view':
        observations[4][:, 1] += 11
    else:
        observations[4][20, 1] += 12
    with pytest.raises(CalibrationQualityError, match='部分棋盤配對誤差過大') as caught:
        solve_stereo_calibration(objects, first, second,
                                  expected.primary.image_size, expected.secondary.image_size)
    report = caught.value.diagnostics
    assert report['failure_stage'] == 'per_view'
    assert all(failure['sample_id'] == 5 for failure in report['failures'])
    # A joint fit can favor the corrupted secondary observation and expose the
    # discrepancy in the primary residual. Report the measured failure, without
    # claiming which camera caused it.
    failed_camera = report['failures'][0]['camera']
    assert all(failure['camera'] == failed_camera for failure in report['failures'])
    assert ('view_rms' if corruption == 'view' else 'corner_max') in report['failed_thresholds']
    assert f'第 5 組{"主鏡頭" if failed_camera == "primary" else "副鏡頭"}' in str(caught.value)
    assert all(f'{report[key]:.3f} px' in str(caught.value)
               for key in ('primary_rms', 'secondary_rms', 'stereo_rms'))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('failed_metrics', [('primary_rms',), ('secondary_rms',), ('stereo_rms',),
                                           ('primary_rms', 'secondary_rms', 'stereo_rms')])
def test_overall_quality_failure_reports_numeric_gates_and_each_sample(monkeypatch, failed_metrics):
    expected, objects, first, second = board_observations(noise=.08)
    original_mono, original_stereo = cv2.calibrateCamera, cv2.stereoCalibrateExtended
    mono_calls = 0

    def mono(*args, **kwargs):
        nonlocal mono_calls
        result = list(original_mono(*args, **kwargs))
        metric = ('primary_rms', 'secondary_rms')[mono_calls]
        mono_calls += 1
        if metric in failed_metrics:
            result[0] = 1.75
        return tuple(result)

    def stereo(*args, **kwargs):
        result = list(original_stereo(*args, **kwargs))
        if 'stereo_rms' in failed_metrics:
            result[0] = 2.25
        return tuple(result)

    monkeypatch.setattr(cv2, 'calibrateCamera', mono)
    monkeypatch.setattr(cv2, 'stereoCalibrateExtended', stereo)
    with pytest.raises(CalibrationQualityError, match='校正重投影誤差過大') as caught:
        solve_stereo_calibration(objects, first, second,
                                  expected.primary.image_size, expected.secondary.image_size)
    report = caught.value.diagnostics
    assert report['failure_stage'] == 'overall_rms'
    assert report['failed_thresholds'] == list(failed_metrics)
    assert report['sample_count'] == 20
    assert report['units'] == 'pixels'
    assert report['limits'] == {'intrinsic_rms': 1.5, 'stereo_rms': 2.0,
                                'view_rms': 3.0, 'corner_max': 8.0}
    assert [view['sample_id'] for view in report['views']] == list(range(1, 21))
    for view in report['views']:
        for camera in ('primary', 'secondary'):
            mono_stats = view[f'mono_{camera}']
            assert 0 <= mono_stats['rms'] <= mono_stats['max']
            joint_stats = view[f'joint_{camera}']
            assert joint_stats['rms'] >= 0
            if report['joint_board_poses']:
                assert joint_stats['rms'] <= joint_stats['max']
            else:
                assert joint_stats['max'] is None
    labels = {'primary_rms': '主鏡頭', 'secondary_rms': '副鏡頭', 'stereo_rms': '雙鏡頭'}
    for metric, label in labels.items():
        assert f'{label} RMS {report[metric]:.3f} px' in str(caught.value)
        if metric in failed_metrics:
            assert f'{label} RMS 超過' in str(caught.value)
    assert all(failure['value'] > failure['limit'] for failure in report['failures'])
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_loaded_calibration_keeps_original_quality_validation():
    data = calibrated_pair().to_dict()
    data['rms'] = 2.25
    with pytest.raises(ValueError, match='校正重投影誤差過大；請重新拍攝') as caught:
        StereoCalibration.from_dict(data)
    assert type(caught.value) is ValueError


def test_solver_keeps_conservative_corner_guard_with_older_opencv(monkeypatch):
    original_extended = cv2.stereoCalibrateExtended

    def legacy_extended(*args, **kwargs):
        result = original_extended(*args, **kwargs)
        return (*result[:9], result[-1])

    monkeypatch.setattr(cv2, 'stereoCalibrateExtended', legacy_extended)
    expected, objects, first, second = board_observations(noise=.08)
    solved = solve_stereo_calibration(objects, first, second,
                                      expected.primary.image_size, expected.secondary.image_size)
    assert solved.rms < .3
    second[4][20, 1] += 12
    with pytest.raises(CalibrationQualityError, match='部分棋盤配對誤差過大') as caught:
        solve_stereo_calibration(objects, first, second,
                                  expected.primary.image_size, expected.secondary.image_size)
    assert not caught.value.diagnostics['joint_board_poses']
    assert caught.value.diagnostics['views'][4]['joint_secondary']['max'] is None


def test_solver_rejects_too_few_static_or_frontoparallel_boards():
    expected, objects, first, second = board_observations()
    args = (expected.primary.image_size, expected.secondary.image_size)
    with pytest.raises(ValueError, match="至少需要"):
        solve_stereo_calibration(objects[:14], first[:14], second[:14], *args)
    with pytest.raises(ValueError, match="位置太集中"):
        solve_stereo_calibration([objects[0]] * 15, [first[0]] * 15, [second[0]] * 15, *args)
    _, objects, first, second = board_observations(frontal=True)
    with pytest.raises(ValueError):
        solve_stereo_calibration(objects, first, second, *args)


def test_solver_rejects_board_moving_between_the_two_cameras():
    expected, objects, first, second = board_observations(noise=.08)
    for index, points in enumerate(second):
        points[:, 1] += 14 * (-1 if index % 2 else 1)
    with pytest.raises(ValueError):
        solve_stereo_calibration(objects, first, second,
                                  expected.primary.image_size, expected.secondary.image_size)


def test_checkerboard_uses_measured_square_edges_and_detects_inner_corners():
    objects = checkerboard_object_points(square_m=.025)
    assert objects.shape == (54, 3)
    assert objects[1] == pytest.approx([.025, 0, 0])
    assert objects[9] == pytest.approx([0, .025, 0])
    board = (np.indices((7, 10)).sum(axis=0) % 2 * 255).astype(np.uint8)
    board = np.pad(np.repeat(np.repeat(board, 40, axis=0), 40, axis=1), 40,
                   constant_values=255)
    corners = detect_checkerboard(board)
    assert corners is not None and corners.shape == (54, 2)
    assert detect_checkerboard(np.full((480, 640, 3), 127, dtype=np.uint8)) is None
    with pytest.raises(ValueError):
        checkerboard_object_points(square_m=25)


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (12, -15, 3), (-8, 10, -4)])
def test_screen_plane_uses_measured_camera_offset_and_tilt(pitch, yaw, roll):
    measured = dict(width_m=.53, height_m=.30, camera_x_m=.265, camera_y_m=-.025,
                    camera_forward_m=.03, pitch_degrees=pitch, yaw_degrees=yaw, roll_degrees=roll)
    plane = ScreenPlane.from_measurements(**measured)
    assert plane.measurements() == pytest.approx(measured, abs=1e-9)
    normal = plane.front_unit
    eye = plane.top_left_m + .25 * plane.right_unit + .18 * plane.down_unit + .75 * normal
    for x, y in ((.04, .05), (.45, .25), (-.01, .1)):
        target = plane.top_left_m + x * plane.right_unit + y * plane.down_unit
        aperture = eye + .55 * (target - eye)
        assert plane.ray_hit(eye, aperture) == pytest.approx((x, y), abs=1e-9)
    loaded = ScreenPlane.from_dict(plane.to_dict())
    assert loaded.plane_id == plane.plane_id


def test_screen_plane_rejects_reversed_or_parallel_ray_and_invalid_basis():
    plane = ScreenPlane.from_measurements(.53, .3, .265, -.025, .03)
    assert plane.ray_hit([0, 0, .7], [0, 0, .8]) is None
    assert plane.ray_hit([0, 0, .7], [.1, 0, .7]) is None
    assert plane.ray_hit([0, 0, .7], [0, 0, -.1]) is None
    assert plane.ray_hit([0, 0, .7], [0, 0, .68]) is None
    assert plane.ray_hit([0, np.nan, .7], [0, 0, .3]) is None
    with pytest.raises(ValueError):
        ScreenPlane([0, 0, 0], [1, 0, 0], [1, 0, 0], .53, .3)


def test_offcenter_camera_reflects_display_right_into_unmirrored_camera_left():
    # The lens is 12 cm from the user's left display edge, 2.5 cm above it,
    # and 3 cm forward. Camera image-right points toward the USER'S left.
    plane = ScreenPlane.from_measurements(.53, .3, .12, -.025, .03)
    assert plane.top_left_m == pytest.approx([.12, .025, -.03])
    assert plane.right_unit == pytest.approx([-1, 0, 0])
    assert plane.down_unit == pytest.approx([0, 1, 0])
    assert plane.front_unit == pytest.approx([0, 0, 1])
    # Physical target: 45 cm to the user's right of top-left, 15 cm down.
    # Its camera x is .12 - .45 = -.33, independently of the plane methods.
    target = np.array([-.33, .175, -.03])
    eye = np.array([0, 0, .70])
    aperture = (eye + target) / 2
    assert plane.ray_hit(eye, aperture) == pytest.approx((.45, .15), abs=1e-9)


@pytest.mark.parametrize("axis,expected", [
    ("pitch", [0, .5, np.sqrt(3) / 2]),  # Camera axis aims DOWN.
    ("yaw", [.5, 0, np.sqrt(3) / 2]),    # Camera axis aims USER-right.
    ("roll", [0, 0, 1]),               # Roll keeps the optical axis fixed.
])
def test_screen_angle_signs_follow_user_facing_display_directions(axis, expected):
    kwargs = {"pitch_degrees": 0, "yaw_degrees": 0, "roll_degrees": 0}
    kwargs[axis + "_degrees"] = 30
    plane = ScreenPlane.from_measurements(.53, .3, .12, -.025, .03, **kwargs)
    # The point 1 m down the optical axis has camera coordinates (0,0,1).
    # Convert with the display basis and remove the measured lens offset.
    offset = np.array([0, 0, 1.]) - plane.top_left_m
    screen_point = np.array([np.dot(offset, direction) for direction in
                             (plane.right_unit, plane.down_unit, plane.front_unit)])
    assert screen_point - [.12, -.025, .03] == pytest.approx(expected, abs=1e-9)
    if axis == "roll":
        # A clockwise camera body turn (as viewed by user) sends its down
        # axis down-left and its top edge toward the user's right.
        offset = np.array([0, 1., 0]) - plane.top_left_m
        screen_point = np.array([np.dot(offset, direction) for direction in
                                 (plane.right_unit, plane.down_unit, plane.front_unit)])
        assert screen_point - [.12, -.025, .03] == pytest.approx([-.5, np.sqrt(3) / 2, 0], abs=1e-9)


def test_yawed_offcenter_camera_ray_hits_independently_measured_physical_target():
    angle = np.radians(20.)
    # Camera optical axis points 20 degrees toward the user's right.
    # Independently constructed CAMERA axes in the left-handed display basis.
    camera_x = np.array([-np.cos(angle), 0, np.sin(angle)])
    camera_y = np.array([0, 1, 0])
    camera_z = np.array([np.sin(angle), 0, np.cos(angle)])
    camera_center = np.array([.12, -.025, .03])
    def camera_point(screen_point):
        delta = screen_point - camera_center
        return np.array([np.dot(delta, direction) for direction in (camera_x, camera_y, camera_z)])
    target_screen = np.array([.45, .15, 0])
    eye_screen = np.array([.26, .12, .75])
    aperture_screen = .5 * (eye_screen + target_screen)
    plane = ScreenPlane.from_measurements(.53, .3, .12, -.025, .03, yaw_degrees=20)
    assert plane.ray_hit(camera_point(eye_screen), camera_point(aperture_screen)) == pytest.approx((.45, .15))
