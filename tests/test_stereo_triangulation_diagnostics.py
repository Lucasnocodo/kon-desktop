from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from kon.stereo import (
    CameraCalibration, StereoCalibration, TriangulationQualityError, triangulate_points,
)


def rig(*, parallel=False):
    first = CameraCalibration((1280, 720), [[800, 0, 640], [0, 805, 360], [0, 0, 1]],
                              [-.12, .04, .001, -.001, 0])
    second = CameraCalibration((960, 720), [[700, 0, 480], [0, 695, 360], [0, 0, 1]],
                               [-.08, .02, -.001, .001, 0])
    rotation = np.eye(3) if parallel else cv2.Rodrigues(np.array([.01, .1, -.015]))[0]
    return StereoCalibration(first, second, rotation, [-.18, 0, 0],
                             primary_id='usb-camera', secondary_id='camo-camera')


def project(calibration, points):
    points = np.asarray(points, dtype=float)
    first = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                              calibration.primary.matrix, calibration.primary.distortion)[0].reshape(-1, 2)
    second = cv2.projectPoints(points, cv2.Rodrigues(calibration.rotation)[0], calibration.translation,
                               calibration.secondary.matrix, calibration.secondary.distortion)[0].reshape(-1, 2)
    return first, second


def assert_strict_json(report):
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_successful_unequal_resolution_depth_reports_same_inputs_and_per_point_evidence():
    calibration = rig()
    points = np.array([[-.06 + .015 * (i % 7), -.04 + .025 * (i // 7), .6 + .005 * i]
                       for i in range(21)])
    first, second = project(calibration, points)
    report = {'stale_image': object()}
    recovered = triangulate_points(calibration, first, second, diagnostics=report)
    assert recovered == pytest.approx(points, abs=2e-6)
    assert report['failing_gate'] is None
    assert report['point_count'] == 21
    assert report['failed_point_indices'] == []
    assert report['calibration_id'] == calibration.calibration_id
    assert report['primary_source_size'] == [1280, 720]
    assert report['secondary_source_size'] == [960, 720]
    assert report['primary_pixels'] == first.tolist()
    assert report['secondary_pixels'] == second.tolist()
    assert report['primary_depth_m'] == pytest.approx(points[:, 2], abs=2e-6)
    secondary_points = points @ calibration.rotation.T + calibration.translation
    assert report['secondary_depth_m'] == pytest.approx(secondary_points[:, 2], abs=2e-6)
    assert min(report['parallax_degrees']) > .5
    assert max(report['primary_reprojection_error_px']) < 1e-4
    assert max(report['secondary_reprojection_error_px']) < 1e-4
    assert report['limits'] == {'max_reprojection_error_px': 3.0, 'min_angle_degrees': .5,
                                'min_depth_m': .05, 'max_depth_m': 4.0}
    assert 'stale_image' not in report
    assert_strict_json(report)


def test_one_wrong_joint_reports_exact_index_and_pixels_that_replay_the_failure():
    calibration = rig()
    points = np.array([[-.06 + .015 * (i % 7), -.04 + .025 * (i // 7), .6 + .005 * i]
                       for i in range(21)])
    first, second = project(calibration, points)
    second[8, 1] += 24
    report = {}
    with pytest.raises(TriangulationQualityError, match='雙鏡頭特徵不一致或不同步') as caught:
        triangulate_points(calibration, first, second, diagnostics=report)
    saved = caught.value.diagnostics
    assert saved == report
    assert saved['failing_gate'] == 'reprojection'
    assert saved['failed_point_indices'] == [8]
    assert saved['worst_point_index'] == 8
    assert saved['worst_camera'] in ('primary', 'secondary')
    assert saved['worst_reprojection_error_px'] > 3
    measured = saved[f"{saved['worst_camera']}_reprojection_error_px"][8]
    assert saved['worst_reprojection_error_px'] == measured
    assert len(saved['parallax_degrees']) == len(saved['primary_depth_m']) == 21
    assert saved['primary_pixels'] == first.tolist()
    assert saved['secondary_pixels'] == second.tolist()
    assert_strict_json(saved)
    # Persisted numeric correspondences reproduce the original gate without a
    # camera, MediaPipe model, screenshot, or an assumed cause of the mismatch.
    replayed = json.loads(json.dumps(saved, allow_nan=False))
    with pytest.raises(TriangulationQualityError) as again:
        triangulate_points(calibration, replayed['primary_pixels'], replayed['secondary_pixels'])
    assert again.value.diagnostics == saved


def test_behind_camera_failure_keeps_depth_and_does_not_invent_unreached_metrics():
    calibration = rig()
    first, second = project(calibration, [[.02, .03, -.8]])
    with pytest.raises(TriangulationQualityError, match='鏡頭後方') as caught:
        triangulate_points(calibration, first, second)
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'depth'
    assert report['failed_point_indices'] == [0]
    assert report['primary_depth_m'][0] == pytest.approx(-.8, abs=2e-6)
    assert report['secondary_depth_m'][0] < 0
    assert report['parallax_degrees'] is None
    assert report['primary_reprojection_error_px'] is None
    assert report['secondary_reprojection_error_px'] is None
    assert_strict_json(report)


def test_far_point_keeps_existing_parallax_gate_and_reports_the_measured_angle():
    calibration = rig()
    first, second = project(calibration, [[.02, .03, 50.]])
    with pytest.raises(TriangulationQualityError, match='夾角太小') as caught:
        triangulate_points(calibration, first, second, max_depth_m=100)
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'parallax'
    assert report['failed_point_indices'] == [0]
    assert 0 < report['parallax_degrees'][0] < .5
    assert report['primary_depth_m'][0] == pytest.approx(50., abs=1e-4)
    assert report['primary_reprojection_error_px'] is None
    assert report['limits']['max_depth_m'] == 100
    assert_strict_json(report)


def test_parallel_rays_fail_homogeneous_gate_before_returning_any_depth():
    calibration = rig(parallel=True)
    first, second = np.array([[640., 360.]]), np.array([[480., 360.]])
    with pytest.raises(TriangulationQualityError, match='視差不足') as caught:
        triangulate_points(calibration, first, second)
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'homogeneous'
    assert report['failed_point_indices'] == [0]
    assert report['primary_depth_m'] is None
    assert_strict_json(report)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_input_is_rejected_and_saved_as_json_null(bad):
    calibration = rig()
    with pytest.raises(TriangulationQualityError, match='有限') as caught:
        triangulate_points(calibration, [[bad, 360]], [[480, 360]])
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'input'
    assert report['primary_pixels'] == [[None, 360.]]
    assert report['secondary_pixels'] == [[480., 360.]]
    assert_strict_json(report)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_solver_output_is_rejected_with_json_safe_depths(monkeypatch, bad):
    calibration = rig()
    first, second = project(calibration, [[.02, .03, .8]])
    monkeypatch.setattr(cv2, 'triangulatePoints', lambda *_args: np.array([[0.], [0.], [bad], [1.]]))
    with np.errstate(invalid='ignore'), pytest.raises(TriangulationQualityError, match='鏡頭後方') as caught:
        triangulate_points(calibration, first, second)
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'depth'
    assert report['primary_depth_m'] == [None]
    assert report['secondary_depth_m'] == [None]
    assert report['failed_point_indices'] == [0]
    assert_strict_json(report)


def test_invalid_threshold_and_mismatched_counts_remain_value_errors_with_evidence():
    calibration = rig()
    first, second = project(calibration, [[.02, .03, .8], [.03, .02, .9]])
    with pytest.raises(TriangulationQualityError, match='重投影誤差門檻') as caught:
        triangulate_points(calibration, first, second, max_reprojection_error_px=float('nan'))
    assert isinstance(caught.value, ValueError)
    report = caught.value.diagnostics
    assert report['failing_gate'] == 'thresholds'
    assert report['limits']['max_reprojection_error_px'] is None
    assert_strict_json(report)
    with pytest.raises(TriangulationQualityError, match='數量不一致') as caught:
        triangulate_points(calibration, first[:1], second)
    assert caught.value.diagnostics['failing_gate'] == 'point_count'
    assert caught.value.diagnostics['primary_pixels'] == first[:1].tolist()
    assert caught.value.diagnostics['secondary_pixels'] == second.tolist()


def test_opencv_failure_reports_the_stage_and_preserves_the_cause(monkeypatch):
    calibration = rig()
    first, second = project(calibration, [[.02, .03, .8]])

    def fail(*_args, **_kwargs):
        raise cv2.error('simulated OpenCV failure')

    monkeypatch.setattr(cv2, 'undistortPoints', fail)
    with pytest.raises(TriangulationQualityError, match='OpenCV 無法三角定位') as caught:
        triangulate_points(calibration, first, second)
    assert caught.value.diagnostics['failing_gate'] == 'undistortion'
    assert isinstance(caught.value.__cause__, cv2.error)
    assert_strict_json(caught.value.diagnostics)
