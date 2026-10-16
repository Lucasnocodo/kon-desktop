import json
import math

import numpy as np
import pytest

from kon.calibration_diagnostics import FitStatus, analyze_calibration
from kon.geometry import ScreenCalibration


def grid():
    return np.array([(x, y) for y in (.18, .5, .82) for x in (.18, .5, .82)])


def test_affine_mirror_skew_passes_and_preserves_all_per_point_diagnostics():
    target = grid()
    raw = target @ np.array([[-.35, .03], [.07, .22]]) + [.12, -.15]
    report = analyze_calibration(raw, target)
    assert report.status is FitStatus.PASSED and report.passed
    assert report.calibration is not None
    assert report.rms_error < 1e-12 and report.max_error < 1e-12
    assert report.design_rank == 3 and report.sample_count == 9
    assert np.asarray(report.predicted_points) == pytest.approx(target)
    assert report.predictions == report.predicted_points
    assert report.per_point_errors == report.point_errors
    assert report.residuals == report.residual_vectors
    assert all(error < 1e-12 for error in report.leave_one_out_errors)
    assert report.suggested_point_index is None
    json.dumps(report.to_dict(), allow_nan=False)


def test_rms_087_max_123_fails_original_average_limit_with_point_errors_available():
    raw = grid()
    noise = np.array([(x, y) for y in (1, -2, 1) for x in (1, -2, 1)]) * .0435
    target = raw + noise
    report = analyze_calibration(raw, target)
    assert report.status is FitStatus.HIGH_RESIDUAL
    assert not report.passed and report.calibration is None
    assert report.rms_error == pytest.approx(.087)
    assert report.max_error == pytest.approx(math.sqrt(8) * .0435)
    assert report.max_error < .16
    assert len(report.point_errors) == len(report.predicted_points) == 9
    assert report.suggested_point_index == 4
    assert "8.7%" in report.reason and "第 5 點" in report.reason
    with pytest.raises(ValueError, match="residual"):
        ScreenCalibration.fit(raw, target, max_rms_error=.08, max_point_error=.16)


def test_single_bad_sample_has_largest_residual_and_is_not_silently_removed():
    raw, target = grid(), grid()
    target[4, 0] += .25
    report = analyze_calibration(raw, target)
    assert report.status is FitStatus.HIGH_RESIDUAL and report.suggested_point_index == 4
    assert report.sample_count == 9 and len(report.point_errors) == 9
    assert report.leave_one_out_errors[4] == pytest.approx(.25)
    assert report.max_error > .16 and report.rms_error < .08
    assert report.calibration is None
    target[4, 0] -= .25
    assert analyze_calibration(raw, target).passed


def test_residual_vectors_point_from_target_to_prediction():
    raw, target = grid(), grid()
    target[4, 0] += .25
    report = analyze_calibration(raw, target)
    assert np.array(report.target_points) + np.array(report.residual_vectors) == pytest.approx(np.array(report.predicted_points))
    assert np.linalg.norm(report.residual_vectors[4]) == pytest.approx(report.point_errors[4])


@pytest.mark.parametrize("raw,target,status", [
    ([], [], FitStatus.INSUFFICIENT_SAMPLES),
    (grid()[:4], grid()[:4], FitStatus.INSUFFICIENT_SAMPLES),
    (grid()[:, :1], grid(), FitStatus.INVALID_SAMPLES),
    ([[math.nan, 0]] * 9, grid(), FitStatus.INVALID_SAMPLES),
    ([[0, 0]] * 9, grid(), FitStatus.INSUFFICIENT_SPREAD),
    (grid(), grid() * .1, FitStatus.SCREEN_COVERAGE),
    (np.c_[np.arange(9), np.arange(9) * 2], grid(), FitStatus.COLLINEAR_RAYS),
])
def test_rejected_data_has_typed_actionable_reasons(raw, target, status):
    report = analyze_calibration(raw, target)
    assert report.status is status and report.reason
    assert not report.passed and report.calibration is None
    assert report.predicted_points == () and report.point_errors == ()
    json.dumps(report.to_dict(), allow_nan=False)


def test_degenerate_mapping_is_distinct_from_collinear_input():
    raw = grid()
    target = np.c_[raw[:, 0], raw[:, 0]]
    report = analyze_calibration(raw, target)
    assert report.status is FitStatus.DEGENERATE_MAPPING
    assert report.design_rank == 3


def test_custom_stricter_limits_are_honored_without_changing_default_thresholds():
    raw, target = grid(), grid()
    target[4, 0] += .02
    assert analyze_calibration(raw, target).passed
    assert not analyze_calibration(raw, target, max_rms_error=.001).passed
    assert analyze_calibration(raw, target).max_rms_error == .08
    assert analyze_calibration(raw, target).max_point_error == .16


@pytest.mark.parametrize("limit", [0, -1, math.nan, math.inf])
def test_invalid_error_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        analyze_calibration(grid(), grid(), max_rms_error=limit)
