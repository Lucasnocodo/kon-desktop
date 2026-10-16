import json
import math

import numpy as np
import pytest

from kon.personal_calibration import DEFAULT_TARGETS, PersonalCalibration


SIZE = (3440,1440)


def affine_raw(targets):
    target = np.array(targets)
    return np.c_[.3-.6*target[:,0]+.03*target[:,1], .05+.3*target[:,1]]


def model():
    return PersonalCalibration.fit(affine_raw(DEFAULT_TARGETS),screen_size=SIZE)


def good_validation(value, frame_samples=True):
    targets = value.validation_targets
    raw = affine_raw(targets)
    frames = [[point+[delta,-delta] for delta in (-.00005,0,.00005,-.00003,0,.00003)] for point in raw]
    return value.validate(raw,targets,screen_size=SIZE,frame_samples=frames if frame_samples else None)


def test_nine_personal_controls_are_adopted_exactly_without_claiming_accuracy():
    value = model()
    for raw,target in zip(value.raw_points,DEFAULT_TARGETS):
        assert value.predict(raw) == pytest.approx(target)
    assert value.sample_count == 9
    assert value.interpolation_rms_error < 1e-12
    assert value.rms_error is None and value.max_error is None
    assert not value.validated and not value.ready_for_live


def test_consistent_ghost_border_maps_frame_edges_and_rejects_unbounded_extrapolation():
    value = model()
    assert value.full_screen_coverage
    assert value.coverage_bounds == (0.,0.,1.,1.)
    for target in ((0,0),(1,0),(0,1),(1,1),(.99,.5)):
        assert value.predict(affine_raw([target])[0]) == pytest.approx(target)
    assert value.predict((10,10)) is None
    assert value.predict(affine_raw([(1.01,.5)])[0]) is None


def test_real_failed_affine_capture_can_be_personally_interpolated_with_safer_diagonals():
    raw = [( .313885,.144149),(.019087,.185385),(-.278105,.152967),
           (.276017,.209476),(.052114,.222768),(-.228049,.237526),
           (.271427,.303387),(.022755,.244130),(-.254646,.363734)]
    value = PersonalCalibration.fit(raw,screen_size=SIZE)
    for point,target in zip(raw,DEFAULT_TARGETS):
        assert value.predict(point) == pytest.approx(target)
    assert value.condition_number <= 25
    assert value.max_jacobian_gain <= 6
    assert value.max_noise_gain_pixels_per_mm > 10  # sensitivity is visible, not hidden
    assert not value.ready_for_live


def test_nonlinear_monotonic_nine_points_need_no_global_affine_fit():
    target = np.array(DEFAULT_TARGETS)
    raw = np.c_[-target[:,0],target[:,1]**1.8]
    value = PersonalCalibration.fit(raw,screen_size=SIZE)
    for point,expected in zip(raw,target):
        assert value.predict(point) == pytest.approx(expected)


@pytest.mark.parametrize("change", ["duplicate","swap","collapse","fold"])
def test_folded_or_collapsed_mappings_are_rejected(change):
    raw = affine_raw(DEFAULT_TARGETS)
    if change == "duplicate":
        raw[7] = raw[4]
    elif change == "swap":
        raw[[3,5]] = raw[[5,3]]
    elif change == "collapse":
        raw[:,1] = .2
    else:
        raw[4] = raw[8]+(.1,.1)
    with pytest.raises(ValueError):
        PersonalCalibration.fit(raw,screen_size=SIZE)


def test_new_holdouts_and_their_frame_errors_enable_live_only_after_verification():
    value = model()
    report = good_validation(value)
    assert report.passed and report.frame_samples_checked
    assert report.sample_count == 9 and value.validated and value.ready_for_live
    assert report.rms_pixels < 1e-9 and report.max_pixels < 1e-9
    assert max(report.point_p95_pixels) < 1
    assert report.point_frame_counts == (6,)*9
    assert value.rms_error == report.rms_error


def test_median_only_success_does_not_skip_required_frame_quality_check():
    value = model()
    report = good_validation(value,frame_samples=False)
    assert report.passed
    assert not report.frame_samples_checked and not value.ready_for_live


def test_point_pixel_errors_use_actual_ultrawide_screen_dimensions():
    value = model()
    targets = np.array(value.validation_targets)
    shifted = targets.copy()
    shifted[4,0] += .02  # 68.8 px, rather than normalized Euclidean .02 claim
    report = value.validate(affine_raw(shifted),targets,screen_size=SIZE)
    assert not report.passed and report.status == "high_error"
    assert report.point_pixel_errors[4] == pytest.approx(68.8)
    assert not value.ready_for_live


def test_training_zero_cannot_hide_bad_independent_holdouts():
    value = model()
    targets = np.array(value.validation_targets)
    shifted = targets.copy()
    shifted[4:8,1] += .04
    report = value.validate(affine_raw(shifted),targets,screen_size=SIZE)
    assert not report.passed
    assert value.interpolation_rms_error < 1e-12 and report.max_pixels > 40


def test_good_medians_with_unstable_frames_do_not_pass():
    value = model()
    targets = value.validation_targets
    raw = affine_raw(targets)
    frames = [[point.copy() for _ in range(6)] for point in raw]
    frames[4] = [raw[4]+[delta,0] for delta in (-.02,.02,-.02,.02,-.02,.02)]
    report = value.validate(raw,targets,screen_size=SIZE,frame_samples=frames)
    assert report.rms_pixels < 1e-9
    assert report.point_p95_pixels[4] > 40
    assert not report.passed and report.status == "unstable_frames"


def test_fewer_than_six_frames_or_missing_required_locations_fail():
    value = model()
    targets = value.validation_targets
    raw = affine_raw(targets)
    frames = [[point.copy() for _ in range(5)] for point in raw]
    report = value.validate(raw,targets,screen_size=SIZE,frame_samples=frames)
    assert not report.passed and report.status == "insufficient_frames"
    repeated = [targets[0]]*9
    report = value.validate(affine_raw(repeated),repeated,screen_size=SIZE)
    assert not report.passed and report.status == "insufficient_coverage"


def test_candidate_and_verified_model_roundtrip_rechecks_saved_validation(tmp_path):
    value = model()
    path = tmp_path/"personal.json"
    value.save(path)
    loaded = PersonalCalibration.load(path)
    assert not loaded.ready_for_live and loaded.rms_error is None
    good_validation(value)
    value.save(path)
    loaded = PersonalCalibration.load(path)
    assert loaded.ready_for_live and loaded.validation.passed
    payload = json.loads(path.read_text())
    payload["validation_request"]["raw_points"][4][0] += .02
    payload["validation"]["passed"] = True  # saved assertion is never trusted
    path.write_text(json.dumps(payload))
    loaded = PersonalCalibration.load(path)
    assert not loaded.ready_for_live and not loaded.validation.passed


@pytest.mark.parametrize("point", [(math.nan,0),(0,math.inf),(0,),"bad"])
def test_invalid_live_points_are_rejected(point):
    assert model().predict(point) is None


def test_validation_json_contains_no_infinite_error_for_missing_support():
    value = model()
    targets = value.validation_targets
    raw = affine_raw(targets)
    raw[4] = (20,20)
    report = value.validate(raw,targets,screen_size=SIZE)
    assert not report.passed and report.status == "outside_support"
    json.dumps(report.to_dict(),allow_nan=False)


def test_saved_geometry_is_checked_again_instead_of_trusting_file_flags(tmp_path):
    value = model()
    path = tmp_path/"personal.json"
    value.save(path)
    payload = json.loads(path.read_text())
    payload["raw_points"][7] = payload["raw_points"][4]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        PersonalCalibration.load(path)


def test_a_failed_new_validation_cannot_leave_previous_live_verification_active():
    value = model()
    good_validation(value)
    assert value.ready_for_live
    with pytest.raises(ValueError):
        value.validate([],[],screen_size=SIZE)
    assert not value.validated and not value.ready_for_live


def test_pixel_validation_must_use_the_same_screen_size_as_mapping_selection():
    value = model()
    with pytest.raises(ValueError,match="螢幕尺寸"):
        value.validate(affine_raw(value.validation_targets),value.validation_targets,screen_size=(1280,720))


def test_missing_mapping_support_is_reported_as_coverage_even_with_frame_samples():
    value = model()
    targets = value.validation_targets
    raw = affine_raw(targets)
    raw[4] = (20,20)
    frames = [[point.copy() for _ in range(6)] for point in raw]
    report = value.validate(raw,targets,screen_size=SIZE,frame_samples=frames)
    assert report.status == "outside_support"
    assert not report.passed and not value.ready_for_live


def test_verified_near_edge_roi_can_enable_live_without_unsafe_ghost_coverage():
    targets = [(x,y) for y in (.03,.5,.97) for x in (.03,.5,.97)]
    value = PersonalCalibration.fit(affine_raw(targets),targets,screen_size=SIZE)
    assert not value.full_screen_coverage  # narrow ghost collars need not be used
    assert value.coverage_bounds == (.03,.03,.97,.97)
    report = good_validation(value)
    assert report.passed and value.ready_for_live
    assert value.predict(affine_raw([(.01,.5)])[0]) is None


def test_central_only_actual_capture_cannot_enable_live_outside_verified_roi():
    raw = [( .313885,.144149),(.019087,.185385),(-.278105,.152967),
           (.276017,.209476),(.052114,.222768),(-.228049,.237526),
           (.271427,.303387),(.022755,.244130),(-.254646,.363734)]
    value = PersonalCalibration.fit(raw,screen_size=SIZE)
    assert not value.full_screen_coverage
    assert value.coverage_bounds == (.18,.18,.82,.82)
    assert not value.ready_for_live
    # Even a synthetically forced passing report cannot expand geometry ROI.
    reference = model()
    value.validation = good_validation(reference)
    value.validated = True
    assert not value.ready_for_live
