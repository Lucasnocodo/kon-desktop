import math

import numpy as np
import pytest

from kon.calibration import CalibrationBuffer


def samples(buffer, start=0., count=6, dt=.06, point=(.1, -.2)):
    for index in range(count):
        buffer.observe(start + index * dt, point)
    return start + (count - 1) * dt


def test_six_unique_stable_source_samples_covering_quarter_second_are_ready():
    buffer = CalibrationBuffer()
    buffer.reset(10.)
    latest = samples(buffer, start=10.)
    state = buffer.evaluate(latest + .05)
    assert state.ready and state.count == state.required == 6
    assert state.point == pytest.approx((.1, -.2))


def test_unique_sample_count_alone_does_not_prove_temporal_stability():
    buffer = CalibrationBuffer()
    latest = samples(buffer, dt=.001)
    state = buffer.evaluate(latest)
    assert not state.ready and state.count == 6
    assert "時間" in state.reason


def test_duplicate_and_out_of_order_samples_never_add_frames_or_override_latest():
    buffer = CalibrationBuffer()
    latest = samples(buffer)
    assert not buffer.observe(latest, None, "duplicate")
    assert not buffer.observe(latest - .1, None, "old")
    state = buffer.evaluate(latest)
    assert state.ready and state.count == 6


def test_invalid_observation_retains_history_but_blocks_until_new_valid_ray():
    buffer = CalibrationBuffer()
    latest = samples(buffer)
    buffer.observe(latest + .06, None, "請讓慣用眼重新入鏡。")
    state = buffer.evaluate(latest + .07)
    assert not state.ready and state.count == 6
    assert state.reason == "請讓慣用眼重新入鏡。"
    buffer.observe(latest + .12, (.1, -.2))
    assert buffer.evaluate(latest + .13).ready


@pytest.mark.parametrize("bad_hit", [(math.nan, 0), (0, math.inf), (0,), "oops", [[0, 0]]])
def test_nonfinite_or_malformed_ray_is_not_a_valid_calibration_sample(bad_hit):
    buffer = CalibrationBuffer()
    latest = samples(buffer)
    buffer.observe(latest + .06, bad_hit)
    state = buffer.evaluate(latest + .07)
    assert not state.ready and state.count == 6


def test_32ms_over_old_strict_age_is_accepted_with_350ms_budget():
    buffer = CalibrationBuffer()
    latest = samples(buffer)
    assert buffer.evaluate(latest + .32).ready
    assert buffer.evaluate(latest + .35).ready
    assert not buffer.evaluate(latest + .351).ready


def test_queued_source_frame_from_previous_marker_is_ignored():
    buffer = CalibrationBuffer()
    samples(buffer, start=10.)
    buffer.reset(11.)
    assert not buffer.observe(10.9, (.1, -.2))
    assert buffer.evaluate(11.).count == 0
    latest = samples(buffer, start=11.)
    assert buffer.evaluate(latest).ready


def test_single_extreme_middle_outlier_does_not_reject_stable_majority():
    buffer = CalibrationBuffer()
    for index in range(8):
        hit = (.5, -.6) if index == 3 else (.1 + index * .00005, -.2)
        buffer.observe(index * .06, hit)
    state = buffer.evaluate(.43)
    assert state.ready and state.count == 8
    assert state.point == pytest.approx((.1002, -.2), abs=.0002)


def test_newest_outlier_waits_for_new_stable_ray_instead_of_capturing_old_aim():
    buffer = CalibrationBuffer()
    latest = samples(buffer, count=7)
    buffer.observe(latest + .06, (.5, -.6))
    assert not buffer.evaluate(latest + .07).ready
    buffer.observe(latest + .12, (.1, -.2))
    assert buffer.evaluate(latest + .13).ready


def test_many_disagreeing_rays_cannot_be_hidden_as_outliers():
    buffer = CalibrationBuffer()
    for index in range(10):
        buffer.observe(index * .06, (.1, -.2) if index < 6 else (.3, -.4))
    assert not buffer.evaluate(.55).ready


def test_moving_hand_is_rejected_without_loosening_12mm_stability_budget():
    buffer = CalibrationBuffer()
    for index in range(8):
        buffer.observe(index * .06, (.1 + index * .012, -.2))
    state = buffer.evaluate(.43)
    assert not state.ready
    assert "移動" in state.reason or "改變" in state.reason


def test_small_camera_noise_is_accepted():
    buffer = CalibrationBuffer()
    rng = np.random.default_rng(42)
    for index in range(8):
        buffer.observe(index * .06, np.array([.1, -.2]) + rng.normal(0, .001, 2))
    assert buffer.evaluate(.43).ready


def test_history_expires_and_future_source_time_is_not_fresh():
    buffer = CalibrationBuffer()
    latest = samples(buffer)
    assert not buffer.evaluate(latest - .01).ready
    state = buffer.evaluate(latest + 2.36)
    assert not state.ready and state.count == 0


@pytest.mark.parametrize("fps", [3, 5, 15])
def test_initial_capture_works_at_three_five_and_fifteen_source_fps(fps):
    buffer = CalibrationBuffer()
    buffer.reset(10.)
    latest = samples(buffer, start=10., dt=1 / fps)
    state = buffer.evaluate(latest + .04)
    assert state.ready and state.count == 6
    assert state.point == pytest.approx((.1, -.2))
    assert latest - 10. < 3.0


def test_longer_low_fps_history_still_requires_a_fresh_actual_ray():
    buffer = CalibrationBuffer()
    latest = samples(buffer, dt=1 / 3)
    assert buffer.evaluate(latest + .35).ready
    assert not buffer.evaluate(latest + .36).ready


@pytest.mark.parametrize("fps", [3, 5, 15])
def test_sustained_changed_aim_cannot_be_captured_by_longer_history(fps):
    buffer = CalibrationBuffer()
    buffer.reset(0.)
    for index in range(6):
        buffer.observe(index / fps, (.1 + index * .012, -.2))
    state = buffer.evaluate(5 / fps + .04)
    assert not state.ready
    assert "移動" in state.reason or "改變" in state.reason


def test_low_fps_old_stable_aim_cannot_mask_new_changed_aim():
    buffer = CalibrationBuffer()
    samples(buffer, dt=1 / 3)
    buffer.observe(2., (.25, -.2))
    assert not buffer.evaluate(2.01).ready
    buffer.observe(7 / 3, (.25, -.2))
    assert not buffer.evaluate(7 / 3 + .01).ready


def test_shorter_history_remains_available_when_explicitly_configured():
    buffer = CalibrationBuffer(window=.8)
    latest = samples(buffer)
    assert buffer.evaluate(latest + .05).ready
    assert buffer.evaluate(latest + 1.16).count == 0


@pytest.mark.parametrize("kwargs", [
    {"min_samples": 1}, {"min_samples": True}, {"min_span": 0},
    {"window": .1}, {"max_age": 2.1}, {"max_jitter": math.nan},
])
def test_invalid_buffer_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CalibrationBuffer(**kwargs)
