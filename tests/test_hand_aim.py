from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kon.hand_aim import HandAimCalibration, HandCursorSmoother


def test_default_comfortable_hand_rectangle_reaches_the_whole_screen():
    model = HandAimCalibration()
    assert not model.personalized
    assert model.bounds == ((.2, .2), (.8, .8))
    assert model.predict((.2, .2)) == pytest.approx((0, 0))
    assert model.predict((.5, .5)) == pytest.approx((.5, .5))
    assert model.predict((.8, .8)) == pytest.approx((1, 1))
    assert model.predict((-.3, 1.3)) == pytest.approx((0, 1))


def test_two_personalized_corners_preserve_mirrored_image_direction():
    model = HandAimCalibration.from_corners((.25, .30), (.75, .85))
    assert model.personalized
    assert model.predict((.25, .30)) == pytest.approx((0, 0))
    assert model.predict((.75, .85)) == pytest.approx((1, 1))
    assert model.predict((.50, .575)) == pytest.approx((.5, .5))
    # Mirroring is already done by the camera worker, never again here.
    assert model.predict((.60, .575))[0] > model.predict((.40, .575))[0]


@pytest.mark.parametrize("left,right", [
    ((.8, .2), (.2, .8)), ((.2, .8), (.8, .2)), ((.5, .5), (.5, .5)),
    ((.2, .2), (.34, .8)), ((.2, .2), (.8, .34)),
    ((-.1, .2), (.8, .8)), ((.2, .2), (1.1, .8)),
    ((math.nan, .2), (.8, .8)), ((.2,), (.8, .8)),
])
def test_corner_enrollment_rejects_reversed_tiny_or_invalid_rectangles(left, right):
    with pytest.raises(ValueError):
        HandAimCalibration.from_corners(left, right)


@pytest.mark.parametrize("point", [None, [], (.5,), (.5, .5, .5), (math.nan, .5), (math.inf, .5)])
def test_bad_hand_cursor_has_no_position(point):
    assert HandAimCalibration().predict(point) is None


def test_minimum_allowed_span_and_both_camera_edges_are_supported():
    model = HandAimCalibration.from_corners((0, .85), (.15, 1))
    assert model.predict((.075, .925)) == pytest.approx((.5, .5))


@pytest.mark.parametrize("personalized", [False, True])
def test_numeric_rectangle_save_and_restore(tmp_path, personalized):
    model = HandAimCalibration.from_corners((.25, .3), (.7, .8)) if personalized else HandAimCalibration()
    path = tmp_path / "hand-aim.json"
    model.save(path)
    restored = HandAimCalibration.load(path)
    assert restored == model
    assert restored.predict((.43, .62)) == pytest.approx(model.predict((.43, .62)))
    assert json.loads(path.read_text())["source"] == "mirrored_image_xy"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("key,value", [
    ("version", True), ("version", 2), ("source", "eye_hand_ray"),
    ("calibration_model", "affine"), ("personalized", "yes"),
    ("top_left", [True, .2]), ("top_left", [math.nan, .2]),
    ("bottom_right", [.22, .8]), ("top_left", [.9, .9]),
])
def test_invalid_saved_rectangle_is_rejected(tmp_path, key, value):
    path = tmp_path / "hand-aim.json"
    HandAimCalibration().save(path)
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="校正檔無效"):
        HandAimCalibration.load(path)


def test_failed_atomic_save_keeps_old_rectangle(tmp_path, monkeypatch):
    path = tmp_path / "hand-aim.json"
    HandAimCalibration().save(path)
    previous = path.read_bytes()
    def fail(*_args):
        raise OSError("simulated storage failure")
    monkeypatch.setattr("kon.hand_aim.os.replace", fail)
    with pytest.raises(OSError):
        HandAimCalibration.from_corners((.25, .25), (.75, .75)).save(path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_stationary_eight_pixel_camera_noise_is_strongly_reduced():
    size = np.array((3440., 1440.))
    center = np.array((.5, .5))
    smoother = HandCursorSmoother()
    smoother.update(0, center, size)
    rng = np.random.default_rng(45)
    raw, filtered = [], []
    for frame in range(1, 301):
        point = center + rng.normal(0, 8, 2) / size
        raw.append(point * size)
        filtered.append(np.array(smoother.update(frame / 15, point, size)) * size)
    raw_rms = np.sqrt(np.mean((np.array(raw)[30:] - center * size) ** 2))
    filtered_rms = np.sqrt(np.mean((np.array(filtered)[30:] - center * size) ** 2))
    assert filtered_rms < raw_rms * .45


def test_median_of_three_rejects_an_isolated_large_camera_spike():
    smoother = HandCursorSmoother()
    size = (3440, 1440)
    for frame in range(16):
        assert smoother.update(frame / 15, (.5, .5), size) == pytest.approx((.5, .5))
    assert smoother.update(16 / 15, (.95, .1), size) == pytest.approx((.5, .5))
    assert smoother.update(17 / 15, (.5, .5), size) == pytest.approx((.5, .5))
    assert smoother.update(18 / 15, (.5, .5), size) == pytest.approx((.5, .5))


def test_fast_intentional_move_reaches_ninety_percent_within_point_two_seconds():
    smoother = HandCursorSmoother()
    size = (3440, 1440)
    for frame in range(16):
        smoother.update(frame / 15, (.5, .5), size)
    responses = [smoother.update(1 + frame / 15, (.8, .5), size) for frame in range(1, 4)]
    assert responses[-1][0] >= .5 + .3 * .9
    assert responses[-1][1] == pytest.approx(.5)
    assert all(.5 <= point[0] <= .8 for point in responses)


def test_slow_intentional_motion_accumulates_through_the_resting_dead_zone():
    size = np.array((3440, 1440))
    center = np.array((.5, .5))
    smoother = HandCursorSmoother()
    smoother.update(0, center, size)
    for frame in range(1, 91):
        point = center + np.array((frame * .3, 0)) / size
        final = np.array(smoother.update(frame / 15, point, size)) * size
    assert final[0] - center[0] * size[0] > 20
    assert point[0] * size[0] - final[0] < 6


def test_resting_dead_zone_is_radial_and_measured_in_screen_pixels():
    size = np.array((3440, 1440))
    center = np.array((.5, .5))
    smoother = HandCursorSmoother()
    smoother.update(0, center, size)
    for frame in range(1, 61):
        point = center + [2 / size[0], 1 / size[1]]
        assert smoother.update(frame / 15, point, size) == pytest.approx(center)


def test_same_pixel_motion_responds_equally_on_wide_and_small_screens():
    def response(size):
        size = np.array(size)
        smoother = HandCursorSmoother()
        for frame in range(16):
            smoother.update(frame / 15, (.2, .2), size)
        output = []
        for frame in range(1, 8):
            point = np.array((.2, .2)) + [400 / size[0], 200 / size[1]]
            result = np.array(smoother.update(1 + frame / 15, point, size)) * size - size * .2
            output.append(result)
        return output
    assert np.array(response((3440, 1440))) == pytest.approx(np.array(response((1920, 1080))))


@pytest.mark.parametrize("timestamp,point,size", [
    (math.nan, (.5, .5), (3440, 1440)),
    (.1, (math.nan, .5), (3440, 1440)),
    (.1, (.5,), (3440, 1440)), (.1, (.5, .5), (0, 1440)),
    (0, (.9, .9), (3440, 1440)), (-.1, (.9, .9), (3440, 1440)),
])
def test_invalid_duplicate_and_out_of_order_samples_do_not_change_filter(timestamp, point, size):
    smoother = HandCursorSmoother()
    smoother.update(0, (.5, .5), (3440, 1440))
    assert smoother.update(timestamp, point, size) is None
    assert smoother.update(.2, (.5, .5), (3440, 1440)) == pytest.approx((.5, .5))


def test_explicit_reset_or_stale_source_gap_starts_at_current_hand_position():
    smoother = HandCursorSmoother()
    smoother.update(0, (.5, .5), (3440, 1440))
    assert smoother.update(.8, (.8, .2), (3440, 1440)) == pytest.approx((.8, .2))
    smoother.reset()
    assert smoother.update(.9, (.2, .8), (3440, 1440)) == pytest.approx((.2, .8))


def test_display_size_change_resets_pixel_history_and_output_is_always_bounded():
    smoother = HandCursorSmoother()
    smoother.update(0, (.2, .2), (3440, 1440))
    assert smoother.update(.1, (.8, .8), (1920, 1080)) == pytest.approx((.8, .8))
    for frame in range(2, 10):
        point = smoother.update(frame / 15, (-1, 2), (1920, 1080))
        assert all(0 <= value <= 1 for value in point)


@pytest.mark.parametrize("kwargs", [{"min_cutoff": 0}, {"beta": -1},
                                   {"derivative_cutoff": math.nan}, {"dead_zone_pixels": -1}, {"max_gap": 0}])
def test_invalid_filter_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HandCursorSmoother(**kwargs)
