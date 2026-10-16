"""Render interpolation only; no camera, gesture gate, audio or file actions."""
import math

import numpy as np
import pytest

from kon.cursor_motion import CursorMotion


def test_first_measurement_is_immediate_and_clear_removes_display_state():
    motion = CursorMotion()
    assert not motion.active and not motion.is_animating
    assert motion.sample(0) is None
    assert motion.push((100, 200), 0)
    assert motion.active and not motion.is_animating
    assert motion.sample(0) == (100, 200)
    motion.clear()
    assert not motion.active and not motion.is_animating
    assert motion.sample(10) is None
    assert motion.push((5, 6), 1)
    assert motion.sample(1) == (5, 6)


def test_latest_measured_position_gets_a_short_smooth_convex_transition():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, -100), .01)
    assert motion.sample(.01) == (0, 0)
    positions = [motion.sample(.01 + frame / 60) for frame in range(1, 7)]
    assert all(0 <= point[0] <= 100 and -100 <= point[1] <= 0 for point in positions)
    assert all(positions[index][0] < positions[index + 1][0] for index in range(len(positions) - 1))
    assert motion.is_animating


def test_follow_response_matches_the_50_to_70ms_latency_budget():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((1000, 500), .01)
    at_50ms = motion.sample(.06)
    at_70ms = motion.sample(.08)
    assert 750 <= at_50ms[0] <= 780
    assert 850 <= at_70ms[0] <= 880
    assert at_50ms[1] == pytest.approx(at_50ms[0] / 2)
    assert at_70ms[1] == pytest.approx(at_70ms[0] / 2)


def test_new_target_interrupts_from_current_position_and_has_no_old_queue():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, 0), .01)
    before = motion.sample(.04)
    motion.push((-100, 50), .04)
    assert motion.sample(.04) == pytest.approx(before)
    after = motion.sample(.05)
    assert -100 <= after[0] < before[0]
    assert 0 < after[1] < 50
    assert motion.sample(.17) == (-100, 50)
    assert not motion.is_animating


def test_repeated_same_target_finishes_exactly_without_an_eternal_timer():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, 200), .01)
    motion.push((100, 200), .06)
    motion.push((100, 200), .11)
    assert motion.sample(.14) == (100, 200)
    assert motion.active and not motion.is_animating
    assert motion.push((100, 200), .15)
    assert not motion.is_animating


def test_terminal_position_is_exact_even_for_small_nonzero_motion():
    motion = CursorMotion()
    motion.push((1, 2), 0)
    motion.push((1.000001, 2.000001), .01)
    assert motion.is_animating
    assert motion.sample(.14) == (1.000001, 2.000001)
    assert not motion.is_animating


def test_exact_terminal_deadline_is_consistent_at_nonzero_ui_clock_origin():
    motion = CursorMotion()
    motion.push((1, 2), 100)
    motion.push((10, 20), 100.01)
    assert motion.sample(100.01 + motion.settle_duration) == (10, 20)
    assert not motion.is_animating


def test_snap_anchors_lock_immediately_and_discards_old_motion():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((1000, 1000), .01)
    motion.sample(.04)
    assert motion.snap((50, 60), .04)
    assert not motion.is_animating
    assert motion.sample(.04) == (50, 60)
    assert motion.sample(1) == (50, 60)


def test_animation_sampling_never_refreshes_measured_arrival_or_predicts_beyond_it():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, 100), .01)
    for frame in range(1, 25):
        point = motion.sample(.01 + frame / 60)
        assert 0 <= point[0] <= 100 and 0 <= point[1] <= 100
    # Render ticks continued, but the measured-arrival gap is still >.35s:
    # the new track starts at the actual new measurement immediately.
    motion.push((500, 600), .5)
    assert motion.sample(.5) == (500, 600)
    assert not motion.is_animating


def test_15fps_measurements_can_have_60hz_display_only_samples():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, 0), 1 / 15)
    display = [motion.sample(1 / 15 + frame / 60) for frame in range(4)]
    assert display[0] == (0, 0)
    assert display[0][0] < display[1][0] < display[2][0] < display[3][0] < 100
    motion.push((200, 0), 2 / 15)
    next_position = motion.sample(2 / 15)
    assert display[-1][0] < next_position[0] < 100


@pytest.mark.parametrize("point", [None, [], (1,), (1, 2, 3), (True, 1), ("1", 2),
                                  (math.nan, 1), (1, math.inf), {"x": 1, "y": 2}])
def test_invalid_points_do_not_corrupt_existing_display(point):
    motion = CursorMotion()
    motion.push((1, 2), 0)
    assert not motion.push(point, .1)
    assert not motion.snap(point, .1)
    assert motion.sample(.05) == (1, 2)


@pytest.mark.parametrize("now", [math.nan, math.inf, True, "1", None, -.1])
def test_invalid_or_backward_times_do_not_change_existing_state(now):
    motion = CursorMotion()
    motion.push((1, 2), 0)
    assert not motion.push((100, 200), now)
    assert not motion.snap((100, 200), now)
    assert motion.sample(now) is None
    assert motion.sample(.01) == (1, 2)


def test_a_late_arrival_cannot_rewind_already_rendered_time():
    motion = CursorMotion()
    motion.push((0, 0), 0)
    motion.push((100, 0), .01)
    latest = motion.sample(.05)
    assert not motion.push((1000, 0), .04)
    assert motion.sample(.05) == latest


def test_large_finite_endpoints_remain_finite_without_subtraction_overflow():
    motion = CursorMotion()
    motion.push((-1e308, 1e308), 0)
    motion.push((1e308, -1e308), .01)
    assert all(math.isfinite(value) for value in motion.sample(.02))
    assert motion.sample(.14) == (1e308, -1e308)


def test_numpy_numeric_points_and_times_are_supported():
    motion = CursorMotion()
    assert motion.push(np.array([100., 200.]), np.float64(0))
    assert motion.sample(np.float64(0)) == (100, 200)


@pytest.mark.parametrize("kwargs", [{"tau": 0}, {"tau": -1}, {"tau": True},
                                   {"tau": "0.035"}, {"tau": math.nan},
                                   {"max_gap": math.inf}, {"max_gap": 0}, {"settle_duration": 0}])
def test_invalid_animation_durations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CursorMotion(**kwargs)
