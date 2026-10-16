"""Time-weighted selection over a rolling three-second history."""

from __future__ import annotations

import math

import pytest

from kon.geometry import TriggerGate


def observe(gate, start, end, target, *, step=.1, include_end=True):
    """Deliver fresh frames; leave a switch boundary for the next target."""
    count = math.ceil((end - start) / step - 1e-9)
    for index in range(count):
        gate.update(round(start + index * step, 10), True, target, True)
    if include_end:
        gate.update(end, True, target, True)


def test_small_pause_tolerance_does_not_reset_continuous_fresh_frames():
    gate = TriggerGate(tracking_grace_seconds=.1)
    for index in range(9):
        gate.update(index * .2, True, "file-a", True)
    assert gate.locked_target == "file-a"
    assert gate.locked_at == pytest.approx(1.6)
    gate.update(1.8, True, "file-a", True)
    assert gate.hear_keyword(1.81, captured_at=1.65) == "file-a"


def observe_weighted(gate, start, end, target, weight, *, step=.1, include_end=True):
    count = math.ceil((end - start) / step - 1e-9)
    for index in range(count):
        gate.update(round(start + index * step, 10), True, target, True, dwell_weight=weight)
    if include_end:
        gate.update(end, True, target, True, dwell_weight=weight)


def test_default_lock_needs_one_and_a_half_seconds_without_waiting_full_three():
    gate = TriggerGate()
    assert gate.dwell_seconds == 3.
    assert gate.minimum_dwell_seconds == 1.5
    assert gate.tracking_grace_seconds == .5
    observe(gate, 0., 1.4, "file-a")
    assert gate.locked_target is None and not gate.ready
    assert gate.progress == pytest.approx(1.4 / 1.5)
    gate.update(1.499, True, "file-a", True)
    assert gate.locked_target is None
    gate.update(1.5, True, "file-a", True)
    assert gate.locked_target == "file-a"
    assert gate.locked_at == 1.5
    assert gate.ready and gate.progress == 1.


def test_custom_history_duration_defaults_to_half_as_minimum():
    gate = TriggerGate(dwell_seconds=.65)
    assert gate.minimum_dwell_seconds == .325
    observe(gate, 0., .3, "file-a")
    assert gate.locked_target is None
    gate.update(.325, True, "file-a", True)
    assert gate.locked_target == "file-a"


@pytest.mark.parametrize("fps", [5, 60])
def test_same_aiming_durations_choose_same_file_at_low_and_high_fps(fps):
    gate = TriggerGate()
    for frame in range(2 * fps + 1):
        now = frame / fps
        target = "file-b" if .6 <= now < 1. else "file-a"
        gate.update(now, True, target, True)
    assert gate.locked_target == "file-a"
    assert gate.locked_at <= 2.


def test_elapsed_seconds_outweigh_number_of_frames():
    gate = TriggerGate()
    # A has many more observations but only .6 seconds. B has fewer frames
    # and 1.5 seconds, so counting frames would choose the wrong file.
    observe(gate, 0., .6, "file-a", step=.01, include_end=False)
    observe(gate, .6, 2., "file-b", step=.25)
    assert gate.locked_target is None
    gate.update(2.1, True, "file-b", True)
    assert gate.locked_target == "file-b"


def test_separate_visits_add_up_without_requiring_one_continuous_hold():
    gate = TriggerGate()
    observe(gate, 0., .6, "file-a", include_end=False)
    observe(gate, .6, .9, "file-b", include_end=False)
    observe(gate, .9, 1.7, "file-a")
    assert gate.locked_target is None
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(1.4)
    gate.update(1.8, True, "file-a", True)
    assert gate.locked_target == "file-a"
    assert gate.locked_at == 1.8


def test_realistic_separate_visits_and_short_gap_lock_at_one_and_a_half_seconds():
    gate = TriggerGate()
    observe(gate, 0., .6, "file-a", include_end=False)
    observe(gate, .6, .9, "file-b", include_end=False)
    observe(gate, .9, 1.4, "file-a")
    gate.invalidate_tracking(1.4)
    assert gate.locked_target is None
    assert gate.leading_dwell_seconds == pytest.approx(1.1)
    gate.update(1.6, True, "file-a", True)
    observe(gate, 1.7, 1.9, "file-a")
    assert gate.locked_target is None  # .6 + .5 + .3; the .2 gap adds nothing.
    gate.update(2., True, "file-a", True)
    assert gate.locked_target == "file-a"  # .6 + .5 + .4 seconds.
    assert gate.locked_at == 2.


def test_last_frame_target_cannot_replace_longest_held_file():
    gate = TriggerGate()
    observe(gate, 0., 1.4, "file-a")
    gate.update(1.5, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(1.51) == "file-a"


def test_initial_empty_space_cannot_lock_and_does_not_delay_new_file():
    gate = TriggerGate()
    observe(gate, 0., 1., None)
    assert gate.locked_target is None
    observe(gate, 1.1, 2.5, "file-a")
    assert gate.locked_target is None
    gate.update(2.6, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_empty_space_is_counted_and_returning_to_file_keeps_previous_votes():
    gate = TriggerGate()
    observe(gate, 0., .6, "file-a", include_end=False)
    observe(gate, .6, .9, None, include_end=False)
    observe(gate, .9, 1.7, "file-a")
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(1.4)
    assert gate.locked_target is None
    gate.update(1.8, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_longest_empty_space_prevents_locking_a_shorter_file_over_minimum():
    gate = TriggerGate(minimum_dwell_seconds=.5)
    observe(gate, 0., .1, "file-a", include_end=False)
    observe(gate, .1, 1.1, None, include_end=False)
    observe(gate, 1.1, 1.5, "file-a")
    assert gate.leading_dwell_seconds == pytest.approx(.5)
    assert gate.locked_target is None and not gate.ready
    observe(gate, 1.6, 2.2, "file-a")
    assert gate.locked_target == "file-a"


def test_exact_tie_after_old_votes_expire_cannot_lock_until_unique_winner():
    gate = TriggerGate(minimum_dwell_seconds=.5)
    observe(gate, 0., .1, "file-a", include_end=False)
    observe(gate, .1, 1.3, None, include_end=False)
    observe(gate, 1.3, 1.9, "file-a", include_end=False)
    observe(gate, 1.9, 2.6, "file-b")
    # Short observed gaps let old empty-space votes age out without adding
    # more dwell. At 3.6, B and empty space each occupied .7 recent seconds.
    for pause, resume in ((2.6, 2.9), (2.9, 3.2), (3.2, 3.5), (3.5, 3.6)):
        gate.invalidate_tracking(pause)
        gate.update(resume, True, None, True)
    assert gate.leading_dwell_seconds == pytest.approx(.7)
    assert gate.locked_target is None and not gate.ready
    assert gate.progress == 1.
    gate.invalidate_tracking(3.6)
    gate.update(3.61, True, "file-b", True)
    assert gate.locked_target == "file-b"


def test_most_visited_file_still_needs_one_and_a_half_recent_seconds():
    gate = TriggerGate()
    observe(gate, 0., 1.2, "file-a", include_end=False)
    observe(gate, 1.2, 2.3, "file-b", include_end=False)
    observe(gate, 2.3, 3., "file-c")
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(1.2)
    assert gate.locked_target is None and not gate.ready


def test_rolling_history_clips_old_partial_segments_instead_of_counting_lifetime():
    gate = TriggerGate()
    observe(gate, 0., 1.4, "file-a", include_end=False)
    observe(gate, 1.4, 2.4, "file-b", include_end=False)
    observe(gate, 2.4, 3., "file-c", include_end=False)
    observe(gate, 3., 4., "file-a")
    # Only A's old [1.0, 1.4] and new [3.0, 4.0] remain. Lifetime A=2.4.
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(1.4)
    assert gate.locked_target is None
    observe(gate, 4.1, 4.4, "file-a")
    assert gate.locked_target is None
    gate.update(4.5, True, "file-a", True)
    assert gate.locked_target == "file-a"


@pytest.mark.parametrize("gap", [.3, .5])
def test_short_tracking_gap_preserves_history_and_excludes_unobserved_time(gap):
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.invalidate_tracking(1.05)
    resume = 1. + gap
    gate.update(resume, True, "file-a", True)
    assert gate.leading_dwell_seconds == pytest.approx(1.)
    observe(gate, resume + .1, resume + .4, "file-a")
    assert gate.locked_target is None
    gate.update(resume + .5, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_tracking_gap_past_grace_clears_recent_history():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.invalidate_tracking(1.1)
    gate.update(1.5001, True, "file-a", True)
    assert gate.leading_dwell_seconds == 0.
    observe(gate, 1.6001, 2.9001, "file-a")
    assert gate.locked_target is None
    gate.update(3.0001, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_repeated_watchdog_pauses_cannot_keep_votes_beyond_grace():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    for timestamp in (1.1, 1.2, 1.3, 1.4, 1.5):
        gate.invalidate_tracking(timestamp)
    assert gate.leading_dwell_seconds == pytest.approx(1.)
    gate.invalidate_tracking(1.5001)
    assert gate.leading_dwell_seconds == 0.


@pytest.mark.parametrize("clear", ["layout", "clear_lock", "disarm"])
def test_explicit_session_change_discards_recent_history(clear):
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    if clear == "layout":
        gate.invalidate_tracking(1.1, discard_dwell=True)
    elif clear == "clear_lock":
        gate.clear_lock(1.1)
    else:
        gate.update(1.1, True, "file-a", False)
    assert gate.leading_dwell_seconds == 0.
    gate.update(1.2, True, "file-a", True)
    observe(gate, 1.3, 2.6, "file-a")
    assert gate.locked_target is None
    gate.update(2.7, True, "file-a", True)
    assert gate.locked_target == "file-a"


@pytest.mark.parametrize("duration,keeps_history", [(.249, True), (.25, False)])
def test_observed_negative_pose_release_boundary_controls_history(duration, keeps_history):
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.update(1.1, False, None, True, release_observed=True)
    gate.update(1.1 + duration, False, None, True, release_observed=True)
    expected = 1. if keeps_history else 0.
    assert gate.leading_dwell_seconds == pytest.approx(expected)
    resume = 1.11 + duration
    gate.update(resume, True, "file-a", True)
    assert gate.leading_dwell_seconds == pytest.approx(expected)
    observe(gate, resume + .1, resume + .4, "file-a")
    assert gate.locked_target is None
    gate.update(resume + .5, True, "file-a", True)
    assert (gate.locked_target == "file-a") is keeps_history


def test_backward_dispatch_time_is_ignored_without_discarding_history():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    assert not gate.update(.5, True, "file-b", True)
    assert gate.leading_dwell_seconds == pytest.approx(1.)
    observe(gate, 1.1, 1.5, "file-a")
    assert gate.locked_target == "file-a"


def test_repeated_or_older_captured_frames_pause_and_keep_history_without_gap_credit():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.update(1.1, True, "file-b", True, observed_at=1.)
    gate.update(1.15, True, "file-b", True, observed_at=.9)
    assert gate.leading_dwell_seconds == pytest.approx(1.)
    assert gate.locked_target is None
    gate.update(1.2, True, "file-a", True, observed_at=1.2)
    observe(gate, 1.3, 1.6, "file-a")
    assert gate.locked_target is None
    gate.update(1.7, True, "file-a", True)
    assert gate.locked_target == "file-a"


@pytest.mark.parametrize("observed_at", [1.2, .7, math.nan, math.inf])
def test_invalid_capture_timestamp_pauses_history_without_crediting_gap(observed_at):
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.update(1.1, True, "file-a", True, observed_at=observed_at)
    assert gate.leading_dwell_seconds == pytest.approx(1.)
    gate.update(1.2, True, "file-a", True)
    observe(gate, 1.3, 1.6, "file-a")
    assert gate.locked_target is None
    gate.update(1.7, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_history_uses_capture_timestamps_despite_changing_processing_delay():
    gate = TriggerGate()
    for frame in range(15):
        captured = frame / 10.
        dispatched = captured + .05 + .2 * captured / 1.5
        gate.update(dispatched, True, "file-a", True, observed_at=captured)
    # Processing spans more than 1.5 seconds; observed dwell is only 1.4.
    assert gate.locked_target is None
    assert gate.leading_dwell_seconds == pytest.approx(1.4)
    gate.update(1.75, True, "file-a", True, observed_at=1.5)
    assert gate.locked_target == "file-a" and gate.locked_at == 1.75


def test_fresh_source_continuity_survives_a_longer_processing_interval():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.update(1.4, True, "file-a", True, observed_at=1.2)
    assert gate.leading_dwell_seconds == pytest.approx(1.2)
    gate.update(1.5, True, "file-a", True, observed_at=1.4)
    assert gate.locked_target is None
    gate.update(1.65, True, "file-a", True, observed_at=1.5)
    assert gate.locked_target == "file-a"


def test_keyword_freshness_uses_actual_capture_time_after_delayed_frame():
    gate = TriggerGate()
    observe(gate, 0., 1.4, "file-a")
    gate.update(1.79, True, "file-a", True, observed_at=1.5)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(1.81, captured_at=1.8) is None
    assert not gate.ready and gate.locked_target == "file-a"
    gate.update(1.9, True, "file-b", True, observed_at=1.85)
    assert gate.hear_keyword(1.91, captured_at=1.9) == "file-a"


def test_capture_gap_cannot_supply_continuous_negative_release_evidence():
    gate = TriggerGate()
    observe(gate, 0., 1.5, "file-a")
    gate.update(1.89, False, None, True, observed_at=1.6)
    gate.update(2.01, False, None, True, observed_at=2.)
    assert gate.locked_target == "file-a" and not gate.ready
    gate.update(2.11, True, "file-b", True, observed_at=2.1)
    assert gate.locked_target == "file-a" and gate.ready


def test_release_duration_uses_capture_times_despite_changing_processing_delay():
    gate = TriggerGate()
    observe(gate, 0., 1.5, "file-a")
    gate.update(1.6, False, None, True, observed_at=1.6)
    gate.update(2.09, False, None, True, observed_at=1.8)
    assert gate.locked_target == "file-a"  # Only .2 seconds of actual release.
    gate.update(2.1, False, None, True, observed_at=1.86)
    assert gate.locked_target is None  # .26 seconds of observed release.


def test_prelock_audio_for_current_shorter_file_cannot_trigger_the_winner():
    gate = TriggerGate(keyword_max_age=5.)
    observe(gate, 0., .6, "file-a", include_end=False)
    observe(gate, .6, .9, "file-b", include_end=False)
    observe(gate, .9, 1.7, "file-a")
    assert gate.hear_keyword(1.71) is None
    gate.update(1.8, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.take_trigger() is None
    assert gate.hear_keyword(1.81, captured_at=1.7) is None
    assert gate.hear_keyword(1.82, captured_at=1.8) == "file-a"


def test_prelock_audio_from_previous_visit_cannot_transfer_back_to_the_winner():
    gate = TriggerGate(keyword_max_age=5.)
    observe(gate, 0., .6, "file-a", include_end=False)
    observe(gate, .6, .9, "file-b", include_end=False)
    observe(gate, .9, 1.8, "file-a")
    assert gate.hear_keyword(1.81, captured_at=.2) is None
    assert gate.hear_keyword(1.82, captured_at=1.4) is None
    assert gate.hear_keyword(1.83, captured_at=1.8) == "file-a"


def test_keyword_before_lock_requires_new_speech_after_minimum_is_reached():
    gate = TriggerGate()
    observe(gate, 0., 1.4, "file-a")
    assert gate.hear_keyword(1.41) is None
    gate.update(1.5, True, "file-a", True)
    assert gate.take_trigger() is None
    assert gate.hear_keyword(1.51) == "file-a"
    assert gate.hear_keyword(1.52) is None


def test_large_movement_clears_completed_lock_and_old_votes():
    gate = TriggerGate(keyword_max_age=10.)
    observe(gate, 0., 1.5, "file-a")
    gate.update(1.6, True, "file-b", True, movement_released=True)
    assert gate.locked_target is None and gate.locked_at is None
    assert gate.leading_dwell_seconds == 0.
    observe(gate, 1.7, 3., "file-b")
    assert gate.locked_target is None
    gate.update(3.1, True, "file-b", True)
    assert gate.locked_target == "file-b"
    assert gate.hear_keyword(3.11, captured_at=1.5) is None
    assert gate.hear_keyword(3.12, captured_at=3.1) == "file-b"


def test_large_movement_discards_partial_votes_even_when_returning_to_same_file():
    gate = TriggerGate()
    observe(gate, 0., 1., "file-a")
    gate.update(1.1, True, "file-a", True, movement_released=True)
    assert gate.leading_dwell_seconds == 0.
    observe(gate, 1.2, 2.5, "file-a")
    assert gate.locked_target is None
    gate.update(2.6, True, "file-a", True)
    assert gate.locked_target == "file-a"


def test_large_movement_cannot_rearm_an_already_consumed_gesture():
    gate = TriggerGate()
    observe(gate, 0., 1.5, "file-a")
    assert gate.hear_keyword(1.51) == "file-a"
    gate.update(1.6, True, "file-b", True, movement_released=True)
    observe(gate, 1.7, 3.1, "file-b")
    assert gate.locked_target is None and gate.progress == 0.
    assert gate.hear_keyword(3.11) is None


def test_aim_weight_only_applies_to_future_intervals_and_locks_after_point_nine_seconds():
    gate = TriggerGate()
    observe(gate, 0., .3, "file-a", include_end=False)
    gate.update(.3, True, "file-a", True, dwell_weight=2.)
    assert gate.leading_dwell_seconds == pytest.approx(.3)
    observe_weighted(gate, .4, .8, "file-a", 2.)
    assert gate.leading_dwell_seconds == pytest.approx(1.3)
    assert gate.locked_target is None
    gate.update(.9, True, "file-a", True, dwell_weight=2.)
    assert gate.locked_target == "file-a"
    assert gate.locked_at == .9  # .3 ordinary + .6 doubled = 1.5 weighted seconds.


@pytest.mark.parametrize("fps", [10, 60])
def test_weighted_seconds_choose_same_file_at_low_and_high_fps(fps):
    gate = TriggerGate()
    for frame in range(round(.9 * fps) + 1):
        now = frame / fps
        gate.update(now, True, "file-a", True, dwell_weight=1. if now < .3 else 2.)
    assert gate.locked_target == "file-a"
    assert gate.locked_at == .9


def test_rolling_clip_retains_original_weight_of_each_partial_segment():
    gate = TriggerGate()
    observe_weighted(gate, 0., .6, "file-a", 2., include_end=False)
    # Keep the boundary observation weighted so the preceding A interval
    # remains 2x; the next B interval drops to 1x at its endpoint.
    gate.update(.6, True, "file-b", True, dwell_weight=2.)
    observe(gate, .7, 1.6, "file-b", include_end=False)
    observe(gate, 1.6, 2.6, "file-c", include_end=False)
    observe(gate, 2.6, 3., None, include_end=False)
    observe_weighted(gate, 3., 3.1, "file-a", 3.)
    # In [0.1, 3.1], A has .5 seconds at 2x and .1 seconds at 3x.
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(1.3)
    assert gate.locked_target is None
    gate.update(3.2, True, "file-a", True, dwell_weight=3.)
    assert gate.leading_dwell_seconds == pytest.approx(1.4)
    assert gate.locked_target is None
    gate.update(3.3, True, "file-a", True, dwell_weight=3.)
    assert gate.locked_target == "file-a"  # .3 * 2 + .3 * 3 = 1.5.


def test_pause_preserves_weighted_votes_without_multiplying_missing_time():
    gate = TriggerGate()
    observe(gate, 0., .3, "file-a", include_end=False)
    observe_weighted(gate, .3, .6, "file-a", 2.)
    assert gate.leading_dwell_seconds == pytest.approx(.9)
    gate.invalidate_tracking(.6)
    gate.update(.8, True, "file-a", True, dwell_weight=2.)
    assert gate.leading_dwell_seconds == pytest.approx(.9)
    observe_weighted(gate, .9, 1., "file-a", 2.)
    assert gate.leading_dwell_seconds == pytest.approx(1.3)
    assert gate.locked_target is None
    gate.update(1.1, True, "file-a", True, dwell_weight=2.)
    assert gate.locked_target == "file-a"


def test_empty_space_always_has_normal_weight():
    gate = TriggerGate()
    observe(gate, 0., .1, "file-a", include_end=False)
    observe_weighted(gate, .1, .4, None, 10., include_end=False)
    observe_weighted(gate, .4, 1., "file-a", 2.)
    assert gate.locked_target is None
    gate.update(1.1, True, "file-a", True, dwell_weight=2.)
    # Empty space receives .3 ordinary seconds, rather than 3 at 10x.
    assert gate.locked_target == "file-a"


def test_weight_drop_does_not_double_the_interval_ending_in_motion():
    gate = TriggerGate()
    observe_weighted(gate, 0., .3, "file-a", 2.)
    assert gate.leading_dwell_seconds == pytest.approx(.6)
    gate.update(.4, True, "file-a", True, dwell_weight=1.)
    assert gate.leading_dwell_seconds == pytest.approx(.7)
    gate.update(.5, True, "file-a", True, dwell_weight=1.)
    assert gate.leading_dwell_seconds == pytest.approx(.8)


def test_weighted_threshold_may_exceed_unweighted_history_length():
    gate = TriggerGate(dwell_seconds=1., minimum_dwell_seconds=1.5)
    observe_weighted(gate, 0., .7, "file-a", 2.)
    assert gate.locked_target is None
    gate.update(.75, True, "file-a", True, dwell_weight=2.)
    assert gate.locked_target == "file-a"


@pytest.mark.parametrize("weight", [-1., 0., .5, math.nan, math.inf, True, False, None, "2"])
def test_invalid_aim_weight_is_rejected_without_changing_dwell(weight):
    gate = TriggerGate()
    observe(gate, 0., .1, "file-a")
    with pytest.raises(ValueError, match="weight"):
        gate.update(.2, True, "file-a", True, dwell_weight=weight)
    assert gate.leading_dwell_seconds == pytest.approx(.1)
    gate.update(.2, True, "file-a", True, dwell_weight=2.)
    assert gate.leading_dwell_seconds == pytest.approx(.2)
