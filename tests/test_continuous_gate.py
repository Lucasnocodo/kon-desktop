"""Successful actions can continue a hold without reusing targets or audio."""

import math

import pytest

from kon.geometry import TriggerGate


def gate():
    return TriggerGate(minimum_dwell_seconds=.5, keyword_max_age=5.)


def observe(value, start, end, target, *, armed=True, **kwargs):
    count = round((end - start) * 10)
    for index in range(count + 1):
        value.update(round(start + index * .1, 10), True, target, armed, **kwargs)


def consume_first(value):
    observe(value, 0., .5, "a")
    assert value.locked_target == "a"
    assert value.hear_keyword(.51) == "a"


def test_completion_allows_different_file_only_after_new_full_dwell_and_new_audio():
    value = gate()
    consume_first(value)
    assert value.complete_action("a", .52)
    assert value.locked_target is None and value.locked_at is None
    assert value.progress == 0 and value.leading_dwell_seconds == 0
    assert value.take_trigger() is None
    # A completion callback cannot itself make the old frame ready.
    assert value.hear_keyword(.53) is None
    observe(value, .6, 1., "b")
    assert value.locked_target is None and not value.ready
    assert value.hear_keyword(1.01) is None  # Speech before the new lock.
    value.update(1.1, True, "b", True)
    assert value.locked_target == "b" and value.ready
    assert value.take_trigger() is None
    assert value.hear_keyword(1.11, captured_at=.51) is None
    assert value.hear_keyword(1.12, captured_at=1.01) is None
    assert value.hear_keyword(1.13) == "b"
    assert value.hear_keyword(1.14) is None


def test_completion_needs_a_consumed_matching_target_and_is_not_repeatable():
    value = gate()
    assert not value.complete_action("a", 0.)
    observe(value, 0., .5, "a")
    assert not value.complete_action("a", .5)  # Locked is not triggered.
    assert value.ready
    assert value.hear_keyword(.51) == "a"
    assert not value.complete_action("b", .52)
    assert value.locked_target == "a" and not value.ready
    assert value.complete_action("a", .53)
    assert not value.complete_action("a", .54)
    observe(value, .6, 1.1, "b")
    assert value.hear_keyword(1.11) == "b"
    assert not value.complete_action("a", 1.12)
    assert value.complete_action("b", 1.13)


@pytest.mark.parametrize("timestamp", [.5, math.nan, math.inf, -math.inf])
def test_invalid_completion_time_cannot_rearm_a_consumed_action(timestamp):
    value = gate()
    consume_first(value)
    assert not value.complete_action("a", timestamp)
    value.clear_lock(.6)
    observe(value, .7, 1.3, "b")
    assert value.locked_target is None and value.progress == 0
    assert value.hear_keyword(1.31) is None


@pytest.mark.parametrize("interruption", ["clear", "disarm", "watchdog"])
def test_ordinary_state_changes_cannot_replace_successful_completion(interruption):
    value = gate()
    consume_first(value)
    if interruption == "clear":
        value.clear_lock(.6)
    elif interruption == "disarm":
        value.update(.6, True, "a", False)
    else:
        value.invalidate_tracking(.6)
    observe(value, .7, 1.3, "b", movement_released=True)
    assert value.locked_target is None and not value.ready
    assert value.hear_keyword(1.31) is None
    # Completion can arrive after animation removal/disarm/tracking changes.
    assert value.complete_action("a", 1.32)
    observe(value, 1.4, 1.9, "b")
    assert value.locked_target == "b" and value.hear_keyword(1.91) == "b"


def test_completed_target_stays_unavailable_during_same_hold_even_after_clear():
    value = gate()
    consume_first(value)
    assert value.complete_action("a", .52)
    observe(value, .6, 1.6, "a")
    assert value.locked_target is None and value.progress == 0
    assert value.hear_keyword(1.61) is None
    value.clear_lock(1.62)
    observe(value, 1.7, 2.2, "b")
    assert value.hear_keyword(2.21) == "b"
    assert value.complete_action("b", 2.22)
    # Remember every consumed file, not just the most recent one.
    observe(value, 2.3, 3.3, "a")
    assert value.locked_target is None and value.progress == 0
    assert value.hear_keyword(3.31) is None


def test_confirmed_physical_release_starts_new_hold_and_allows_same_file():
    value = gate()
    consume_first(value)
    assert value.complete_action("a", .52)
    for stamp in (.6, .8, .86):
        value.update(stamp, False, None, True, release_observed=True)
    observe(value, .9, 1.4, "a")
    assert value.locked_target == "a"
    assert value.hear_keyword(1.41) == "a"


@pytest.mark.parametrize("kind", ["brief", "uncertain", "gap"])
def test_unproven_release_cannot_make_consumed_file_available_again(kind):
    value = gate()
    consume_first(value)
    assert value.complete_action("a", .52)
    if kind == "brief":
        value.update(.6, False, None, True, release_observed=True)
        value.update(.8, False, None, True, release_observed=True)
    elif kind == "uncertain":
        for stamp in (.6, .8, .86):
            value.update(stamp, False, None, True, release_observed=False)
    else:
        value.update(.6, False, None, True, release_observed=True)
        value.update(1., False, None, True, release_observed=True)
    observe(value, 1.1, 1.7, "a")
    assert value.locked_target is None and not value.ready
    assert value.hear_keyword(1.71) is None


def test_successful_completion_after_release_still_requires_new_dwell_and_audio():
    value = gate()
    consume_first(value)
    for stamp in (.6, .8, .86):
        value.update(stamp, False, None, True, release_observed=True)
    assert not value.complete_action("b", .87)
    assert not value.complete_action("a", .5)
    assert value.complete_action("a", .88)
    assert not value.complete_action("a", .89)
    observe(value, .9, 1.3, "b")
    assert value.locked_target is None
    value.update(1.4, True, "b", True)
    assert value.locked_target == "b" and value.ready
    assert value.hear_keyword(1.41, captured_at=.51) is None
    assert value.hear_keyword(1.42) == "b"


def test_next_trigger_replaces_completion_identity_so_old_callback_cannot_rearm_it():
    value = gate()
    consume_first(value)
    for stamp in (.6, .8, .86):
        value.update(stamp, False, None, True, release_observed=True)
    observe(value, .9, 1.4, "b")
    assert value.hear_keyword(1.41) == "b"
    assert not value.complete_action("a", 1.42)
    assert value.hear_keyword(1.43) is None
    assert value.complete_action("b", 1.44)
