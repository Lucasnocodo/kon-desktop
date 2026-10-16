"""Bounded tolerance for recognition noise without hiding a real hand release."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from kon.geometry import GestureResult
from kon.gesture_stability import GestureStabilizer


def result(*, positive=True, score=None, reason=None, source="world"):
    if score is None:
        score = .65 if positive else .58
    if reason is None:
        reason = "personal gesture recognized" if positive else "personal gesture mismatch"
    return GestureResult(score, positive, reason=reason, source=source)


@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("reason,positive_score,miss_score", [
    ("personal gesture mismatch", .61, .58),
    ("finger pattern", .51, .46),
    ("thumb contact or finger margin", .51, .46),
])
def test_repeated_threshold_jitter_stays_recognized(fps, reason, positive_score, miss_score):
    stabilizer = GestureStabilizer()
    # Two noisy observations between actual matches reproduce a held pose
    # repeatedly crossing the classifier boundary at different frame rates.
    for index in range(fps * 2):
        positive = index % 3 == 0
        observation = result(positive=positive,
                             score=positive_score if positive else miss_score,
                             reason="recognized" if positive else reason)
        assert stabilizer.update(observation, index / fps) is True


@pytest.mark.parametrize("reason,minimum_score", [
    ("personal gesture mismatch", .52),
    ("finger pattern", .40),
    ("thumb contact or finger margin", .40),
])
def test_near_match_score_boundary_is_inclusive(reason, minimum_score):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=False, score=minimum_score, reason=reason), .1) is True
    assert stabilizer.update(result(positive=False, score=minimum_score - .001, reason=reason), .11) is False
    # A clearly different pose clears the retained match immediately.
    assert stabilizer.update(result(positive=False, score=minimum_score, reason=reason), .12) is False


def test_near_misses_do_not_renew_the_last_actual_match_grace():
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    for timestamp in (.05, .10, .15, .20):
        assert stabilizer.update(result(positive=False), timestamp) is True
    assert stabilizer.update(result(positive=False), .201) is False
    assert stabilizer.update(result(positive=False), .25) is False


def test_new_actual_match_renews_grace_without_an_acquisition_delay():
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=False), .10) is True
    assert stabilizer.update(result(), .15) is True
    assert stabilizer.update(result(positive=False), .30) is True
    assert stabilizer.update(result(positive=False), .351) is False
    assert stabilizer.update(result(), .36) is True


@pytest.mark.parametrize("reason,score", [
    ("invalid landmarks", .9),
    ("degenerate palm", .9),
    ("personal gesture landmarks", .9),
    ("personal gesture source", .9),
    ("personal gesture handedness", .9),
    ("", .9),
    ("unknown failure", .9),
    ("personal gesture mismatch", .1),
    ("finger pattern", .1),
    ("thumb contact or finger margin", .1),
    ("personal gesture mismatch", math.nan),
])
def test_hard_failure_immediately_clears_retained_recognition(reason, score):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=False, reason=reason, score=score), .05) is False
    assert stabilizer.update(result(positive=False), .10) is False
    assert stabilizer.update(result(), .15) is True


@pytest.mark.parametrize("reason", [
    "personal gesture mismatch", "finger pattern", "thumb contact or finger margin",
])
def test_an_unmatched_observation_cannot_acquire_recognition(reason):
    stabilizer = GestureStabilizer()
    for timestamp in (0., .05, .10):
        assert stabilizer.update(result(positive=False, reason=reason, score=.99), timestamp) is False


@pytest.mark.parametrize("positive", [False, True])
def test_invalid_landmark_shape_clears_even_a_raw_positive(positive):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=positive), .05, shape_valid=False) is False
    assert stabilizer.update(result(positive=False), .10) is False


def test_changing_tracked_hand_clears_previous_hands_grace():
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=False), .05, same_hand=False) is False
    assert stabilizer.update(result(positive=False), .10) is False
    assert stabilizer.update(result(), .15) is True


def test_switching_landmark_source_cannot_reuse_previous_match():
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(source="world"), 0.) is True
    assert stabilizer.update(result(positive=False, source="normalized"), .05) is False
    assert stabilizer.update(result(positive=False, source="world"), .10) is False
    assert stabilizer.update(result(source="normalized"), .15) is True
    assert stabilizer.update(result(positive=False, source="normalized"), .20) is True


@dataclass(frozen=True)
class Profile:
    label: str = "same calibration"


def test_replacing_an_equal_profile_clears_previous_profiles_grace():
    stabilizer = GestureStabilizer()
    original, replacement = Profile(), Profile()
    assert original == replacement and original is not replacement
    assert stabilizer.update(result(), 0., profile=original) is True
    assert stabilizer.update(result(positive=False), .05, profile=original) is True
    assert stabilizer.update(result(positive=False), .10, profile=replacement) is False
    assert stabilizer.update(result(positive=False), .15, profile=original) is False
    assert stabilizer.update(result(), .20, profile=replacement) is True
    assert stabilizer.update(result(positive=False), .25, profile=replacement) is True


@pytest.mark.parametrize("first_profile,second_profile", [(None, Profile()), (Profile(), None)])
def test_installing_or_removing_personal_profile_clears_previous_grace(first_profile, second_profile):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0., profile=first_profile) is True
    assert stabilizer.update(result(positive=False), .05, profile=second_profile) is False


@pytest.mark.parametrize("invalid_timestamp", [1., .99, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("positive", [False, True])
def test_duplicate_reverse_or_nonfinite_time_clears_history_and_rejects_frame(invalid_timestamp, positive):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 1.) is True
    assert stabilizer.update(result(positive=positive), invalid_timestamp) is False
    assert stabilizer.update(result(positive=False), 1.1) is False
    assert stabilizer.update(result(), 1.2) is True


@pytest.mark.parametrize("timestamps", [(1., 1., 1.), (.9, .95, 1.), (math.nan, .9, 1.)])
def test_repeated_invalid_frames_cannot_reprime_recognition_before_time_recovers(timestamps):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 1.) is True
    for timestamp in timestamps:
        assert stabilizer.update(result(), timestamp) is False
    assert stabilizer.update(result(positive=False), 1.01) is False
    assert stabilizer.update(result(), 1.02) is True


@pytest.mark.parametrize("positive", [False, True])
def test_large_frame_gap_discards_grace_and_accepts_fresh_positive(positive):
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 0.) is True
    assert stabilizer.update(result(positive=positive), 1.) is positive
    assert stabilizer.update(result(positive=False), 1.05) is positive


def test_reset_clears_match_and_timestamp_history():
    stabilizer = GestureStabilizer()
    assert stabilizer.update(result(), 100.) is True
    stabilizer.reset()
    assert stabilizer.update(result(positive=False), 0.) is False
    assert stabilizer.update(result(), .05) is True
    assert stabilizer.update(result(positive=False), .10) is True
