"""Stable visual recognition never substitutes for raw action evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kon.geometry import TriggerGate
from kon.vision import TrackingFrame
from test_target_lock import lock_first, session


BASE_FRAME = TrackingFrame(
    0., (), (), .65, True, None, None, (.25, .5), "追蹤狀態",
    hand_detected=True, depth_status="ready", gesture_reason="personal gesture recognized",
    gesture_source="world", handedness="Right", raw_pose_ok=True,
)


def send(window, clock, timestamp, *, stable_pose=True, raw_pose=True,
         point=(.25, .5), reason=None, depth="ready"):
    clock["now"] = timestamp
    observation = replace(
        BASE_FRAME, timestamp=timestamp, pose_ok=stable_pose, raw_pose_ok=raw_pose,
        gesture_score=.65 if raw_pose else .58,
        gesture_reason=reason or ("personal gesture recognized" if raw_pose else "personal gesture mismatch"),
        raw_hit=point, depth_status=depth, hand_detected=depth != "no_hand",
    )
    window.on_tracking(observation)


def test_short_classifier_jitter_keeps_labels_and_cursor_but_blocks_voice_actions(session):
    window, clock = session
    lock_first(window, clock)
    recognized_label = window.pose_label.text()
    recognized_hint = window.gesture_hint_label.text()
    locked_cursor = window.overlay.point
    fired = []
    window.summon = fired.append

    for timestamp in (100.8, 100.85, 100.89):
        send(window, clock, timestamp, raw_pose=False, point=(.27, .5))
        assert window.pose_label.text() == recognized_label
        assert window.gesture_hint_label.text() == recognized_hint
        assert window.overlay.point == locked_cursor
        assert window.aim_point is not None
        assert window.overlay.target_id == window.gate.locked_target == "a"
        assert window.overlay.progress == 1
        assert not window.voice_active and not window.gate.ready
        window.on_keyword(timestamp, "kon")
        assert fired == []


def test_actual_match_resumes_retained_lock_with_fresh_audio_only(session):
    window, clock = session
    lock_first(window, clock)
    fired = []
    window.summon = fired.append
    old_audio = 100.71

    send(window, clock, 100.8, raw_pose=False)
    window.on_keyword(100.8, "kon")
    assert fired == [] and not window.voice_active

    send(window, clock, 100.9)
    assert window.gate.locked_target == window.voice_target == "a"
    assert window.gate.ready and window.voice_active
    window.on_keyword(old_audio, "kon")
    window.on_keyword(100.8, "kon")
    assert fired == []
    window.on_keyword(100.9, "kon")
    assert fired == ["a"]


def test_stable_pose_keeps_dwell_continuous_while_raw_misses_prevent_readiness():
    gate = TriggerGate(dwell_seconds=.4)
    assert gate.update(0., True, "a", True, raw_pose_ok=True) is False
    assert gate.update(.1, True, "a", True, raw_pose_ok=False, release_observed=True) is False
    assert gate.leading_dwell_seconds == pytest.approx(.1)
    assert gate.progress == pytest.approx(.5)
    # The stabilized pose completes selection during the short noisy interval,
    # while neither direct speech nor a queued action can consume that lock.
    assert gate.update(.2, True, "a", True, raw_pose_ok=False, release_observed=True) is False
    assert gate.locked_target == "a"
    assert gate.hear_keyword(.2) is None
    assert gate.take_trigger(now=.2) is None
    assert gate.update(.25, True, "a", True, raw_pose_ok=True) is True
    assert gate.hear_keyword(.25) == "a"


def test_gate_resuming_raw_match_requires_audio_from_resumed_interval():
    gate = TriggerGate(dwell_seconds=.2)
    gate.update(1., True, "a", True, raw_pose_ok=True)
    assert gate.update(1.1, True, "a", True, raw_pose_ok=True) is True
    assert gate.update(1.15, True, "a", True, raw_pose_ok=False, release_observed=True) is False
    assert gate.update(1.2, True, "a", True, raw_pose_ok=True) is True
    assert gate.locked_target == "a"
    assert gate.hear_keyword(1.21, captured_at=1.12) is None
    assert gate.hear_keyword(1.21, captured_at=1.17) is None
    assert gate.hear_keyword(1.21, captured_at=1.2) == "a"


def test_physical_release_timer_starts_at_first_raw_miss_before_visual_grace_expires(session):
    window, clock = session
    lock_first(window, clock)
    # Stabilization may retain a display pose for the first 200 ms. Release
    # must still finish 250 ms after the first actual negative observation.
    for timestamp in (100.8, 100.9, 100.99):
        send(window, clock, timestamp, raw_pose=False)
        assert window.gate.locked_target == "a"
        assert not window.voice_active
    send(window, clock, 101.04, stable_pose=False, raw_pose=False, point=None)
    assert window.gate.locked_target == "a"
    send(window, clock, 101.05, stable_pose=False, raw_pose=False, point=None)
    assert window.gate.locked_target is None
    assert window.overlay.target_id is None and not window.voice_active


def test_missing_aim_during_visual_grace_still_counts_observed_raw_release(session):
    window, clock = session
    lock_first(window, clock)
    recognized_label = window.pose_label.text()
    # The hand shape is observed, but the eye/ray is unusable. Missing aim
    # must not reset the release timer during the retained display pose.
    for timestamp in (100.8, 100.9, 100.99):
        send(window, clock, timestamp, raw_pose=False, point=None, depth="eye_unusable")
        assert window.pose_label.text() == recognized_label
        assert window.aim_point is None
        assert window.gate.locked_target == "a"
        assert not window.gate.ready and not window.voice_active
    send(window, clock, 101.04, stable_pose=False, raw_pose=False,
         point=None, depth="eye_unusable")
    assert window.gate.locked_target == "a"
    send(window, clock, 101.05, stable_pose=False, raw_pose=False,
         point=None, depth="eye_unusable")
    assert window.gate.locked_target is None
    assert window.overlay.target_id is None and not window.voice_active


def test_consumed_hold_cannot_rearm_through_short_classifier_jitter(session):
    window, clock = session
    lock_first(window, clock)
    fired = []
    window.summon = fired.append
    window.on_keyword(100.7, "kon")
    assert fired == ["a"]

    for index in range(8):
        timestamp = 100.8 + index * .1
        send(window, clock, timestamp, raw_pose=bool(index % 2))
        window.on_keyword(timestamp, "kon")
        assert window.gate.locked_target == "a"
        assert not window.gate.ready and not window.voice_active
        assert fired == ["a"]


@pytest.mark.parametrize("stable_pose", [False, True])
def test_uncertain_data_cannot_complete_physical_release_or_rearm_consumption(session, stable_pose):
    window, clock = session
    lock_first(window, clock)
    fired = []
    window.summon = fired.append
    window.on_keyword(100.7, "kon")

    for timestamp in (100.8, 100.95, 101.1, 101.25, 101.4):
        send(window, clock, timestamp, stable_pose=stable_pose, raw_pose=False,
             point=(.25, .5) if stable_pose else None,
             reason="personal gesture landmarks", depth="hand_pnp_failed")
        assert window.gate.locked_target == "a"
        assert not window.voice_active
    send(window, clock, 101.5)
    window.on_keyword(101.5, "kon")
    assert window.gate.locked_target == "a"
    assert not window.gate.ready and fired == ["a"]


@pytest.mark.parametrize("stable_pose", [False, True])
def test_gate_release_observed_false_preserves_lock_despite_prolonged_raw_failure(stable_pose):
    gate = TriggerGate(dwell_seconds=.2)
    gate.update(0., True, "a", True, raw_pose_ok=True)
    assert gate.update(.1, True, "a", True, raw_pose_ok=True) is True
    for timestamp in (.2, .3, .4, .5, .6):
        assert gate.update(timestamp, stable_pose, "a", True,
                           raw_pose_ok=False, release_observed=False) is False
        assert gate.locked_target == "a"
        assert gate.hear_keyword(timestamp) is None
    assert gate.update(.7, True, "a", True, raw_pose_ok=True) is True


@pytest.mark.parametrize("explicit_none", [False, True])
def test_absent_raw_evidence_preserves_legacy_pose_readiness_and_release(explicit_none):
    gate = TriggerGate(dwell_seconds=.2)
    options = {"raw_pose_ok": None} if explicit_none else {}
    assert gate.update(0., True, "a", True, **options) is False
    assert gate.update(.1, True, "a", True, **options) is True
    assert gate.update(.2, False, None, True, **options) is False
    assert gate.locked_target == "a"
    assert gate.update(.45, False, None, True, **options) is False
    assert gate.locked_target is None
    assert gate.update(.5, True, "a", True, **options) is False
    assert gate.update(.6, True, "a", True, **options) is True
