"""Real template threshold chatter through the camera worker, without devices."""

from dataclasses import replace

import numpy as np
import pytest

from kon.geometry import GestureResult
from kon.vision import CameraWorker
from test_gesture_calibration import open_hand, personal_hand, profile
from test_vision import WIDTH, HEIGHT, _results, projected_hand


def personal_results(offset=0., *, side="Right", opened=False):
    points = open_hand() if opened else personal_hand()
    points[4, 0] += offset
    # Uniform metric scaling leaves the enrolled joint features unchanged.
    return _results(((points, (points - points[0]) * .32),), labels=(side,))


@pytest.mark.parametrize("mode", ["hand", "eye_ray"])
def test_real_personal_threshold_jitter_preserves_pose_and_raw_diagnostics(mode):
    worker = CameraWorker(gesture_profile=profile(), aim_mode=mode)
    for index in range(60):
        # A 0.01 image-width thumb estimate change straddles the real model's
        # existing boundary; neither the saved model nor tolerance is changed.
        raw_match = index % 3 == 0
        hand, face = personal_results(.09 if raw_match else .10)
        frame = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1 + index / 30)
        assert frame.pose_ok
        assert frame.raw_pose_ok is raw_match
        assert frame.gesture_reason == ("personal gesture recognized" if raw_match
                                        else "personal gesture mismatch")
        assert frame.gesture_score == pytest.approx(.6076 if raw_match else .5622, abs=.0001)
        if mode == "hand":
            assert frame.hand_cursor is not None


def test_worker_grace_expires_from_last_real_match_and_clear_open_hand_rejects_immediately():
    worker = CameraWorker(gesture_profile=profile(), aim_mode="hand")
    good, face = personal_results()
    near, _ = personal_results(.10)
    assert worker._make_tracking(good, face, WIDTH, HEIGHT, 1.).pose_ok
    for timestamp in (1.05, 1.10, 1.15, 1.20):
        assert worker._make_tracking(near, face, WIDTH, HEIGHT, timestamp).pose_ok
    expired = worker._make_tracking(near, face, WIDTH, HEIGHT, 1.21)
    assert not expired.pose_ok and not expired.raw_pose_ok
    assert worker._make_tracking(good, face, WIDTH, HEIGHT, 1.25).pose_ok
    opened, _ = personal_results(opened=True)
    released = worker._make_tracking(opened, face, WIDTH, HEIGHT, 1.26)
    assert not released.pose_ok and not released.raw_pose_ok
    assert released.gesture_score == 0


@pytest.mark.parametrize("failure", ["side", "missing_world", "nan", "collapsed", "no_hand"])
def test_hard_tracking_failures_never_borrow_previous_personal_match(failure):
    worker = CameraWorker(gesture_profile=profile(), aim_mode="hand")
    hand, face = personal_results()
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.).pose_ok
    bad, _ = personal_results(.10)
    if failure == "side":
        bad.handedness[0][0].category_name = "Left"
    elif failure == "missing_world":
        bad.hand_world_landmarks = []
    elif failure == "nan":
        bad.hand_world_landmarks[0][4].x = np.nan
    elif failure == "collapsed":
        bad.hand_world_landmarks[0][4] = bad.hand_world_landmarks[0][3]
    else:
        bad.hand_landmarks = []
    rejected = worker._make_tracking(bad, face, WIDTH, HEIGHT, 1.05)
    assert not rejected.pose_ok and not rejected.raw_pose_ok
    near, _ = personal_results(.10)
    assert not worker._make_tracking(near, face, WIDTH, HEIGHT, 1.10).pose_ok


@pytest.mark.parametrize("change", ["profile", "source_size", "inference_size", "wrist"])
def test_worker_context_changes_discard_near_miss_grace(change):
    model = profile()
    worker = CameraWorker(gesture_profile=model, aim_mode="hand")
    hand, face = personal_results()
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.).pose_ok
    near, _ = personal_results(.10)
    width, height, source_size = WIDTH, HEIGHT, None
    if change == "profile":
        worker.set_gesture_profile(replace(model))
    elif change == "source_size":
        source_size = (1280, 720)
    elif change == "inference_size":
        width, height = 640, 360
    else:
        for point in near.hand_landmarks[0]:
            point.x -= .23
    rejected = worker._make_tracking(near, face, width, height, 1.05, source_size=source_size)
    assert not rejected.pose_ok and not rejected.raw_pose_ok


def test_generic_near_miss_requires_usable_landmarks_and_same_source(monkeypatch):
    current = {"result": GestureResult(.8, True, reason="recognized", source="world")}
    monkeypatch.setattr("kon.vision.classify_gao", lambda *_args, **_kwargs: current["result"])
    normalized, world, *_ = projected_hand()
    hand, face = _results(((normalized, world),))
    worker = CameraWorker(aim_mode="hand")
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.).pose_ok
    current["result"] = GestureResult(.46, False, reason="finger pattern", source="world")
    near = worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.05)
    assert near.pose_ok and near.raw_pose_ok is False
    hand.hand_world_landmarks[0][4] = hand.hand_world_landmarks[0][3]
    assert not worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.10).pose_ok
    current["result"] = GestureResult(.8, True, reason="recognized", source="world")
    hand, face = _results(((normalized, world),))
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.15).pose_ok
    current["result"] = GestureResult(.46, False, reason="finger pattern", source="normalized")
    assert not worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.20).pose_ok


@pytest.mark.parametrize("invalid_time", [1., .99, np.nan, np.inf])
def test_invalid_source_times_cannot_reprime_pose_even_when_frame_size_changes(invalid_time):
    worker = CameraWorker(gesture_profile=profile(), aim_mode="hand")
    hand, face = personal_results()
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.).pose_ok
    for _ in range(2):
        invalid = worker._make_tracking(hand, face, 640, 360, invalid_time, source_size=(1280, 720))
        assert not invalid.pose_ok and not invalid.raw_pose_ok
        assert invalid.depth_status == "invalid_timestamp"
        assert worker._last_source_timestamp == 1.
    near, _ = personal_results(.10)
    assert not worker._make_tracking(near, face, WIDTH, HEIGHT, 1.05).pose_ok
    assert worker._make_tracking(hand, face, WIDTH, HEIGHT, 1.10).pose_ok
