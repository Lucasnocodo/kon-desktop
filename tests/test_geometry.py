from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kon.geometry import (
    APERTURE_INDICES, ScreenCalibration, TriggerGate,
    classify_gao, ray_screen_intersection, ring_center,
)


def open_hand():
    return np.array([
        (.50, .80, 0), (.45, .75, 0), (.40, .70, 0), (.37, .65, 0), (.35, .60, 0),
        (.47, .60, 0), (.47, .50, 0), (.47, .42, 0), (.47, .35, 0),
        (.50, .58, 0), (.50, .47, 0), (.50, .38, 0), (.50, .30, 0),
        (.53, .60, 0), (.53, .50, 0), (.53, .42, 0), (.53, .35, 0),
        (.56, .62, 0), (.56, .54, 0), (.56, .47, 0), (.56, .40, 0),
    ])


def gao_hand():
    points = open_hand()
    points[12] = (.50, .62, 0)
    points[16] = (.53, .65, 0)
    points[4] = (.515, .635, 0)
    return points


def test_gao_requires_the_distinct_finger_pattern_and_thumb_contact():
    result = classify_gao(gao_hand())
    assert result.is_gao
    assert result.score > .9
    ily = gao_hand()
    ily[4] = open_hand()[4]
    assert not classify_gao(ily).is_gao


@pytest.mark.parametrize("shape", ["open", "fist", "pinch", "ok", "middle_pinch", "victory"])
def test_other_poses_cannot_trigger(shape):
    points = open_hand()
    if shape == "fist":
        points[[8, 12, 16, 20], 1] = [.62, .62, .65, .68]
        points[4] = points[8]
    elif shape == "pinch":
        points[4] = points[8]
    elif shape == "ok":
        points[8] = (.47, .65, 0)
        points[4] = points[8]
    elif shape == "middle_pinch":
        points[4] = points[12]
    elif shape == "victory":
        points[16, 1], points[20, 1] = .65, .68
    assert not classify_gao(points).is_gao


@pytest.mark.parametrize("angle", [0, .7, math.pi / 2, math.pi, -2.2])
@pytest.mark.parametrize("scale", [.1, 1, 10])
def test_pose_is_rotation_scale_and_translation_invariant(angle, scale):
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    points = gao_hand() @ rotation.T * scale + [2, -5, .3]
    assert classify_gao(points).is_gao
    assert classify_gao(points).score == pytest.approx(classify_gao(gao_hand()).score)


def test_aspect_correction_works_for_normalized_image_landmarks():
    points = gao_hand()
    normalized = points / [1.8, 1, 1.8]
    assert classify_gao(normalized, image_aspect_ratio=1.8).score == pytest.approx(classify_gao(points).score)


@pytest.mark.parametrize("points", [[], [[0, 0, 0]] * 21, [[math.nan, 0, 0]] * 21, [[0, 0]] * 21])
def test_bad_landmarks_fail_closed(points):
    assert not classify_gao(points).is_gao


def forward_curled_world_hand():
    """The folded fingers point forward, so tip radius exceeds PIP radius."""
    return np.array([
        (0, 0, 0), (-.30, -.25, -.10), (-.25, -.65, -.20),
        (0, -.99, -.48), (.06, -1, -.50),
        (-.15, -.65, 0), (-.15, -.95, 0), (-.15, -1.20, 0), (-.15, -1.40, 0),
        (0, -.65, 0), (0, -1, 0), (0, -1, -.30), (0, -1, -.50),
        (.15, -.65, 0), (.15, -1, 0), (.15, -1, -.30), (.15, -1, -.50),
        (.30, -.55, 0), (.30, -.85, 0), (.30, -1.05, 0), (.30, -1.25, 0),
    ])


def test_real_joint_curl_passes_when_wrist_distance_order_is_misleading():
    from touch.geometry import ily_pose

    points = forward_curled_world_hand()
    assert not ily_pose([tuple(point) for point in points])
    for tip, pip in ((12, 10), (16, 14)):
        assert np.linalg.norm(points[tip]) > np.linalg.norm(points[pip])
    result = classify_gao(open_hand(), world_points=points)
    assert result.is_gao
    assert result.source == "world"
    assert dict(result.finger_metrics)["middle_bend_degrees"] == pytest.approx(90)
    assert result.reason == "recognized"


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("scale", [.01, 1, 10])
def test_world_joint_geometry_is_invariant_to_full_3d_rotation(axis, scale):
    angle = 1.1
    rotation = np.eye(3)
    first, second = [(1, 2), (0, 2), (0, 1)][axis]
    rotation[first, first] = rotation[second, second] = math.cos(angle)
    rotation[first, second] = -math.sin(angle)
    rotation[second, first] = math.sin(angle)
    points = forward_curled_world_hand() @ rotation.T * scale + [2, -5, .3]
    expected = classify_gao([], world_points=forward_curled_world_hand())
    result = classify_gao([], image_aspect_ratio=math.nan, world_points=points)
    assert result.is_gao
    assert result.source == "world"
    assert result.score == pytest.approx(expected.score)
    assert result.thumb_contact_ratio == pytest.approx(expected.thumb_contact_ratio)


def test_thumb_shaft_and_ip_contact_do_not_require_thumb_tip_contact():
    points = gao_hand()
    points[3] = (.515, .635, 0)
    points[4] = (.60, .56, 0)
    palm_length = np.linalg.norm(points[9] - points[0])
    old_tip_contact = max(np.linalg.norm(points[4] - points[12]),
                          np.linalg.norm(points[4] - points[16])) / palm_length
    assert old_tip_contact > .428
    result = classify_gao(points)
    assert result.is_gao
    assert result.thumb_contact_ratio < .15


def test_thumb_pointing_outward_cannot_pass_by_contacting_only_its_base():
    points = gao_hand()
    points[2] = (.515, .635, 0)
    points[3] = (.45, .60, 0)
    points[4] = (.38, .56, 0)
    result = classify_gao(points)
    assert not result.is_gao
    assert result.thumb_contact_ratio < .15
    assert dict(result.finger_metrics)["thumb_opposition"] < 0
    assert "拇指朝外" in result.hint


@pytest.mark.parametrize("indices,base,lengths,label", [
    ((5, 6, 7, 8), (.47, .60, 0), (.10, .08, .07), "食指"),
    ((17, 18, 19, 20), (.56, .62, 0), (.08, .07, .07), "小指"),
])
@pytest.mark.parametrize("angles,accepted", [
    ((0, 60, 60), True),
    ((0, 45, 90), True),
    ((0, 70, 70), False),
    ((0, 50, 100), False),
])
def test_raised_fingers_allow_natural_bend_but_reject_deep_curl(indices, base, lengths, label, angles, accepted):
    points = gao_hand()
    chain = [np.array(base)]
    for length, degrees in zip(lengths, angles):
        angle = math.radians(degrees)
        chain.append(chain[-1] + length * np.array([math.sin(angle), -math.cos(angle), 0]))
    points[list(indices)] = chain
    result = classify_gao([], world_points=points)
    assert result.is_gao is accepted
    if not accepted:
        assert result.reason == "finger pattern"
        assert label in result.hint


@pytest.mark.parametrize("bend,accepted", [(50, True), (45, False)])
def test_folded_fingers_allow_a_looser_curl_with_thumb_contact(bend, accepted):
    points = forward_curled_world_hand()
    angle = math.radians(bend)
    direction = np.array([0, -math.cos(angle), -math.sin(angle)])
    for pip, dip, tip in ((10, 11, 12), (14, 15, 16)):
        points[dip] = points[pip] + .3 * direction
        points[tip] = points[pip] + .5 * direction
    folded_center = (points[12] + points[16]) / 2
    points[3] = folded_center + [0, .01, .02]
    points[4] = folded_center
    result = classify_gao([], world_points=points)
    assert result.is_gao is accepted
    if accepted:
        assert .50 <= result.score < .60
    else:
        assert result.reason == "finger pattern"
        assert "中指" in result.hint and "無名指" in result.hint


@pytest.mark.parametrize("finger,indices,tip,label", [
    ("index", (6, 7, 8), 8, "食指"),
    ("pinky", (18, 19, 20), 20, "小指"),
])
def test_bent_raised_finger_cannot_pass_a_large_wrist_distance_margin(finger, indices, tip, label):
    points = gao_hand()
    if finger == "index":
        points[list(indices)] = [(.47, .50, 0), (.60, .50, 0), (.60, .30, 0)]
    else:
        points[list(indices)] = [(.56, .54, 0), (.66, .54, 0), (.66, .32, 0)]
    radial_margin = (np.linalg.norm(points[tip] - points[0])
                     - np.linalg.norm(points[indices[0]] - points[0])) / np.linalg.norm(points[9] - points[0])
    assert radial_margin > .188
    result = classify_gao(points)
    assert not result.is_gao
    assert result.reason == "finger pattern"
    assert label in result.hint
    assert dict(result.finger_metrics)[finger + "_bend_degrees"] == pytest.approx(90)


def test_valid_world_pose_failure_never_retries_a_positive_image_pose():
    result = classify_gao(gao_hand(), world_points=open_hand() * .3)
    assert not result.is_gao
    assert result.source == "world"
    assert "中指" in result.hint and "無名指" in result.hint


@pytest.mark.parametrize("world", [None, [], [[0, 0, 0]] * 21,
                                   [[math.nan, 0, 0]] * 21, [[0, 0]] * 21])
def test_unavailable_or_invalid_world_points_use_the_image_fallback(world):
    result = classify_gao(gao_hand(), world_points=world)
    assert result.is_gao
    assert result.source == "normalized"


@pytest.mark.parametrize("start,end,label", [(6, 7, "食指"), (10, 11, "中指"),
                                             (14, 15, "無名指"), (18, 19, "小指"),
                                             (2, 3, "拇指"), (3, 4, "拇指")])
def test_degenerate_world_finger_bones_fail_without_image_pose_rescue(start, end, label):
    points = gao_hand()
    points[end] = points[start] + [1e-7, 0, 0]
    result = classify_gao(gao_hand(), world_points=points)
    assert not result.is_gao
    assert result.source == "world"
    assert label in result.hint and "追蹤不完整" in result.hint


def test_contact_feedback_identifies_the_fingertip_that_is_too_far():
    points = forward_curled_world_hand()
    # Preserve joint angles/folding while displacing only the ring chain.
    points[13:17, 0] += .45
    result = classify_gao([], world_points=points)
    assert not result.is_gao
    assert result.reason == "thumb contact or finger margin"
    assert "無名指未靠到" in result.hint
    assert "中指未靠到" not in result.hint


@pytest.mark.parametrize("depth,accepted", [(.45, True), (.48, True), (.51, False), (.55, False)])
def test_relaxed_thumb_contact_still_requires_both_fingers_within_half_a_palm(depth, accepted):
    points = gao_hand()
    points[2:5, 2] = depth * np.linalg.norm(points[9] - points[0])
    result = classify_gao([], world_points=points)
    assert result.is_gao is accepted
    if accepted:
        assert .428 < result.thumb_contact_ratio <= .50
        assert result.score >= .50
    else:
        assert result.thumb_contact_ratio > .50
        assert result.score < .50
        assert "未靠到拇指指腹" in result.hint


def test_finite_landmarks_with_overflowing_distances_fail_closed():
    points = gao_hand()
    points[8] = (1e300, 0, 0)
    with np.errstate(over="ignore"):
        result = classify_gao(gao_hand(), world_points=points)
    assert not result.is_gao
    assert result.source == "world"
    assert math.isfinite(result.score)


def test_ring_center_uses_documented_aperture_boundary():
    points = gao_hand()
    assert ring_center(points) == pytest.approx(points[list(APERTURE_INDICES)].mean(axis=0))
    assert ring_center([]) is None
    assert ring_center([[0, 0, 0]] * 21) is None


def test_ray_is_actual_eye_through_aperture_intersection():
    # Aperture halfway between the eye and plane: its XY offset doubles.
    assert ray_screen_intersection((.10, .20, .6), (.15, .25, .3)) == pytest.approx((.20, .30))
    # Moving the eye while preserving the aperture changes the aimed point.
    assert ray_screen_intersection((0, .20, .6), (.15, .25, .3)) == pytest.approx((.30, .30))


@pytest.mark.parametrize("eye,aperture", [
    ((0, 0, .6), (0, 0, .65)), ((0, 0, .6), (0, 0, .6)),
    ((0, 0, .6), (0, 0, .59)), ((0, 0, .6), (0, 0, -.1)),
    ((0, 0, 6), (0, 0, 3)), ((0, 0, .6), (5, 0, .3)),
    ((math.nan, 0, .6), (0, 0, .3)), ((0, .6), (0, 0, .3)),
])
def test_ray_rejects_degenerate_or_implausible_depth(eye, aperture):
    assert ray_screen_intersection(eye, aperture) is None


def calibration_samples():
    screen = np.array([(x, y) for y in (0, .5, 1) for x in (0, .5, 1)])
    # Includes mirror and skew rather than a convenient identity mapping.
    raw = screen @ np.array([[-.35, .03], [.07, .22]]) + [.12, -.15]
    return raw, screen


def test_affine_calibration_recovers_mirror_skew_and_reports_quality(tmp_path):
    raw, screen = calibration_samples()
    model = ScreenCalibration.fit(raw, screen)
    assert model.sample_count == 9
    assert model.rms_error < 1e-12
    assert model.max_error < 1e-12
    for point, expected in zip(raw, screen):
        assert model.predict(point) == pytest.approx(expected)
    path = tmp_path / "calibration.json"
    model.save(path)
    restored = ScreenCalibration.load(path)
    assert restored.predict(raw[4]) == pytest.approx((.5, .5))
    assert not path.with_name(path.name + ".tmp").exists()
    assert model.predict((20, 30)) is None


def test_calibration_accepts_small_measurement_noise():
    raw, screen = calibration_samples()
    raw += np.random.default_rng(42).normal(0, .002, raw.shape)
    model = ScreenCalibration.fit(raw, screen)
    assert 0 < model.rms_error < .04


def test_calibration_rejects_collinear_samples():
    _, screen = calibration_samples()
    raw = np.c_[np.arange(9) * .1, np.arange(9) * .2]
    with pytest.raises(ValueError, match="collinear"):
        ScreenCalibration.fit(raw, screen)


@pytest.mark.parametrize("limit", [0, -1, math.nan, math.inf])
def test_invalid_calibration_error_limits_are_rejected(limit):
    raw, screen = calibration_samples()
    with pytest.raises(ValueError, match="limits"):
        ScreenCalibration.fit(raw, screen, max_rms_error=limit)


def test_calibration_rejects_bad_correspondences_and_corrupted_saved_data(tmp_path):
    raw, screen = calibration_samples()
    screen[[0, 8]] = screen[[8, 0]]
    with pytest.raises(ValueError, match="residual"):
        ScreenCalibration.fit(raw, screen)
    raw, screen = calibration_samples()
    path = tmp_path / "calibration.json"
    ScreenCalibration.fit(raw, screen).save(path)
    payload = json.loads(path.read_text())
    payload["input_scale"] = [0, 0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ScreenCalibration.load(path)


def hold(gate, start=0., end=.7, target="file-a", armed=True):
    for timestamp in np.arange(start, end + .0001, .1):
        gate.update(float(timestamp), True, target, armed)


def test_gate_requires_target_dwell_before_immediate_keyword_trigger():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate, end=.6)
    assert not gate.ready
    assert gate.progress == pytest.approx(.6 / .65)
    gate.update(.7, True, "file-a", True)
    assert gate.ready
    assert gate.hear_keyword(.71) == "file-a"
    assert gate.hear_keyword(.72) is None
    assert gate.take_trigger() is None


def test_keyword_during_dwell_requires_new_speech_after_lock_finishes():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    assert gate.hear_keyword(.05) is None
    hold(gate, start=.1, end=.7)
    assert gate.take_trigger() is None
    assert gate.ready
    assert gate.hear_keyword(.71) == "file-a"
    assert gate.hear_keyword(.72) is None
    assert not gate.ready


def test_stale_keyword_cannot_trigger_after_long_dwell():
    gate = TriggerGate(dwell_seconds=2., minimum_dwell_seconds=2.)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.01)
    hold(gate, start=.1, end=2.1)
    assert gate.ready
    assert gate.take_trigger() is None
    assert gate.hear_keyword(2.11) == "file-a"


def test_stale_pose_blocks_keyword_but_completed_lock_resumes_on_the_same_file():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    acquired = gate.locked_at
    assert gate.hear_keyword(1.01) is None
    assert gate.locked_target == "file-a"
    gate.update(1.02, True, "file-b", True)
    assert gate.progress == 1
    assert gate.ready
    assert gate.locked_target == "file-a"
    assert gate.locked_at == acquired
    assert gate.hear_keyword(1.03, captured_at=.6) is None
    assert gate.hear_keyword(1.04, captured_at=1.03) == "file-a"


def test_target_change_keeps_observation_window_but_drops_old_keyword():
    gate = TriggerGate(dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.1)
    gate.update(.2, True, "file-b", True)
    assert gate.locked_target is None and not gate.ready
    hold(gate, start=.3, end=.7, target="file-b")
    assert gate.locked_target == "file-b"
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.71) == "file-b"


def test_consumed_practice_pose_cannot_fire_live_until_debounced_release():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"  # Practice consumes same gate.
    gate.update(.72, True, "file-a", False)
    gate.update(.73, True, "file-b", True)  # Mode/arming change is not release.
    hold(gate, start=.8, end=1.5, target="file-b")
    assert gate.hear_keyword(1.51) is None
    gate.update(1.52, False, None, True)
    gate.update(1.60, True, "file-b", True)  # Brief tracking blip cannot rearm.
    hold(gate, start=1.7, end=2.3, target="file-b")
    assert gate.hear_keyword(2.31) is None
    gate.update(2.4, False, None, True)
    gate.update(2.6, False, None, True)
    gate.update(2.7, False, None, True)
    hold(gate, start=2.8, end=3.5, target="file-b")
    assert gate.hear_keyword(3.51) == "file-b"


def test_disarm_drops_pending_speech_and_restarts_candidate():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    gate.update(.1, True, "file-a", False)
    assert gate.hear_keyword(.15) is None
    hold(gate, start=.2, end=.9)
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.91) == "file-a"


def test_missing_camera_frames_do_not_count_as_a_deliberate_release():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    gate.update(.8, False, None, True)
    # No continuous release observations: the camera stops for two seconds.
    hold(gate, start=2.8, end=3.5)
    assert gate.hear_keyword(3.51) is None


def test_losing_aim_target_without_releasing_hand_cannot_rearm_consumption():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    for timestamp in (.8, .9, 1.0, 1.1):
        gate.update(timestamp, True, None, True)  # Eye hidden or off-screen aim.
    hold(gate, start=1.2, end=1.9, target="file-b")
    assert gate.hear_keyword(1.91) is None


def test_prelock_speech_cannot_trigger_or_transfer_to_a_jittered_aim_target():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.7)
    gate.update(.8, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.81) == "file-a"
    hold(gate, start=.9, end=1.5, target="file-b")
    assert gate.hear_keyword(1.51) is None


def test_before_lock_missing_aim_and_brief_tracking_gap_keep_recent_votes():
    gate = TriggerGate()
    hold(gate, end=.3)
    assert gate.locked_target is None and gate.locked_at is None
    gate.update(.4, True, None, True)
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(.4)
    gate.update(.5, True, "file-b", True)
    assert gate.locked_target is None
    assert gate.hear_keyword(.81) is None  # Candidate pose became stale.
    gate.update(.82, True, "file-c", True)
    assert gate.leading_target == "file-a"
    assert gate.leading_dwell_seconds == pytest.approx(.4)
    assert gate.locked_target is None
    assert gate.hear_keyword(1.4) is None  # Past the tracking grace period.
    gate.update(1.41, True, "file-c", True)
    assert gate.progress == 0 and gate.locked_target is None


def test_completed_lock_retains_same_file_through_missing_and_different_aim():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    acquired = gate.locked_at
    for now, target in ((.8, None), (.9, "file-b"), (1., None), (1.1, "file-c")):
        assert gate.update(now, True, target, True)
        assert gate.locked_target == "file-a"
        assert gate.locked_at == acquired
        assert gate.progress == 1
    assert gate.hear_keyword(1.11, captured_at=.8) == "file-a"


def test_brief_observed_non_command_pose_pauses_without_losing_completed_lock():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    gate.update(.8, False, None, True, release_observed=True)
    assert not gate.ready
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(.81) is None
    gate.update(.9, False, None, True, release_observed=True)
    assert gate.locked_target == "file-a"
    assert gate.update(.95, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(.96, captured_at=.75) is None
    assert gate.hear_keyword(.97, captured_at=.96) == "file-a"


def test_continuous_observed_release_clears_lock_and_rearms_consumed_pose():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    for now in (.8, .9, 1.0):
        gate.update(now, False, None, True, release_observed=True)
        assert gate.locked_target == "file-a"
    gate.update(1.06, False, None, True, release_observed=True)
    assert gate.locked_target is None and gate.locked_at is None
    hold(gate, start=1.1, end=1.8, target="file-b")
    assert gate.locked_target == "file-b"
    assert gate.hear_keyword(1.81) == "file-b"


def test_unknown_tracking_false_never_releases_or_swaps_the_completed_lock():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    gate.update(.8, False, None, True, release_observed=True)
    for now in (.9, 1., 1.1, 1.2, 1.3):
        gate.update(now, False, None, True, release_observed=False)
        assert gate.locked_target == "file-a"
        assert not gate.ready
    assert gate.update(1.4, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(1.41, captured_at=.7) is None
    assert gate.hear_keyword(1.42, captured_at=1.4) == "file-a"


def test_camera_gap_cannot_extend_negative_release_evidence_or_count_as_hold():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    gate.update(.8, False, None, True, release_observed=True)
    gate.update(1.2, False, None, True, release_observed=True)
    assert gate.locked_target == "file-a"
    assert not gate.ready
    assert gate.update(1.25, True, "file-b", True)
    assert gate.locked_target == "file-a"


def test_one_negative_frame_then_positive_endpoint_cannot_unlock_or_rearm():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    gate.update(.8, False, None, True, release_observed=True)
    gate.update(1.06, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(1.07) is None


def test_watchdog_preserves_lock_and_consumption_after_postlock_speech():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.7)
    assert gate.hear_keyword(.71) == "file-a"
    gate.invalidate_tracking(.72)
    assert gate.locked_target == "file-a"
    assert not gate.ready
    assert gate.take_trigger() is None
    gate.update(.8, True, "file-b", True)
    assert gate.locked_target == "file-a"
    assert gate.hear_keyword(.81) is None  # The original action remains consumed.


def test_explicit_ui_clear_lock_discards_target_but_preserves_consumption():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.7)
    assert gate.hear_keyword(.71) == "file-a"
    gate.clear_lock(.72)
    assert gate.locked_target is None and gate.locked_at is None
    assert not gate.ready and gate.progress == 0
    assert gate.take_trigger() is None
    hold(gate, start=.8, end=1.5, target="file-b")
    assert gate.locked_target is None and gate.locked_at is None
    assert gate.progress == 0 and not gate.ready
    assert gate.hear_keyword(1.51) is None
    for now in (1.6, 1.8, 1.9):
        gate.update(now, False, None, True, release_observed=True)
    hold(gate, start=2., end=2.7, target="file-b")
    assert gate.locked_target == "file-b"
    assert gate.hear_keyword(2.71) == "file-b"


def test_removed_consumed_target_cannot_retarget_after_short_negative_blip():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    gate.clear_lock(.72)
    gate.update(.8, False, None, True, release_observed=True)
    hold(gate, start=.9, end=1.6, target="file-b")
    assert gate.locked_target is None
    assert gate.progress == 0
    assert gate.hear_keyword(1.61) is None


def test_disarm_preserves_lock_but_audio_requires_new_arming_and_freshness_epoch():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    gate.update(.8, True, "file-b", False)
    assert gate.locked_target == "file-a" and not gate.ready
    assert gate.update(.9, True, "file-b", True)
    assert gate.hear_keyword(.91, captured_at=.7) is None
    assert gate.hear_keyword(.92, captured_at=.9) == "file-a"


def test_prelock_speech_cannot_be_dispatched_after_pose_goes_stale():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.7)
    assert gate.take_trigger(now=1.01) is None


def test_fresh_lock_requires_new_speech_instead_of_queued_prelock_speech():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    hold(gate, start=.1, end=.7)
    assert gate.take_trigger(now=.71) is None
    assert gate.take_trigger(now=.72) is None
    assert gate.hear_keyword(.73) == "file-a"


def test_repeated_camera_watchdog_invalidation_never_rearms_consumed_pose():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    for timestamp in (.8, .9, 1., 1.1, 1.2):
        gate.invalidate_tracking(timestamp)
        assert not gate.ready
        assert gate.progress == 0
    hold(gate, start=1.3, end=2.)
    assert gate.hear_keyword(2.01) is None
    # Only actual observed release, after tracking returns, can rearm.
    gate.update(2.1, False, None, True)
    gate.update(2.3, False, None, True)
    gate.update(2.4, False, None, True)
    hold(gate, start=2.5, end=3.2)
    assert gate.hear_keyword(3.21) == "file-a"


def test_tracking_invalidation_cancels_a_partly_observed_release():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.71) == "file-a"
    gate.update(.8, False, None, True)
    gate.invalidate_tracking(.9)
    hold(gate, start=1., end=1.7)
    assert gate.hear_keyword(1.71) is None


def test_tracking_invalidation_drops_pending_keyword_and_trigger():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., True, "file-a", True)
    gate.hear_keyword(.05)
    gate.invalidate_tracking(.1)
    hold(gate, start=.2, end=.9)
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.91) == "file-a"

    other = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    other.update(0., True, "file-a", True)
    other.hear_keyword(.05)
    hold(other, start=.1, end=.7)
    assert other.hear_keyword(.71) == "file-a"
    other.invalidate_tracking(.72)
    assert other.take_trigger() is None
    hold(other, start=.8, end=1.5)
    assert other.hear_keyword(1.51) is None


def test_keyword_without_active_target_and_backward_time_are_ignored():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    gate.update(0., False, None, True)
    assert gate.hear_keyword(.05) is None
    hold(gate, start=.1, end=.8)
    assert gate.hear_keyword(.79) is None
    assert gate.hear_keyword(math.nan) is None
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.81) == "file-a"


def test_delayed_recognition_requires_audio_captured_after_lock():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    # Captured before latest camera frame; recognition finishes afterward.
    assert gate.hear_keyword(.75, captured_at=.4) is None
    assert gate.hear_keyword(.76, captured_at=.75) == "file-a"
    assert gate.hear_keyword(.77, captured_at=.75) is None


def test_delayed_keyword_before_dwell_completes_does_not_trigger_on_next_frame():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate, end=.6)
    assert gate.hear_keyword(.62, captured_at=.2) is None
    gate.update(.7, True, "file-a", True)
    assert gate.take_trigger() is None
    assert gate.hear_keyword(.71, captured_at=.2) is None
    assert gate.hear_keyword(.72, captured_at=.7) == "file-a"


def test_delayed_keyword_cannot_transfer_to_new_target():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate, end=.1)
    gate.update(.2, True, "file-b", True)
    hold(gate, start=.3, end=1.1, target="file-b")
    assert gate.hear_keyword(1.12, captured_at=.1) is None
    assert gate.hear_keyword(1.13, captured_at=.5) is None
    assert gate.hear_keyword(1.14, captured_at=1.) == "file-b"


def test_delayed_keyword_cannot_survive_disarm_and_rearm():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate, end=.2)
    gate.update(.3, True, "file-a", False)
    hold(gate, start=.4, end=1.1)
    assert gate.hear_keyword(1.12, captured_at=.1) is None
    assert gate.hear_keyword(1.13, captured_at=.5) is None
    assert gate.hear_keyword(1.14, captured_at=1.11) == "file-a"


@pytest.mark.parametrize("captured_at", [-1, .9, math.nan, math.inf])
def test_future_stale_or_invalid_audio_source_times_are_rejected(captured_at):
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate)
    assert gate.hear_keyword(.8, captured_at=captured_at) is None
    assert gate.ready


def test_keyword_at_max_age_boundary_is_accepted_but_older_one_is_rejected():
    gate = TriggerGate(dwell_seconds=.65, minimum_dwell_seconds=.65)
    hold(gate, end=2.2)
    assert gate.hear_keyword(2.21, captured_at=.7) is None
    assert gate.hear_keyword(2.22, captured_at=.72) == "file-a"
