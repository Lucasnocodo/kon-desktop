from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from kon.geometry import classify_gao
from kon.gesture_calibration import (
    FEATURE_NAMES, GestureAngleCalibration, GestureCalibration, GestureEnrollmentBuffer,
    add_gesture_angle, extract_gesture_features,
)


def open_hand():
    return np.array([
        (.50, .80, 0), (.45, .75, 0), (.40, .70, 0), (.37, .65, 0), (.35, .60, 0),
        (.47, .60, 0), (.47, .50, 0), (.47, .42, 0), (.47, .35, 0),
        (.50, .58, 0), (.50, .47, 0), (.50, .38, 0), (.50, .30, 0),
        (.53, .60, 0), (.53, .50, 0), (.53, .42, 0), (.53, .35, 0),
        (.56, .62, 0), (.56, .54, 0), (.56, .47, 0), (.56, .40, 0),
    ])


def personal_hand():
    points = open_hand()
    points[12] = (.50, .62, 0)
    points[16] = (.53, .65, 0)
    points[3] = (.515, .635, 0)
    # The fingertip projects past the contact point, unlike the stock pose.
    points[4] = (.65, .61, .04)
    return points


def frames(points=None, count=16, noise=0., seed=10):
    points = personal_hand() if points is None else points
    rng = np.random.default_rng(seed)
    return [points + rng.normal(0, noise, points.shape) for _ in range(count)]


def profile(points=None):
    return GestureCalibration.fit(frames(points), handedness="Right").confirm(
        frames(points, noise=.0002), handedness="right")


def test_personal_template_does_not_require_the_strict_classifier_to_accept():
    points = personal_hand()
    assert not classify_gao(points, world_points=points).is_gao
    model = profile(points)
    assert model.confirmed
    assert model.sample_count == 32
    assert model.confirmation_sample_count == 16
    result = model.match([], world_points=points, handedness="Right")
    assert result.is_gao
    assert result.score > .98
    assert result.reason == "personal gesture recognized"
    assert result.source == "world"
    assert "個人" in result.hint


@pytest.mark.parametrize("pose", ["open", "fist", "index_pinch", "ok", "middle_pinch", "victory", "ily"])
def test_substantially_different_poses_do_not_match_the_personal_template(pose):
    points = open_hand()
    if pose == "fist":
        points[[8, 12, 16, 20], 1] = [.62, .62, .65, .68]
        points[4] = points[8]
    elif pose == "index_pinch":
        points[4] = points[8]
    elif pose == "ok":
        points[8] = (.47, .65, 0)
        points[4] = points[8]
    elif pose == "middle_pinch":
        points[4] = points[12]
    elif pose == "victory":
        points[16, 1], points[20, 1] = .65, .68
    elif pose == "ily":
        points[12] = (.50, .62, 0)
        points[16] = (.53, .65, 0)
    result = profile().match(points, source="world", handedness="right")
    assert not result.is_gao
    assert result.reason == "personal gesture mismatch"


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("scale", [.01, 1, 10])
def test_features_and_matching_ignore_rotation_translation_and_uniform_scale(axis, scale):
    angle = .9
    rotation = np.eye(3)
    first, second = ((1, 2), (0, 2), (0, 1))[axis]
    rotation[first, first] = rotation[second, second] = math.cos(angle)
    rotation[first, second] = -math.sin(angle)
    rotation[second, first] = math.sin(angle)
    moved = personal_hand() @ rotation.T * scale + [3, -7, .4]
    assert extract_gesture_features(moved) == pytest.approx(extract_gesture_features(personal_hand()))
    assert profile().match(moved, source="world", handedness="right").is_gao


def test_image_features_correct_aspect_but_source_mismatch_never_falls_back():
    points = personal_hand()
    assert extract_gesture_features(points / [1.8, 1, 1.8], source="normalized", aspect_ratio=1.8) == pytest.approx(
        extract_gesture_features(points))
    result = profile().match(points, source="normalized", handedness="right")
    assert not result.is_gao
    assert result.reason == "personal gesture source"
    result = profile().match(points, world_points=open_hand(), handedness="right")
    assert not result.is_gao


@pytest.mark.parametrize("side", ["left", "unknown", "invalid"])
def test_enrolled_hand_side_is_required(side):
    result = profile().match(personal_hand(), source="world", handedness=side)
    assert not result.is_gao


def test_confirmation_requires_same_source_side_and_same_pose():
    provisional = GestureCalibration.fit(frames(), handedness="right")
    for kwargs in ({"source": "normalized", "handedness": "right"},
                   {"source": "world", "handedness": "left"}):
        with pytest.raises(ValueError, match="相同"):
            provisional.confirm(frames(), **kwargs)
    with pytest.raises(ValueError, match="不同"):
        provisional.confirm(frames(open_hand()), handedness="right")
    with pytest.raises(ValueError, match="已完成"):
        profile().confirm(frames(), handedness="right")


@pytest.mark.parametrize("bad", [[], [[0, 0, 0]] * 21, [[math.nan, 0, 0]] * 21, [[0, 0]] * 21])
def test_invalid_data_cannot_enroll_or_match(bad):
    with pytest.raises(ValueError):
        extract_gesture_features(bad)
    with pytest.raises(ValueError):
        GestureCalibration.fit([bad] * 16)
    assert not profile().match(bad, source="world", handedness="right").is_gao


def test_short_bone_and_noisy_changing_poses_cannot_expand_match_tolerance():
    broken = personal_hand()
    broken[7] = broken[6]
    with pytest.raises(ValueError, match="食指"):
        GestureCalibration.fit(frames(broken))
    changing = frames(count=12) + frames(open_hand(), count=12)
    with pytest.raises(ValueError, match="變動|跳動|改變"):
        GestureCalibration.fit(changing)


def test_small_camera_noise_and_an_isolated_spike_do_not_spoil_the_hold():
    samples = frames(count=24, noise=.0006)
    samples[8] = open_hand()
    model = GestureCalibration.fit(samples, handedness="right")
    assert model.sample_count == 23
    confirmed = model.confirm(frames(noise=.0006, seed=34), handedness="right")
    assert confirmed.match(personal_hand(), source="world", handedness="right").is_gao


@pytest.mark.parametrize("landmark,offset,outside", [(4, .12, .15), (12, .10, .12)])
def test_saved_template_accepts_looser_contact_positions_with_a_bounded_limit(tmp_path, landmark, offset, outside):
    model = profile()
    path = tmp_path / "profile.json"
    model.save(path)
    saved = path.read_bytes()
    restored = GestureCalibration.load(path)
    points = personal_hand()
    points[landmark, 2] += offset
    result = restored.match(points, source="world", handedness="right")
    assert result.is_gao and result.score >= .60
    points[landmark, 2] += outside - offset
    assert not restored.match(points, source="world", handedness="right").is_gao
    assert restored == model and path.read_bytes() == saved


def test_confirmation_and_live_preview_accept_the_same_looser_contact_pose():
    provisional = GestureCalibration.fit(frames(), handedness="right")
    points = personal_hand()
    points[12, 2] += .10
    assert provisional.match(points, source="world", handedness="right").is_gao
    confirmed = provisional.confirm(frames(points), handedness="right")
    assert confirmed.confirmed
    assert confirmed.match(points, source="world", handedness="right").is_gao
    assert confirmed.match(personal_hand(), source="world", handedness="right").is_gao


def test_saved_confirmed_numeric_template_survives_restart(tmp_path):
    path = tmp_path / "profile.json"
    model = profile()
    model.save(path)
    payload = json.loads(path.read_text())
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert "samples" not in payload and "images" not in payload
    restored = GestureCalibration.load(path)
    assert restored == model
    assert restored.match(personal_hand(), source="world", handedness="right").is_gao
    assert not list(tmp_path.glob("*.tmp"))


def test_unconfirmed_model_cannot_replace_a_saved_profile(tmp_path):
    path = tmp_path / "profile.json"
    profile().save(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="第二次"):
        GestureCalibration.fit(frames()).save(path)
    assert path.read_bytes() == original


def test_failed_atomic_replacement_preserves_previous_profile_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    profile().save(path)
    original = path.read_bytes()
    def fail_replace(*_args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr("kon.gesture_calibration.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk"):
        profile().save(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [
    ("version", 99), ("version", True), ("confirmed", False), ("center", [0] * 23),
    ("center", [True] * 24),
    ("center", [math.nan] * 24), ("bandwidth", [0] * 24),
    ("bandwidth", [10] * 24), ("sample_count", True),
    ("sample_count", 20), ("confirmation_sample_count", 0),
    ("source", "other"), ("handedness", "both"), ("feature_names", []),
])
def test_corrupted_profile_is_rejected(tmp_path, field, value):
    path = tmp_path / "profile.json"
    profile().save(path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="校正檔無效"):
        GestureCalibration.load(path)


def fill(buffer, start=0., count=16, step=1 / 15, points=None, side="right"):
    for index, point in enumerate(frames(points, count)):
        buffer.observe(start + index * step, point, handedness=side)
    return start + (count - 1) * step


def test_buffer_requires_unique_fresh_frames_covering_a_full_second():
    buffer = GestureEnrollmentBuffer()
    buffer.reset(10.)
    assert not buffer.observe(9.9, personal_hand(), handedness="right")
    for index in range(12):
        assert buffer.observe(10 + index / 30, personal_hand(), handedness="right")
        assert not buffer.observe(10 + index / 30, personal_hand(), handedness="right")
    assert not buffer.evaluate(10.4).ready
    assert buffer.evaluate(10.4).count == 12
    fill(buffer, start=10.4, count=16)
    state = buffer.evaluate(11.4)
    assert state.ready
    assert state.span >= 1
    assert state.source == "world" and state.handedness == "right"
    assert len(state.samples) == state.count >= 12


def test_buffer_supports_twelve_fresh_unique_frames_at_three_fps():
    buffer = GestureEnrollmentBuffer()
    end = fill(buffer, count=12, step=1 / 3)
    state = buffer.evaluate(end + .1)
    assert state.ready
    assert state.count == 12
    assert state.span == pytest.approx(11 / 3)


def test_new_capture_excludes_old_frames_and_cannot_reuse_confirmation():
    buffer = GestureEnrollmentBuffer()
    end = fill(buffer)
    first = buffer.evaluate(end)
    assert first.ready
    buffer.reset(end + .1)
    assert not buffer.observe(end, personal_hand(), handedness="right")
    assert not buffer.evaluate(end + .1).ready
    second_end = fill(buffer, start=end + .2)
    second = buffer.evaluate(second_end)
    assert second.ready
    confirmed = GestureCalibration.fit(first.samples, source=first.source, handedness=first.handedness).confirm(
        second.samples, source=second.source, handedness=second.handedness)
    assert confirmed.confirmed


def test_camera_loss_future_time_or_latest_invalid_frame_blocks_capture():
    buffer = GestureEnrollmentBuffer()
    end = fill(buffer)
    assert buffer.evaluate(end).ready
    assert not buffer.evaluate(end - .01).ready
    assert not buffer.evaluate(end + .36).ready
    buffer.observe(end + .1, None, handedness="right")
    assert not buffer.evaluate(end + .1).ready
    buffer.observe(end + .2, personal_hand(), handedness="right")
    assert buffer.evaluate(end + .2).ready


def test_latest_spike_and_long_camera_gap_do_not_capture_an_old_majority():
    buffer = GestureEnrollmentBuffer()
    end = fill(buffer, count=24)
    buffer.observe(end + .1, open_hand(), handedness="right")
    assert not buffer.evaluate(end + .1).ready
    buffer.observe(end + .2, personal_hand(), handedness="right")
    assert buffer.evaluate(end + .2).ready
    buffer.observe(end + 1., personal_hand(), handedness="right")
    state = buffer.evaluate(end + 1.)
    assert not state.ready and state.count == 1


@pytest.mark.parametrize("kwargs", [{"source": "normalized", "handedness": "right"},
                                   {"handedness": "unknown"}])
def test_capture_requires_world_tracking_and_a_known_hand(kwargs):
    buffer = GestureEnrollmentBuffer()
    for index in range(16):
        buffer.observe(index / 15, personal_hand(), **kwargs)
    state = buffer.evaluate(1.)
    assert not state.ready
    assert state.count == 0


def test_changing_hand_resets_capture_instead_of_mixing_hands():
    buffer = GestureEnrollmentBuffer()
    end = fill(buffer)
    buffer.observe(end + .1, personal_hand(), handedness="left")
    state = buffer.evaluate(end + .1)
    assert not state.ready and state.count == 1
    assert state.handedness == "left"


def test_continuously_changing_pose_cannot_be_enrolled_as_stable():
    buffer = GestureEnrollmentBuffer()
    for index in range(24):
        points = personal_hand() if index % 2 else open_hand()
        buffer.observe(index / 15, points, handedness="right")
    assert not buffer.evaluate(23 / 15).ready


def alternate_fold_view():
    """Another estimated view of the same middle/ring-folded hand pattern."""
    points = personal_hand()
    points[10] = (.50, .48, .09)
    points[11] = (.50, .55, .045)
    points[14] = (.53, .52, .09)
    points[15] = (.53, .585, .045)
    return points


def two_angles():
    original = profile()
    alternate = profile(alternate_fold_view())
    return original, alternate, add_gesture_angle(original, alternate)


def test_appended_angle_matches_either_template_without_averaging_or_widening():
    original, alternate, combined = two_angles()
    assert not original.match(alternate_fold_view(), source="world", handedness="right").is_gao
    assert not alternate.match(personal_hand(), source="world", handedness="right").is_gao
    assert combined.match(personal_hand(), source="world", handedness="right").is_gao
    assert combined.match([], world_points=alternate_fold_view(), handedness="Right").is_gao
    assert combined.models[0] is original and combined.models[1] is alternate
    assert combined.models[0].center == original.center
    assert combined.models[1].center == alternate.center
    assert combined.models[0].bandwidth == original.bandwidth
    assert combined.models[1].bandwidth == alternate.bandwidth
    assert original.angle_count == 1 and combined.angle_count == 2
    assert combined.source == "world" and combined.handedness == "right" and combined.confirmed
    assert combined.sample_count == original.sample_count + alternate.sample_count


def test_two_angle_centers_do_not_create_an_unrecorded_averaged_template(monkeypatch):
    original, alternate, combined = two_angles()
    midpoint = tuple((np.array(original.center) + alternate.center) / 2)
    monkeypatch.setattr("kon.gesture_calibration.extract_gesture_features", lambda *_args, **_kwargs: midpoint)
    assert not original.match([], source="world", handedness="right").is_gao
    assert not alternate.match([], source="world", handedness="right").is_gao
    assert not combined.match([], source="world", handedness="right").is_gao


def test_all_zero_scores_still_use_the_closest_angle_failure_hint(monkeypatch):
    original, alternate, combined = two_angles()
    features = list(alternate.center)
    features[0:2] = [-1, 0]  # Index mismatch, while middle/ring remain near angle two.
    monkeypatch.setattr("kon.gesture_calibration.extract_gesture_features", lambda *_args, **_kwargs: tuple(features))
    first = original.match([], source="world", handedness="right")
    second = alternate.match([], source="world", handedness="right")
    result = combined.match([], source="world", handedness="right")
    assert first.score == second.score == result.score == 0
    assert "食指" not in first.hint
    assert "食指" in second.hint
    assert result.hint == second.hint


def test_any_passed_angle_wins_even_at_different_peak_and_group_tolerance_edges(monkeypatch):
    reference = profile()
    query = reference.center
    width = reference.bandwidth
    peak_center = list(query)
    peak_center[20] += width[20] * 1.35 * (2 + 1.5e-9)
    peak_failure = replace(reference, center=tuple(peak_center))
    group_center = list(query)
    for index in range(16, 24):
        sign = -1 if index == 23 else 1  # Keep thumb straightness within [0,1].
        group_center[index] += sign * width[index] * 1.35 * (1 + .9e-9)
    group_pass = replace(reference, center=tuple(group_center))
    combined = add_gesture_angle(peak_failure, group_pass)
    monkeypatch.setattr("kon.gesture_calibration.extract_gesture_features", lambda *_args, **_kwargs: query)
    assert not peak_failure.match([], source="world", handedness="right").is_gao
    assert group_pass.match([], source="world", handedness="right").is_gao
    assert combined.match([], source="world", handedness="right").is_gao


@pytest.mark.parametrize("pose", ["open", "fist", "ok", "ily"])
def test_multiple_folded_views_still_reject_substantially_different_shapes(pose):
    points = open_hand()
    if pose == "fist":
        points[[8, 12, 16, 20], 1] = [.62, .62, .65, .68]
        points[4] = points[8]
    elif pose == "ok":
        points[8] = (.47, .65, 0)
        points[4] = points[8]
    elif pose == "ily":
        points[12] = (.50, .62, 0)
        points[16] = (.53, .65, 0)
    assert not two_angles()[2].match(points, source="world", handedness="right").is_gao


@pytest.mark.parametrize("kwargs,reason", [
    ({"source": "normalized", "handedness": "right"}, "personal gesture source"),
    ({"source": "world", "handedness": "left"}, "personal gesture handedness"),
    ({"source": "world", "handedness": "unknown"}, "personal gesture handedness"),
])
def test_angle_set_never_relaxes_source_or_hand_side(kwargs, reason):
    combined = two_angles()[2]
    result = combined.match(personal_hand(), **kwargs)
    assert not result.is_gao and result.reason == reason


def test_invalid_world_data_does_not_try_a_valid_image_or_another_angle():
    combined = two_angles()[2]
    result = combined.match(personal_hand(), world_points=[[math.nan, 0, 0]] * 21, handedness="right")
    assert not result.is_gao and result.reason == "personal gesture landmarks"


@pytest.mark.parametrize("kind", ["new_unconfirmed", "old_unconfirmed", "wrong_side", "wrong_source"])
def test_invalid_append_preserves_previous_profile_file_and_template(tmp_path, kind):
    original = profile()
    path = tmp_path / "profile.json"
    original.save(path)
    previous = path.read_bytes()
    old, new = original, profile(alternate_fold_view())
    if kind == "new_unconfirmed":
        new = GestureCalibration.fit(frames(), handedness="right")
    elif kind == "old_unconfirmed":
        old = GestureCalibration.fit(frames(), handedness="right")
    elif kind == "wrong_side":
        new = GestureCalibration.fit(frames(), handedness="left").confirm(frames(), handedness="left")
    else:
        new = GestureCalibration.fit(frames(), source="normalized", handedness="right").confirm(
            frames(), source="normalized", handedness="right")
    with pytest.raises(ValueError):
        add_gesture_angle(old, new)
    assert path.read_bytes() == previous
    assert original.angle_count == 1
    assert original.match(personal_hand(), source="world", handedness="right").is_gao


def test_appending_returns_new_immutable_angle_set_and_caps_at_five():
    original, alternate, initial = two_angles()
    expanded = add_gesture_angle(initial, original)
    assert initial.angle_count == 2 and expanded.angle_count == 3
    assert expanded.models[:2] == initial.models
    for _ in range(2):
        expanded = add_gesture_angle(expanded, alternate)
    assert expanded.angle_count == 5
    with pytest.raises(ValueError, match="最多.*5"):
        add_gesture_angle(expanded, original)
    with pytest.raises(FrozenInstanceError):
        expanded.models = ()


def test_angle_set_rejects_shallow_mutable_template_fields():
    original = profile()
    mutable = replace(original, center=list(original.center))
    with pytest.raises(ValueError, match="不可變"):
        add_gesture_angle(original, mutable)


def test_version_one_load_and_version_two_round_trip_preserve_exact_old_data(tmp_path):
    original, alternate, combined = two_angles()
    path = tmp_path / "profile.json"
    original.save(path)
    old_payload = json.loads(path.read_text())
    old_loaded = GestureCalibration.load(path)
    assert isinstance(old_loaded, GestureCalibration)
    assert old_loaded == original and old_loaded.angle_count == 1
    combined.save(path)
    new_payload = json.loads(path.read_text())
    assert new_payload["version"] == 2 and new_payload["angle_count"] == 2
    assert new_payload["models"][0] == old_payload
    restored = GestureCalibration.load(path)
    assert isinstance(restored, GestureAngleCalibration)
    assert restored.models == (original, alternate)
    assert restored.angle_count == 2
    assert restored.match(personal_hand(), source="world", handedness="right").is_gao
    assert restored.match(alternate_fold_view(), source="world", handedness="right").is_gao
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_multi_angle_save_preserves_original_v1_bytes(tmp_path, monkeypatch):
    original, _, combined = two_angles()
    path = tmp_path / "profile.json"
    original.save(path)
    previous = path.read_bytes()
    def fail_replace(*_args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr("kon.gesture_calibration.os.replace", fail_replace)
    with pytest.raises(OSError):
        combined.save(path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [
    ("version", 99), ("version", True), ("models", []), ("models", None),
    ("angle_count", True), ("angle_count", 3), ("confirmed", False),
    ("source", "normalized"), ("handedness", "left"), ("model", "other"),
])
def test_corrupted_multi_angle_outer_payload_is_rejected(tmp_path, field, value):
    path = tmp_path / "profile.json"
    two_angles()[2].save(path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="校正檔無效"):
        GestureCalibration.load(path)


@pytest.mark.parametrize("change", ["one_angle", "six_angles", "unconfirmed", "wrong_side", "wrong_source",
                                    "wide_bandwidth", "nan_center", "nested_version_two"])
def test_corrupted_multi_angle_inner_template_is_rejected(tmp_path, change):
    path = tmp_path / "profile.json"
    two_angles()[2].save(path)
    payload = json.loads(path.read_text())
    if change == "one_angle":
        payload["models"] = payload["models"][:1]
        payload["angle_count"] = 1
    elif change == "six_angles":
        payload["models"] = payload["models"] * 3
        payload["angle_count"] = 6
    else:
        model = payload["models"][1]
        if change == "unconfirmed":
            model["confirmed"] = False
        elif change == "wrong_side":
            model["handedness"] = "left"
        elif change == "wrong_source":
            model["source"] = "normalized"
        elif change == "wide_bandwidth":
            model["bandwidth"] = [10] * 24
        elif change == "nan_center":
            model["center"][0] = math.nan
        else:
            model["version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="校正檔無效"):
        GestureCalibration.load(path)
