"""Standard and supplemental angle enrollment commit only when complete."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.calibration_store import gesture_profile_path
from kon.gesture_calibration import STANDARD_GESTURE_ANGLES, GestureCalibration, GestureAngleCalibration
from kon.vision import TrackingFrame
from test_gesture_calibration import personal_hand
from test_hand_only_app import FakeWorker


class FakeCamera(FakeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = kwargs.get("gesture_profile")

    def set_gesture_profile(self, profile):
        self.profile = profile


@pytest.fixture
def session(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.CameraWorker", FakeCamera)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeWorker)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    points = personal_hand()
    original = GestureCalibration.fit([points] * 12, source="world", handedness="right").confirm(
        [points] * 12, source="world", handedness="right")
    path = gesture_profile_path(tmp_path)
    original.save(path)
    previous = path.read_bytes()
    window = Window(aim_mode="hand")
    assert window.gesture_profile == original
    yield window, clock, window.gesture_profile, path, previous
    window.close()
    application.processEvents()


def held_angle(window, clock, *, direction=None, handedness="Right", count=13):
    axis, angle = ((0, 0), (0, -.35), (0, .35), (1, -.35), (1, .35))[direction] if direction is not None else (1, .35)
    rotation = np.eye(3)
    first, second = (1, 2) if axis == 0 else (0, 2)
    rotation[first, first] = rotation[second, second] = np.cos(angle)
    rotation[first, second], rotation[second, first] = -np.sin(angle), np.sin(angle)
    points = tuple(map(tuple, np.asarray(personal_hand()) @ rotation.T))
    start = clock["now"] + .01
    frames = []
    for index in range(count):
        timestamp = start + index * .1
        clock["now"] = timestamp + .01
        frame = TrackingFrame(
            timestamp, (), (), 0, False, None, None, None, "目前舊手勢尚未通過",
            hand_detected=True, hand_world_points=points, handedness=handedness, aiming_mode="hand")
        frames.append(frame)
        window.on_tracking(frame)
    return frames


def release_review_hand(window, clock):
    timestamp = clock["now"] + .01
    clock["now"] = timestamp + .01
    window.on_tracking(TrackingFrame(
        timestamp, (), (), 0, False, None, None, None, "手已移出畫面",
        hand_detected=False, depth_status="no_hand", aiming_mode="hand"))


def pass_fox_review(window, clock):
    assert window.gesture_review_profile is not None and not window.gesture_review_passed
    release_review_hand(window, clock)
    assert window.gesture_review_released
    held_angle(window, clock, direction=0)
    assert window.gesture_review_passed and window.gesture_dialog.record_button.isEnabled()


def begin_standard_review(window, clock):
    window.toggle_sensors()
    window.gesture_button.click()
    for index in range(5):
        complete_standard_angle(window, clock, index)
    assert window.gesture_calibrating and window.gesture_review_profile.angle_count == 5
    assert window.gesture_dialog._review_mode and not window.gesture_dialog.record_button.isEnabled()


def complete_standard_angle(window, clock, index):
    assert window.gesture_angle_index == index
    window.capture_gesture()
    held_angle(window, clock, direction=index)
    assert window.gesture_dialog.step == 1
    window.capture_gesture()
    held_angle(window, clock, direction=index)


def start_append(window):
    window.toggle_sensors()  # Inert workers only.
    assert window.gesture_angle_button.isEnabled()
    window.gesture_angle_button.click()
    assert window.gesture_calibrating and window.gesture_append


def test_new_angle_saves_only_after_two_holds_preserves_old_and_restarts(session):
    window, clock, original, path, previous = session
    aiming = window.calibration
    start_append(window)
    window.capture_gesture()
    held_angle(window, clock)
    assert window.gesture_dialog.step == 1
    assert window.gesture_profile == original and path.read_bytes() == previous
    assert window.camera.profile == original
    window.capture_gesture()
    held_angle(window, clock)
    assert not window.gesture_calibrating and not window.gesture_append
    assert isinstance(window.gesture_profile, GestureAngleCalibration)
    assert window.gesture_profile.angle_count == 2
    assert window.gesture_profile.models[0] == original
    assert window.camera.profile is window.gesture_profile
    assert window.calibration is aiming
    assert "2 個角度" in window.gesture_profile_label.text()
    loaded = GestureCalibration.load(path)
    assert loaded == window.gesture_profile
    window.stop_sensors()
    reopened = Window(aim_mode="hand")
    try:
        assert reopened.gesture_profile == loaded
        assert "2 個角度" in reopened.gesture_profile_label.text()
        assert not reopened.gesture_angle_button.isEnabled()
    finally:
        reopened.close()


def test_cancel_new_angle_keeps_original_file_and_running_camera_profile(session):
    window, clock, original, path, previous = session
    start_append(window)
    window.capture_gesture()
    held_angle(window, clock)
    window.cancel_gesture_calibration()
    assert window.gesture_profile == original and path.read_bytes() == previous
    assert window.camera.profile == original
    assert window.gesture_base_profile is None and not window.gesture_append


def test_new_angle_save_failure_retains_original_and_can_retry(session, monkeypatch):
    window, clock, original, path, previous = session
    start_append(window)
    window.capture_gesture()
    held_angle(window, clock)
    original_save = GestureAngleCalibration.save
    def fail_save(*_args):
        raise OSError("測試磁碟無法儲存")
    monkeypatch.setattr(GestureAngleCalibration, "save", fail_save)
    window.capture_gesture()
    held_angle(window, clock)
    assert window.gesture_calibrating and window.gesture_dialog.step == 1
    assert "磁碟" in window.gesture_dialog.feedback_label.text()
    assert window.gesture_profile == original and window.camera.profile == original
    assert path.read_bytes() == previous
    monkeypatch.setattr(GestureAngleCalibration, "save", original_save)
    window.capture_gesture()
    held_angle(window, clock)
    assert window.gesture_profile.angle_count == 2


def test_regular_recalibration_button_still_starts_replacement_enrollment(session):
    window, _, original, path, previous = session
    window.toggle_sensors()
    window.gesture_button.click()
    assert window.gesture_calibrating and not window.gesture_append
    assert window.gesture_base_profile is None
    assert window.gesture_staged_profile is None and window.gesture_angle_index == 0
    assert "中間" in window.gesture_dialog.step_label.text()
    assert "1 / 5" in window.gesture_dialog.angle_label.text()
    window.cancel_gesture_calibration()
    assert window.gesture_profile == original and path.read_bytes() == previous


def test_standard_flow_records_all_five_angles_before_saving_and_reload(session):
    window, clock, original, path, previous = session
    aiming = window.calibration
    window.toggle_sensors()
    window.gesture_button.click()
    for index, direction in enumerate(STANDARD_GESTURE_ANGLES):
        assert direction in window.gesture_dialog.step_label.text()
        assert f"{index + 1} / 5" in window.gesture_dialog.angle_label.text()
        complete_standard_angle(window, clock, index)
        if index < 4:
            assert window.gesture_calibrating and window.gesture_dialog.step == 0
            assert window.gesture_staged_profile.angle_count == index + 1
            assert window.gesture_candidate is None
            assert window.gesture_profile is original and window.camera.profile is original
            assert path.read_bytes() == previous
    assert window.gesture_calibrating and window.gesture_review_profile.angle_count == 5
    assert window.gesture_profile is original and window.camera.profile is original
    assert path.read_bytes() == previous
    assert "試做" in window.gesture_dialog.step_label.text()
    assert not window.gesture_dialog.record_button.isEnabled()
    pass_fox_review(window, clock)
    assert window.gesture_profile is original and path.read_bytes() == previous
    window.gesture_dialog.record_button.click()
    assert not window.gesture_calibrating and window.gesture_dialog is None
    assert window.gesture_staged_profile is None and window.gesture_candidate is None
    assert window.gesture_profile.angle_count == 5
    assert window.camera.profile is window.gesture_profile and window.calibration is aiming
    assert not window.gesture_angle_button.isEnabled() and window.gesture_angle_button.isHidden()
    loaded = GestureCalibration.load(path)
    assert loaded == window.gesture_profile
    assert all(model.confirmed and model.sample_count >= 24 for model in loaded.models)
    window.stop_sensors()
    reopened = Window(aim_mode="hand")
    try:
        assert reopened.gesture_profile == loaded
        assert "5 個角度" in reopened.gesture_profile_label.text()
    finally:
        reopened.close()


def test_new_user_also_gets_all_five_angles_from_the_main_button(session):
    window, clock, _, path, _ = session
    path.unlink()
    window.gesture_profile = None
    window.update_gesture_profile_label()
    window.toggle_sensors()
    window.gesture_button.click()
    for index in range(5):
        complete_standard_angle(window, clock, index)
        if index < 4:
            assert not path.exists() and window.gesture_profile is None
            assert window.camera.profile is None
    assert not path.exists() and window.gesture_profile is None and window.camera.profile is None
    assert window.gesture_review_profile.angle_count == 5
    pass_fox_review(window, clock)
    assert not path.exists()
    window.gesture_dialog.record_button.click()
    assert GestureCalibration.load(path).angle_count == 5


def test_cancel_later_standard_angle_discards_staging_and_preserves_old_profile(session):
    window, clock, original, path, previous = session
    window.toggle_sensors()
    window.gesture_button.click()
    complete_standard_angle(window, clock, 0)
    complete_standard_angle(window, clock, 1)
    window.capture_gesture()
    held_angle(window, clock, direction=2)
    window.cancel_gesture_calibration()
    assert window.gesture_profile is original and window.camera.profile is original
    assert path.read_bytes() == previous
    assert window.gesture_staged_profile is None and window.gesture_candidate is None
    window.gesture_button.click()
    assert window.gesture_angle_index == 0 and window.gesture_staged_profile is None


def test_standard_angle_transition_requires_two_new_explicit_captures(session):
    window, clock, original, path, previous = session
    window.toggle_sensors()
    window.gesture_button.click()
    complete_standard_angle(window, clock, 0)
    held_angle(window, clock, direction=1)  # A stable hold before clicking cannot advance.
    assert window.gesture_dialog.step == 0 and window.gesture_candidate is None
    window.capture_gesture()
    assert not window.gesture_buffer.evaluate(clock["now"]).ready
    assert window.gesture_candidate is None
    held_angle(window, clock, direction=1)
    assert window.gesture_dialog.step == 1
    held_angle(window, clock, direction=1)
    assert window.gesture_angle_index == 1 and window.gesture_staged_profile.angle_count == 1
    assert window.gesture_profile is original and path.read_bytes() == previous


def test_later_angle_rejects_switching_hands_before_confirmation(session):
    window, clock, original, path, previous = session
    window.toggle_sensors()
    window.gesture_button.click()
    complete_standard_angle(window, clock, 0)
    window.capture_gesture()
    held_angle(window, clock, direction=1, handedness="Left")
    assert window.gesture_angle_index == 1 and window.gesture_candidate is None
    assert window.gesture_staged_profile.angle_count == 1
    assert "相同的手" in window.gesture_dialog.feedback_label.text()
    assert window.gesture_profile is original and path.read_bytes() == previous
    complete_standard_angle(window, clock, 1)
    assert window.gesture_angle_index == 2


def test_final_standard_save_failure_can_retry_without_duplicating_an_angle(session, monkeypatch):
    window, clock, original, path, previous = session
    begin_standard_review(window, clock)
    candidate = window.gesture_review_profile
    assert candidate.angle_count == 5
    pass_fox_review(window, clock)
    save = GestureAngleCalibration.save
    def fail_save(*_args):
        raise OSError("測試磁碟無法儲存")
    monkeypatch.setattr(GestureAngleCalibration, "save", fail_save)
    window.gesture_dialog.record_button.click()
    assert window.gesture_calibrating and window.gesture_dialog._review_mode
    assert window.gesture_review_profile is candidate and window.gesture_review_passed
    assert window.gesture_review_profile.angle_count == 5
    assert window.gesture_dialog.record_button.isEnabled()
    assert window.gesture_profile is original and window.camera.profile is original
    assert path.read_bytes() == previous and "磁碟" in window.gesture_dialog.feedback_label.text()
    monkeypatch.setattr(GestureAngleCalibration, "save", save)
    window.gesture_dialog.record_button.click()  # Retry the same reviewed candidate without another hold.
    assert not window.gesture_calibrating and window.gesture_profile.angle_count == 5
    assert window.gesture_profile is candidate and window.camera.profile is candidate
    assert GestureCalibration.load(path) == window.gesture_profile


@pytest.mark.parametrize("review_passed", [False, True])
def test_cancel_standard_review_preserves_original_file_and_camera_profile(session, review_passed):
    window, clock, original, path, previous = session
    aiming = window.calibration
    begin_standard_review(window, clock)
    if review_passed:
        pass_fox_review(window, clock)
    window.gesture_dialog.reject()
    assert not window.gesture_calibrating and window.gesture_dialog is None
    assert window.gesture_profile is original and window.camera.profile is original
    assert path.read_bytes() == previous and window.calibration is aiming
    assert window.gesture_review_profile is None and window.gesture_staged_profile is None


def test_fox_review_cannot_save_before_fresh_release_and_stable_new_hold(session):
    window, clock, original, path, previous = session
    begin_standard_review(window, clock)
    candidate = window.gesture_review_profile
    window.save_reviewed_gesture()  # The save guard also applies to direct or queued signals.
    held_angle(window, clock, direction=0)
    assert not window.gesture_review_released and not window.gesture_review_passed
    assert not window.gesture_dialog.record_button.isEnabled()
    window.save_reviewed_gesture()
    assert path.read_bytes() == previous and window.gesture_profile is original

    # Missing data alone does not establish a measured release.
    timestamp = clock["now"] + .01
    clock["now"] = timestamp + .01
    window.on_tracking(TrackingFrame(
        timestamp, (), (), 0, False, None, None, None, "等待追蹤", aiming_mode="hand"))
    assert not window.gesture_review_released
    # An old absence observation cannot authorize the release either.
    window.on_tracking(TrackingFrame(
        timestamp - .5, (), (), 0, False, None, None, None, "舊的無手畫面",
        depth_status="no_hand", aiming_mode="hand"))
    assert not window.gesture_review_released

    release_review_hand(window, clock)
    assert window.gesture_review_released
    held_angle(window, clock, direction=0, count=11)
    assert window.gesture_review_count == 11 and not window.gesture_review_passed
    assert not window.gesture_dialog.record_button.isEnabled()
    window.save_reviewed_gesture()
    assert window.gesture_review_profile is candidate
    assert path.read_bytes() == previous and window.camera.profile is original
    held_angle(window, clock, direction=0, count=1)
    assert window.gesture_review_passed
    assert path.read_bytes() == previous  # Passing the trial still requires an explicit save.
    window.gesture_dialog.record_button.click()
    assert window.gesture_profile is candidate and GestureCalibration.load(path) == candidate


@pytest.mark.parametrize("invalid", ["duplicate", "reversed", "stale", "future"])
def test_duplicate_replayed_and_invalid_review_frames_cannot_pass_or_preserve_stable_count(session, invalid):
    window, clock, original, path, previous = session
    begin_standard_review(window, clock)
    release_review_hand(window, clock)
    frames = held_angle(window, clock, direction=0, count=6)
    last_timestamp = frames[-1].timestamp
    assert window.gesture_review_count == 6 and not window.gesture_review_passed
    for _ in range(13):
        clock["now"] += .01
        timestamp = {"duplicate": last_timestamp, "reversed": last_timestamp - .1,
                     "stale": clock["now"] - .5, "future": clock["now"] + .1}[invalid]
        window.on_tracking(replace(frames[-1], timestamp=timestamp))
        assert not window.gesture_review_passed and window.gesture_review_count == 0
        assert window.gesture_review_since is None
    assert not window.gesture_dialog.record_button.isEnabled()
    assert window.gesture_review_timestamp == last_timestamp
    window.save_reviewed_gesture()
    assert window.gesture_profile is original and window.camera.profile is original
    assert path.read_bytes() == previous
    # Recovery starts a completely new measured hold after the rejected samples.
    held_angle(window, clock, direction=0, count=11)
    assert not window.gesture_review_passed and window.gesture_review_count == 11
    held_angle(window, clock, direction=0, count=1)
    assert window.gesture_review_passed and path.read_bytes() == previous
