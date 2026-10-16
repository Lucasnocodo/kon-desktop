import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import CALIBRATION_POINTS, Window
from kon.calibration_store import gesture_profile_path
from kon.gesture_calibration import GestureAngleCalibration
from kon.vision import TrackingFrame
from test_gesture_calibration import open_hand, personal_hand


class RecordingSpeech:
    def __init__(self):
        self.activations = []

    def set_active(self, enabled):
        self.activations.append(enabled)

    def stop(self):
        pass

    def wait(self, timeout):
        return True


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(application, monkeypatch, tmp_path):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window = Window(aim_mode="eye_ray")
    window.running = True
    window.speech = RecordingSpeech()
    yield window, clock
    window.stop_sensors()
    window.close()


def aim_frames(window, clock, raw_hit=(.1, .2), *, message="瞄準線已建立", count=6):
    start = clock["now"] + .01
    for index in range(count):
        timestamp = start + index * .22
        clock["now"] = timestamp + .01
        window.on_tracking(TrackingFrame(timestamp, (), (), 0, False,
                                         None, None, raw_hit, message))


def gesture_frames(window, clock, *, points=None, handedness="Right"):
    points = personal_hand() if points is None else points
    start = clock["now"] + .01
    for index in range(13):
        timestamp = start + index * .1
        clock["now"] = timestamp + .01
        window.on_tracking(TrackingFrame(
            timestamp, (), (), 0, False, None, None, None, "",
            hand_detected=True, hand_world_points=points, handedness=handedness))


def test_aim_steps_show_current_guidance_and_never_arm_recognizer(session):
    window, clock = session
    window.voice_active = window.armed = True
    window.begin_calibration()
    assert "慣用眼" in window.overlay.calibration_feedback
    assert not any(window.speech.activations)
    for index in range(2):
        window.capture_calibration()
        assert "正在取樣" in window.overlay.calibration_feedback
        deadline = window.calibration_capture_deadline
        window.capture_calibration()
        assert window.calibration_capture_deadline == deadline
        aim_frames(window, clock)
        assert window.calibration_index == index + 1
        assert f"第 {index + 1} 點已記錄" in window.overlay.calibration_feedback
        window.on_keyword(clock["now"], "kon")
    assert not window.voice_active and not window.armed and not window.gate.ready
    assert not any(window.speech.activations)


def test_tracking_failure_displays_actual_problem_without_recording_a_point(session):
    window, clock = session
    window.begin_calibration()
    window.capture_calibration()
    reason = "沒有偵測到手部；請讓整隻手完整入鏡"
    aim_frames(window, clock, raw_hit=None, message=reason, count=20)
    assert reason in window.overlay._calibration_state_label.text()
    assert not window.calibration_raw
    assert not window.overlay.calibration_ready
    assert window.calibration_index == 0


def test_recovered_tracking_replaces_obsolete_correction_with_current_state(session):
    window, clock = session
    window.begin_calibration()
    aim_frames(window, clock, raw_hit=None, message="沒有偵測到手部；請讓整隻手完整入鏡", count=1)
    assert "整隻手" in window.overlay._calibration_state_label.text()
    aim_frames(window, clock, raw_hit=None, message="右眼可能閉合或被遮擋；請睜眼並避開手部遮擋", count=1)
    assert "被遮擋" in window.overlay._calibration_state_label.text()
    aim_frames(window, clock)
    assert window.overlay.calibration_ready
    assert "被遮擋" not in window.overlay._calibration_state_label.text()
    assert not window.calibration_raw  # Ready frames still need a record request.


def test_aim_completion_is_shown_only_after_model_is_saved(session, tmp_path):
    window, clock = session
    window.begin_calibration()
    for x, y in CALIBRATION_POINTS:
        window.capture_calibration()
        aim_frames(window, clock, raw_hit=(x * .2 - .1, y * .2 - .1))
    assert not window.calibrating
    assert (tmp_path / ".kon/calibration.json").is_file()
    assert "校正已保存" in window.notice.text()


@pytest.mark.parametrize("failure", ["geometry", "save"])
def test_failed_aim_displays_failure_without_claiming_completion(session, monkeypatch, tmp_path, failure):
    window, clock = session
    window.begin_calibration()
    if failure == "save":
        def cannot_save(*args, **kwargs):
            raise OSError("disk unavailable")
        monkeypatch.setattr("kon.app.save_calibration", cannot_save)
    for x, y in CALIBRATION_POINTS:
        window.capture_calibration()
        aim_frames(window, clock, raw_hit=(.1, .2) if failure == "geometry" else (x * .2 - .1, y * .2 - .1))
    assert not window.calibrating and window.calibration is None
    assert not (tmp_path / ".kon/calibration.json").exists()
    assert "未建立" in window.notice.text() if failure == "geometry" else "無法儲存" in window.notice.text()


def test_standard_gesture_five_angles_have_fresh_holds_and_only_saved_completion(session, tmp_path):
    window, clock = session
    window.voice_active = True
    window.begin_gesture_calibration()
    assert "中間" in window.gesture_dialog.angle_label.text()
    for angle_index in range(5):
        window.capture_gesture()
        assert "新影格" in window.gesture_dialog.feedback_label.text()
        gesture_frames(window, clock)
        assert window.gesture_candidate is not None
        assert "第一次已記錄" in window.gesture_dialog.feedback_label.text()
        assert window.gesture_profile is None
        window.capture_gesture()
        gesture_frames(window, clock)
        if angle_index < 4:
            assert window.gesture_calibrating and window.gesture_profile is None
            assert window.gesture_angle_index == angle_index + 1
            assert not gesture_profile_path(tmp_path).exists()
    assert window.gesture_profile.confirmed and window.gesture_profile.angle_count == 5
    assert gesture_profile_path(tmp_path).is_file()
    assert "手圈已保存並套用" in window.notice.text()
    assert not window.voice_active and not any(window.speech.activations)


def test_gesture_watchdog_waits_for_a_real_observation_after_record_request(session):
    window, clock = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    clock["now"] += .1
    window.check_freshness()
    assert not window.gesture_buffer.has_observation
    assert window.gesture_capture_deadline is not None
    assert "新影格" in window.gesture_dialog.feedback_label.text()
    gesture_frames(window, clock)
    assert window.gesture_candidate is not None


@pytest.mark.parametrize("problem", ["hand", "shape"])
def test_gesture_confirmation_displays_actual_problem_and_keeps_first_hold(session, problem):
    window, clock = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    gesture_frames(window, clock)
    candidate = window.gesture_candidate
    window.capture_gesture()
    gesture_frames(window, clock, handedness="Left" if problem == "hand" else "Right",
                   points=open_hand() if problem == "shape" else personal_hand())
    assert window.gesture_calibrating and window.gesture_profile is None
    assert window.gesture_candidate is candidate
    feedback = window.gesture_dialog.feedback_label.text()
    assert "相同的手" in feedback if problem == "hand" else "手勢與第一次不同" in feedback
    assert window.gesture_dialog.step == 1


def test_later_gesture_angle_rejects_different_hand_without_losing_saved_staging(session):
    window, clock = session
    window.begin_gesture_calibration()
    for _ in range(2):
        window.capture_gesture()
        gesture_frames(window, clock)
    first_angle = window.gesture_staged_profile
    window.capture_gesture()
    gesture_frames(window, clock, handedness="Left")
    assert window.gesture_angle_index == 1 and window.gesture_candidate is None
    assert window.gesture_staged_profile is first_angle and window.gesture_profile is None
    assert "相同的手" in window.gesture_dialog.feedback_label.text()
    window.capture_gesture()
    gesture_frames(window, clock)
    assert window.gesture_candidate is not None and window.gesture_dialog.step == 1


def test_gesture_save_failure_shows_error_and_allows_retry_without_false_completion(session, monkeypatch):
    window, clock = session
    window.begin_gesture_calibration()
    for _ in range(4):
        window.capture_gesture()
        gesture_frames(window, clock)
        window.capture_gesture()
        gesture_frames(window, clock)
    window.capture_gesture()
    gesture_frames(window, clock)
    original_save = GestureAngleCalibration.save
    def cannot_save(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(GestureAngleCalibration, "save", cannot_save)
    window.capture_gesture()
    gesture_frames(window, clock)
    assert window.gesture_calibrating and window.gesture_profile is None
    assert window.gesture_angle_index == 4 and window.gesture_staged_profile.angle_count == 4
    assert "disk unavailable" in window.gesture_dialog.feedback_label.text()
    monkeypatch.setattr(GestureAngleCalibration, "save", original_save)
    window.capture_gesture()
    gesture_frames(window, clock)
    assert window.gesture_profile.angle_count == 5
    assert "手圈已保存並套用" in window.notice.text()


@pytest.mark.parametrize("leave", ["cancel", "control", "pause", "mode", "screen", "close"])
@pytest.mark.parametrize("mode", ["aim", "gesture"])
def test_leaving_calibration_clears_pending_capture_and_disarms_commands(session, leave, mode):
    window, _ = session
    if mode == "aim":
        window.begin_calibration()
        window.capture_calibration()
    else:
        window.begin_gesture_calibration()
        window.capture_gesture()
    if leave == "cancel":
        (window.cancel_calibration if mode == "aim" else window.cancel_gesture_calibration)()
    elif leave == "control":
        window.return_to_control()
    elif leave == "pause":
        window.stop_sensors()
    elif leave == "mode":
        window.mode.setCurrentIndex(1)
    elif leave == "screen":
        window.screen_changed()
    else:
        window.close()
    assert not window.calibrating and not window.gesture_calibrating
    assert window.calibration_capture_deadline is None and window.gesture_capture_deadline is None
    assert not window.voice_active and not window.armed


def test_switching_calibration_modes_clears_the_previous_capture(session):
    window, _ = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    window.begin_calibration()
    assert not window.gesture_calibrating and window.gesture_capture_deadline is None
    assert window.calibrating and "慣用眼" in window.overlay.calibration_feedback
    window.begin_gesture_calibration()
    assert not window.calibrating and window.calibration_capture_deadline is None
    assert window.gesture_calibrating and "中間" in window.gesture_dialog.angle_label.text()
