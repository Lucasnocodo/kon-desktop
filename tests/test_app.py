import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
import pytest

from kon.app import CALIBRATION_POINTS, Window
from kon.desktop import DesktopTarget
from kon.geometry import ScreenCalibration
from kon.calibration_diagnostics import analyze_calibration
from kon.personal_calibration import PersonalCalibration
from kon.vision import TrackingFrame


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    value = Window(aim_mode="eye_ray")
    yield value
    value.stop_sensors()
    value.close()


def test_opening_control_room_does_not_start_sensors(window):
    assert window.camera is None and window.speech is None
    assert not window.running and not window.armed
    assert window.mode.currentIndex() == 0


def test_preview_is_always_virtual_even_in_desktop_mode(window):
    window.mode.setCurrentIndex(1)
    window.preview_animation()
    assert window.overlay.practice
    assert window.overlay.eating.snapshot is None
    assert window.pending_action is None
    assert not window.armed


def test_cancel_before_swallow_invalidates_real_action(window):
    window.pending_action = (window.action_serial, "file", True)
    window.overlay.play(DesktopTarget("file", "file", (100, 100)))
    previous = window.action_serial
    window.return_to_control()
    assert window.action_serial != previous
    assert window.pending_action is None
    assert window.overlay.eating is None


def test_keyword_before_voice_arm_or_after_source_ttl_is_ignored(window, monkeypatch):
    fired = []
    window.summon = fired.append
    window.voice_active = True
    window.voice_armed_at = 95.0
    monkeypatch.setattr("kon.app.time.monotonic", lambda: 100.0)
    window.on_keyword(94.9, "高")
    window.on_keyword(96.0, "高")
    assert fired == []


def test_old_scan_callback_cannot_activate_targets(window):
    # Cancelling the control-room session changes the identity used by asynchronous scans.
    previous = window.action_serial
    window.return_to_control()
    assert previous != window.action_serial and not window.testing


def test_stale_camera_watchdog_drops_aim_without_claiming_gesture_release(window, monkeypatch):
    calls = []
    class Gate:
        def invalidate_tracking(self, now):
            calls.append(now)
    window.running = True
    window.gate = Gate()
    window.latest = TrackingFrame(95.0, (), (), 0, False, None, None, None, "")
    monkeypatch.setattr("kon.app.time.monotonic", lambda: 100.0)
    window.check_freshness()
    assert calls == [100.0]
    assert window.overlay.point is None and not window.voice_active
    assert "追蹤暫時中斷" in window.gesture_hint_label.text()


def test_calibrated_aim_cannot_arm_voice_when_tracked_hand_shape_fails(window, monkeypatch):
    # A successful position calibration and a high score alone cannot turn
    # an unrecognized hand shape into an actionable desktop target.
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    window.calibration = ScreenCalibration.fit(samples, samples)
    window.running = window.testing = window.armed = True
    window.targets = [DesktopTarget("a", "a", (window.overlay.width() / 2, window.overlay.height() / 2))]
    fired = []
    window.summon = fired.append
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    for index in range(12):
        clock["now"] = 100.0 + index * .1
        window.on_tracking(TrackingFrame(
            clock["now"], (), (), .95, False, None, None, (.5, .5), "手形未通過",
            hand_detected=True, gesture_reason="pinky extension",
            gesture_hint="小指尚未伸直，請讓小指清楚入鏡。",
        ))
    window.on_keyword(clock["now"], "kon")
    assert window.overlay.point is None
    assert not window.voice_active and not window.gate.ready
    assert fired == []
    assert "小指尚未伸直" in window.gesture_hint_label.text()


def test_hand_shape_diagnostic_does_not_claim_calibration_failure(window, monkeypatch):
    window.running = True
    monkeypatch.setattr("kon.app.time.monotonic", lambda: 100.0)
    window.on_tracking(TrackingFrame(
        100.0, (), (), .4, False, None, None, (.5, .5), "瞄準射線已建立",
        hand_detected=True, gesture_hint="無名指尚未彎起，請把它壓向拇指。",
    ))
    assert "無名指" in window.gesture_hint_label.text()
    assert "重新校正" not in window.gesture_hint_label.text()


def test_real_mode_requires_calibration_and_running_speech(window):
    window.mode.setCurrentIndex(1)
    window.begin_targets()
    assert not window.armed and not window.testing


def test_actual_trash_job_runs_only_after_matching_swallow_event(window):
    jobs = []
    window.job = lambda function, callback: jobs.append(function)
    sentinel = object()
    target = DesktopTarget("a", "a", (100, 100), sentinel)
    window.pending_action = (window.action_serial, "other", True)
    window.on_bite(target)
    assert not jobs
    window.pending_action = (window.action_serial, "a", True)
    window.on_bite(target)
    assert len(jobs) == 1
    window.on_bite(target)
    assert len(jobs) == 1


def test_virtual_target_disappears_after_swallow(window):
    target = DesktopTarget("demo", "demo", (100, 100))
    window.targets = window.overlay.targets = [target]
    window.on_bite(target)
    assert not window.targets and not window.overlay.targets


def test_cancelled_queued_trash_never_calls_native_file_operation(window, monkeypatch):
    calls = []
    monkeypatch.setattr("kon.app.trash_target", calls.append)
    serial = window.action_serial
    window.return_to_control()
    assert window.trash_if_current(serial, object()) is None
    assert calls == []


def test_return_to_control_invalidates_without_synthetic_hand_release(window):
    calls = []
    class Gate:
        def invalidate_tracking(self, now):
            calls.append(now)
    window.gate = Gate()
    window.return_to_control()
    assert len(calls) == 1


def test_saved_nondefault_settings_restore_before_calibration(window, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    (tmp_path / ".kon").mkdir()
    config = window.config() | {"eye": "left", "camera": 2, "fov": 78.0}
    (tmp_path / ".kon/calibration-meta.json").write_text(json.dumps(config))
    window._restore_settings()
    assert window.config() == config


def test_default_voice_trigger_uses_loud_short_sound_with_optional_kon(window):
    assert window.trigger_setting == "kon"
    assert window.trigger_input.text() == "kon"
    assert window.voice_trigger_mode == window.voice_trigger_mode_combo.currentData() == "loud_sound"
    assert "大聲唸一個短音" in window.voice_instructions.text()
    assert "不需發音校正" in window.pronunciation_label.text()
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("phrase"))
    assert "kon" in window.voice_instructions.text()


def test_legacy_voice_preferences_default_to_loud_sound_and_keep_calibration(window, tmp_path):
    import json
    from kon.pronunciation import PronunciationProfile
    profile = PronunciationProfile.create("invoke", "en", ["invoke"] * 3, ["invoke"], "invoke")
    profile.save(tmp_path)
    preferences = json.loads((tmp_path / ".kon/preferences.json").read_text())
    preferences["camera_settings"] = {"eye": "left", "camera": 2, "fov": 79.5}
    (tmp_path / ".kon/preferences.json").write_text(json.dumps(preferences))
    window._load_preferences()
    assert window.voice_trigger_mode == "loud_sound"
    assert window.pronunciation_profile == profile and window.trigger_setting == "invoke"
    assert "不必辨識成指定文字" in window.pronunciation_label.text()


def test_voice_mode_selection_persists_and_preserves_other_preferences(window, tmp_path):
    import json
    from kon.pronunciation import PronunciationProfile
    profile = PronunciationProfile.create("invoke", "en", ["invoke"] * 3, ["invoke"], "invoke")
    profile.save(tmp_path)
    destination = tmp_path / ".kon/preferences.json"
    before = json.loads(destination.read_text())
    before.update(camera_settings={"eye": "left", "camera": 2, "fov": 79.5}, other={"keep": True})
    destination.write_text(json.dumps(before))
    window._load_preferences()
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("phrase"))
    assert json.loads(destination.read_text()) == before | {"voice_trigger_mode": "phrase"}
    reloaded = Window(aim_mode="eye_ray")
    try:
        assert reloaded.voice_trigger_mode == reloaded.voice_trigger_mode_combo.currentData() == "phrase"
        assert reloaded.pronunciation_profile == profile
        assert "invoke" in reloaded.voice_instructions.text()
        assert reloaded.trigger_changed()
        assert json.loads(destination.read_text())["voice_trigger_mode"] == "phrase"
    finally:
        reloaded.close()


def test_live_voice_mode_change_disarms_and_invalidates_pending_file_actions(window, tmp_path, monkeypatch):
    class LiveSpeech:
        def __init__(self):
            self.active = True
            self.configuration = None
        def isRunning(self):
            return True
        def set_active(self, value):
            self.active = value
        def configure_trigger(self, setting, **configuration):
            assert not self.active
            self.configuration = (setting, configuration)
        def stop(self):
            self.active = False
        def wait(self, _timeout):
            return True
    worker = window.speech = LiveSpeech()
    window.running = window.armed = window.testing = window.voice_active = True
    window.pending_action = (window.action_serial, "real-file", True)
    previous = window.action_serial
    calls = []
    monkeypatch.setattr(window, "summon", calls.append)
    monkeypatch.setattr("kon.app.time.monotonic", lambda: 100.0)
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("phrase"))
    assert window.running and not window.armed and not window.testing and not window.voice_active
    assert window.pending_action is None and window.action_serial > previous
    assert window.gate.locked_target is None
    assert worker.configuration == ("kon", {"language": None, "accepted_phrases": None, "trigger_mode": "phrase"})
    window.on_keyword(99.9, "大聲短音")
    assert calls == []
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("loud_sound"))
    assert worker.configuration[1]["trigger_mode"] == "loud_sound"
    assert "大聲唸一個短音" in window.summon_caption.text()


def test_loud_mode_target_hints_explain_short_sound_and_phrase_mode_uses_setting(window, monkeypatch):
    window.activate_targets([DesktopTarget("a", "練習檔案", (100, 100))])
    assert "大聲唸一個短音" in window.overlay.hint
    assert "kon" not in window.overlay.hint
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("phrase"))
    window.activate_targets([DesktopTarget("a", "練習檔案", (100, 100))])
    assert "請說 kon" in window.overlay.hint


def test_custom_voice_setting_persists_without_invalidating_aim(window, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    calibration = object()
    window.calibration = calibration
    window.trigger_input.setText("kon, con")
    assert window.trigger_changed()
    assert window.trigger_display == "kon"
    assert window.calibration is calibration
    saved = json.loads((tmp_path / ".kon/preferences.json").read_text())
    assert saved["voice_trigger"] == "kon, con"
    window.trigger_input.setText("go")
    window._load_preferences()
    assert window.trigger_input.text() == "kon, con"


def test_empty_voice_setting_keeps_previous_phrase(window, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    window.trigger_input.setText("  ")
    assert not window.trigger_changed()
    assert window.trigger_setting == "kon"
    assert window.trigger_input.text() == "kon"


def test_voice_setting_is_locked_while_sensors_run(window):
    window.running = True
    window.trigger_input.setText("吃掉")
    assert not window.trigger_changed()
    assert window.trigger_setting == "kon"


@pytest.fixture
def calibration_session(window, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.running = True
    window.begin_calibration()
    return window, clock


def tracking(timestamp, raw_hit=(.1, .2), pose_ok=False, message="眼睛與手孔可定位"):
    return TrackingFrame(timestamp, (), (), 0, pose_ok, None, None, raw_hit, message)


def steady_calibration_frames(window, clock, start=100.01, age=.01, raw_hit=(.1, .2)):
    start = max(start, clock["now"] + .01)
    for index in range(6):
        timestamp = start + index * .22
        clock["now"] = timestamp + age
        window.on_tracking(tracking(timestamp, raw_hit))


def test_calibration_can_record_usable_ray_without_command_pose(calibration_session):
    window, clock = calibration_session
    steady_calibration_frames(window, clock)
    assert window.overlay.calibration_ready
    assert window.isMinimized()
    window.capture_calibration()
    assert not window.calibration_raw
    steady_calibration_frames(window, clock, start=clock["now"] + .01)
    assert window.calibration_raw == [pytest.approx((.1, .2))]
    assert window.overlay.calibration_step == 1
    assert "第 1 點已記錄" in window.overlay.calibration_feedback
    assert window.calibration_capture_deadline is None
    assert not window.armed and not window.voice_active


def test_record_request_waits_for_stable_frames_and_advances_once(calibration_session):
    window, clock = calibration_session
    window.capture_calibration()
    assert window.calibration_capture_deadline == 106.0
    assert "正在取樣" in window.overlay.calibration_feedback
    steady_calibration_frames(window, clock)
    assert len(window.calibration_raw) == 1
    # Queued data captured before the second marker cannot record that marker.
    window.on_tracking(tracking(100.2))
    window.capture_calibration()
    assert len(window.calibration_raw) == 1
    assert not window.overlay.calibration_ready


def test_calibration_accepts_fresh_point_three_two_tenths_old(calibration_session):
    window, clock = calibration_session
    window.capture_calibration()
    steady_calibration_frames(window, clock, age=.32)
    window.capture_calibration()
    assert len(window.calibration_raw) == 1


def test_record_timeout_reason_survives_following_camera_frame(calibration_session):
    window, clock = calibration_session
    window.on_tracking(tracking(100.0, None, message="右眼虹膜太小，請靠近鏡頭"))
    window.capture_calibration()
    clock["now"] = 106.01
    window.on_tracking(tracking(106.0, None, message="右眼虹膜太小，請靠近鏡頭"))
    feedback = window.overlay.calibration_feedback
    assert "尚未記錄" in feedback and "虹膜太小" in feedback
    clock["now"] = 106.1
    window.on_tracking(tracking(106.09, None, message="右眼虹膜太小，請靠近鏡頭"))
    assert window.overlay.calibration_feedback == feedback
    assert window.calibration_capture_deadline is None
    assert not window.calibration_raw


def test_expired_request_does_not_capture_after_gui_delay(calibration_session):
    window, clock = calibration_session
    window.capture_calibration()
    clock["now"] = 105.7
    steady_calibration_frames(window, clock, start=105.71)
    assert not window.calibration_raw
    assert window.calibration_capture_deadline is None
    assert "尚未記錄" in window.overlay.calibration_feedback


def test_camera_preview_is_forwarded_to_current_calibration(calibration_session):
    window, clock = calibration_session
    image = QImage(96, 54, QImage.Format_RGBA8888)
    image.fill(QColor("#3f88a6"))
    forwarded = []
    window.overlay.set_calibration_preview = forwarded.append
    window.on_frame(image)
    assert forwarded == [image]
    assert window.preview.image == image
    window.cancel_calibration()
    forwarded.clear()
    window.on_frame(image)
    assert forwarded == []


def test_cancel_pending_calibration_ignores_late_camera_samples(calibration_session):
    window, clock = calibration_session
    window.capture_calibration()
    window.cancel_calibration()
    steady_calibration_frames(window, clock)
    assert not window.calibrating and window.calibration_capture_deadline is None
    assert not window.calibration_raw
    assert window.overlay.calibration_point is None


def test_calibration_does_not_arm_desktop_deletion(window, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.mode.setCurrentIndex(1)
    window.running = window.armed = window.testing = True
    window.pending_action = (window.action_serial, "file", True)
    operations = []
    window.job = lambda *args: operations.append(args)
    window.begin_calibration()
    window.capture_calibration()
    steady_calibration_frames(window, clock)
    window.on_keyword(clock["now"], "go")
    assert not window.armed and not window.testing and not window.voice_active
    assert window.pending_action is None and operations == []


def test_bad_nine_point_fit_disables_previous_test_button(calibration_session):
    window, clock = calibration_session
    window.calibration_raw = [(.1, .2)] * 8
    window.calibration = object()
    window.test_button.setEnabled(True)
    window.capture_calibration()
    steady_calibration_frames(window, clock)
    assert window.calibration is None and not window.test_button.isEnabled()
    assert not window.calibrating and "個人校正未建立" in window.notice.text()


def test_nine_distinct_points_finish_and_restore_control_room(calibration_session, tmp_path):
    window, clock = calibration_session
    for index, (x, y) in enumerate(CALIBRATION_POINTS):
        window.capture_calibration()
        steady_calibration_frames(window, clock, start=100.01 + index * .6,
                                  raw_hit=(x * .2 - .1, y * .2 - .1))
    assert not window.calibrating and not window.isMinimized()
    assert window.calibration.sample_count == 9
    assert window.calibration.predict((0, 0)) == pytest.approx((.5, .5))
    assert window.test_button.isEnabled() and "已保存" in window.cal_label.text()
    assert window.calibration.rms_error is None and not window.calibration.ready_for_live
    assert (tmp_path / ".kon/calibration.json").is_file()
    assert (tmp_path / ".kon/calibration-meta.json").is_file()


def test_capture_never_uses_steady_hold_recorded_before_key(calibration_session):
    window, clock = calibration_session
    steady_calibration_frames(window, clock, raw_hit=(.5, .5))
    window.capture_calibration()
    assert not window.calibration_raw
    window.on_tracking(tracking(clock["now"] - .1, (.5, .5)))
    assert not window.calibration_raw
    steady_calibration_frames(window, clock, start=clock["now"] + .01, raw_hit=(.1, .2))
    assert window.calibration_raw == [pytest.approx((.1, .2))]


def test_duplicate_record_request_does_not_restart_capture(calibration_session):
    window, clock = calibration_session
    window.capture_calibration()
    deadline = window.calibration_capture_deadline
    clock["now"] = 100.1
    window.on_tracking(tracking(100.09))
    window.capture_calibration()
    assert window.calibration_capture_deadline == deadline
    assert window.calibration_buffer.evaluate(clock["now"]).count == 1


def failed_samples():
    points = [(x * .2 - .1, y * .2 - .1) for x, y in CALIBRATION_POINTS]
    points[4] = (.06, .02)
    return points


def test_retry_replaces_only_selected_point_and_revalidates_all_nine(calibration_session, tmp_path):
    window, clock = calibration_session
    window.calibration_raw = failed_samples()
    previous = window.calibration_raw.copy()
    window.calibration_diagnostics = analyze_calibration(previous, CALIBRATION_POINTS)
    assert not window.calibration_diagnostics.passed
    window.retry_calibration_point(4)
    assert window.overlay.calibration_step == 4 and len(window.calibration_raw) == 9
    window.capture_calibration()
    steady_calibration_frames(window, clock, raw_hit=(0, 0))
    assert window.calibration_diagnostics.passed and window.calibration is not None
    assert window.calibration_raw[4] == pytest.approx((0, 0))
    assert all(window.calibration_raw[index] == previous[index] for index in range(9) if index != 4)
    assert (tmp_path / ".kon/calibration-last-attempt.json").is_file()


def test_failed_attempt_is_saved_and_can_be_recovered_after_restart(calibration_session, tmp_path):
    import json
    window, clock = calibration_session
    window.calibration_raw = failed_samples()[:8]
    window.capture_calibration()
    steady_calibration_frames(window, clock, raw_hit=failed_samples()[8])
    assert not window.calibration_diagnostics.passed and not window.calibrating
    saved = json.loads((tmp_path / ".kon/calibration-last-attempt.json").read_text())
    assert len(saved["raw_points"]) == len(saved["fit"]["point_errors"]) == 9
    assert saved["point_details"][8]["source_sample_count"] == 6
    assert "影像" not in saved and window.calibration is None
    assert len(list((tmp_path / ".kon/calibration-attempts").glob("*.json"))) == 1
    restarted = Window(aim_mode="eye_ray")
    try:
        assert len(restarted.calibration_raw) == 9
        assert restarted.calibration is None and not restarted.calibration_diagnostics.passed
        assert not restarted.cal_report_button.isHidden()
    finally:
        restarted.stop_sensors()
        restarted.close()


def test_changed_camera_config_invalidates_stored_retry_samples(window, tmp_path):
    from kon.calibration_history import save_attempt
    samples = failed_samples()
    save_attempt(tmp_path, samples, CALIBRATION_POINTS,
                 analyze_calibration(samples, CALIBRATION_POINTS), window.config())
    window.fov.setValue(82.15)
    window._load_calibration_attempt()
    assert not window.calibration_raw and window.calibration_diagnostics is None


def test_camera_preferences_restore_before_loading_calibration(window, tmp_path):
    import json
    (tmp_path / ".kon").mkdir(exist_ok=True)
    (tmp_path / ".kon/preferences.json").write_text(json.dumps({
        "voice_trigger": "go", "camera_settings": {"eye": "right", "camera": 0, "fov": 82.15}}))
    window._restore_settings()
    assert window.fov.value() == 82.15
    assert window.trigger_changed()
    saved = json.loads((tmp_path / ".kon/preferences.json").read_text())
    assert saved["camera_settings"]["fov"] == 82.15


def test_latest_failed_attempt_cannot_erase_saved_accepted_calibration(window, tmp_path):
    import json
    from kon.calibration_history import save_attempt
    raw = [(x * .2 - .1, y * .2 - .1) for x, y in CALIBRATION_POINTS]
    ScreenCalibration.fit(raw, CALIBRATION_POINTS).save(tmp_path / ".kon/calibration.json")
    (tmp_path / ".kon/calibration-meta.json").write_text(json.dumps(window.config()))
    failed = failed_samples()
    save_attempt(tmp_path, failed, CALIBRATION_POINTS,
                 analyze_calibration(failed, CALIBRATION_POINTS), window.config())
    restored = Window(aim_mode="eye_ray")
    try:
        assert restored.calibration is not None
        assert restored.calibration.predict(raw[4]) == pytest.approx((.5, .5))
        assert restored.calibration_diagnostics is None
        assert not restored.calibrating
        assert "已保存" in restored.cal_label.text()
    finally:
        restored.stop_sensors()
        restored.close()


class ReadySpeech:
    def isRunning(self):
        return True
    def set_active(self, _enabled):
        pass
    def stop(self):
        pass
    def wait(self, _timeout):
        return True


def install_linear_personal(window, targets=CALIBRATION_POINTS):
    window.calibration_raw = [(x * .2 - .1, y * .2 - .1) for x, y in targets]
    window.calibration_screen_points = list(targets)
    window.calibration = PersonalCalibration.fit(window.calibration_raw, targets,
                                                screen_size=window.config()["screen"])
    window.calibration_diagnostics = analyze_calibration(window.calibration_raw, targets)
    window.running = True
    window.speech = ReadySpeech()
    window.update_personal_controls()
    return window.calibration


def feed_validation(window, clock, *, final_offset=(0, 0)):
    targets = window.calibration.validation_targets
    for index, (x, y) in enumerate(targets):
        window.capture_calibration()
        start = clock["now"] + .01
        if index == len(targets) - 1:
            x += final_offset[0]
            y += final_offset[1]
        for sample in range(17):
            timestamp = start + sample * .07
            clock["now"] = timestamp + .01
            window.on_tracking(tracking(timestamp, (x * .2 - .1, y * .2 - .1)))


def test_rejected_global_fit_can_restore_exact_personal_points_for_practice(window, tmp_path):
    from kon.calibration_history import save_attempt
    raw = [(.314, .144), (.019, .185), (-.278, .153),
           (.276, .209), (.052, .223), (-.228, .238),
           (.271, .303), (.023, .244), (-.255, .364)]
    diagnostic = analyze_calibration(raw, CALIBRATION_POINTS)
    assert not diagnostic.passed
    save_attempt(tmp_path, raw, CALIBRATION_POINTS, diagnostic, window.config())
    window._load_calibration_attempt()
    model = window.calibration
    assert isinstance(model, PersonalCalibration) and not model.ready_for_live
    assert model.predict(raw[7]) == pytest.approx(CALIBRATION_POINTS[7])
    window.running = True
    window.speech = ReadySpeech()
    window.begin_targets()
    assert window.testing and window.overlay.practice
    assert all(target.snapshot is None for target in window.targets)


def test_saved_personal_mapping_can_read_desktop_without_forced_validation(window):
    install_linear_personal(window)
    window.mode.setCurrentIndex(1)
    jobs = []
    window.job = lambda *args: jobs.append(args)
    window.begin_targets()
    assert not window.calibration_validating and not window.calibrating
    assert not window.testing and not window.armed and not window.voice_active
    assert len(jobs) == 1
    inside = DesktopTarget("file", "file", (window.overlay.width() * .5, window.overlay.height() * .5), object())
    window.activate_targets([inside])
    assert window.testing and window.armed


def test_validation_freezes_nine_controls_and_checks_fresh_frame_quality(calibration_session, tmp_path):
    window, clock = calibration_session
    model = install_linear_personal(window)
    controls = window.calibration_raw.copy()
    window.begin_validation()
    feed_validation(window, clock)
    assert not window.calibrating and model.ready_for_live
    assert model.validation.frame_samples_checked
    assert min(model.validation.point_frame_counts) >= 6
    assert max(model.validation.point_p95_pixels) < 1e-8
    assert window.calibration_raw == controls
    restored = PersonalCalibration.load(tmp_path / ".kon/calibration.json")
    assert restored.ready_for_live


def test_failed_fresh_verification_keeps_personal_mapping_for_virtual_practice(calibration_session):
    window, clock = calibration_session
    model = install_linear_personal(window)
    controls = window.calibration_raw.copy()
    window.begin_validation()
    feed_validation(window, clock, final_offset=(.12, 0))
    assert window.calibration is model and not model.ready_for_live
    assert not window.calibrating and not window.armed
    assert window.calibration_raw == controls
    assert "九點個人對應仍已採用" in window.notice.text()
    window.begin_targets()
    assert window.testing and window.overlay.practice


def test_cancelling_reverification_does_not_restore_old_file_permission(calibration_session, tmp_path):
    window, clock = calibration_session
    model = install_linear_personal(window)
    goals = model.validation_targets
    raws = [(x * .2 - .1, y * .2 - .1) for x, y in goals]
    model.validate(raws, goals, screen_size=window.config()["screen"],
                   frame_samples=[[point] * 6 for point in raws])
    assert model.ready_for_live
    window.begin_validation()
    window.cancel_calibration()
    assert not model.ready_for_live
    assert not PersonalCalibration.load(tmp_path / ".kon/calibration.json").ready_for_live


def test_desktop_activation_rejects_summoning_outside_verified_region(window):
    goals = [(x, y) for y in (.01, .5, .99) for x in (.01, .5, .99)]
    model = install_linear_personal(window, goals)
    checks = model.validation_targets
    raws = [(x * .2 - .1, y * .2 - .1) for x, y in checks]
    model.validate(raws, checks, screen_size=window.config()["screen"],
                   frame_samples=[[point] * 6 for point in raws])
    assert model.ready_for_live and not model.full_screen_coverage
    window.mode.setCurrentIndex(1)
    inside = DesktopTarget("inside", "inside", (window.overlay.width() * .5, window.overlay.height() * .5), object())
    outside = DesktopTarget("outside", "outside", (0, window.overlay.height() * .5), object())
    window.activate_targets([inside, outside])
    assert window.targets == [inside, outside]
    window.summon(outside.id)
    assert not window.testing
    assert "超出已保存的瞄準範圍" in window.notice.text()


def test_empty_desktop_scan_does_not_claim_calibration_failure(window):
    install_linear_personal(window)
    window.mode.setCurrentIndex(1)
    window.activate_targets([])
    assert not window.testing and not window.armed
    assert "沒有讀到" in window.notice.text()
    assert "一般檔案" in window.notice.text()
    assert "校正" not in window.notice.text()


def test_desktop_files_outside_calibration_get_specific_guidance(window):
    install_linear_personal(window, [(x, y) for y in (.01, .5, .99) for x in (.01, .5, .99)])
    window.mode.setCurrentIndex(1)
    outside = DesktopTarget("outside", "outside", (0, window.overlay.height() * .5), object())
    window.activate_targets([outside])
    assert window.testing and window.armed
    assert "已讀到 1 個" in window.notice.text()
    assert "已校正範圍外" in window.notice.text()
    assert "狐狸召喚限已校正範圍" in window.notice.text()
    assert "Finder 權限" not in window.notice.text()


def personal_hand_points():
    from test_gesture_calibration import personal_hand
    return tuple(tuple(float(value) for value in point) for point in personal_hand())


def gesture_frames(window, clock, *, points=None, count=13):
    points = personal_hand_points() if points is None else points
    start = clock["now"] + .01
    for index in range(count):
        timestamp = start + index * .1
        clock["now"] = timestamp + .01
        window.on_tracking(TrackingFrame(
            timestamp, (), (), 0, False, None, None, None, "內建手势未通過",
            hand_detected=True, hand_world_points=points, handedness="Right",
        ))


def test_personal_gesture_five_angles_save_and_restore_without_aim_reset(window, tmp_path, monkeypatch):
    from kon.calibration_store import gesture_profile_path, save_calibration
    from kon.gesture_calibration import GestureCalibration
    aim = install_linear_personal(window)
    save_calibration(tmp_path, aim, window.config())
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.begin_gesture_calibration()
    # Frames before either explicit click cannot fulfill that hold.
    gesture_frames(window, clock)
    assert window.gesture_candidate is None
    window.capture_gesture()
    assert window.gesture_candidate is None
    gesture_frames(window, clock)
    assert window.gesture_candidate is not None and window.gesture_dialog.step == 1
    assert not gesture_profile_path(tmp_path).exists()
    window.capture_gesture()
    assert window.gesture_calibrating and window.gesture_profile is None
    gesture_frames(window, clock)
    assert window.gesture_calibrating and window.gesture_profile is None
    for angle_index in range(1, 5):
        assert window.gesture_angle_index == angle_index
        assert window.gesture_staged_profile.angle_count == angle_index
        # New-angle preview frames cannot fulfill either explicit capture.
        gesture_frames(window, clock)
        assert window.gesture_candidate is None and window.gesture_dialog.step == 0
        assert not gesture_profile_path(tmp_path).exists()
        window.capture_gesture()
        gesture_frames(window, clock)
        assert window.gesture_candidate is not None and window.gesture_dialog.step == 1
        assert not gesture_profile_path(tmp_path).exists()
        window.capture_gesture()
        gesture_frames(window, clock)
    assert not window.gesture_calibrating and window.gesture_dialog is None
    assert window.calibration is aim
    assert not window.armed and not window.voice_active
    profile = GestureCalibration.load(gesture_profile_path(tmp_path))
    assert profile.confirmed and profile.angle_count == 5
    assert profile.match(personal_hand_points(), source="world", handedness="Right").is_gao
    reopened = Window(aim_mode="eye_ray")
    try:
        assert reopened.gesture_profile.confirmed
        assert reopened.gesture_profile.angle_count == 5
        assert reopened.calibration.raw_points == aim.raw_points
        assert not reopened.calibrating and not reopened.gesture_calibrating
        assert reopened.camera is None and reopened.speech is None
        reopened.running, reopened.speech = True, ReadySpeech()
        reopened.mode.setCurrentIndex(1)
        jobs = []
        reopened.job = lambda *args: jobs.append(args)
        reopened.begin_targets()
        assert len(jobs) == 1 and not reopened.calibrating
    finally:
        reopened.stop_sensors()
        reopened.close()


def test_cancel_gesture_replacement_preserves_saved_profile_and_aim(window, tmp_path, monkeypatch):
    from kon.calibration_store import gesture_profile_path
    from kon.gesture_calibration import GestureCalibration
    points = personal_hand_points()
    old = GestureCalibration.fit([points] * 12, handedness="Right").confirm([points] * 12, handedness="Right")
    old.save(gesture_profile_path(tmp_path))
    previous_bytes = gesture_profile_path(tmp_path).read_bytes()
    window.gesture_profile = old
    aim = install_linear_personal(window)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.begin_gesture_calibration()
    window.capture_gesture()
    gesture_frames(window, clock)
    window.cancel_gesture_calibration()
    gesture_frames(window, clock)
    assert window.gesture_profile is old and window.calibration is aim
    assert gesture_profile_path(tmp_path).read_bytes() == previous_bytes
    assert not window.gesture_calibrating and not window.armed


def test_gesture_capture_timeout_and_duplicate_request_do_not_replace_profile(window, monkeypatch):
    window.running = True
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.begin_gesture_calibration()
    window.capture_gesture()
    deadline = window.gesture_capture_deadline
    clock["now"] = 101.0
    window.capture_gesture()
    assert window.gesture_capture_deadline == deadline
    clock["now"] = deadline + .01
    window.check_freshness()
    assert window.gesture_capture_deadline is None
    assert "尚未記錄" in window.gesture_dialog.feedback_label.text()
    assert window.gesture_profile is None and window.gesture_candidate is None


def test_starting_aim_calibration_cancels_personal_gesture_capture(window):
    window.running = True
    window.begin_gesture_calibration()
    window.capture_gesture()
    window.begin_calibration()
    assert not window.gesture_calibrating and window.gesture_dialog is None
    assert window.gesture_capture_deadline is None
    assert window.calibrating


def test_sensor_restart_installs_saved_personal_gesture_without_opening_devices(window, monkeypatch):
    from kon.gesture_calibration import GestureCalibration
    points = personal_hand_points()
    profile = GestureCalibration.fit([points] * 12, handedness="Right").confirm([points] * 12, handedness="Right")
    window.gesture_profile = profile

    class QuietSignal:
        def connect(self, _callback):
            pass

    class FakeWorker:
        def __init__(self, *_args, gesture_profile=None, **_kwargs):
            self.gesture_profile = gesture_profile
            self.configuration = _kwargs
            self.running = False
            for name in ("frame", "tracking", "status", "error", "transcript", "keyword"):
                setattr(self, name, QuietSignal())
        def start(self):
            self.running = True
        def stop(self):
            self.running = False
        def isRunning(self):
            return self.running
        def set_active(self, _value):
            pass
        def wait(self, _timeout):
            return True

    monkeypatch.setattr("kon.app.CameraWorker", FakeWorker)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeWorker)
    window.toggle_sensors()
    assert window.camera.gesture_profile is profile
    assert window.speech.configuration["trigger_mode"] == "loud_sound"
    assert window.running and window.gesture_button.isEnabled()
    assert not window.calibrating and not window.gesture_calibrating


def test_personal_gesture_enrollment_accepts_valid_frame_age_point_three_two(window, monkeypatch):
    window.running = True
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window.begin_gesture_calibration()
    window.capture_gesture()
    for index in range(13):
        timestamp = 100.01 + index * .1
        clock["now"] = timestamp + .32
        window.on_tracking(TrackingFrame(
            timestamp, (), (), 0, False, None, None, None, "",
            hand_detected=True, hand_world_points=personal_hand_points(), handedness="Right"))
    assert window.gesture_candidate is not None and window.gesture_dialog.step == 1


def test_cancelled_new_aim_capture_restores_saved_mapping_without_restart(window, tmp_path):
    from kon.calibration_store import save_calibration
    original = install_linear_personal(window)
    save_calibration(tmp_path, original, window.config())
    window.begin_calibration()
    assert window.calibration is None
    window.cancel_calibration()
    assert window.calibration.raw_points == original.raw_points
    assert window.test_button.isEnabled()
