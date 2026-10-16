"""Hand-only app flows with fake workers/clock; no camera, mic or real files."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget
from kon.geometry import ScreenCalibration, TriggerGate
from kon.hand_aim import HandAimCalibration
from kon.vision import TrackingFrame


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.started = False
        self.active = False
        for signal in ("frame", "tracking", "status", "error", "transcript", "keyword", "finished"):
            setattr(self, signal, FakeSignal())

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started

    def stop(self):
        self.started = False

    def wait(self, _timeout):
        return True

    def set_active(self, active):
        self.active = bool(active)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    # Even accidental sensor startup within a test uses inert Python objects.
    monkeypatch.setattr("kon.app.CameraWorker", FakeWorker)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeWorker)
    window = Window(aim_mode="hand")
    window.stable_dwell_multiplier = 1
    # Acquisition timing has dedicated default-policy integration coverage.
    window.gate = TriggerGate(dwell_seconds=.65, keyword_max_age=2.5)
    yield window, clock
    window.close()
    application.processEvents()


def send(window, clock, timestamp, cursor=(.35, .5), *, pose=True,
         depth="ready", reason="recognized", arrival=None, aiming_mode="hand", raw=None, score=None):
    clock["now"] = timestamp if arrival is None else arrival
    window.on_tracking(TrackingFrame(
        timestamp, (), (), score if score is not None else (1 if pose else 0), pose, None, None, raw, "手圈追蹤",
        hand_detected=depth != "no_hand", depth_status=depth, gesture_reason=reason,
        gesture_source="world", handedness="Right", hand_cursor=cursor, aiming_mode=aiming_mode))


def set_targets(window):
    window.running = window.testing = window.armed = True
    width, height = window.overlay.width(), window.overlay.height()
    window.targets = window.overlay.targets = [
        DesktopTarget("a", "A.txt", (width * .25, height * .5), radius=40),
        DesktopTarget("b", "B.txt", (width * .75, height * .5), radius=40),
    ]


def lock_first(window, clock):
    set_targets(window)
    for index in range(8):
        send(window, clock, 100 + index * .1)
    assert window.gate.locked_target == "a"
    assert window.voice_active and window.voice_target == "a"


def corner_frames(window, clock, cursor, *, count=7, pose=False):
    # Corner range enrollment needs the circle position, not a passing command
    # pose or any eye/3D screen intersection.
    start = clock["now"] + .01
    for index in range(count):
        timestamp = start + index * .1
        send(window, clock, timestamp, cursor, pose=pose, reason="finger pattern", arrival=timestamp + .01)


def legacy_eye_files(window, tmp_path):
    folder = tmp_path / ".kon"
    folder.mkdir(exist_ok=True)
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    ScreenCalibration.fit(samples, samples).save(folder / "calibration.json")
    (folder / "calibration-meta.json").write_text(json.dumps(window.config()))
    (folder / "calibration-last-attempt.json").write_text(json.dumps({"version": 1, "raw_points": samples}))
    return {path: path.read_bytes() for path in folder.glob("calibration*.json")}


def test_explicit_hand_model_allows_practice_without_eye_or_nine_point_calibration(session):
    window, _ = session
    assert window.aim_mode == "hand"
    assert isinstance(window.calibration, HandAimCalibration)
    assert not window.calibration.personalized
    assert window.camera is None and window.speech is None
    assert not window.calibrating and not window.hand_range_calibrating
    assert "可直接開始" in window.cal_label.text()
    window.running = True
    window.speech = FakeWorker()
    window.speech.start()
    window.begin_targets()
    assert window.testing and window.armed
    assert window.overlay.practice and len(window.targets) > 0
    assert not window.calibrating and not window.hand_range_calibrating


def test_startup_passes_hand_mode_to_fake_camera_and_never_opens_devices(session):
    window, _ = session
    window.toggle_sensors()
    assert isinstance(window.camera, FakeWorker)
    assert isinstance(window.speech, FakeWorker)
    assert window.camera.started and window.speech.started
    assert window.camera.kwargs["aim_mode"] == "hand"
    assert window.camera.kwargs["gesture_profile"] is window.gesture_profile
    assert window.test_button.isEnabled()
    assert isinstance(window.calibration, HandAimCalibration)


def test_mirrored_circle_maps_once_and_ignores_eye_ray_data(session):
    window, clock = session
    window.running = True
    send(window, clock, 100, (.65, .5), raw=(.1, .1))
    expected = (.75 * window.overlay.width(), .5 * window.overlay.height())
    assert window.overlay.point == pytest.approx(expected)
    assert window.latest.eye_xyz is None and window.latest.face_points == ()
    # Mirroring .65 a second time would select .25 instead of .75.
    assert window.overlay.point[0] > window.overlay.width() / 2


def test_sticky_file_and_anchor_survive_small_hand_cursor_jitter(session):
    window, clock = session
    lock_first(window, clock)
    for index in range(10):
        send(window, clock, 100.8 + index * .1, (.36, .5))
    assert window.gate.locked_target == "a"
    assert window.overlay.target_id == "a"
    assert window.overlay.point == window.targets[0].center
    assert window.overlay.progress == 1
    assert window.voice_active and window.voice_target == "a"
    fired = []
    window.summon = fired.append
    window.on_keyword(clock["now"], "kon")
    assert fired == ["a"]


def test_large_hand_cursor_move_releases_lock_and_waits_for_a_new_window(session, monkeypatch):
    window, clock = session
    monkeypatch.setattr(window.hand_smoother, "update", lambda _time, point, _size: point)
    lock_first(window, clock)
    send(window, clock, 100.8, (.65, .5))
    assert window.gate.locked_target is None and not window.voice_active
    assert window.overlay.target_id == "b" and window.overlay.progress == 0
    for index in range(1, 4):
        send(window, clock, 100.8 + index * .1, (.65, .5))
        assert window.gate.locked_target is None
    send(window, clock, 101.2, (.65, .5))
    assert window.gate.locked_target == "b" and window.voice_target == "b"


def test_hand_mode_locks_accumulated_visits_early_within_recent_three_seconds(session, monkeypatch):
    window, clock = session
    window.gate = TriggerGate(keyword_max_age=2.5)
    monkeypatch.setattr(window.hand_smoother, "update", lambda _time, point, _size: point)
    set_targets(window)
    for index in range(19):
        send(window, clock, 100 + index * .1, (.65, .5) if 6 <= index < 9 else (.35, .5))
        if index < 18:
            assert window.gate.locked_target is None and not window.voice_active
    assert window.gate.locked_target == "a" and window.voice_target == "a"
    assert window.gate.locked_at == 101.8
    assert window.overlay.point == window.targets[0].center


def test_valid_hand_cursor_and_high_score_cannot_lock_or_arm_voice_when_pose_is_false(session):
    window, clock = session
    set_targets(window)
    fired = []
    window.summon = fired.append
    for index in range(12):
        send(window, clock, 100 + index * .1, (.35, .5), pose=False,
             score=.95, reason="personal gesture mismatch")
    window.on_keyword(clock["now"], "kon")
    assert window.gate.locked_target is None and not window.gate.ready
    assert not window.voice_active and window.overlay.point is None
    assert window.overlay.target_id is None
    assert fired == []


def test_future_source_timestamp_pauses_actions_without_releasing_or_swapping_lock(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 101.0, (.65, .5), arrival=100.8)
    assert not window.voice_active and not window.gate.ready
    assert window.gate.locked_target == "a"
    assert window.overlay.target_id == "a" and window.overlay.point == window.targets[0].center
    fired = []
    window.summon = fired.append
    window.on_keyword(100.8, "kon")
    assert fired == []


def test_missing_hand_cursor_pauses_speech_but_retains_lock_and_rejects_old_audio(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.8, None, depth="hand_cursor_unusable")
    assert window.gate.locked_target == "a" and window.overlay.target_id == "a"
    assert window.overlay.point == window.targets[0].center
    assert window.overlay.progress == 1 and not window.voice_active
    send(window, clock, 100.9, (.36, .5))
    assert window.gate.ready and window.voice_target == "a"
    fired = []
    window.summon = fired.append
    window.on_keyword(100.71, "kon")
    assert fired == []
    window.on_keyword(100.9, "kon")
    assert fired == ["a"]


def test_fresh_no_hand_releases_only_after_debounce_then_new_file_can_lock(session):
    window, clock = session
    lock_first(window, clock)
    for timestamp in (100.8, 100.95):
        send(window, clock, timestamp, None, pose=False, reason="", depth="no_hand")
        assert window.gate.locked_target == "a" and not window.voice_active
    send(window, clock, 101.06, None, pose=False, reason="", depth="no_hand")
    assert window.gate.locked_target is None and window.overlay.target_id is None
    for index in range(8):
        send(window, clock, 101.2 + index * .1, (.65, .5))
    assert window.gate.locked_target == "b" and window.voice_target == "b"


def test_stale_no_hand_or_synthetic_absence_cannot_release_a_file(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.5, None, pose=False, reason="", depth="no_hand", arrival=101.0)
    assert window.gate.locked_target == "a" and not window.voice_active
    for timestamp in (101.1, 101.25, 101.4):
        send(window, clock, timestamp, None, pose=False, reason="", depth="")
    assert window.gate.locked_target == "a" and window.overlay.target_id == "a"


@pytest.mark.parametrize("failure", ["wrong_mode", "invalid_filter", "duplicate_timestamp"])
def test_missing_valid_screen_position_never_keeps_voice_active(session, monkeypatch, failure):
    window, clock = session
    lock_first(window, clock)
    if failure == "invalid_filter":
        monkeypatch.setattr(window.hand_smoother, "update", lambda *_args: None)
        send(window, clock, 100.8, (.65, .5))
    elif failure == "duplicate_timestamp":
        send(window, clock, 100.7, (.65, .5), arrival=100.8)
    else:
        send(window, clock, 100.8, (.65, .5), aiming_mode="eye_ray", raw=(.75, .5))
    assert not window.voice_active and not window.gate.ready
    assert window.gate.locked_target == "a"


def test_legacy_eye_profile_does_not_replace_default_hand_model_or_change_its_files(session, tmp_path):
    window, _ = session
    legacy = legacy_eye_files(window, tmp_path)
    reopened = Window(aim_mode="hand")
    try:
        assert reopened.aim_mode == "hand"
        assert isinstance(reopened.calibration, HandAimCalibration)
        assert not reopened.calibration.personalized
        assert reopened.calibration.bounds == ((.2, .2), (.8, .8))
        assert not reopened.calibrating
        for path, data in legacy.items():
            assert path.read_bytes() == data
    finally:
        reopened.close()


def test_optional_two_fresh_corner_holds_save_reload_without_touching_legacy_eye_files(session, tmp_path):
    window, clock = session
    legacy = legacy_eye_files(window, tmp_path)
    original = window.calibration
    window.running = True
    window.begin_calibration()
    assert window.hand_range_calibrating and not window.calibrating
    assert "眼" not in window.hand_range_dialog.reason_label.text()
    corner_frames(window, clock, (.25, .30))
    assert window.hand_range_points == []  # Pre-button stability is not capture.
    window.capture_hand_range()
    assert window.calibration is original
    corner_frames(window, clock, (.25, .30))
    assert window.hand_range_dialog.step == 1
    assert len(window.hand_range_points) == 1
    assert not window.hand_aim_path().exists()
    window.capture_hand_range()
    corner_frames(window, clock, (.75, .85))
    assert not window.hand_range_calibrating and window.hand_range_dialog is None
    assert window.calibration.personalized
    assert window.calibration.top_left == pytest.approx((.25, .30))
    assert window.calibration.bottom_right == pytest.approx((.75, .85))
    saved = HandAimCalibration.load(window.hand_aim_path())
    assert saved == window.calibration
    for path, data in legacy.items():
        assert path.read_bytes() == data
    reopened = Window(aim_mode="hand")
    try:
        assert reopened.aim_mode == "hand"
        assert reopened.calibration == saved
        assert not reopened.hand_range_calibrating and not reopened.calibrating
        assert reopened.camera is None and reopened.speech is None
        for path, data in legacy.items():
            assert path.read_bytes() == data
    finally:
        reopened.close()


def test_cancelled_corner_adjustment_preserves_old_personalized_data_and_legacy_files(session, tmp_path):
    window, clock = session
    legacy = legacy_eye_files(window, tmp_path)
    original = HandAimCalibration.from_corners((.3, .25), (.85, .8))
    original.save(window.hand_aim_path())
    window.calibration = original
    previous_bytes = window.hand_aim_path().read_bytes()
    window.running = True
    window.begin_hand_range()
    window.capture_hand_range()
    corner_frames(window, clock, (.25, .2))
    assert window.hand_range_dialog.step == 1
    window.cancel_hand_range()
    assert not window.hand_range_calibrating and window.hand_range_dialog is None
    assert window.calibration is original
    assert window.hand_aim_path().read_bytes() == previous_bytes
    for path, data in legacy.items():
        assert path.read_bytes() == data


def test_invalid_second_corner_keeps_old_range_and_allows_retry(session):
    window, clock = session
    original = HandAimCalibration.from_corners((.3, .25), (.85, .8))
    original.save(window.hand_aim_path())
    window.calibration = original
    previous = window.hand_aim_path().read_bytes()
    window.running = True
    window.begin_hand_range()
    window.capture_hand_range()
    corner_frames(window, clock, (.25, .25))
    window.capture_hand_range()
    corner_frames(window, clock, (.3, .3))  # Too little comfortable movement.
    assert window.hand_range_calibrating and window.hand_range_dialog.step == 1
    assert window.calibration is original
    assert window.hand_aim_path().read_bytes() == previous
    assert "範圍太小" in window.hand_range_dialog.feedback_label.text()
    window.capture_hand_range()
    corner_frames(window, clock, (.75, .8))
    assert not window.hand_range_calibrating
    assert window.calibration != original


def test_two_corner_timeout_cannot_save_a_partial_mapping(session):
    window, clock = session
    original = window.calibration
    window.running = True
    window.begin_hand_range()
    window.capture_hand_range()
    corner_frames(window, clock, (.25, .25), count=3)
    clock["now"] += 6
    window.check_freshness()
    assert window.hand_range_deadline is None
    assert window.hand_range_calibrating and not window.hand_range_points
    assert window.calibration is original
    assert not window.hand_aim_path().exists()


def test_corner_save_failure_preserves_installed_model_and_previous_file(session, monkeypatch):
    window, clock = session
    original = HandAimCalibration.from_corners((.3, .25), (.85, .8))
    original.save(window.hand_aim_path())
    window.calibration = original
    previous = window.hand_aim_path().read_bytes()
    window.running = True
    window.begin_hand_range()
    window.capture_hand_range()
    corner_frames(window, clock, (.25, .25))
    def fail_save(*_args):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(HandAimCalibration, "save", fail_save)
    window.capture_hand_range()
    corner_frames(window, clock, (.75, .8))
    assert window.hand_range_calibrating and window.hand_range_dialog.step == 1
    assert window.calibration is original
    assert window.hand_aim_path().read_bytes() == previous
    assert "storage failure" in window.hand_range_dialog.feedback_label.text()
