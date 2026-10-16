"""Eye-through-circle and optional hand-only aiming retain separate profiles."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.calibration_store import save_calibration
from kon.desktop import DesktopTarget
from kon.geometry import ScreenCalibration, TriggerGate
from kon.hand_aim import HandAimCalibration
from kon.vision import TrackingFrame
from test_hand_only_app import FakeWorker


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def environment(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.CameraWorker", FakeWorker)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeWorker)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    windows = []
    def make(*, aim_mode=None):
        value = Window() if aim_mode is None else Window(aim_mode=aim_mode)
        value.stable_dwell_multiplier = 1
        # Mode routing and pause/resume checks use an explicit short window.
        value.gate = TriggerGate(dwell_seconds=.65, keyword_max_age=2.5)
        windows.append(value)
        return value
    yield make, clock
    for value in reversed(windows):
        value.close()
    application.processEvents()


def preferences_path(tmp_path):
    folder = tmp_path / ".kon"
    folder.mkdir(exist_ok=True)
    return folder / "preferences.json"


def identity_eye_calibration():
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    return ScreenCalibration.fit(samples, samples)


def choose_mode(window, mode):
    index = window.aim_method.findData(mode)
    assert index >= 0
    window.aim_method.setCurrentIndex(index)


def send(window, clock, timestamp, *, raw=(.25, .5), cursor=(.65, .5),
         pose=True, depth="ready", aiming_mode=None):
    clock["now"] = timestamp
    window.on_tracking(TrackingFrame(
        timestamp, (), (), 1 if pose else 0, pose, None, None, raw, "瞄準追蹤",
        hand_detected=depth != "no_hand", depth_status=depth, gesture_reason="recognized" if pose else "",
        hand_cursor=cursor, aiming_mode=window.aim_mode if aiming_mode is None else aiming_mode))


def eye_lock(window, clock):
    window.calibration = identity_eye_calibration()
    window.running = window.testing = window.armed = True
    width, height = window.overlay.width(), window.overlay.height()
    window.targets = window.overlay.targets = [
        DesktopTarget("a", "A.txt", (width * .25, height * .5), radius=40),
        DesktopTarget("b", "B.txt", (width * .75, height * .5), radius=40),
    ]
    for index in range(8):
        send(window, clock, 100 + index * .1)
    assert window.gate.locked_target == "a" and window.voice_target == "a"


def test_missing_preference_defaults_to_eye_through_hand_circle(environment):
    make, _ = environment
    window = make()
    assert window.aim_mode == "eye_ray"
    assert window.aim_method.currentData() == "eye_ray"
    assert window.aim_method.findData("hand") >= 0
    assert window.camera is None and window.speech is None
    assert not window.running and not window.calibrating


@pytest.mark.parametrize("saved", [{}, {"aim_mode": None}, {"aim_mode": "invalid"}, {"aim_mode": 42}, None])
def test_invalid_or_missing_saved_mode_defaults_to_eye_without_opening_devices(environment, tmp_path, saved):
    preferences_path(tmp_path).write_text(json.dumps(saved))
    make, _ = environment
    window = make()
    assert window.aim_mode == "eye_ray"
    assert window.aim_method.currentData() == "eye_ray"
    assert window.camera is None and window.speech is None


def test_corrupt_preferences_default_to_eye_mode(environment, tmp_path):
    preferences_path(tmp_path).write_text("not JSON")
    make, _ = environment
    assert make().aim_mode == "eye_ray"


@pytest.mark.parametrize("saved", ["hand", "eye_ray"])
def test_saved_aim_mode_is_restored_on_restart(environment, tmp_path, saved):
    preferences_path(tmp_path).write_text(json.dumps({"aim_mode": saved, "voice_trigger": "kon"}))
    make, _ = environment
    window = make()
    assert window.aim_mode == saved
    assert window.aim_method.currentData() == saved
    assert window.camera is None and window.speech is None


@pytest.mark.parametrize("saved,explicit", [("hand", "eye_ray"), ("eye_ray", "hand")])
def test_explicit_constructor_mode_takes_precedence_without_rewriting_saved_choice(environment, tmp_path, saved, explicit):
    path = preferences_path(tmp_path)
    path.write_text(json.dumps({"aim_mode": saved, "unrelated": {"value": 12}}))
    original = path.read_bytes()
    make, _ = environment
    window = make(aim_mode=explicit)
    assert window.aim_mode == explicit
    assert window.aim_method.currentData() == explicit
    assert path.read_bytes() == original


def test_switch_saves_only_mode_and_restart_uses_it(environment, tmp_path):
    original = {"aim_mode": "eye_ray", "voice_trigger": "kon", "voice_trigger_mode": "phrase",
                "unrelated": {"numbers": [1, 2, 3], "label": "保留"}}
    path = preferences_path(tmp_path)
    path.write_text(json.dumps(original, ensure_ascii=False))
    make, _ = environment
    window = make()
    choose_mode(window, "hand")
    assert window.aim_mode == "hand"
    assert isinstance(window.calibration, HandAimCalibration)
    assert json.loads(path.read_text()) == original | {"aim_mode": "hand"}
    reopened = make()
    assert reopened.aim_mode == "hand"
    assert reopened.aim_method.currentData() == "hand"
    assert reopened.camera is None and reopened.speech is None


def test_paused_mode_switch_restores_both_calibrations_without_changing_their_bytes(environment, tmp_path):
    make, _ = environment
    window = make(aim_mode="eye_ray")
    eye = identity_eye_calibration()
    save_calibration(tmp_path, eye, window.config())
    hand = HandAimCalibration.from_corners((.25, .3), (.75, .85))
    hand.save(window.hand_aim_path())
    original = {path: path.read_bytes() for path in (tmp_path / ".kon").rglob("*.json")}
    window._load_calibration()
    choose_mode(window, "hand")
    assert window.calibration == hand
    assert not window.armed and not window.testing
    assert window.camera is None and window.speech is None
    choose_mode(window, "eye_ray")
    assert isinstance(window.calibration, ScreenCalibration)
    assert window.calibration.predict((.25, .5)) == pytest.approx((.25, .5))
    for path, data in original.items():
        assert path.read_bytes() == data


def test_mode_switch_clears_existing_session_lock_and_voice(environment):
    make, clock = environment
    window = make(aim_mode="eye_ray")
    eye_lock(window, clock)
    window.running = False  # Sensor stop is established; switching is now permitted.
    choose_mode(window, "hand")
    assert window.aim_mode == "hand"
    assert window.gate.locked_target is None
    assert not window.armed and not window.testing and not window.voice_active
    assert window.camera is None and window.speech is None


def test_attempting_to_switch_running_sensors_restores_selector_and_preserves_profile_and_preference(environment, tmp_path):
    make, _ = environment
    window = make(aim_mode="eye_ray")
    window.calibration = identity_eye_calibration()
    profile = window.calibration
    window.toggle_sensors()
    camera, speech = window.camera, window.speech
    previous = preferences_path(tmp_path).read_bytes()
    choose_mode(window, "hand")
    assert window.aim_mode == "eye_ray"
    assert window.aim_method.currentData() == "eye_ray"
    assert window.calibration is profile
    assert window.camera is camera and window.speech is speech
    assert camera.isRunning() and speech.isRunning()
    assert camera.kwargs["aim_mode"] == "eye_ray"
    assert preferences_path(tmp_path).read_bytes() == previous


@pytest.mark.parametrize("mode", ["eye_ray", "hand"])
def test_selected_mode_reaches_fake_camera_without_opening_real_hardware(environment, mode):
    make, _ = environment
    window = make(aim_mode=mode)
    window.toggle_sensors()
    assert isinstance(window.camera, FakeWorker)
    assert window.camera.kwargs["aim_mode"] == mode
    assert isinstance(window.speech, FakeWorker)


def test_eye_mode_uses_raw_eye_ray_hit_and_ignores_hand_cursor(environment):
    make, clock = environment
    window = make(aim_mode="eye_ray")
    window.calibration = identity_eye_calibration()
    window.running = True
    send(window, clock, 100, raw=(.25, .5), cursor=(.65, .5))
    assert window.overlay.point == pytest.approx((window.overlay.width() * .25, window.overlay.height() * .5))


def test_missing_eye_ray_retains_sticky_file_but_pauses_commands_despite_valid_hand_cursor(environment):
    make, clock = environment
    window = make(aim_mode="eye_ray")
    eye_lock(window, clock)
    send(window, clock, 100.8, raw=None, cursor=(.65, .5), depth="eye_unusable")
    assert window.gate.locked_target == "a"
    assert window.overlay.point == window.targets[0].center and window.overlay.target_id == "a"
    assert not window.gate.ready and not window.voice_active
    send(window, clock, 100.9, raw=(.27, .5), cursor=(.2, .2))
    assert window.gate.ready and window.voice_target == "a"


@pytest.mark.parametrize("mode,other", [("eye_ray", "hand"), ("hand", "eye_ray")])
def test_queued_frame_from_other_mode_pauses_but_cannot_replace_latest_sample_or_lock(environment, mode, other):
    make, clock = environment
    window = make(aim_mode=mode)
    window.calibration = identity_eye_calibration() if mode == "eye_ray" else HandAimCalibration()
    window.running = window.testing = window.armed = True
    width, height = window.overlay.width(), window.overlay.height()
    window.targets = window.overlay.targets = [DesktopTarget("a", "A.txt", (width * .25, height * .5), radius=40)]
    for index in range(8):
        send(window, clock, 100 + index * .1, raw=(.25, .5), cursor=(.35, .5))
    assert window.gate.locked_target == "a" and window.voice_active
    previous = window.latest
    send(window, clock, 100.8, raw=(.75, .5), cursor=(.65, .5), aiming_mode=other)
    assert window.latest is previous
    assert window.gate.locked_target == "a" and not window.voice_active
    assert not window.gate.ready
    assert window.overlay.target_id == "a" and window.overlay.point == window.targets[0].center


@pytest.mark.parametrize("mode,expected", [("eye_ray", (.25, .5)), ("hand", (.75, .5))])
def test_both_modes_apply_strong_pixel_filter_after_their_own_mapping(environment, monkeypatch, mode, expected):
    make, clock = environment
    window = make(aim_mode=mode)
    window.calibration = identity_eye_calibration() if mode == "eye_ray" else HandAimCalibration()
    window.running = True
    calls = []
    def filter_point(timestamp, normalized, screen_size):
        calls.append((timestamp, normalized, screen_size))
        return (.42, .61)
    monkeypatch.setattr(window.hand_smoother, "update", filter_point)
    send(window, clock, 100, raw=(.25, .5), cursor=(.65, .5))
    assert len(calls) == 1
    assert calls[0][0] == 100
    assert calls[0][1] == pytest.approx(expected)
    assert calls[0][2] == (window.overlay.width(), window.overlay.height())
    assert window.overlay.point == pytest.approx((window.overlay.width() * .42, window.overlay.height() * .61))
