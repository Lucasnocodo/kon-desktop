"""Locked desktop selection survives aiming noise; no devices/files are touched."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget
from kon.geometry import ScreenCalibration, TriggerGate
from kon.vision import TrackingFrame


@pytest.fixture
def session(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    window = Window(aim_mode="eye_ray")
    window.stable_dwell_multiplier = 1
    # These cases exercise release, speech and file anchors after acquisition;
    # the real recent-three-seconds voting policy is covered in test_grid_app.
    window.gate = TriggerGate(dwell_seconds=.65, keyword_max_age=2.5)
    monkeypatch.setattr(window.hand_smoother, "update", lambda _time, point, _size: point)
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    window.calibration = ScreenCalibration.fit(samples, samples)
    window.running = window.testing = window.armed = True
    width, height = window.overlay.width(), window.overlay.height()
    window.targets = window.overlay.targets = [
        DesktopTarget("a", "A.txt", (width * .25, height * .5), radius=40),
        DesktopTarget("b", "B.txt", (width * .75, height * .5), radius=40),
    ]
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    yield window, clock
    window.stop_sensors()
    window.close()
    application.processEvents()


def send(window, clock, timestamp, raw=(.25, .5), *, pose=True, reason="recognized", depth="ready", world=()):
    clock["now"] = timestamp
    window.on_tracking(TrackingFrame(
        timestamp, (), (), 1 if pose else 0, pose, None, None, raw, "追蹤狀態",
        hand_detected=depth != "no_hand", depth_status=depth, gesture_reason=reason,
        hand_world_points=world, gesture_source="world", handedness="Right"))


def lock_first(window, clock):
    for index in range(8):
        send(window, clock, 100 + index * .1)
    assert window.gate.locked_target == "a" and window.voice_active


def test_completed_lock_keeps_file_and_crosshair_during_small_aim_jitter(session):
    window, clock = session
    lock_first(window, clock)
    for index in range(10):
        send(window, clock, 100.8 + index * .1, raw=(.27, .5))
    assert window.gate.locked_target == "a"
    assert window.overlay.target_id == "a"
    assert window.overlay.point == window.targets[0].center
    assert window.overlay.progress == 1
    assert window.voice_active and window.voice_target == "a"
    fired = []
    window.summon = fired.append
    window.on_keyword(clock["now"], "kon")
    assert fired == ["a"]


def test_large_aim_move_releases_lock_and_starts_a_new_selection_window(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.8, raw=(.75, .5))
    assert window.gate.locked_target is None
    assert not window.voice_active and window.overlay.progress == 0
    assert window.overlay.target_id == "b"
    for index in range(1, 4):
        send(window, clock, 100.8 + index * .1, raw=(.75, .5))
        assert window.gate.locked_target is None
    send(window, clock, 101.2, raw=(.75, .5))
    assert window.gate.locked_target == "b" and window.voice_target == "b"


@pytest.mark.parametrize("offset,releases", [((2, 2), False), ((3, 0), True), ((0, 3), True)])
@pytest.mark.parametrize("screen_size", [(3440, 1440), (1920, 1080), (800, 800)])
def test_movement_release_uses_two_surrounding_cell_rings(session, offset, releases, screen_size):
    window, clock = session
    window.overlay.resize(*screen_size)
    width, height = screen_size
    window.targets = window.overlay.targets = [DesktopTarget("a", "A.txt", (width * .25, height * .5))]
    lock_first(window, clock)
    size = (window.overlay.width(), window.overlay.height())
    cell = window.grid_selector.cell_for(window.targets[0].center, size)
    destination = (cell[0] + offset[0], cell[1] + offset[1])
    x, y, width, height = window.grid_selector.cell_bounds(destination, size)
    raw = ((x + width / 2) / size[0], (y + height / 2) / size[1])
    send(window, clock, 100.8, raw=raw)
    assert (window.gate.locked_target is None) is releases
    assert window.voice_active is not releases


def test_large_movement_cannot_rearm_a_consumed_gesture(session):
    window, clock = session
    lock_first(window, clock)
    fired = []
    window.summon = fired.append
    window.on_keyword(100.7, "kon")
    assert fired == ["a"]
    for index in range(9):
        send(window, clock, 100.8 + index * .1, raw=(.75, .5))
    window.on_keyword(clock["now"], "kon")
    assert fired == ["a"]
    assert window.gate.locked_target != "b" and not window.gate.ready


def test_missing_ray_keeps_visual_lock_but_pauses_speech_until_fresh_ray_returns(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.8, raw=None, depth="eye_unusable")
    assert window.overlay.target_id == "a" and window.overlay.point == window.targets[0].center
    assert window.overlay.progress == 1 and not window.voice_active
    send(window, clock, 100.9, raw=(.27, .5))
    assert window.gate.ready and window.voice_active and window.voice_target == "a"


def test_real_absence_releases_only_after_debounce_then_new_target_can_lock(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.8, raw=None, pose=False, reason="", depth="no_hand")
    assert window.gate.locked_target == "a" and not window.voice_active
    send(window, clock, 100.95, raw=None, pose=False, reason="", depth="no_hand")
    send(window, clock, 101.06, raw=None, pose=False, reason="", depth="no_hand")
    assert window.gate.locked_target is None and window.overlay.target_id is None
    for index in range(8):
        send(window, clock, 101.2 + index * .1, raw=(.75, .5))
    assert window.gate.locked_target == "b" and window.voice_target == "b"


def test_valid_opened_hand_releases_but_a_brief_recognition_error_does_not(session):
    window, clock = session
    lock_first(window, clock)
    send(window, clock, 100.8, pose=False, reason="personal gesture mismatch")
    assert not window.voice_active and window.gate.locked_target == "a"
    send(window, clock, 100.9, raw=(.27, .5))
    assert window.gate.locked_target == "a" and window.voice_active
    for timestamp in (101.0, 101.15, 101.26):
        send(window, clock, timestamp, pose=False, reason="personal gesture mismatch")
    assert window.gate.locked_target is None


@pytest.mark.parametrize("reason,depth", [("personal gesture landmarks", "hand_pnp_failed"),
                                        ("finger pattern", "hand_pnp_failed"), ("", "")])
def test_uncertain_or_synthetic_status_frames_cannot_release_a_lock(session, reason, depth):
    window, clock = session
    lock_first(window, clock)
    for timestamp in (100.8, 100.95, 101.1, 101.25):
        send(window, clock, timestamp, raw=None, pose=False, reason=reason, depth=depth)
    assert window.gate.locked_target == "a" and window.overlay.target_id == "a"
    assert not window.voice_active


def test_stale_watchdog_preserves_lock_and_blocks_old_audio_on_resume(session):
    window, clock = session
    lock_first(window, clock)
    old_audio = 100.71
    clock["now"] = 101.1
    window.check_freshness()
    assert window.overlay.target_id == "a" and not window.voice_active
    send(window, clock, 101.2, raw=(.27, .5))
    assert window.voice_target == "a" and window.gate.ready
    fired = []
    window.summon = fired.append
    window.on_keyword(old_audio, "kon")
    assert fired == []
    window.on_keyword(101.2, "kon")
    assert fired == ["a"]


def test_session_cancel_clears_lock_and_never_carries_it_to_new_target_list(session):
    window, clock = session
    lock_first(window, clock)
    window.return_to_control()
    assert window.gate.locked_target is None and not window.voice_active
    window.activate_targets([DesktopTarget("new", "new.txt", (100, 100))])
    assert window.gate.locked_target is None


def test_real_file_position_check_uses_locked_anchor_not_new_aim(session, monkeypatch):
    window, clock = session
    original = window.targets[0]
    real = DesktopTarget(original.id, original.name, original.center, object(), original.radius)
    window.targets[0] = real
    window.overlay.practice = False
    lock_first(window, clock)
    send(window, clock, 100.8, raw=(.27, .5))
    # Even if display code changes its cursor later, the saved file center
    # anchors Finder's identity/location/overlap checks for this lock.
    window.overlay.point = window.targets[1].center
    checks = []
    played = []
    monkeypatch.setattr("kon.app.validate_target_position", lambda target, point, size: checks.append((target.id, point)) or target)
    monkeypatch.setattr(window.overlay, "play", played.append)
    window.job = lambda function, done: done(function())
    window.summon("a")
    assert checks == [("a", real.center)]
    assert window.pending_action[1] == "a"
    assert played == [real]
