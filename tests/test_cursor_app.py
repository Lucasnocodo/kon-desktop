"""Display ticks never become tracking/action samples; all devices are fakes."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget
from kon.geometry import ScreenCalibration, TriggerGate
from kon.vision import TrackingFrame
from test_hand_only_app import FakeWorker


class PreviewWorker(FakeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_enabled = True
        self.preview_changes = []

    def set_preview_enabled(self, enabled):
        self.preview_enabled = bool(enabled)
        self.preview_changes.append(bool(enabled))




@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.CameraWorker", PreviewWorker)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeWorker)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window = Window(aim_mode="eye_ray")
    # Isolate presentation timing from the production observation-window policy.
    window.gate = TriggerGate(dwell_seconds=.8, minimum_dwell_seconds=.8, keyword_max_age=2.5)
    window.stable_dwell_multiplier = 1
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    window.calibration = ScreenCalibration.fit(samples, samples)
    # Deterministic measured positions isolate the new presentation channel;
    # the real numerical measurement filter has its own jitter/lag tests.
    monkeypatch.setattr(window.hand_smoother, "update", lambda _timestamp, point, _size: point)
    window.camera, window.speech = PreviewWorker(), FakeWorker()
    window.camera.start()
    window.speech.start()
    window.running = True
    width, height = window.overlay.width(), window.overlay.height()
    window.activate_targets([
        DesktopTarget("a", "A.txt", (width * .25, height * .5)),
        DesktopTarget("b", "B.txt", (width * .75, height * .5)),
    ])
    assert window.overlay.isVisible()
    yield window, clock
    window.close()
    application.processEvents()


def send(window, clock, timestamp, raw=(.25, .5), *, pose=True, depth="ready"):
    clock["now"] = timestamp
    window.on_tracking(TrackingFrame(
        timestamp, (), (), 1 if pose else 0, pose, None, None, raw, "追蹤狀態",
        hand_detected=depth != "no_hand", depth_status=depth,
        gesture_reason="recognized" if pose else "", aiming_mode="eye_ray"))


def moving_to_second(window, clock):
    send(window, clock, 100)
    send(window, clock, 100.1, (.75, .5))


def jitter_lock_second(window, clock):
    moving_to_second(window, clock)
    for index in range(2, 10):
        send(window, clock, 100 + index * .1, (.75 + (.005 if index % 2 else -.005), .5))
    assert window.gate.locked_target == "b"


def test_measured_target_changes_before_rendered_cursor_arrives_and_render_is_precise_60hz(session):
    window, clock = session
    moving_to_second(window, clock)
    first, second = window.targets
    assert window.aim_point == pytest.approx(second.center)
    assert window.overlay.point == pytest.approx(first.center)
    # A leads the cumulative preview, while actual measured aim already picks
    # B's grid cell independently of the interpolated display point.
    assert window.overlay.target_id == window.gate.leading_target == "a"
    assert window.gate.leading_dwell_seconds == pytest.approx(.1)
    assert window.grid_selector.active_cell == window.grid_selector.cell_for(second.center, (window.overlay.width(), window.overlay.height()))
    assert window.gate.locked_target is None and not window.voice_active
    assert window.cursor_timer.isActive()
    assert window.cursor_timer.interval() == 16 and window.cursor_timer.timerType() == Qt.PreciseTimer
    clock["now"] = 100.14
    window.animate_cursor()
    assert first.center[0] < window.overlay.point[0] < second.center[0]
    assert window.aim_point == pytest.approx(second.center)
    assert window.overlay.target_id == "a"
    assert window.gate.leading_dwell_seconds == pytest.approx(.1)


def test_display_ticks_cannot_increase_dwell_lock_voice_or_latest_source_timestamp(session):
    window, clock = session
    send(window, clock, 100)
    send(window, clock, 100.1)
    latest = window.latest
    progress = window.gate.progress
    assert 0 < progress < 1
    for index in range(1, 49):
        clock["now"] = 100.1 + index / 60
        window.animate_cursor()
    assert window.latest is latest and window.latest.timestamp == 100.1
    assert window.gate.progress == progress
    assert window.gate.locked_target is None and not window.gate.ready
    assert not window.voice_active and not window.speech.active


def test_stale_unlocked_source_stops_render_and_clears_aim_without_extrapolation(session):
    window, clock = session
    moving_to_second(window, clock)
    latest = window.latest
    clock["now"] = 100.46
    window.animate_cursor()
    assert window.latest is latest
    assert window.overlay.point is None and window.aim_point is None
    assert not window.cursor_timer.isActive() and not window.cursor_motion.active
    assert window.gate.locked_target is None and not window.voice_active


@pytest.mark.parametrize("pose,depth", [(True, "eye_unusable"), (False, "no_hand")])
def test_fresh_source_failure_clears_cursor_and_animation_but_retains_accumulated_preview(session, pose, depth):
    window, clock = session
    moving_to_second(window, clock)
    assert window.cursor_motion.is_animating
    send(window, clock, 100.15, None, pose=pose, depth=depth)
    assert window.overlay.point is None and window.aim_point is None
    assert not window.cursor_timer.isActive() and not window.cursor_motion.active
    assert window.overlay.target_id == window.gate.leading_target == "a"
    assert window.overlay.progress == pytest.approx(.1 / .8)
    assert "A.txt" in window.overlay.hint and "已保留" in window.overlay.hint
    assert not window.voice_active and not window.speech.active


def test_completed_lock_snaps_immediately_to_file_center_and_stops_animation(session):
    window, clock = session
    moving_to_second(window, clock)
    for index in range(2, 9):
        send(window, clock, 100 + index * .1, (.75 + (.005 if index % 2 else -.005), .5))
    assert window.gate.locked_target is None and window.cursor_timer.isActive()
    send(window, clock, 100.9, (.745, .5))
    second = window.targets[1]
    assert window.gate.locked_target == "b" and window.voice_target == "b"
    assert window.aim_point != second.center  # Actual aim remains its measured value.
    assert window.overlay.point == second.center
    assert not window.cursor_timer.isActive() and not window.cursor_motion.is_animating


def test_stale_locked_source_keeps_exact_center_and_pauses_voice_without_new_measurements(session):
    window, clock = session
    jitter_lock_second(window, clock)
    latest = window.latest
    center = window.targets[1].center
    clock["now"] = 101.3
    window.check_freshness()
    window.animate_cursor()
    assert window.latest is latest
    assert window.gate.locked_target == "b" and not window.gate.ready
    assert window.overlay.point == center and window.aim_point is None
    assert not window.voice_active and not window.speech.active
    assert not window.cursor_timer.isActive() and not window.cursor_motion.is_animating


def test_camera_preview_is_paused_in_target_session_restored_in_control_and_kept_for_minimized_calibration(session):
    window, _ = session
    assert not window.camera.preview_enabled  # activate_targets already ran.
    image = QImage(8, 8, QImage.Format_RGBA8888)
    image.fill(QColor("red"))
    before = window.preview.image
    window.on_frame(image)  # An old preview callback queued before disable.
    assert window.preview.image is before
    window.return_to_control(stop_sound=False)
    assert window.camera.preview_enabled
    window.on_frame(image)
    assert window.preview.image is image
    window.begin_calibration()
    assert window.calibrating and window.isMinimized()
    assert window.camera.preview_enabled
    next_image = QImage(8, 8, QImage.Format_RGBA8888)
    next_image.fill(QColor("blue"))
    window.on_frame(next_image)
    assert window.preview.image is next_image
