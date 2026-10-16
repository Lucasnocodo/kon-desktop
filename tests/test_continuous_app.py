"""Continuous selection uses confirmed outcomes and fresh targets, without devices."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget, FileSnapshot, TrashReceipt
from kon.hand_aim import HandAimCalibration
from kon.selection_settings import SelectionSettings
from kon.vision import TrackingFrame


class ControlledPool:
    """Run the real Job's done/failed handling only when the test requests it."""
    def __init__(self):
        self.jobs = []

    def start(self, job):
        self.jobs.append(job)

    def run_next(self):
        assert self.jobs
        self.jobs.pop(0).run()


@pytest.fixture
def session(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    clock = {"now": 100.}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])

    def forbidden_device(*_args, **_kwargs):
        pytest.fail("A continuous-operation test must never open devices")

    monkeypatch.setattr("kon.app.CameraWorker", forbidden_device)
    monkeypatch.setattr("kon.app.SpeechWorker", forbidden_device)
    value = Window(aim_mode="hand")
    value.watchdog.stop()
    value.selection_apply_settings(SelectionSettings(window_seconds=.5, minimum_score=.5, stable_multiplier=1))
    value.calibration = HandAimCalibration((0., 0.), (1., 1.))
    monkeypatch.setattr(value.hand_smoother, "update", lambda _time, point, _size: point)
    pool = ControlledPool()
    value.pool = pool
    played = []

    def play(target):
        played.append(target)
        value.overlay.eating = target

    monkeypatch.setattr(value.overlay, "play", play)
    events = {"validated": [], "trashed": [], "scanned": [], "scan_result": [], "fail": None}

    def validate(target, point, size):
        events["validated"].append((target, point, size))
        if events["fail"] == "validate":
            raise RuntimeError("position validation failed")
        return target

    def trash(snapshot):
        events["trashed"].append(snapshot)
        if events["fail"] == "trash":
            raise RuntimeError("trash failed")
        return TrashReceipt(snapshot.path, tmp_path / "Trash" / snapshot.path.name, 1, 2)

    def scan(size):
        events["scanned"].append(size)
        if events["fail"] == "scan":
            raise RuntimeError("desktop refresh failed")
        return events["scan_result"]

    monkeypatch.setattr("kon.app.validate_target_position", validate)
    monkeypatch.setattr("kon.app.trash_target", trash)
    monkeypatch.setattr("kon.app.scan_desktop", scan)
    harness = SimpleNamespace(window=value, clock=clock, pool=pool, played=played,
                              events=events, root=tmp_path)
    yield harness
    value.close()
    application.processEvents()


def setup_targets(session, *, real=False):
    window = session.window
    window.mode.setCurrentIndex(1 if real else 0)
    width, height = window.overlay.width(), window.overlay.height()
    targets = []
    for index, (name, x) in enumerate((("A.txt", .25), ("B.txt", .75))):
        path = session.root / "Desktop" / name
        snapshot = FileSnapshot(path, 1, index + 1, 10, 1) if real else None
        targets.append(DesktopTarget(str(path) if real else name, name,
                                     (width * x, height * .5), snapshot))
    window.running = True
    window.activate_targets(targets)
    return targets


def send(session, timestamp, point=(.25, .5)):
    session.clock["now"] = timestamp
    session.window.on_tracking(TrackingFrame(
        timestamp, (), (), 1., True, None, None, None, "手圈追蹤正常",
        hand_detected=True, depth_status="ready", gesture_reason="recognized",
        gesture_source="world", handedness="Right", hand_cursor=point, aiming_mode="hand"))


def hold(session, start, end, point=(.25, .5)):
    for index in range(round((end - start) * 10) + 1):
        send(session, round(start + index * .1, 10), point)


def trigger_first(session, *, real=False):
    targets = setup_targets(session, real=real)
    hold(session, 100., 100.4)
    assert session.window.gate.locked_target is None
    send(session, 100.5)
    assert session.window.gate.locked_target == targets[0].id
    session.clock["now"] = 100.51
    session.window.on_keyword(100.51, "短音")
    assert not session.window.armed and not session.window.voice_active
    if real:
        assert len(session.pool.jobs) == 1 and not session.played
        session.pool.run_next()
        assert session.events["validated"][0][1] == targets[0].center
    assert session.played == [targets[0]]
    return targets


def finish_animation(session, timestamp):
    session.clock["now"] = timestamp
    session.window.overlay.eating = None
    session.window.on_animation_done()


def bite(session, target, timestamp=100.6):
    session.clock["now"] = timestamp
    session.window.on_bite(target)


def test_practice_success_keeps_hand_and_requires_new_half_second_lock_and_new_audio(session):
    first, second = trigger_first(session)
    bite(session, first)
    assert not session.window.armed
    finish_animation(session, 100.7)
    window = session.window
    assert window.armed and window.testing and window.active_action is None
    assert window.targets == [second] and not session.pool.jobs
    assert window.gate.locked_target is None and window.gate.progress == 0
    hold(session, 100.8, 101.2, (.75, .5))
    assert window.gate.locked_target is None and not window.voice_active
    session.clock["now"] = 101.21
    window.on_keyword(101.21, "鎖定前的短音")
    send(session, 101.3, (.75, .5))
    assert window.gate.locked_target == second.id and window.voice_active
    session.clock["now"] = 101.31
    window.on_keyword(100.51, "上一個檔案的短音")
    window.on_keyword(101.21, "鎖定前的短音")
    assert session.played == [first]
    session.clock["now"] = 101.32
    window.on_keyword(101.32, "新的短音")
    assert session.played == [first, second]
    assert window.camera is None and window.speech is None


def test_animation_without_successful_bite_does_not_rearm(session):
    first, _second = trigger_first(session)
    finish_animation(session, 100.7)
    assert not session.window.armed and not session.window.voice_active
    assert first in session.window.targets
    hold(session, 100.8, 101.4, (.75, .5))
    assert not session.window.gate.ready and not session.window.voice_active
    assert session.played == [first]


def test_practice_waits_for_bite_when_animation_finishes_first(session):
    first, second = trigger_first(session)
    finish_animation(session, 100.6)
    assert not session.window.armed
    bite(session, first, 100.7)
    assert session.window.armed and session.window.testing
    assert session.window.targets == [second]
    assert session.window.active_action is None and not session.pool.jobs
    hold(session, 100.8, 101.2, (.75, .5))
    assert session.window.gate.locked_target is None
    send(session, 101.3, (.75, .5))
    assert session.window.gate.locked_target == second.id


def test_physical_release_during_animation_does_not_cancel_successful_continuation(session):
    first, second = trigger_first(session)
    for timestamp in (100.6, 100.8, 100.86):
        session.clock["now"] = timestamp
        session.window.on_tracking(TrackingFrame(
            timestamp, (), (), 0., False, None, None, None, "手已離開畫面",
            hand_detected=False, depth_status="no_hand", aiming_mode="hand"))
    bite(session, first, 100.9)
    finish_animation(session, 101.)
    assert session.window.armed and session.window.testing
    assert session.window.targets == [second] and session.window.active_action is None
    hold(session, 101.1, 101.5, (.75, .5))
    assert session.window.gate.locked_target is None
    send(session, 101.6, (.75, .5))
    assert session.window.gate.locked_target == second.id


@pytest.mark.parametrize("order", ["trash_first", "animation_first"])
def test_real_completion_waits_for_trash_animation_and_fresh_scan_before_resuming(session, order):
    first, second = trigger_first(session, real=True)
    fresh_second = DesktopTarget(second.id, second.name,
                                 (session.window.overlay.width() * .6,
                                  session.window.overlay.height() * .25),
                                 FileSnapshot(second.snapshot.path, 1, 99, 20, 2))
    session.events["scan_result"] = [fresh_second]
    bite(session, first)
    bite(session, first, 100.61)  # Duplicate animation signal cannot trash twice.
    assert len(session.pool.jobs) == 1
    if order == "trash_first":
        session.clock["now"] = 100.7
        session.pool.run_next()
        assert not session.pool.jobs and not session.window.armed
        assert not session.events["scanned"]
        finish_animation(session, 100.8)
    else:
        finish_animation(session, 100.7)
        assert len(session.pool.jobs) == 1 and not session.events["trashed"]
        assert not session.window.armed and not session.events["scanned"]
        session.clock["now"] = 100.8
        session.pool.run_next()
    assert session.events["trashed"] == [first.snapshot]
    assert len(session.pool.jobs) == 1  # Fresh scan is now queued, not completed.
    assert session.window.active_action.refreshing
    assert not session.window.armed and not session.window.voice_active
    assert not session.events["scanned"]
    hold(session, 100.9, 101.5, (.75, .5))
    assert not session.window.gate.ready and not session.window.voice_active
    session.pool.run_next()
    assert len(session.events["scanned"]) == 1
    assert session.window.targets == [fresh_second]
    assert session.window.targets[0] is fresh_second
    assert session.window.active_action is None and session.window.armed
    assert session.window.gate.locked_target is None and session.window.gate.progress == 0
    hold(session, 101.6, 102., (.6, .25))
    assert session.window.gate.locked_target is None
    send(session, 102.1, (.6, .25))
    assert session.window.gate.locked_target == fresh_second.id


@pytest.mark.parametrize("stage", ["validate", "trash", "scan"])
@pytest.mark.parametrize("cancel", ["control", "speech_error"])
def test_cancelled_session_ignores_late_success_callbacks_and_never_rearms(session, stage, cancel):
    if stage == "validate":
        targets = setup_targets(session, real=True)
        hold(session, 100., 100.5)
        session.clock["now"] = 100.51
        session.window.on_keyword(100.51, "短音")
    else:
        targets = trigger_first(session, real=True)
        bite(session, targets[0])
        if stage == "scan":
            finish_animation(session, 100.7)
            session.clock["now"] = 100.8
            session.pool.run_next()
            session.events["scan_result"] = [targets[1]]
    # Model a function which finished before cancellation, whose UI callback
    # is delivered afterward. This includes an actual confirmed Trash receipt.
    assert len(session.pool.jobs) == 1
    delayed = session.pool.jobs.pop(0)
    result = delayed.function()
    session.clock["now"] = 100.9
    if cancel == "control":
        session.window.return_to_control()
    else:
        session.window.speech_error("microphone disconnected")
    serial = session.window.action_serial
    delayed.signals.done.emit(result)
    assert session.window.action_serial == serial
    assert not session.window.testing and not session.window.armed
    assert not session.window.voice_active and session.window.active_action is None
    assert not session.pool.jobs
    if stage == "validate":
        assert not session.played


def test_cancel_before_queued_trash_runs_does_not_touch_file_or_rearm(session):
    first, _second = trigger_first(session, real=True)
    bite(session, first)
    session.window.return_to_control()
    session.pool.run_next()
    assert not session.events["trashed"] and not session.events["scanned"]
    assert not session.window.testing and not session.window.armed
    assert session.window.active_action is None


@pytest.mark.parametrize("stage", ["validate", "trash", "scan"])
def test_failed_real_job_returns_to_control_without_rearming_consumed_hold(session, stage):
    if stage == "validate":
        targets = setup_targets(session, real=True)
        hold(session, 100., 100.5)
        session.clock["now"] = 100.51
        session.window.on_keyword(100.51, "短音")
    else:
        targets = trigger_first(session, real=True)
        bite(session, targets[0])
        if stage == "scan":
            finish_animation(session, 100.7)
            session.clock["now"] = 100.8
            session.pool.run_next()
    session.events["fail"] = stage
    session.pool.run_next()
    assert not session.window.testing and not session.window.armed
    assert not session.window.voice_active and session.window.active_action is None
    assert not session.pool.jobs
    # A normal new UI session does not undo the consumed hold after a failure.
    session.window.activate_targets([targets[1]])
    hold(session, 101., 101.6, (.75, .5))
    assert session.window.gate.locked_target is None and not session.window.gate.ready
    assert not session.window.voice_active


def test_duplicate_animation_done_queues_only_one_desktop_refresh(session):
    first, second = trigger_first(session, real=True)
    session.events["scan_result"] = [second]
    bite(session, first)
    session.pool.run_next()
    finish_animation(session, 100.7)
    finish_animation(session, 100.71)
    assert len(session.pool.jobs) == 1
    session.pool.run_next()
    finish_animation(session, 100.72)
    assert len(session.events["scanned"]) == 1 and not session.pool.jobs
    assert session.window.armed and session.window.active_action is None


def test_old_animation_completion_cannot_finish_the_next_animation(session):
    first, second = trigger_first(session)
    bite(session, first)
    finish_animation(session, 100.7)
    hold(session, 100.8, 101.3, (.75, .5))
    session.clock["now"] = 101.31
    session.window.on_keyword(101.31, "新的短音")
    state = session.window.active_action
    assert state.target_id == second.id and not state.animation_done
    assert session.window.overlay.eating is second
    session.window.on_animation_done()  # Delayed duplicate for the old animation.
    assert session.window.active_action is state and not state.animation_done
    bite(session, second, 101.4)
    assert state.succeeded and not session.window.armed
    finish_animation(session, 101.5)
    assert not session.window.testing and session.window.active_action is None
