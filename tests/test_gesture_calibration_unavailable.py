"""Stereo geometry failures remain visible and cannot train a hand template."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.calibration_store import gesture_profile_path
from kon.gesture_calibration import GestureEnrollmentBuffer
from kon.vision import TrackingFrame
from test_gesture_calibration import personal_hand, profile
from test_gesture_angles_app import FakeCamera
from test_hand_only_app import FakeWorker
from test_stereo_app import install


@pytest.mark.parametrize('points,side', [(None, 'unknown'), (None, 'right'), (personal_hand(), 'right')])
def test_unavailable_geometry_reason_precedes_handedness_and_never_adds_samples(points, side):
    buffer = GestureEnrollmentBuffer()
    for index in range(16):
        assert buffer.observe(index / 15, points, handedness=side,
                              unavailable_reason='其中一台鏡頭的慣用眼被遮擋')
    state = buffer.evaluate(1)
    assert not state.ready and state.count == 0
    assert state.reason == '其中一台鏡頭的慣用眼被遮擋'


def test_failure_blocks_ready_hold_without_incrementing_then_fresh_valid_geometry_can_recover():
    buffer = GestureEnrollmentBuffer()
    for index in range(13):
        buffer.observe(index * .1, personal_hand(), handedness='right')
    assert buffer.evaluate(1.2).ready
    previous_count = buffer.evaluate(1.2).count
    buffer.observe(1.3, None, handedness='unknown', unavailable_reason='雙鏡頭重投影誤差過大')
    failed = buffer.evaluate(1.3)
    assert not failed.ready and failed.count == previous_count
    assert failed.reason == '雙鏡頭重投影誤差過大'
    buffer.observe(1.4, personal_hand(), handedness='right')
    assert buffer.evaluate(1.4).ready


def test_old_failure_does_not_replace_current_fresh_observation():
    buffer = GestureEnrollmentBuffer()
    buffer.reset(10)
    buffer.observe(10.1, personal_hand(), handedness='right')
    assert not buffer.observe(10.0, None, unavailable_reason='舊錯誤')
    assert '舊錯誤' not in buffer.evaluate(10.1).reason
    assert buffer.evaluate(10.1).count == 1


@pytest.fixture
def session(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr('kon.app.ROOT', tmp_path)
    monkeypatch.setattr('kon.app.CameraWorker', FakeCamera)
    monkeypatch.setattr('kon.app.StereoCameraWorker', FakeCamera)
    monkeypatch.setattr('kon.app.SpeechWorker', FakeWorker)
    old = profile()
    path = gesture_profile_path(tmp_path)
    old.save(path)
    previous = path.read_bytes()
    clock = {'now': 100.0}
    monkeypatch.setattr('kon.app.time.monotonic', lambda: clock['now'])
    window = Window(aim_mode='eye_ray')
    window.acquisition.setCurrentIndex(window.acquisition.findData('stereo'))
    install(window)
    window.toggle_sensors()
    yield window, clock, old, path, previous
    window.close()
    application.processEvents()


def deliver(window, clock, *, points=(), reason='', message='', side='unknown', acquisition='stereo'):
    clock['now'] += .1
    window.on_tracking(TrackingFrame(
        clock['now'] - .01, (), (), 0, False, None, None, None, message,
        hand_detected=True, depth_status='stereo_unusable' if len(points) == 0 else 'ready',
        hand_depth_error=reason, hand_world_points=points, handedness=side,
        acquisition_mode=acquisition))


@pytest.mark.parametrize('depth_error,message,expected', [
    ('虹膜三角定位失敗', '通用追蹤訊息', '虹膜三角定位失敗'),
    ('', '兩台鏡頭的左右手對應不一致', '兩台鏡頭的左右手對應不一致'),
    ('', '', '雙鏡頭尚未取得有效的 3D 手部關節，請依追蹤提示調整。'),
])
def test_paused_stereo_enrollment_shows_actual_geometry_failure_and_preserves_saved_gesture(session, depth_error, message, expected):
    window, clock, old, path, previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    for _ in range(16):
        deliver(window, clock, reason=depth_error, message=message)
    state = window.gesture_buffer.evaluate(clock['now'])
    assert state.count == 0 and not state.ready and state.reason == expected
    assert window.gesture_dialog.reason_label.text() == expected
    assert window.gesture_candidate is None and window.gesture_dialog.step == 0
    assert window.gesture_profile == old and window.camera.profile == old
    assert path.read_bytes() == previous


def test_second_hold_geometry_failure_keeps_candidate_and_displays_specific_reason(session):
    window, clock, old, path, previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    for _ in range(13):
        deliver(window, clock, points=personal_hand(), side='Right', message='舊手勢尚未通過')
    candidate = window.gesture_candidate
    assert candidate is not None and window.gesture_dialog.step == 1
    window.capture_gesture()
    for _ in range(16):
        deliver(window, clock, reason='雙鏡頭手部重建失敗')
    assert window.gesture_candidate is candidate
    assert window.gesture_buffer.evaluate(clock['now']).count == 0
    assert window.gesture_dialog.reason_label.text() == '雙鏡頭手部重建失敗'
    assert window.gesture_dialog.match_label.text() == '雙鏡頭手部重建失敗'
    assert window.gesture_profile == old and path.read_bytes() == previous


def test_monocular_missing_hand_keeps_existing_handedness_instruction(session):
    window, clock, _old, _path, _previous = session
    previous = window.acquisition.blockSignals(True)
    window.acquisition.setCurrentIndex(window.acquisition.findData('single'))
    window.acquisition.blockSignals(previous)
    window.begin_gesture_calibration()
    window.capture_gesture()
    deliver(window, clock, reason='不適用的深度訊息', message='不適用的單鏡頭訊息', acquisition='single')
    assert '尚未辨識左右手' in window.gesture_buffer.evaluate(clock['now']).reason


def test_each_stereo_start_receives_unique_local_numeric_diagnostic_path(session, tmp_path):
    window, _clock, _old, _path, _previous = session
    first = window.camera.kwargs['diagnostics_path']
    assert first.parent == tmp_path / '.kon/stereo-live-diagnostics'
    assert first.suffix == '.json' and len(first.stem) == 32
    assert not first.exists()  # Inert workers never write diagnostics or open hardware.
    window.stop_sensors()
    window.toggle_sensors()
    second = window.camera.kwargs['diagnostics_path']
    assert second.parent == first.parent and second != first and not second.exists()
