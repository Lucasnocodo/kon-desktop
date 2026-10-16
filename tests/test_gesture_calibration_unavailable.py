"""Incomplete hand observations cannot train or replace a saved template."""
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


@pytest.mark.parametrize('points,side', [(None, 'unknown'), (None, 'right'), (personal_hand(), 'right')])
def test_unavailable_geometry_reason_precedes_handedness_and_never_adds_samples(points, side):
    buffer = GestureEnrollmentBuffer()
    for index in range(16):
        assert buffer.observe(index / 15, points, handedness=side,
                              unavailable_reason='手部關節資料缺失')
    state = buffer.evaluate(1)
    assert not state.ready and state.count == 0
    assert state.reason == '手部關節資料缺失'


def test_failure_blocks_ready_hold_without_incrementing_then_fresh_valid_geometry_can_recover():
    buffer = GestureEnrollmentBuffer()
    for index in range(13):
        buffer.observe(index * .1, personal_hand(), handedness='right')
    assert buffer.evaluate(1.2).ready
    previous_count = buffer.evaluate(1.2).count
    buffer.observe(1.3, None, handedness='unknown', unavailable_reason='手部關節資料無效')
    failed = buffer.evaluate(1.3)
    assert not failed.ready and failed.count == previous_count
    assert failed.reason == '手部關節資料無效'
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
    monkeypatch.setattr('kon.app.SpeechWorker', FakeWorker)
    old = profile()
    path = gesture_profile_path(tmp_path)
    old.save(path)
    previous = path.read_bytes()
    clock = {'now': 100.0}
    monkeypatch.setattr('kon.app.time.monotonic', lambda: clock['now'])
    window = Window(aim_mode='eye_ray')
    window.toggle_sensors()
    yield window, clock, old, path, previous
    window.close()
    application.processEvents()


def deliver(window, clock, *, points=(), reason='', message='', side='unknown'):
    clock['now'] += .1
    window.on_tracking(TrackingFrame(
        clock['now'] - .01, (), (), 0, False, None, None, None, message,
        hand_detected=len(points) == 21, depth_status='hand_depth_unusable' if len(points) != 21 else 'eye_unusable',
        hand_depth_error=reason, hand_world_points=points, handedness=side))


@pytest.mark.parametrize('side,expected', [
    ('unknown', '尚未辨識左右手'),
    ('Right', '需要完整且有效的 21 個手部關節'),
])
def test_incomplete_enrollment_shows_actual_problem_and_preserves_saved_gesture(session, side, expected):
    window, clock, old, path, previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    for _ in range(16):
        deliver(window, clock, side=side)
    state = window.gesture_buffer.evaluate(clock['now'])
    assert state.count == 0 and not state.ready and expected in state.reason
    assert expected in window.gesture_dialog.reason_label.text()
    assert window.gesture_candidate is None and window.gesture_dialog.step == 0
    assert window.gesture_profile == old and window.camera.profile == old
    assert path.read_bytes() == previous


def test_hand_shape_enrollment_does_not_require_usable_eye_ray(session):
    window, clock, old, path, previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    for _ in range(13):
        deliver(window, clock, points=personal_hand(), side='Right', message='慣用眼尚未取得')
    assert window.gesture_candidate is not None and window.gesture_dialog.step == 1
    assert window.gesture_profile == old and path.read_bytes() == previous
    assert not window.voice_active and not window.armed


def test_second_hold_missing_hand_keeps_candidate_and_requires_new_valid_observations(session):
    window, clock, old, path, previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    for _ in range(13):
        deliver(window, clock, points=personal_hand(), side='Right')
    candidate = window.gesture_candidate
    assert candidate is not None and window.gesture_dialog.step == 1
    window.capture_gesture()
    for _ in range(16):
        deliver(window, clock, side='Right')
    assert window.gesture_candidate is candidate
    assert window.gesture_buffer.evaluate(clock['now']).count == 0
    assert '21 個手部關節' in window.gesture_dialog.reason_label.text()
    assert '等待新的完整手部畫面' in window.gesture_dialog.match_label.text()
    assert window.gesture_profile == old and path.read_bytes() == previous


def test_missing_hand_keeps_existing_handedness_instruction(session):
    window, clock, _old, _path, _previous = session
    window.begin_gesture_calibration()
    window.capture_gesture()
    deliver(window, clock, reason='不適用的深度訊息', message='不適用的單鏡頭訊息')
    assert '尚未辨識左右手' in window.gesture_buffer.evaluate(clock['now']).reason
