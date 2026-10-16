"""Real calibration dialog sequencing with fake frames and no camera access."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import json
import threading
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import kon.stereo_dialog as module
from kon.stereo import ScreenPlane
from kon.stereo_capture import CameraSource, CapturedFrame, FramePair
from kon.stereo_samples import StereoSampleSession


@pytest.fixture(scope='module')
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application, tmp_path):
    value = module.StereoCalibrationDialog(0, 1, screen_size_mm=(600, 340), checkpoint_dir=tmp_path / 'samples')
    yield value
    value.reject()
    application.processEvents()


def _pair(first, second):
    image = np.zeros((90, 120, 3), np.uint8)
    return FramePair(CapturedFrame(first, image, 1), CapturedFrame(second, image, 1))


def _sample(shift=0):
    points = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(np.float32) * 20 + 100
    points[:, 0] += shift
    return {'primary': points, 'secondary': points - (10, 0),
            'primary_size': (640, 480), 'secondary_size': (640, 480),
            'primary_id': 'usb-camera', 'secondary_id': 'iphone-camera'}


def _wait_for(application, predicate, timeout=2000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(5)
        application.processEvents()
    assert predicate()


def test_constructor_opens_no_camera_or_discovery(application, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Constructing a dialog must not access cameras')
    monkeypatch.setattr(module, 'DualCapture', forbidden)
    monkeypatch.setattr(module, 'list_camera_sources', forbidden)
    value = module.StereoCalibrationDialog(0, 1)
    assert value._capture_worker is None
    assert value.calibration_result is None and value.screen_plane_result is None
    assert not value.record_button.isEnabled()
    value.reject()


def test_fresh_sample_requires_both_frames_after_click_and_in_sync():
    request = module._FreshSampleRequest()
    assert request.request(10)
    assert not request.request(10.1)
    assert not request.is_fresh(_pair(9.999, 10.010), 10.05)
    assert not request.is_fresh(_pair(10.001, 10.030), 10.05)
    assert not request.is_fresh(_pair(10.001, 10.010), 10.4)
    assert not request.is_fresh(_pair(10.101, 10.110), 10.05)
    assert request.is_fresh(_pair(10.001, 10.010), 10.05)
    request.cancel()
    assert not request.is_fresh(_pair(10.001, 10.010), 10.05)
    assert not request.finish(10)
    assert request.request(11)
    assert not request.finish(10)
    assert request.pending_at == 11


def test_capture_worker_skips_pair_read_before_click(application, monkeypatch):
    worker = module._CaptureWorker(0, 1, 'usb', 'phone')
    samples, failures, captures = [], [], []
    stale = _pair(time.monotonic() - .02, time.monotonic() - .015)
    fresh_times = []

    class FakeCapture:
        failure = ''
        def __init__(self, primary, secondary, **kwargs):
            assert primary.device_id == 'usb' and secondary.device_id == 'phone'
            self.calls = 0
            captures.append(self)
            self.closed = False
        def start(self):
            pass
        def take(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return stale
            if self.calls == 2:
                now = time.monotonic()
                fresh_times.append(now)
                return _pair(now, now)
            worker.stop()
            return None
        def close(self):
            self.closed = True
            return True

    monkeypatch.setattr(module, 'DualCapture', FakeCapture)
    monkeypatch.setattr(module, 'list_camera_sources', lambda: [CameraSource(0, 'USB', 'usb'), CameraSource(1, 'Phone', 'phone')])
    corners = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(np.float32)
    monkeypatch.setattr(module, 'detect_checkerboard', lambda *args: corners)
    worker.sample_ready.connect(samples.append)
    worker.failed.connect(failures.append)
    assert worker.request_sample()
    worker.start()
    _wait_for(application, lambda: not worker.isRunning())
    application.processEvents()
    assert captures[0].closed and captures[0].calls == 3
    assert failures == [] and len(samples) == 1
    assert samples[0]['primary_id'] == 'usb'
    assert samples[0]['secondary_id'] == 'phone'
    assert samples[0]['primary_size'] == (120, 90)


def test_capture_rejects_changed_device_before_open(application, monkeypatch):
    monkeypatch.setattr(module, 'list_camera_sources', lambda: [CameraSource(0, 'Other', 'other'), CameraSource(1, 'Phone', 'phone')])
    calls = []
    monkeypatch.setattr(module, 'DualCapture', lambda *args, **kwargs: calls.append(args))
    worker = module._CaptureWorker(0, 1, 'usb', 'phone')
    errors = []
    worker.failed.connect(errors.append)
    worker.start()
    _wait_for(application, lambda: not worker.isRunning())
    application.processEvents()
    assert calls == []
    assert errors and '鏡頭編號已改變' in errors[0]


def test_record_rejects_duplicate_pose_and_clears_solve(dialog):
    dialog._record_pending = True
    dialog._record_pair(_sample())
    assert len(dialog._samples) == 1
    dialog._record_pending = True
    dialog._record_pair(_sample(2))
    assert len(dialog._samples) == 1
    assert '太接近' in dialog.status_label.text()
    dialog._calibration = SimpleNamespace(rms=.4)
    dialog._record_pending = True
    dialog._record_pair(_sample(60))
    assert len(dialog._samples) == 2 and dialog._calibration is None
    assert not dialog.solve_button.isEnabled()


def test_square_size_change_discards_collected_data(dialog):
    dialog._samples.append(_sample())
    dialog._calibration = SimpleNamespace(rms=.4)
    dialog.square_mm.setValue(20)
    assert dialog._samples == [] and dialog._calibration is None
    assert '0 / 15' in dialog.progress_label.text()


def test_screen_measurements_converted_to_meters_and_reconfirmation_required(dialog):
    fields = dialog.plane_fields
    assert fields['width_m'].value() == 60
    assert fields['height_m'].value() == 34
    assert fields['camera_x_m'].value() == 30
    assert fields['camera_y_m'].value() == -2
    dialog.measurements_confirmed.setChecked(True)
    fields['camera_y_m'].setValue(-3.5)
    fields['camera_forward_m'].setValue(2)
    fields['yaw_degrees'].setValue(4)
    assert not dialog.measurements_confirmed.isChecked()
    measurements = dialog._make_screen_plane().measurements()
    assert measurements['width_m'] == pytest.approx(.6)
    assert measurements['height_m'] == pytest.approx(.34)
    assert measurements['camera_y_m'] == pytest.approx(-.035)
    assert measurements['camera_forward_m'] == pytest.approx(.02)
    assert measurements['yaw_degrees'] == pytest.approx(4)


def test_existing_measured_plane_is_editable(application):
    plane = ScreenPlane.from_measurements(.58, .33, .29, -.04, .015,
                                          pitch_degrees=3, yaw_degrees=-4, roll_degrees=2)
    value = module.StereoCalibrationDialog(0, 1, initial_plane=plane)
    restored = value._make_screen_plane()
    assert restored.top_left_m == pytest.approx(plane.top_left_m)
    assert restored.right_unit == pytest.approx(plane.right_unit)
    assert not value.measurements_confirmed.isChecked()
    value.reject()


def test_cannot_apply_until_solved_and_measurements_confirmed(dialog):
    emitted = []
    dialog.calibrated.connect(lambda *args: emitted.append(args))
    dialog.accept()
    assert emitted == [] and dialog.calibration_result is None
    calibration = SimpleNamespace(rms=.25)
    dialog._solved(calibration)
    assert not dialog.accept_button.isEnabled()
    assert '0.250 px' in dialog.status_label.text()
    assert '不代表實際瞄準誤差' in dialog.status_label.text()
    dialog.measurements_confirmed.setChecked(True)
    assert dialog.accept_button.isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.Accepted
    assert emitted == [(calibration, dialog.screen_plane_result)]
    assert dialog.calibration_result is calibration


def test_cancel_waits_for_camera_owner_before_dialog_finishes(dialog, application, monkeypatch):
    opened, release = threading.Event(), threading.Event()
    instances = []

    class BlockingCleanupCapture:
        failure = ''
        def __init__(self, *args, **kwargs):
            self.stop = kwargs['stop_event']
            self.closes = 0
            instances.append(self)
        def start(self):
            opened.set()
        def take(self, timeout):
            self.stop.wait(.005)
            return None
        def close(self):
            self.closes += 1
            return release.is_set()

    monkeypatch.setattr(module, 'DualCapture', BlockingCleanupCapture)
    monkeypatch.setattr(module, 'list_camera_sources', lambda: [CameraSource(0, 'USB', 'usb'), CameraSource(1, 'Phone', 'phone')])
    rejected, applied = [], []
    dialog.rejected.connect(lambda: rejected.append(1))
    dialog.calibrated.connect(lambda *args: applied.append(args))
    dialog.show()
    dialog._toggle_preview()
    _wait_for(application, opened.is_set)
    worker = dialog._capture_worker
    dialog._request_sample()
    assert dialog._record_pending
    dialog.reject()
    QTest.qWait(20)
    application.processEvents()
    assert rejected == [] and applied == []
    assert dialog._capture_worker is worker and worker.isRunning()
    assert worker.requests.pending_at is None
    assert not dialog.cancel_button.isEnabled()
    release.set()
    _wait_for(application, lambda: dialog._capture_worker is None)
    assert rejected == [1] and applied == []
    assert dialog.calibration_result is None and not dialog.isVisible()
    assert instances[0].closes >= 2


def test_solver_uses_measured_square_and_preserves_camera_identity(application, monkeypatch):
    captured = []
    result = SimpleNamespace(rms=.1)
    def solve(objects, first, second, size1, size2, **kwargs):
        captured.append((objects, first, second, size1, size2, kwargs))
        return result
    monkeypatch.setattr(module, 'solve_stereo_calibration', solve)
    samples = [_sample(index * 35) for index in range(15)]
    worker = module._SolveWorker(samples, 20, 0, 1)
    results = []
    worker.succeeded.connect(results.append)
    worker.start()
    _wait_for(application, lambda: not worker.isRunning())
    application.processEvents()
    assert results == [result]
    objects, first, second, size1, size2, kwargs = captured[0]
    assert len(objects) == 15
    assert np.max(objects[0][:, 0]) == pytest.approx(.16)
    assert kwargs['primary_id'] == 'usb-camera'
    assert kwargs['secondary_id'] == 'iphone-camera'
    assert size1 == size2 == (640, 480)


def test_cancel_during_solver_discards_result_and_waits_for_thread(dialog, application, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_solver(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(rms=.1)

    monkeypatch.setattr(module, 'solve_stereo_calibration', slow_solver)
    dialog._samples = [_sample(index * 35) for index in range(15)]
    dialog._update_controls()
    applied, rejected = [], []
    dialog.calibrated.connect(lambda *args: applied.append(args))
    dialog.rejected.connect(lambda: rejected.append(1))
    dialog.show()
    dialog._begin_solve()
    _wait_for(application, entered.is_set)
    dialog.reject()
    application.processEvents()
    assert dialog._solve_worker is not None and rejected == []
    assert dialog.calibration_result is None
    release.set()
    _wait_for(application, lambda: dialog._solve_worker is None)
    assert rejected == [1] and applied == []
    assert dialog._calibration is None and dialog.calibration_result is None


def test_export_board_has_detectable_9_by_6_corners_and_physical_scale():
    image = module.checkerboard_print_image(25, pixels_per_square=64)
    pointer = image.bits()
    pixels = np.frombuffer(pointer, np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()]
    found, corners = cv2.findChessboardCorners(pixels, (9, 6))
    assert found and len(corners) == 54
    assert 64 / image.dotsPerMeterX() * 1000 == pytest.approx(25, abs=.01)
    assert image.dotsPerMeterX() == image.dotsPerMeterY()


def accept_sample(dialog, sample=None):
    dialog._record_pending = True
    dialog._record_pair(_sample() if sample is None else sample)


def setup_load(dialog, tmp_path, monkeypatch, *, sources=None, confirmed=QMessageBox.Yes, session=None):
    session = session or StereoSampleSession(23, 4, 7, (_sample(),))
    path = tmp_path / 'restore.json'
    session.save(path)
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    discovery = Mock(return_value=sources if sources is not None else
                     [CameraSource(0, 'USB', 'usb-camera'), CameraSource(1, 'Camo', 'iphone-camera')])
    monkeypatch.setattr(module, 'list_camera_sources', discovery)
    question = Mock(return_value=confirmed)
    monkeypatch.setattr(module.QMessageBox, 'question', question)
    return path, discovery, question


def test_accepted_pair_checkpoints_and_cancel_keeps_corner_data(dialog):
    path = dialog.checkpoint_path
    assert not path.exists()
    accept_sample(dialog)
    session = StereoSampleSession.load(path)
    assert len(session.samples) == 1
    assert session.samples[0]['primary_id'] == 'usb-camera'
    assert session.square_mm == 25
    assert '已自動保存 1' in dialog.checkpoint_label.text()
    previous = path.read_bytes()
    accept_sample(dialog, _sample(2))
    assert path.read_bytes() == previous  # Duplicate poses never replace the checkpoint.
    dialog.reject()
    assert path.read_bytes() == previous


def test_clear_starts_another_session_without_destroying_saved_pairs(dialog):
    accept_sample(dialog)
    old_path = dialog.checkpoint_path
    previous = old_path.read_bytes()
    dialog.square_mm.setValue(23)
    assert old_path.read_bytes() == previous
    assert dialog.checkpoint_path != old_path and not dialog.checkpoint_path.exists()
    accept_sample(dialog, _sample(60))
    assert StereoSampleSession.load(dialog.checkpoint_path).square_mm == 23
    assert old_path.read_bytes() == previous


def test_checkpoint_io_failure_preserves_accepted_memory_and_warns(dialog, monkeypatch):
    monkeypatch.setattr(module.StereoSampleSession, 'save', Mock(side_effect=OSError('disk full')))
    accept_sample(dialog)
    assert len(dialog._samples) == 1 and dialog._calibration is None
    assert '自動保存失敗' in dialog.status_label.text()
    assert 'disk full' in dialog.checkpoint_label.text()


def test_explicit_save_exports_current_corners_without_capturing(dialog, tmp_path, monkeypatch):
    accept_sample(dialog)
    destination = tmp_path / 'my-corners'
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', lambda *args: (str(destination), ''))
    forbidden = Mock(side_effect=AssertionError('Save must not open cameras'))
    monkeypatch.setattr(module, 'DualCapture', forbidden)
    dialog.save_samples_button.click()
    saved = StereoSampleSession.load(destination.with_suffix('.json'))
    assert len(saved.samples) == 1
    np.testing.assert_allclose(saved.samples[0]['primary'], dialog._samples[0]['primary'])
    forbidden.assert_not_called()


def test_confirmed_load_matches_native_identity_not_old_indices_and_does_not_apply(dialog, tmp_path, monkeypatch):
    accept_sample(dialog, _sample(60))
    old_checkpoint = dialog.checkpoint_path
    dialog._calibration = SimpleNamespace(rms=.1)
    dialog.measurements_confirmed.setChecked(True)
    _path, discovery, question = setup_load(dialog, tmp_path, monkeypatch)
    forbidden = Mock(side_effect=AssertionError('Load must not open cameras'))
    monkeypatch.setattr(module, 'DualCapture', forbidden)

    dialog.load_samples_button.click()

    discovery.assert_called_once()
    question.assert_called_once()
    assert 'Camo' in question.call_args.args[2]
    assert dialog.square_mm.value() == 23
    assert dialog.primary_source == 0 and dialog.secondary_source == 1
    assert dialog.primary_id == 'usb-camera' and dialog.secondary_id == 'iphone-camera'
    assert dialog._calibration is None and dialog.calibration_result is None
    assert not dialog.measurements_confirmed.isChecked()
    assert len(dialog._samples) == 1
    np.testing.assert_allclose(dialog._samples[0]['primary'], _sample()['primary'])
    assert old_checkpoint.exists() and dialog.checkpoint_path != old_checkpoint
    assert StereoSampleSession.load(dialog.checkpoint_path).primary_source == 0
    forbidden.assert_not_called()


@pytest.mark.parametrize('failure', ['decline', 'different_device', 'missing_id', 'invalid_file'])
def test_unconfirmed_or_invalid_load_preserves_all_current_calibration_memory(dialog, tmp_path, monkeypatch, failure):
    accept_sample(dialog, _sample(60))
    old_samples, old_path = dialog._samples, dialog.checkpoint_path
    old_calibration = SimpleNamespace(rms=.1)
    dialog._calibration = old_calibration
    dialog.measurements_confirmed.setChecked(True)
    sources = [CameraSource(0, 'USB', 'usb-camera'), CameraSource(1, 'Camo', 'iphone-camera')]
    if failure == 'different_device':
        sources[1] = CameraSource(1, 'New Camo', 'new-device')
    if failure == 'missing_id':
        sources[1] = CameraSource(1, 'Camo', '')
    path, _discovery, question = setup_load(dialog, tmp_path, monkeypatch, sources=sources,
                                           confirmed=QMessageBox.No if failure == 'decline' else QMessageBox.Yes)
    if failure == 'invalid_file':
        payload = json.loads(path.read_text())
        payload['samples'][0]['primary'][0] = [-1, 0]
        path.write_text(json.dumps(payload))

    dialog._load_samples()

    assert dialog._samples is old_samples and dialog._calibration is old_calibration
    assert dialog.checkpoint_path == old_path and dialog.square_mm.value() == 25
    assert dialog.measurements_confirmed.isChecked()
    if failure != 'decline':
        question.assert_not_called()


def test_load_is_disabled_during_preview_and_never_interrupts_capture(dialog, monkeypatch):
    worker = SimpleNamespace(stop=Mock())
    dialog._capture_worker = worker
    dialog._update_controls()
    pick = Mock()
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', pick)
    try:
        assert not dialog.load_samples_button.isEnabled()
        dialog._load_samples()
        pick.assert_not_called()
        worker.stop.assert_not_called()
    finally:
        dialog._capture_worker = None


@pytest.mark.parametrize('change', ['size', 'identity'])
def test_first_new_pair_after_load_checks_actual_capture_size_and_identity(dialog, tmp_path, monkeypatch, change):
    setup_load(dialog, tmp_path, monkeypatch)
    dialog._load_samples()
    previous = dialog.checkpoint_path.read_bytes()
    sample = _sample(60)
    if change == 'size':
        sample['secondary_size'] = (800, 600)
    else:
        sample['secondary_id'] = 'another-camera'
    worker = SimpleNamespace(stop=Mock())
    dialog._capture_worker = worker
    try:
        accept_sample(dialog, sample)
        worker.stop.assert_called_once()
        assert len(dialog._samples) == 1
        assert dialog.checkpoint_path.read_bytes() == previous
        assert '尺寸已改變' in dialog.status_label.text()
    finally:
        dialog._capture_worker = None
        dialog._capture_stopping = False


def test_solver_failure_persists_numeric_diagnostics_and_keeps_checkpoint(dialog, monkeypatch):
    accept_sample(dialog)
    checkpoint = dialog.checkpoint_path.read_bytes()
    diagnostics = {'schema_version': 1, 'sample_count': 1, 'primary_rms': 1.6,
                   'secondary_rms': .4, 'stereo_rms': 1.2, 'failures': [{'sample_id': 1, 'value': 1.6}]}

    class QualityError(ValueError):
        pass

    error = QualityError('主鏡頭 RMS 1.6 px 超過 1.5 px')
    error.diagnostics = diagnostics
    monkeypatch.setattr(module, 'solve_stereo_calibration', Mock(side_effect=error))
    worker = module._SolveWorker(dialog._samples, 25, 0, 1, checkpoint_path=dialog.checkpoint_path)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert worker.diagnostics_path.exists()
    report = json.loads(worker.diagnostics_path.read_text())
    assert report['diagnostics'] == diagnostics
    assert report['sample_checkpoint'] == str(dialog.checkpoint_path)
    assert dialog.checkpoint_path.read_bytes() == checkpoint
    assert len(dialog._samples) == 1 and '1.6 px' in failures[0] and '數值診斷已保存' in failures[0]
