"""Headless dual-camera app flows. Every sensor worker is inert Python code."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.stereo_settings import StereoRig
from test_stereo_settings import calibration, plane


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeWorker:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.started = self.active = False
        self.stopped = False
        type(self).instances.append(self)
        for name in ("frame", "tracking", "status", "error", "transcript", "keyword", "finished"):
            setattr(self, name, FakeSignal())

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started

    def stop(self):
        self.started = False
        self.stopped = True

    def wait(self, _timeout):
        return True

    def set_active(self, active):
        self.active = bool(active)

    def set_preview_enabled(self, _enabled):
        pass


class FakeMonoWorker(FakeWorker):
    instances = []


class FakeStereoWorker(FakeWorker):
    instances = []


class FakeSpeechWorker(FakeWorker):
    instances = []


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def environment(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.CameraWorker", FakeMonoWorker)
    monkeypatch.setattr("kon.app.StereoCameraWorker", FakeStereoWorker)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeSpeechWorker)
    for worker in (FakeMonoWorker, FakeStereoWorker, FakeSpeechWorker):
        worker.instances.clear()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Automatic camera access is forbidden in app tests")

    monkeypatch.setattr("kon.stereo_capture.cv2.VideoCapture", forbidden)
    monkeypatch.setattr("kon.app.list_camera_sources", forbidden)
    windows = []

    def make():
        window = Window(aim_mode="eye_ray")
        windows.append(window)
        return window

    yield make
    for window in reversed(windows):
        window.close()
    application.processEvents()


def select_stereo(window):
    window.acquisition.setCurrentIndex(window.acquisition.findData("stereo"))
    assert window.using_stereo


def install(window, *, baseline=.20):
    value = calibration(primary=window.camera_index.value(),
                        secondary=window.secondary_camera_index.value(), baseline=baseline)
    screen = plane()
    window.install_stereo_calibration(value, screen)
    assert window.stereo_rig is not None
    return value, screen


def test_control_room_defaults_to_single_and_never_constructs_workers(environment):
    window = environment()
    assert window.acquisition.currentData() == "single"
    assert not window.using_stereo
    assert window.camera is None and window.speech is None
    assert not window.running
    assert not FakeMonoWorker.instances and not FakeStereoWorker.instances and not FakeSpeechWorker.instances


def test_single_camera_startup_remains_available_without_stereo_rig(environment):
    window = environment()
    window.toggle_sensors()
    assert window.running
    assert isinstance(window.camera, FakeMonoWorker)
    assert isinstance(window.speech, FakeSpeechWorker)
    assert window.camera.started and window.speech.started
    assert window.camera.kwargs["aim_mode"] == "eye_ray"
    assert not FakeStereoWorker.instances
    window.stop_sensors()
    assert window.camera.stopped and window.speech.stopped


def test_stereo_selection_without_calibration_blocks_both_sensor_workers(environment):
    window = environment()
    select_stereo(window)
    window.toggle_sensors()
    assert not window.running
    assert window.camera is None and window.speech is None
    assert not FakeStereoWorker.instances and not FakeSpeechWorker.instances
    assert "校正" in window.notice.text()


def test_same_camera_index_cannot_open_stereo_calibration_or_tracking(environment):
    window = environment()
    select_stereo(window)
    window.secondary_camera_index.setValue(window.camera_index.value())
    window.begin_stereo_calibration()
    assert window.stereo_dialog is None
    assert "不同" in window.notice.text()
    window.toggle_sensors()
    assert not window.running
    assert not FakeStereoWorker.instances and not FakeSpeechWorker.instances


def test_saved_rig_requires_physical_setup_confirmation_before_starting(environment):
    window = environment()
    select_stereo(window)
    install(window)
    window.rig_confirm.setChecked(False)
    window.toggle_sensors()
    assert not window.running
    assert window.camera is None and window.speech is None
    assert "確認" in window.notice.text()
    assert not FakeStereoWorker.instances and not FakeSpeechWorker.instances


def test_confirmed_rig_starts_stereo_worker_with_saved_geometry_then_stops(environment):
    window = environment()
    select_stereo(window)
    camera, screen = install(window)
    assert window.rig_confirm.isChecked()
    window.toggle_sensors()
    assert window.running
    assert isinstance(window.camera, FakeStereoWorker)
    assert isinstance(window.speech, FakeSpeechWorker)
    assert not FakeMonoWorker.instances
    assert window.camera.args[0].calibration_id == camera.calibration_id
    assert window.camera.args[1].plane_id == screen.plane_id
    assert window.camera.args[2] == "right"
    assert window.camera.started and window.speech.started
    assert not window.acquisition.isEnabled()
    assert not window.rig_confirm.isEnabled()
    window.stop_sensors()
    assert not window.running
    assert window.camera.stopped and window.speech.stopped
    assert window.acquisition.isEnabled()


@pytest.mark.parametrize("source", ["primary", "secondary"])
def test_changing_either_source_invalidates_current_rig_and_confirmation(environment, source):
    window = environment()
    select_stereo(window)
    install(window)
    if source == "primary":
        window.camera_index.setValue(2)
    else:
        window.secondary_camera_index.setValue(2)
    assert window.stereo_rig is None
    assert not window.rig_confirm.isChecked()
    window.toggle_sensors()
    assert not window.running
    assert not FakeStereoWorker.instances and not FakeSpeechWorker.instances


def test_screen_change_stops_active_stereo_and_clears_confirmation(environment):
    window = environment()
    select_stereo(window)
    install(window)
    window.toggle_sensors()
    camera, speech = window.camera, window.speech
    window.screen_changed()
    assert not window.running
    assert camera.stopped and speech.stopped
    assert not window.rig_confirm.isChecked()
    assert not window.armed


def test_install_failure_preserves_saved_and_active_rig(environment, monkeypatch):
    window = environment()
    select_stereo(window)
    install(window)
    original = window.stereo_rig
    original_bytes = window.stereo_path().read_bytes()
    window.rig_confirm.setChecked(False)

    def denied(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(StereoRig, "save", denied)
    window.install_stereo_calibration(calibration(baseline=.25), plane())
    assert window.stereo_rig is original
    assert window.stereo_path().read_bytes() == original_bytes
    assert not window.rig_confirm.isChecked()
    assert "保留舊設定" in window.notice.text()


def test_restart_restores_acquisition_secondary_and_rig_without_accessing_devices(environment):
    window = environment()
    window.secondary_camera_index.setValue(3)
    select_stereo(window)
    camera, _ = install(window)
    restored = environment()
    assert restored.acquisition.currentData() == "stereo"
    assert restored.secondary_camera_index.value() == 3
    assert restored.stereo_rig.calibration.calibration_id == camera.calibration_id
    assert not restored.rig_confirm.isChecked()
    assert restored.camera is None and restored.speech is None
    assert not FakeStereoWorker.instances and not FakeSpeechWorker.instances


def test_starting_sensors_preserves_stereo_choice_for_next_restart(environment, tmp_path):
    window = environment()
    window.secondary_camera_index.setValue(3)
    select_stereo(window)
    camera, _ = install(window)
    window.toggle_sensors()
    assert window.running
    window.stop_sensors()
    saved = json.loads((tmp_path / ".kon/camera-settings.json").read_text())
    assert saved["acquisition_mode"] == "stereo"
    assert saved["secondary_camera"] == 3
    restored = environment()
    assert restored.using_stereo
    assert restored.secondary_camera_index.value() == 3
    assert restored.stereo_rig.calibration.calibration_id == camera.calibration_id


def test_mono_and_stereo_app_configuration_have_separate_identity(environment):
    window = environment()
    mono = window.config()
    assert mono.get("acquisition_mode", "single") == "single"
    select_stereo(window)
    value, screen = install(window)
    stereo = window.config()
    assert stereo["acquisition_mode"] == "stereo"
    assert stereo["secondary_camera"] == 1
    assert stereo["stereo_calibration_id"] == value.calibration_id
    assert stereo["screen_plane_id"] == screen.plane_id
    window.acquisition.setCurrentIndex(window.acquisition.findData("single"))
    assert window.config() == mono


def test_restart_restores_camera_indices_above_nine(environment):
    window = environment()
    window.camera_index.setValue(12)
    window.secondary_camera_index.setValue(13)
    select_stereo(window)
    value, _ = install(window)
    restored = environment()
    assert restored.using_stereo
    assert restored.camera_index.value() == 12
    assert restored.secondary_camera_index.value() == 13
    assert restored.stereo_rig.calibration.calibration_id == value.calibration_id
    assert restored.camera is None and restored.speech is None
