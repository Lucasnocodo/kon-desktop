"""Device choices reach both live capture paths without touching audio hardware."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from test_pronunciation_app import FakeCamera, FakeSpeech


MICROPHONE = {"name": "EMEET C960", "hostapi": "Core Audio"}


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def environment(application, monkeypatch, tmp_path):
    FakeSpeech.instances = []
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeSpeech)
    monkeypatch.setattr("kon.app.CameraWorker", FakeCamera)
    devices = Mock(return_value=[])
    monkeypatch.setattr("kon.app.list_microphones", devices)
    stream = Mock(side_effect=AssertionError("App controls must never open an audio stream"))
    monkeypatch.setattr("kon.speech.sd.InputStream", stream)
    windows = []

    def make():
        window = Window()
        # Execute refresh jobs synchronously, with real Job signals and handlers.
        window.pool = SimpleNamespace(start=lambda job: job.run())
        windows.append(window)
        return window

    yield make, devices, stream
    for window in reversed(windows):
        window.stop_sensors()
        for worker in FakeSpeech.instances:
            worker.hold_after_stop = False
            worker.finish()
        window.close()
    application.processEvents()


def select_microphone(window, setting=MICROPHONE):
    window.microphone_combo.addItem(setting["name"], setting)
    window.microphone_combo.setCurrentIndex(window.microphone_combo.count() - 1)
    assert window.microphone_setting == setting


def test_opening_controls_does_not_enumerate_or_open_audio(environment):
    make, devices, stream = environment
    window = make()

    devices.assert_not_called()
    stream.assert_not_called()
    assert window.microphone_setting is None
    assert "iPhone" in window.microphone_combo.currentText()
    assert window.microphone_combo.isEnabled()
    assert not FakeSpeech.instances


def test_explicit_refresh_lists_choices_without_opening_capture(environment, monkeypatch):
    make, devices, stream = environment
    window = make()
    devices.return_value = [SimpleNamespace(index=4, name=MICROPHONE["name"])]
    monkeypatch.setattr("kon.app.encode_microphone", lambda _device: MICROPHONE)

    window.refresh_microphones_button.click()

    devices.assert_called_once()
    stream.assert_not_called()
    assert window.microphone_combo.count() == 2
    assert window.microphone_combo.itemData(1) == MICROPHONE
    assert window.microphone_setting is None


def test_selected_microphone_persists_and_reloads_without_enumeration(environment, tmp_path):
    make, devices, stream = environment
    window = make()
    select_microphone(window)
    path = tmp_path / ".kon/preferences.json"
    assert json.loads(path.read_text())["microphone"] == MICROPHONE

    reloaded = make()

    assert reloaded.microphone_setting == MICROPHONE
    assert reloaded.microphone_combo.currentData() == MICROPHONE
    devices.assert_not_called()
    stream.assert_not_called()


def test_missing_selected_device_stays_selected_instead_of_falling_back(environment):
    make, devices, _stream = environment
    window = make()
    select_microphone(window)
    devices.return_value = []

    window.refresh_microphones()

    assert window.microphone_setting == MICROPHONE
    assert window.microphone_combo.currentData() == MICROPHONE
    assert "未連接" in window.microphone_combo.currentText()


@pytest.mark.parametrize("selected", [False, True])
def test_normal_capture_receives_saved_microphone_and_locks_choice(environment, selected):
    make, _devices, _stream = environment
    window = make()
    if selected:
        select_microphone(window)

    window.toggle_sensors()

    assert window.running
    assert window.speech.configuration["microphone_setting"] == (MICROPHONE if selected else None)
    assert not window.microphone_combo.isEnabled()
    assert not window.refresh_microphones_button.isEnabled()
    window.stop_sensors()
    assert window.microphone_combo.isEnabled()
    assert window.refresh_microphones_button.isEnabled()


def test_pronunciation_capture_receives_saved_microphone_and_locks_choice(environment):
    make, _devices, _stream = environment
    window = make()
    select_microphone(window)

    window.begin_pronunciation_calibration()
    assert not window.microphone_combo.isEnabled()
    assert not window.refresh_microphones_button.isEnabled()
    window.capture_pronunciation()

    assert window.pronunciation_worker.configuration["microphone_setting"] == MICROPHONE
    assert window.pronunciation_worker.isRunning()
    window.cancel_pronunciation_calibration()
    assert window.microphone_combo.isEnabled()
    assert window.refresh_microphones_button.isEnabled()


@pytest.mark.parametrize("activity", ["normal", "pronunciation"])
def test_active_capture_cannot_be_switched_programmatically(environment, activity):
    make, devices, _stream = environment
    window = make()
    select_microphone(window)
    if activity == "normal":
        window.toggle_sensors()
    else:
        window.begin_pronunciation_calibration()

    window.microphone_combo.setCurrentIndex(0)
    window.refresh_microphones()

    assert window.microphone_setting == MICROPHONE
    assert window.microphone_combo.currentData() == MICROPHONE
    devices.assert_not_called()


def test_choice_stays_locked_until_previous_audio_worker_finishes(environment):
    make, _devices, _stream = environment
    window = make()
    window.begin_pronunciation_calibration()
    window.capture_pronunciation()
    worker = window.pronunciation_worker
    worker.hold_after_stop = True

    window.cancel_pronunciation_calibration()

    assert not window.microphone_combo.isEnabled()
    assert not window.refresh_microphones_button.isEnabled()
    worker.finish()
    assert window.microphone_combo.isEnabled()
    assert window.refresh_microphones_button.isEnabled()
