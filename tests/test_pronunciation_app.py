"""Real pronunciation dialog interactions with local sensors replaced by signals."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog

from kon.app import Window
from kon.calibration_store import gesture_profile_path, save_calibration
from kon.geometry import ScreenCalibration
from kon.pronunciation import PronunciationProfile
from kon.vision import TrackingFrame
from test_gesture_calibration import profile as gesture_profile


class FakeSpeech(QObject):
    sample = Signal(int, str)
    sample_failed = Signal(int, str)
    status = Signal(str)
    error = Signal(str)
    finished = Signal()
    keyword = Signal(float, str)
    transcript = Signal(str)
    instances = []

    def __init__(self, root=None, parent=None, **configuration):
        super().__init__(parent)
        self.configuration = configuration
        self.requests = []
        self.configurations = []
        self.activations = []
        self.cancel_count = self.stop_count = 0
        self.waits = []
        self.pending = None
        self.running = self.active = False
        self.stopping = False
        self.hold_after_stop = False
        self.instances.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def request_sample(self, request, *, language):
        if self.stopping:
            return False
        self.requests.append((request, language))
        self.pending = request
        self.active = True
        return True

    def deliver(self, request, text):
        if request == self.pending:
            self.pending = None
            self.active = False
        self.sample.emit(request, text)

    def cancel_sample(self):
        self.cancel_count += 1
        self.pending = None
        self.active = False

    def configure_trigger(self, setting, **configuration):
        assert not self.active
        self.configurations.append((setting, configuration))

    def set_active(self, enabled):
        self.activations.append(enabled)
        self.active = enabled

    def stop(self):
        self.stop_count += 1
        self.stopping = True
        self.active = False
        if not self.hold_after_stop:
            self.finish()

    def finish(self):
        was_running = self.running
        self.running = False
        if was_running:
            self.finished.emit()

    def wait(self, timeout):
        self.waits.append(timeout)
        return not self.running


class FakeCamera(QObject):
    frame = Signal(object)
    tracking = Signal(object)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def isRunning(self):
        return self.running

    def wait(self, timeout):
        return not self.running


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(application, monkeypatch, tmp_path):
    FakeSpeech.instances = []
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    monkeypatch.setattr("kon.app.SpeechWorker", FakeSpeech)

    def unexpected_camera(*args, **kwargs):
        pytest.fail("Pronunciation enrollment opened a camera")

    monkeypatch.setattr("kon.app.CameraWorker", unexpected_camera)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window = Window()
    yield window, clock
    window.stop_sensors()
    for worker in FakeSpeech.instances:
        worker.hold_after_stop = False
        worker.finish()
    window.close()
    application.processEvents()


def click(widget):
    assert widget.isEnabled()
    if isinstance(widget, QCheckBox):
        QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(9, widget.height() // 2))
    else:
        QTest.mouseClick(widget, Qt.LeftButton)
    QApplication.instance().processEvents()


def begin(window, phrase="kon", language="en"):
    click(window.pronunciation_button)
    dialog = window.pronunciation_dialog
    assert dialog is not None and dialog.isVisible()
    dialog.phrase_input.setText(phrase)
    dialog.language_combo.setCurrentIndex(dialog.language_combo.findData(language))
    return dialog


def record(window, dialog, text):
    click(dialog.record_button)
    worker = window.pronunciation_worker
    request = window.pronunciation_request[0]
    assert dialog.pending and not dialog.phrase_input.isEnabled()
    worker.deliver(request, text)
    QApplication.instance().processEvents()
    return worker, request


def collect(window, dialog, samples=("Kon!", "Con.", "Go")):
    for text in samples:
        worker, _request = record(window, dialog, text)
    assert len(window.pronunciation_samples) == len(dialog.samples) == 3
    assert not dialog.record_button.isEnabled()
    assert not dialog.test_button.isEnabled()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.selected_aliases()
    return worker


def select(dialog, *aliases):
    for checkbox in dialog.candidate_checkboxes:
        if checkbox.text() in aliases:
            click(checkbox)
    assert dialog.selected_aliases() == aliases


def verify(window, dialog, heard):
    click(dialog.test_button)
    request = window.pronunciation_request[0]
    window.pronunciation_worker.deliver(request, heard)
    QApplication.instance().processEvents()


def existing_profile(root):
    value = PronunciationProfile.create("invoke", "en", ["invoke"] * 3, ["invoke"], "invoke")
    value.save(root)
    return value


def test_independent_microphone_enrollment_saves_selected_aliases_and_preserves_other_calibration(session, tmp_path):
    window, _clock = session
    points = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    aim = window.calibration = ScreenCalibration.fit(points, points)
    gesture = window.gesture_profile = gesture_profile()
    save_calibration(tmp_path, aim, window.config())
    gesture.save(gesture_profile_path(tmp_path))
    preserved = {path: path.read_bytes() for path in (tmp_path / ".kon").rglob("*") if path.is_file()}
    preferences_path = tmp_path / ".kon/preferences.json"
    preferences = {"camera_settings": {"eye": "left", "camera": 2, "fov": 79.5}, "other": {"keep": True}}
    preferences_path.write_text(json.dumps(preferences))
    dialog = begin(window)
    assert window.camera is window.speech is None and not window.running
    assert FakeSpeech.instances == []
    worker = collect(window, dialog)
    assert len(FakeSpeech.instances) == 1
    assert worker.requests == [(1, "en"), (2, "en"), (3, "en")]
    select(dialog, "kon", "con")
    verify(window, dialog, "Go")
    assert not dialog.apply_button.isEnabled() and window.pronunciation_verification is None
    verify(window, dialog, "Con!")
    assert dialog.apply_button.isEnabled()
    click(dialog.apply_button)
    assert not window.pronunciation_calibrating and dialog.result() == QDialog.Accepted
    assert window.trigger_setting == "kon, con" and window.trigger_language == "en"
    assert window.calibration is aim and window.gesture_profile is gesture
    assert worker.stop_count == 1 and window.pronunciation_worker is None
    assert all(path.read_bytes() == before for path, before in preserved.items())
    data = json.loads(preferences_path.read_text())
    assert data["camera_settings"] == preferences["camera_settings"]
    assert data["other"] == preferences["other"]
    assert PronunciationProfile.load(tmp_path) == window.pronunciation_profile
    assert window.pronunciation_phrases() == ("kon", "con")
    assert not list((tmp_path / ".kon").rglob("*.wav"))
    reloaded = Window()
    try:
        assert reloaded.pronunciation_profile == window.pronunciation_profile
        assert reloaded.trigger_setting == "kon, con" and reloaded.trigger_language == "en"
        assert "已保存" in reloaded.pronunciation_label.text()
    finally:
        reloaded.close()


def test_live_worker_is_reused_and_configured_with_exact_selected_phrases_and_language(session):
    window, _clock = session
    window.voice_trigger_mode_combo.setCurrentIndex(window.voice_trigger_mode_combo.findData("phrase"))
    window.running = True
    worker = window.speech = FakeSpeech()
    worker.start()
    window.voice_active = True
    dialog = begin(window, phrase="召喚狐狸", language="zh")
    assert worker.activations and not any(worker.activations)
    assert collect(window, dialog, ("召喚狐狸。", "招喚狐狸", "去")) is worker
    assert len(FakeSpeech.instances) == 1
    select(dialog, "招喚狐狸")
    verify(window, dialog, "招喚狐狸。")
    click(dialog.apply_button)
    assert worker.configurations == [
        ("召喚狐狸, 招喚狐狸", {"language": "zh", "accepted_phrases": ("召喚狐狸", "招喚狐狸"), "trigger_mode": "phrase"})
    ]
    assert not worker.active and worker.isRunning() and worker.stop_count == 0
    assert worker.requests == [(1, "zh"), (2, "zh"), (3, "zh"), (4, "zh")]
    assert not window.armed and not window.testing


def test_primary_phrase_remains_valid_when_only_a_recorded_alias_is_selected(session):
    window, _clock = session
    dialog = begin(window, phrase="summon")
    collect(window, dialog, ("Some on", "Some on", "Come on"))
    select(dialog, "some on")
    verify(window, dialog, "Summon!")
    assert dialog.apply_button.isEnabled()
    click(dialog.apply_button)
    assert window.pronunciation_phrases() == ("summon", "some on")


def test_calibration_disarms_file_actions_and_ignores_keyword_and_tracking(session, monkeypatch):
    window, clock = session
    window.running = window.armed = window.testing = True
    window.voice_active = True
    window.speech = FakeSpeech()
    window.speech.start()
    window.pending_action = (window.action_serial, "real-file", True)
    initial_serial = window.action_serial
    actions = []
    monkeypatch.setattr(window, "job", lambda *args: actions.append("job"))
    monkeypatch.setattr(window, "summon", lambda *args: actions.append("summon"))
    dialog = begin(window)
    assert not window.armed and not window.testing and not window.voice_active
    assert window.pending_action is None and window.action_serial > initial_serial
    record(window, dialog, "kon")
    window.voice_active = True  # Even a queued result from the former live session is ignored.
    window.voice_armed_at = clock["now"] - .1
    window.on_keyword(clock["now"], "kon")
    window.on_tracking(TrackingFrame(clock["now"], (), (), 1., True, None, None, (.5, .5), ""))
    window.set_voice(True)
    assert actions == [] and not window.speech.active
    assert window._jobs == set()


def test_cancel_preserves_saved_profile_and_late_result_cannot_enter_new_session(session, tmp_path):
    window, _clock = session
    original = existing_profile(tmp_path)
    window._load_preferences()
    before = (tmp_path / ".kon/preferences.json").read_bytes()
    dialog = begin(window)
    click(dialog.record_button)
    worker, request = window.pronunciation_worker, window.pronunciation_request[0]
    click(dialog.cancel_button)
    assert not window.pronunciation_calibrating and window.pronunciation_profile == original
    assert worker.stop_count == 1
    worker.deliver(request, "con")
    newer = begin(window)
    click(newer.record_button)
    pending = window.pronunciation_request
    worker.deliver(request, "con")
    worker.sample_failed.emit(request, "late failure")
    assert window.pronunciation_request == pending and window.pronunciation_samples == []
    assert (tmp_path / ".kon/preferences.json").read_bytes() == before


def test_restart_discards_pending_result_and_requires_three_new_samples(session):
    window, _clock = session
    dialog = begin(window)
    record(window, dialog, "kon")
    click(dialog.record_button)
    worker, old_request = window.pronunciation_worker, window.pronunciation_request[0]
    click(dialog.restart_button)
    assert window.pronunciation_samples == dialog.samples == []
    assert window.pronunciation_request is None and worker.cancel_count == 1
    click(dialog.record_button)
    new_request = window.pronunciation_request[0]
    worker.deliver(old_request, "con")
    worker.sample_failed.emit(old_request, "stale")
    assert window.pronunciation_request[0] == new_request and window.pronunciation_samples == []
    worker.deliver(new_request, "kon")
    assert window.pronunciation_samples == ["kon"] and not dialog.test_button.isEnabled()


def test_timeout_cancels_capture_and_keeps_previous_profile_for_retry(session, tmp_path):
    window, clock = session
    original = existing_profile(tmp_path)
    window._load_preferences()
    before = (tmp_path / ".kon/preferences.json").read_bytes()
    dialog = begin(window)
    click(dialog.record_button)
    worker, request = window.pronunciation_worker, window.pronunciation_request[0]
    clock["now"] += 30.1
    window.check_freshness()
    assert worker.cancel_count == 1 and window.pronunciation_request is None
    assert dialog.record_button.isEnabled() and not dialog.pending
    worker.deliver(request, "invoke")
    assert window.pronunciation_samples == [] and window.pronunciation_profile == original
    assert (tmp_path / ".kon/preferences.json").read_bytes() == before
    record(window, dialog, "invoke")
    assert window.pronunciation_samples == ["invoke"]


def test_storage_failure_unlocks_dialog_and_can_retry_without_new_verification(session, tmp_path, monkeypatch):
    window, _clock = session
    dialog = begin(window)
    collect(window, dialog)
    select(dialog, "con")
    verify(window, dialog, "con")
    original_save = PronunciationProfile.save

    def fail_save(self, root):
        raise OSError("disk unavailable")

    monkeypatch.setattr(PronunciationProfile, "save", fail_save)
    click(dialog.apply_button)
    assert window.pronunciation_calibrating and window.pronunciation_profile is None
    assert not dialog.pending and dialog.apply_button.isEnabled()
    assert dialog.cancel_button.isEnabled() and "disk unavailable" in dialog.feedback_label.text()
    assert not (tmp_path / ".kon/preferences.json").exists()
    monkeypatch.setattr(PronunciationProfile, "save", original_save)
    click(dialog.apply_button)
    assert not window.pronunciation_calibrating
    assert window.trigger_setting == "kon, con"


@pytest.mark.parametrize("edit", ["alias", "phrase", "language"])
def test_choice_edits_require_a_new_test_and_cannot_save_old_approval(session, tmp_path, edit):
    window, _clock = session
    dialog = begin(window)
    collect(window, dialog)
    select(dialog, "con")
    verify(window, dialog, "con")
    assert dialog.apply_button.isEnabled()
    if edit == "alias":
        checkbox = next(item for item in dialog.candidate_checkboxes if item.text() == "go")
        click(checkbox)
    elif edit == "phrase":
        dialog.phrase_input.setText("invoke")
    else:
        dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("zh"))
    assert not dialog.apply_button.isEnabled()
    window.save_pronunciation()
    assert window.pronunciation_calibrating and window.pronunciation_profile is None
    assert not (tmp_path / ".kon/preferences.json").exists()
    if edit != "alias":
        assert window.pronunciation_samples == dialog.samples == []


def test_unchanged_preferences_keep_enrollment_and_manual_trigger_edit_invalidates_it(session, tmp_path):
    window, _clock = session
    saved = existing_profile(tmp_path)
    window._load_preferences()
    assert window.trigger_changed()
    assert window.pronunciation_profile == saved and PronunciationProfile.load(tmp_path) == saved
    window.trigger_input.setText("go")
    assert window.trigger_changed()
    assert window.pronunciation_profile is None and window.trigger_language is None
    data = json.loads((tmp_path / ".kon/preferences.json").read_text())
    assert data["voice_trigger"] == "go"
    assert "voice_pronunciation" not in data and "voice_language" not in data


def test_stopping_previous_live_microphone_blocks_a_second_microphone(session):
    window, _clock = session
    old = window.speech = FakeSpeech()
    old.start()
    old.hold_after_stop = True
    old.stop()
    dialog = begin(window)
    click(dialog.record_button)
    assert len(FakeSpeech.instances) == 1 and window.pronunciation_worker is None
    assert window.pronunciation_request is None and not dialog.pending
    assert "停止" in dialog.feedback_label.text()
    old.finish()
    record(window, dialog, "kon")
    assert len(FakeSpeech.instances) == 2


def test_owned_worker_retirement_blocks_reopening_and_close_waits_for_it(session):
    window, _clock = session
    dialog = begin(window)
    click(dialog.record_button)
    worker = window.pronunciation_worker
    worker.hold_after_stop = True
    click(dialog.cancel_button)
    assert window._retiring_pronunciation_workers == [worker]
    another = begin(window)
    click(another.record_button)
    assert len(FakeSpeech.instances) == 1 and window.pronunciation_request is None
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and worker.waits == [2500]
    worker.finish()
    assert window._retiring_pronunciation_workers == []
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_sensor_start_passes_restored_exact_profile_to_worker(session, tmp_path, monkeypatch):
    window, _clock = session
    saved = PronunciationProfile.create("kon", "zh", ["kon"] * 3, ["kon"], "kon")
    saved.save(tmp_path)
    window._load_preferences()
    monkeypatch.setattr("kon.app.CameraWorker", FakeCamera)
    window.toggle_sensors()
    assert window.running
    assert window.speech.configuration == {
        "trigger_setting": "kon", "language": "zh", "accepted_phrases": ("kon",),
        "trigger_mode": "loud_sound",
        "microphone_setting": None,
    }
    assert PronunciationProfile.load(tmp_path) == saved
