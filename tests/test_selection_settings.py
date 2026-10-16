"""Selection tuning validates saved data and never starts real sensors."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget
from kon.selection_settings import SelectionSettings, save_selection_settings


@pytest.fixture
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    value = Window(aim_mode="hand")
    yield value
    value.close()
    application.processEvents()


@pytest.mark.parametrize("field,value", [
    ("window_seconds", 0), ("window_seconds", 11),
    ("minimum_score", 0), ("minimum_score", 51),
    ("stable_seconds", .01), ("stable_seconds", 3),
    ("stable_multiplier", .9), ("stable_multiplier", 6),
    ("movement_speed_limit", .001), ("movement_speed_limit", .6),
    ("release_cells", -1), ("release_cells", 6), ("release_cells", 2.0),
    ("tracking_grace_seconds", .05), ("tracking_grace_seconds", 3),
    ("window_seconds", True), ("stable_seconds", "0.3"),
    ("stable_multiplier", float("nan")), ("minimum_score", float("inf")),
])
def test_invalid_saved_field_falls_back_without_discarding_other_valid_fields(field, value):
    saved = {"release_cells": 4, "stable_seconds": .7, field: value}
    settings = SelectionSettings.from_mapping(saved)
    assert getattr(settings, field) == getattr(SelectionSettings(), field)
    if field != "release_cells":
        assert settings.release_cells == 4
    if field != "stable_seconds":
        assert settings.stable_seconds == .7
    with pytest.raises(ValueError):
        replace(SelectionSettings(), **{field: value})


@pytest.mark.parametrize("saved", [None, [], "bad", True])
def test_non_object_preferences_use_defaults(saved):
    assert SelectionSettings.from_mapping(saved) == SelectionSettings()


def test_weighted_threshold_can_exceed_real_window_but_not_maximum_weighted_score():
    assert SelectionSettings(window_seconds=2, minimum_score=3, stable_multiplier=2).minimum_score == 3
    with pytest.raises(ValueError):
        SelectionSettings(window_seconds=2, minimum_score=4.1, stable_multiplier=2)
    recovered = SelectionSettings.from_mapping({"window_seconds": .5, "stable_multiplier": 1, "minimum_score": 7})
    assert recovered.window_seconds == .5 and recovered.stable_multiplier == 1
    assert recovered.minimum_score == .5


def test_save_merges_existing_preferences_atomically_and_preserves_other_sections(tmp_path):
    destination = tmp_path / ".kon/preferences.json"
    destination.parent.mkdir()
    previous = {"aim_mode": "hand", "voice_trigger": "go", "camera_settings": {"fov": 82.15},
                "unknown_future_preference": [1, "keep"]}
    destination.write_text(json.dumps(previous))
    settings = SelectionSettings(stable_seconds=.5, release_cells=3)
    save_selection_settings(destination, settings)
    saved = json.loads(destination.read_text())
    assert {key: saved[key] for key in previous} == previous
    assert saved["selection_settings"] == settings.to_mapping()
    assert not list(destination.parent.glob(".selection-*.tmp"))


def test_failed_atomic_replace_retains_original_file_and_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "preferences.json"
    destination.write_text('{"voice_trigger":"keep"}')
    original = destination.read_bytes()
    def denied_replace(self, target):
        raise PermissionError("save denied")
    monkeypatch.setattr(Path, "replace", denied_replace)
    with pytest.raises(PermissionError, match="save denied"):
        save_selection_settings(destination, SelectionSettings())
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(".selection-*.tmp"))


def test_initialization_loads_settings_without_writing_or_starting_sensors(application, tmp_path, monkeypatch):
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    destination = tmp_path / ".kon/preferences.json"
    destination.parent.mkdir()
    settings = SelectionSettings(window_seconds=2, minimum_score=3, stable_seconds=.4,
                                 stable_multiplier=2.5, movement_speed_limit=.025,
                                 release_cells=3, tracking_grace_seconds=.8)
    destination.write_text(json.dumps({"selection_settings": settings.to_mapping(), "voice_trigger": "go"}))
    original = destination.read_bytes()
    value = Window(aim_mode="hand")
    try:
        assert value.selection_settings == settings
        assert value.gate.dwell_seconds == 2 and value.gate.minimum_dwell_seconds == 3
        assert value.gate.tracking_grace_seconds == .8
        assert value.stable_dwell_seconds == .4 and value.stable_dwell_multiplier == 2.5
        assert value.hand_movement_speed_limit == .025 and value.lock_release_cells == 3
        assert value.camera is None and value.speech is None and not value.running
        assert destination.read_bytes() == original
        for field, control in value.selection_controls.items():
            assert control.value() == getattr(settings, field)
    finally:
        value.close()


def test_selection_load_is_independent_of_malformed_voice_preferences(window, tmp_path):
    destination = tmp_path / ".kon/preferences.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({"voice_trigger": [], "selection_settings": {"stable_seconds": .75}}))
    window._load_preferences()
    assert window.stable_dwell_seconds == .75
    assert destination.read_text().startswith('{"voice_trigger": []')


def test_controls_fold_and_can_be_operated_with_keyboard(window):
    assert window.selection_settings_panel.isHidden()
    QTest.keyClick(window.selection_settings_button, Qt.Key_Space)
    assert not window.selection_settings_panel.isHidden()
    control = window.selection_controls["stable_seconds"]
    assert control.accessibleName() and control.toolTip()
    assert not control.keyboardTracking()
    QTest.keyClick(control, Qt.Key_Up)
    assert window.stable_dwell_seconds == pytest.approx(.35)
    QTest.keyClick(window.selection_settings_button, Qt.Key_Space)
    assert window.selection_settings_panel.isHidden()


def test_lower_window_or_multiplier_caps_threshold_and_saves_effective_values(window, tmp_path):
    window.selection_controls["window_seconds"].setValue(.5)
    assert window.selection_settings.minimum_score == 1
    assert window.selection_controls["minimum_score"].maximum() == 1
    window.selection_controls["stable_multiplier"].setValue(1)
    assert window.selection_settings.minimum_score == .5
    assert window.gate.minimum_dwell_seconds == .5
    saved = json.loads((tmp_path / ".kon/preferences.json").read_text())
    assert saved["selection_settings"]["minimum_score"] == .5
    assert "最近 0.5 秒" in window.selection_instruction()
    assert "加權" in window.summon_caption.text() and "門檻 0.5 秒" in window.summon_caption.text()


def test_editing_live_settings_clears_lock_voice_and_history_without_rearming_or_stopping_camera(window, monkeypatch):
    class Sensor:
        stopped = False
        def stop(self):
            self.stopped = True
        def wait(self, _timeout):
            return True
        def set_active(self, active):
            self.active = active
    monkeypatch.setattr("kon.app.time.monotonic", lambda: 100.)
    window.camera = Sensor()
    window.speech = Sensor()
    window.running = window.testing = window.armed = window.voice_active = True
    gate = window.gate
    gate._latched = True
    gate._locked_target = "a"
    gate._pending = "a"
    window.targets = [DesktopTarget("a", "A.txt", (100, 100))]
    gate._dwell_times["a"] = 1
    window.selection_controls["release_cells"].setValue(3)
    assert window.gate is gate and gate._latched
    assert gate.locked_target is None and not gate._dwell_times and gate.take_trigger() is None
    assert window.running and not window.camera.stopped
    assert window.testing and not window.voice_active and not window.speech.active
    assert window.lock_release_cells == 3


def test_editing_settings_cancels_pending_animation_without_stopping_sensors(window):
    window.running = True
    target = DesktopTarget("a", "A.txt", (100, 100))
    window.overlay.play(target)
    window.pending_action = (window.action_serial, "a", True)
    serial = window.action_serial
    window.selection_controls["minimum_score"].setValue(2)
    assert window.overlay.eating is None and window.pending_action is None
    assert window.action_serial != serial and window.running


def test_new_gate_uses_loaded_settings_and_dynamic_copy_tracks_voice_mode(window):
    window.selection_controls["window_seconds"].setValue(4)
    window.selection_controls["minimum_score"].setValue(2.2)
    gate = window.selection_gate()
    assert gate.dwell_seconds == 4 and gate.minimum_dwell_seconds == 2.2
    assert "最近 4 秒" in window.footnote.text() and "門檻 2.2 秒" in window.gesture_instructions.text()
    window.voice_trigger_mode = "phrase"
    window.trigger_display = "go"
    window.update_voice_labels()
    assert "門檻 2.2 秒" in window.summon_caption.text() and "說 go" in window.summon_caption.text()


def test_save_failure_keeps_applied_tuning_and_reports_unsaved_state(window, monkeypatch):
    def fail_save(*_args):
        raise PermissionError("readonly preferences")
    monkeypatch.setattr("kon.app.save_selection_settings", fail_save)
    window.selection_controls["stable_multiplier"].setValue(3)
    assert window.stable_dwell_multiplier == 3
    assert "已套用" in window.notice.text() and "無法保存" in window.notice.text()
