"""Wizard progress and practice are UI state, independent of camera input."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kon.gesture_calibration_dialog import GestureCalibrationDialog


@pytest.fixture
def wizard():
    app = QApplication.instance() or QApplication([])
    dialog = GestureCalibrationDialog()
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    yield dialog, app
    dialog.close()


def test_sampling_progress_uses_effective_requirement(wizard):
    dialog, _ = wizard
    dialog.set_state(5, 12, "請保持手形。", False, pending=True)
    assert dialog.sample_progress.maximum() == 12
    assert dialog.sample_progress.value() == 5
    assert "5 筆" in dialog.sample_progress.accessibleDescription()
    dialog.set_state(20, 12, "完成", True)
    assert dialog.sample_progress.value() == 12
    dialog.set_step(1)
    assert dialog.sample_progress.value() == 0


def test_angle_track_distinguishes_completed_current_and_upcoming(wizard):
    dialog, _ = wizard
    dialog.set_standard_angle_mode(2)
    assert [marker.objectName() for marker in dialog.angle_markers] == [
        "angleComplete", "angleComplete", "angleCurrent", "angleUpcoming", "angleUpcoming",
    ]
    assert "同一手形" in dialog.action_label.text()
    assert "只調整手掌角度" in dialog.shape_rule_label.text()


def test_long_feedback_scrolls_without_growing_past_typical_screen(wizard):
    dialog, app = wizard
    dialog.set_standard_angle_mode(3)
    dialog.set_step(1)
    dialog.set_feedback("中指與第一次記錄的彎曲角度不同，請先放開再重新比出。" * 12)
    dialog.resize(960, 700)
    app.processEvents()
    assert dialog.height() <= 740
    assert dialog.width() <= 1000
    assert dialog.preview_label.width() >= 400
    assert dialog.preview_label.height() >= 220
    assert dialog.content_scroll.verticalScrollBar().maximum() > 0
    assert dialog.record_button.geometry().bottom() < dialog.height()


def test_review_needs_fresh_match_and_emits_save_once(wizard):
    dialog, app = wizard
    samples, saves = [], []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.save_requested.connect(lambda: saves.append(1))
    dialog.set_review_mode("靈狐")
    assert dialog.title_label.text() == "試做並儲存「靈狐」"
    assert dialog.record_button.text() == "儲存並使用"
    assert not dialog.record_button.isEnabled()
    dialog._request_sample()
    assert not saves and not samples
    dialog.set_review_match(True, "已辨識到剛才的手形。")
    assert dialog.record_button.isEnabled()
    dialog.set_review_match(False, "手已離開鏡頭，請重新比出。")
    assert not dialog.record_button.isEnabled()
    dialog.set_review_match(True)
    dialog.record_button.setFocus()
    app.processEvents()
    QTest.keyClick(dialog.record_button, Qt.Key_Return)
    dialog._request_sample()
    assert saves == [1] and not samples
    assert dialog.pending and not dialog.record_button.isEnabled()


def test_pending_capture_shortcuts_do_not_save_or_recapture(wizard):
    dialog, app = wizard
    samples, saves = [], []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.save_requested.connect(lambda: saves.append(1))
    dialog.record_button.click()
    for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
        QTest.keyClick(dialog, key)
    app.processEvents()
    assert samples == [1] and saves == []


def test_review_retry_invalidates_match_and_returns_to_capture(wizard):
    dialog, _ = wizard
    retries = []
    dialog.retry_requested.connect(lambda: retries.append(1))
    dialog.set_standard_angle_mode(4)
    dialog.set_review_mode("靈狐")
    dialog.set_review_match(True)
    dialog.retry_button.click()
    assert retries == [1] and not dialog.record_button.isEnabled()
    dialog.set_standard_angle_mode(0)
    dialog.set_step(0)
    assert not dialog._review_mode
    assert dialog.retry_button.isHidden()
    assert not dialog.sample_progress.isHidden()
    assert "記錄中間角度" in dialog.record_button.text()
