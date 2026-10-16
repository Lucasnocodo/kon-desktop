"""Optional hand-position range UI; no camera, recognition, or storage."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from kon.hand_aim_dialog import HandAimDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application):
    value = HandAimDialog()
    value.show()
    value.activateWindow()
    application.processEvents()
    yield value
    value.close()


def test_range_setup_is_optional_and_explains_mirrored_camera_positions(dialog):
    assert "選用" in dialog.scope_label.text()
    assert "未調整時可直接使用預設範圍" in dialog.scope_label.text()
    assert "鏡像預覽" in dialog.preview_note_label.text()
    assert "左上" in dialog.instructions_label.text()
    assert "記錄左上位置" in dialog.record_button.text()
    assert "九點" not in dialog.scope_label.text()
    assert "手形" not in dialog.instructions_label.text()


def test_right_bottom_step_starts_new_position_capture(dialog):
    dialog.set_state(6, 8, "左上位置已取得", True)
    dialog.set_feedback("左上位置已記錄。接著記錄你舒服的右下位置。")
    dialog.set_step(1)
    assert "2 / 2" in dialog.step_label.text()
    assert "右下" in dialog.instructions_label.text()
    assert "比前一步更靠右、更靠下" in dialog.instructions_label.text()
    assert "0 / 8" in dialog.progress_label.text()
    assert "左上位置已記錄" in dialog.feedback_label.text()
    assert "記錄右下位置" in dialog.record_button.text()
    assert not dialog.ready and not dialog.pending


def test_unready_button_starts_wait_and_duplicate_click_is_suppressed(dialog):
    samples = []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.set_state(0, 6, "手圈仍在移動", False)
    assert dialog.record_button.isEnabled()
    QTest.mouseClick(dialog.record_button, Qt.LeftButton)
    dialog._request_sample()
    assert samples == [1] and dialog.pending
    assert not dialog.record_button.isEnabled()
    dialog.set_state(0, 6, "調整後再記錄", False, pending=False)
    dialog.record_button.click()
    assert samples == [1, 1]


@pytest.mark.parametrize("key", [Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter])
@pytest.mark.parametrize("child", ["record_button", "cancel_button"])
def test_keyboard_requests_one_sample_with_either_child_focused(dialog, application, key, child):
    samples, exits = [], []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.exit_requested.connect(lambda: exits.append(1))
    widget = getattr(dialog, child)
    widget.setFocus()
    application.processEvents()
    QTest.keyClick(widget, key)
    assert samples == [1] and not exits


def test_auto_repeat_does_not_request_another_position(dialog, application):
    samples = []
    dialog.sample_requested.connect(lambda: samples.append(1))
    for event_type in (QEvent.KeyPress, QEvent.KeyRelease):
        application.sendEvent(dialog, QKeyEvent(event_type, Qt.Key_Space, Qt.NoModifier, " ", True, 3))
    assert not samples
    assert all(not shortcut.autoRepeat() for shortcut in dialog._shortcuts)


def test_feedback_survives_live_tracking_updates(dialog):
    dialog.set_feedback("範圍太小：右下位置需要更靠右、更靠下，請重新記錄。")
    dialog.set_state(2, 6, "等待新的穩定位置", False, pending=True)
    assert "範圍太小" in dialog.feedback_label.text()
    assert "等待新的穩定位置" in dialog.reason_label.text()
    assert dialog.feedback_label.textFormat() == dialog.reason_label.textFormat() == Qt.PlainText


@pytest.mark.parametrize("action", ["escape", "cancel", "close"])
def test_exit_paths_emit_once_without_a_position_sample(dialog, action):
    samples, exits = [], []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.exit_requested.connect(lambda: exits.append(1))
    if action == "escape":
        QTest.keyClick(dialog, Qt.Key_Escape)
    elif action == "cancel":
        dialog.cancel_button.click()
    else:
        dialog.close()
    dialog.reject()
    dialog.close()
    assert exits == [1] and not samples
    assert dialog.result() == QDialog.Rejected


def test_success_accepts_without_exit_signal(dialog):
    exits = []
    dialog.exit_requested.connect(lambda: exits.append(1))
    dialog.finish_success()
    dialog.finish_success()
    dialog.close()
    assert not exits and dialog.result() == QDialog.Accepted


def test_preview_is_copied_and_not_mirrored_again(dialog, application):
    image = QImage(960, 540, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor("#14121d"))
    image.setPixelColor(0, 0, QColor("#f5ab80"))
    dialog.set_preview(image)
    image.fill(QColor("#00ff00"))
    assert dialog.preview_label.image.pixelColor(0, 0) == QColor("#f5ab80")
    assert dialog.preview_label.image.pixelColor(959, 0) == QColor("#14121d")
    dialog.resize(700, 780)
    application.processEvents()
    pixmap = dialog.preview_label.pixmap()
    assert abs(pixmap.width() / pixmap.height() - 16 / 9) < .02


def test_minimum_layout_keeps_range_reason_and_feedback_readable(dialog, application):
    dialog.set_step(1)
    dialog.set_state(2, 6, "手圈仍在移動；請在鏡像預覽中選擇更靠右、更靠下的位置，並停住片刻。", False)
    dialog.set_feedback("右下位置距離左上位置太近，還不足以覆蓋舒適的移動範圍；調整後再按記錄。")
    dialog.resize(560, 640)
    application.processEvents()
    for label in (dialog.scope_label, dialog.instructions_label, dialog.preview_note_label,
                  dialog.reason_label, dialog.feedback_label, dialog.capture_hint_label):
        needed = label.fontMetrics().boundingRect(
            QRect(0, 0, label.contentsRect().width(), 10000), Qt.TextWordWrap, label.text()).height()
        assert label.contentsRect().height() >= needed
    assert dialog.preview_label.height() >= 220
    rectangles = [dialog.layout().itemAt(i).geometry() for i in range(dialog.layout().count())]
    assert all(first.bottom() < second.top() for first, second in zip(rectangles, rectangles[1:]))
    assert dialog.rect().contains(dialog.record_button.geometry())


@pytest.mark.parametrize("step", [-1, 2, True])
def test_invalid_step_is_rejected(dialog, step):
    with pytest.raises(ValueError):
        dialog.set_step(step)
