"""Personal gesture enrollment interactions without sensors or file actions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from kon.gesture_calibration_dialog import GestureCalibrationDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application):
    value = GestureCalibrationDialog()
    value.show()
    value.activateWindow()
    value.setFocus()
    application.processEvents()
    yield value
    value.close()


def test_separate_step_does_not_claim_to_relearn_aim(dialog):
    assert "九點瞄準校正會保留" in dialog.scope_label.text()
    assert "1 / 2" in dialog.step_label.text()
    assert "記錄我的手勢" in dialog.record_button.text()
    assert dialog.record_button.isEnabled()


def test_click_when_unready_starts_pending_once(dialog):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append("sample"))
    dialog.set_state(0, 6, "整隻手尚未入鏡", False)
    assert dialog.record_button.isEnabled()
    QTest.mouseClick(dialog.record_button, Qt.LeftButton)
    dialog._request_sample()
    assert calls == ["sample"]
    assert dialog.pending and not dialog.record_button.isEnabled()


@pytest.mark.parametrize("key", [Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter])
def test_shortcuts_record_once_with_button_focused(dialog, application, key):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append("sample"))
    dialog.record_button.setFocus()
    application.processEvents()
    QTest.keyClick(dialog.record_button, key)
    assert calls == ["sample"]


def test_shortcut_works_when_cancel_child_has_focus(dialog, application):
    samples, exits = [], []
    dialog.sample_requested.connect(lambda: samples.append(1))
    dialog.exit_requested.connect(lambda: exits.append(1))
    dialog.cancel_button.setFocus()
    application.processEvents()
    QTest.keyClick(dialog.cancel_button, Qt.Key_Space)
    assert samples == [1] and exits == []


def test_auto_repeat_does_not_request_a_sample(dialog, application):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append(1))
    for event_type in (QEvent.KeyPress, QEvent.KeyRelease):
        application.sendEvent(dialog, QKeyEvent(event_type, Qt.Key_Space, Qt.NoModifier, " ", True, 2))
    assert calls == []
    assert all(not shortcut.autoRepeat() for shortcut in dialog._shortcuts)


def test_second_hold_resets_count_and_requests_fresh_capture(dialog):
    samples = []
    dialog.sample_requested.connect(lambda: samples.append(dialog.step))
    dialog.set_state(9, 9, "第一次記錄已完成", True)
    dialog.set_feedback("第一次已記錄，請放開後重新比一次。")
    dialog.set_step(1)
    assert "2 / 2" in dialog.step_label.text()
    assert "先放開" in dialog.instructions_label.text()
    assert "同一隻手" in dialog.instructions_label.text()
    assert "新的畫面" in dialog.instructions_label.text()
    assert "0 / 9" in dialog.progress_label.text()
    assert not dialog.ready and not dialog.pending
    assert "第一次已記錄" in dialog.feedback_label.text()
    assert "確認手勢" in dialog.record_button.text()
    dialog.record_button.click()
    assert samples == [1]


def test_match_status_is_only_visible_for_the_second_hold(dialog):
    assert dialog.match_label.isHidden()
    dialog.set_match_status("與第一次記錄相符")
    assert dialog.match_label.isHidden()
    dialog.set_step(1)
    assert dialog.match_label.isVisible()
    assert dialog.match_label.text() == "重新比出手勢後，這裡會顯示與第一次的比對結果"
    assert dialog.match_label.textFormat() == Qt.PlainText
    dialog.set_step(0)
    assert dialog.match_label.isHidden()


def test_standard_angles_progress_through_two_fresh_holds_each(dialog):
    samples = []
    dialog.sample_requested.connect(lambda: samples.append((dialog._standard_angle_index, dialog.step)))
    angles = ("中間", "上", "下", "左", "右")
    for index, angle in enumerate(angles):
        dialog.set_standard_angle_mode(index)
        dialog.set_step(0)
        assert dialog.angle_label.isVisible()
        assert f"角度 {index + 1} / 5：{angle}" in dialog.angle_label.text()
        assert "中間 → 上 → 下 → 左 → 右" in dialog.angle_label.text()
        assert f"{angle}角度" in dialog.step_label.text() and "1 / 2" in dialog.step_label.text()
        assert "同一隻手維持相同手形" in dialog.instructions_label.text()
        assert "手與鏡頭之間的角度" in dialog.instructions_label.text()
        assert "不需要移動瞄準目標" in dialog.instructions_label.text()
        if index:
            assert f"向{angle}傾斜" in dialog.instructions_label.text()
        else:
            assert "正面朝向鏡頭" in dialog.instructions_label.text()
        assert f"記錄{angle}角度" in dialog.record_button.text()
        assert dialog.match_label.isHidden()
        assert "0 / 6" in dialog.progress_label.text()
        dialog.record_button.click()
        dialog.set_state(6, 6, "第一次已完成", True)
        dialog.set_step(1)
        assert f"{angle}角度" in dialog.step_label.text() and "2 / 2" in dialog.step_label.text()
        assert f"剛才的{angle}角度" in dialog.instructions_label.text()
        assert "先放開手勢" in dialog.instructions_label.text()
        assert "只與此角度的第一次記錄比對" in dialog.instructions_label.text()
        assert dialog.match_label.isVisible()
        assert "此角度第一次" in dialog.match_label.text()
        assert f"確認{angle}角度" in dialog.record_button.text()
        assert "0 / 6" in dialog.progress_label.text()
        assert not dialog.ready and not dialog.pending
        dialog.record_button.click()
    assert samples == [(index, step) for index in range(5) for step in (0, 1)]


def test_standard_angles_explain_atomic_save_and_preserved_calibration(dialog):
    dialog.set_standard_angle_mode(0)
    assert "五個角度" in dialog.title_label.text()
    assert "瞄準校正會保留" in dialog.scope_label.text()
    assert "五個角度完成後一次保存" in dialog.scope_label.text()
    assert "取消會保留原手勢" in dialog.scope_label.text()


def test_standard_angle_mode_keeps_active_capture_and_comparison(dialog):
    dialog.set_standard_angle_mode(1)
    dialog.set_step(1)
    dialog.set_feedback("上角度第一次已記錄，請放開後再重做一次。")
    dialog.set_state(5, 12, "等待更多新畫面。", False, pending=True)
    dialog.set_match_status("與這個角度的第一次記錄相符")
    progress = dialog.progress_label.text()
    dialog.set_standard_angle_mode(1)
    assert dialog.match_label.text() == "與這個角度的第一次記錄相符"
    assert dialog.progress_label.text() == progress
    assert "第一次已記錄" in dialog.feedback_label.text()
    assert dialog.pending and not dialog.ready and not dialog.record_button.isEnabled()


@pytest.mark.parametrize("index", [-1, 5, True, 1.0, None])
def test_invalid_standard_angle_does_not_change_mode(dialog, index):
    dialog.set_standard_angle_mode(2)
    before = tuple(label.text() for label in (dialog.angle_label, dialog.step_label, dialog.scope_label))
    with pytest.raises(ValueError):
        dialog.set_standard_angle_mode(index)
    assert dialog._standard_angle_index == 2
    assert tuple(label.text() for label in (dialog.angle_label, dialog.step_label, dialog.scope_label)) == before


def test_append_angle_mode_can_replace_standard_mode(dialog):
    dialog.set_standard_angle_mode(4)
    dialog.set_step(1)
    dialog.set_append_angle_mode(2)
    assert dialog.angle_label.isHidden()
    assert "增加手勢角度" in dialog.title_label.text()
    assert "剛才的新角度" in dialog.instructions_label.text()
    assert "確認新角度" in dialog.record_button.text()
    assert dialog.match_label.text() == "重新比出手勢後，這裡會顯示與第一次的比對結果"
    dialog.set_standard_angle_mode(0)
    assert dialog.angle_label.isVisible()
    assert "確認中間角度" in dialog.record_button.text()
    assert "此角度第一次" in dialog.match_label.text()


def test_append_angle_first_hold_keeps_existing_angles_and_records_a_new_direction(dialog):
    samples = []
    dialog.sample_requested.connect(lambda: samples.append(dialog.step))
    dialog.set_append_angle_mode(2)
    assert dialog.title_label.text() == "增加手勢角度"
    assert "已有2個角度會保留" in dialog.scope_label.text()
    assert "同一隻手維持同樣手形" in dialog.scope_label.text()
    assert "另一方向" in dialog.scope_label.text()
    assert "1 / 2" in dialog.step_label.text()
    assert "轉到常用的另一角度後記錄" in dialog.instructions_label.text()
    assert dialog.match_label.isHidden()
    assert "記錄這個角度" in dialog.record_button.text()
    dialog.record_button.click()
    assert samples == [0] and dialog.pending


@pytest.mark.parametrize("mode_first", [True, False])
def test_append_angle_mode_updates_second_hold_before_or_after_step_change(dialog, mode_first):
    if mode_first:
        dialog.set_append_angle_mode(3)
    dialog.set_step(1)
    dialog.set_feedback("這個新角度第一次已記錄，原有角度仍保留。")
    dialog.set_state(5, 12, "請保持新角度，等待新的穩定畫面。", False, pending=True)
    dialog.set_match_status("與第一次記錄相符")
    progress = dialog.progress_label.text()
    if not mode_first:
        dialog.set_append_angle_mode(3)
    assert "已有3個角度會保留" in dialog.scope_label.text()
    assert "2 / 2" in dialog.step_label.text()
    assert "先放開手勢" in dialog.instructions_label.text()
    assert "剛才的新角度" in dialog.instructions_label.text()
    assert "新角度的第一次記錄比對" in dialog.instructions_label.text()
    assert dialog.match_label.isVisible() and dialog.match_label.text() == "與第一次記錄相符"
    assert dialog.progress_label.text() == progress
    assert "原有角度仍保留" in dialog.feedback_label.text()
    assert dialog.pending and not dialog.ready and not dialog.record_button.isEnabled()


def test_append_angle_long_instructions_and_match_fit_at_minimum_width(dialog, application):
    dialog.set_append_angle_mode(4)
    dialog.set_step(1)
    dialog.set_feedback("尚未確認：這次與新角度的第一次記錄不同。請放開手勢，再從同一個新角度重新比出。")
    dialog.set_match_status("中指與第一次記錄的彎曲角度不同，請維持同樣手形，再從剛才的新角度重新比出。")
    dialog.resize(560, 660)
    application.processEvents()
    for label in (dialog.instructions_label, dialog.scope_label, dialog.match_label, dialog.feedback_label):
        needed = label.fontMetrics().boundingRect(
            QRect(0, 0, label.contentsRect().width(), 10000), Qt.TextWordWrap, label.text()).height()
        assert label.contentsRect().height() >= needed
    assert dialog.preview_label.height() >= 220


@pytest.mark.parametrize("index", range(5))
def test_standard_angle_instructions_and_live_feedback_fit_at_minimum_width(dialog, application, index):
    dialog.set_standard_angle_mode(index)
    dialog.set_step(1)
    dialog.set_state(4, 6, "手形與這個角度的第一次記錄尚未一致；請用同一隻手，放開後重新比出。", False)
    dialog.set_feedback("確認逾時：沒有取得足夠的穩定新畫面。這個角度尚未完成，請放開後重新比出並記錄。")
    dialog.set_match_status("中指、無名指與此角度第一次記錄的彎曲姿勢不同；請使用同一隻手，從剛才角度重做相同手形。")
    dialog.resize(560, 660)
    application.processEvents()
    for label in (dialog.angle_label, dialog.step_label, dialog.instructions_label,
                  dialog.scope_label, dialog.reason_label, dialog.feedback_label,
                  dialog.capture_hint_label, dialog.match_label):
        needed = label.fontMetrics().boundingRect(
            QRect(0, 0, label.contentsRect().width(), 10000), Qt.TextWordWrap, label.text()).height()
        assert label.contentsRect().height() >= needed
    assert dialog.preview_label.height() >= 220
    assert dialog.record_button.isVisible() and dialog.cancel_button.isVisible()


def test_live_match_result_never_replaces_feedback_or_capture_progress(dialog):
    dialog.set_step(1)
    dialog.set_feedback("確認逾時：請重新比出你的召喚手勢。")
    dialog.set_state(5, 12, "請再保持手形片刻，等待更多新畫面。", False, pending=True)
    progress, reason, feedback = (label.text() for label in
                                  (dialog.progress_label, dialog.reason_label, dialog.feedback_label))
    dialog.set_match_status("與第一次記錄相符")
    assert dialog.match_label.text() == "與第一次記錄相符"
    assert dialog.pending and not dialog.ready and not dialog.record_button.isEnabled()
    dialog.set_match_status("中指與第一次記錄的姿勢不同，請用同一隻手重新比出。")
    assert "中指" in dialog.match_label.text()
    assert dialog.progress_label.text() == progress
    assert dialog.reason_label.text() == reason
    assert dialog.feedback_label.text() == feedback
    dialog.set_state(6, 12, "已取得更多新画面，請繼續保持。", False, pending=True)
    assert "中指" in dialog.match_label.text()
    assert dialog.feedback_label.text() == feedback


def test_live_reason_preserves_previous_action_feedback(dialog):
    dialog.set_feedback("確認逾時：這次與第一筆的手形差異過大，請重新比出。")
    dialog.set_state(3, 6, "手部仍在移動，請保持片刻。", False)
    assert "確認逾時" in dialog.feedback_label.text()
    assert "手部仍在移動" in dialog.reason_label.text()
    assert dialog.feedback_label.textFormat() == dialog.reason_label.textFormat() == Qt.PlainText
    dialog.set_state(4, 6, "請讓手指完整入鏡。", False, pending=True)
    assert "確認逾時" in dialog.feedback_label.text()
    assert not dialog.record_button.isEnabled()


def test_pending_failure_can_be_retried_after_parent_updates_state(dialog):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append(1))
    dialog.record_button.click()
    dialog.set_state(0, 6, "八秒內沒有取得穩定畫面，請重新取樣。", False)
    dialog.record_button.click()
    assert calls == [1, 1]


@pytest.mark.parametrize("action", ["escape", "cancel", "close"])
def test_cancel_paths_emit_exit_once_without_sample(dialog, action):
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


def test_success_accepts_without_cancellation_signal(dialog):
    exits = []
    dialog.exit_requested.connect(lambda: exits.append(1))
    dialog.finish_success()
    dialog.finish_success()
    dialog.close()
    assert not exits and dialog.result() == QDialog.Accepted


def test_camera_preview_is_copied_and_preserves_aspect_ratio(dialog, application):
    image = QImage(960, 540, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor("#f5ab80"))
    dialog.set_preview(image)
    image.fill(QColor("#111018"))
    assert dialog.preview_label.image.pixelColor(0, 0) == QColor("#f5ab80")
    dialog.resize(700, 800)
    application.processEvents()
    pixmap = dialog.preview_label.pixmap()
    assert abs(pixmap.width() / pixmap.height() - 16 / 9) < .02
    assert pixmap.width() <= dialog.preview_label.width()
    assert pixmap.height() <= dialog.preview_label.height()


def test_empty_preview_has_visible_camera_guidance(dialog):
    dialog.set_preview(QImage())
    assert "等待鏡頭畫面" in dialog.preview_label.text()


def test_minimum_dialog_keeps_live_reason_and_feedback_visible(dialog, application):
    dialog.set_step(1)
    dialog.set_state(4, 6, "手形與第一次記錄尚未一致；請用同一隻手，放開後重新比出，讓中指、無名指與拇指清楚入鏡。", False)
    dialog.set_feedback("確認逾時：沒有取得足夠的穩定新畫面。請先放開手勢，再用同一隻手重新比出並記錄。")
    dialog.set_match_status("中指、無名指與第一次記錄的彎曲姿勢不同；请使用同一隻手，再次比出原來的召喚手勢，並讓拇指接觸位置清楚入鏡。")
    dialog.resize(560, 660)
    application.processEvents()
    for label in (dialog.instructions_label, dialog.scope_label, dialog.reason_label,
                  dialog.feedback_label, dialog.capture_hint_label, dialog.match_label):
        needed = label.fontMetrics().boundingRect(
            QRect(0, 0, label.contentsRect().width(), 10000), Qt.TextWordWrap, label.text()).height()
        assert label.contentsRect().height() >= needed
    assert dialog.preview_label.height() >= 220
    assert dialog.height() >= dialog.layout().minimumSize().height()
    assert dialog.record_button.isVisible() and dialog.cancel_button.isVisible()


@pytest.mark.parametrize("step", [-1, 2, True])
def test_invalid_step_does_not_change_enrollment(dialog, step):
    with pytest.raises(ValueError):
        dialog.set_step(step)
    assert dialog.step == 0
