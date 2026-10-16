"""User choices and fresh test gating without a microphone or file operations."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from kon.pronunciation_dialog import PronunciationDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application):
    value = PronunciationDialog(trigger_setting="KON, con", language="en")
    value.show()
    value.activateWindow()
    application.processEvents()
    yield value
    value.close()


def _choose(dialog):
    dialog.set_samples(["KON!", "Con.", "kon"])
    dialog.candidate_checkboxes[0].setChecked(True)


def _pass(dialog):
    _choose(dialog)
    dialog.test_button.click()
    dialog.set_test_result(True, "KON!")


def test_initial_choice_and_audio_scope(dialog):
    assert dialog.desired_phrase() == "kon"
    assert dialog.selected_language() == "en"
    assert "你習慣的方式唸三次" in dialog.scope_label.text()
    assert "才會收集音訊" in dialog.capture_hint_label.text()
    assert "不會開啟或移動任何檔案" in dialog.test_hint_label.text()
    assert "0 / 3" in dialog.progress_label.text()
    assert not dialog.test_button.isEnabled() and not dialog.apply_button.isEnabled()
    assert dialog.selected_aliases() == ()


def test_record_is_single_request_and_parent_can_release_failure(dialog):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append(1))
    dialog.record_button.click()
    dialog._request_sample()
    assert calls == [1]
    assert dialog.pending and not dialog.record_button.isEnabled()
    assert not dialog.phrase_input.isEnabled() and not dialog.language_combo.isEnabled()
    dialog.set_capture_state(False, "沒有聽到，請重試。")
    dialog.record_button.click()
    assert calls == [1, 1]


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space])
def test_phrase_typing_cannot_request_capture(dialog, application, key):
    calls = []
    dialog.sample_requested.connect(lambda: calls.append(1))
    dialog.phrase_input.setFocus()
    application.processEvents()
    QTest.keyClick(dialog.phrase_input, key)
    assert calls == [] and not dialog.pending
    assert dialog.isVisible()


def test_three_independent_samples_normalize_and_require_user_selection(dialog):
    dialog.set_samples(["ＫＯＮ!", "Con."])
    assert "2 / 3" in dialog.progress_label.text()
    assert not dialog.candidate_checkboxes and not dialog.test_button.isEnabled()
    dialog.set_samples(["ＫＯＮ!", "Con.", "kon"])
    assert dialog.samples == ["kon", "con", "kon"]
    assert [checkbox.text() for checkbox in dialog.candidate_checkboxes] == ["kon", "con"]
    assert dialog.selected_aliases() == () and not dialog.test_button.isEnabled()
    assert not dialog.record_button.isEnabled()
    assert "第 3 次：kon" == dialog.sample_labels[2].text()
    dialog.candidate_checkboxes[1].setChecked(True)
    assert dialog.selected_aliases() == ("con",)
    assert dialog.test_button.isEnabled() and not dialog.apply_button.isEnabled()


def test_empty_capture_cannot_count_as_a_completed_utterance(dialog):
    dialog.set_samples(["kon", "...", "con", " "])
    assert dialog.samples == ["kon", "con"]
    assert "2 / 3" in dialog.progress_label.text()
    assert dialog.record_button.isEnabled() and not dialog.test_button.isEnabled()


def test_unrequested_verdict_and_early_save_do_not_apply(dialog):
    saves = []
    dialog.save_requested.connect(lambda: saves.append(1))
    _choose(dialog)
    dialog.set_test_result(True, "kon")
    dialog._request_save()
    dialog.finish_success()
    assert not saves and not dialog.apply_button.isEnabled()
    assert dialog.isVisible()


def test_pass_requires_fresh_test_and_selected_complete_result(dialog):
    tests, saves, exits = [], [], []
    dialog.test_requested.connect(lambda: tests.append(1))
    dialog.save_requested.connect(lambda: saves.append(1))
    dialog.exit_requested.connect(lambda: exits.append(1))
    _choose(dialog)
    dialog.test_button.click()
    dialog._request_test()
    assert tests == [1] and dialog.pending
    dialog.set_test_result(True, "con")
    assert not dialog.apply_button.isEnabled()
    dialog.test_button.click()
    dialog.set_capture_state(False)
    dialog.set_test_result(True, "KON!")
    assert dialog.apply_button.isEnabled()
    dialog.apply_button.click()
    dialog._request_save()
    assert saves == [1] and dialog.pending
    dialog.finish_success()
    dialog.close()
    assert dialog.result() == QDialog.Accepted and not exits


def test_failing_fresh_test_stays_retryable(dialog):
    _choose(dialog)
    dialog.test_button.click()
    dialog.set_test_result(False, "something else")
    assert "尚未通過" in dialog.feedback_label.text()
    assert not dialog.pending and dialog.test_button.isEnabled()
    assert not dialog.apply_button.isEnabled()


def test_fresh_test_accepts_named_phrase_alongside_selected_alias(dialog):
    dialog.set_samples(["con", "con", "con"])
    dialog.candidate_checkboxes[0].setChecked(True)
    assert dialog.selected_aliases() == ("con",)
    dialog.test_button.click()
    dialog.set_test_result(True, "KON!")
    assert dialog.apply_button.isEnabled()


def test_retesting_discards_previous_pass_and_save_failure_can_retry(dialog):
    _pass(dialog)
    dialog.test_button.click()
    assert not dialog.apply_button.isEnabled()
    dialog.set_test_result(True, "kon")
    saves = []
    dialog.save_requested.connect(lambda: saves.append(1))
    dialog.apply_button.click()
    dialog.set_capture_state(False, "保存失敗，請重試。")
    assert dialog.apply_button.isEnabled()
    dialog.apply_button.click()
    assert saves == [1, 1]


def test_changing_selected_alias_invalidates_pass_without_losing_samples(dialog):
    _pass(dialog)
    dialog.candidate_checkboxes[1].setChecked(True)
    assert dialog.selected_aliases() == ("kon", "con")
    assert dialog.samples == ["kon", "con", "kon"]
    assert not dialog.apply_button.isEnabled() and dialog.test_button.isEnabled()


@pytest.mark.parametrize("choice", ["phrase", "language"])
def test_phrase_and_language_edits_restart_and_discard_stale_result(dialog, choice):
    restarts = []
    dialog.restart_requested.connect(lambda: restarts.append(1))
    _choose(dialog)
    dialog.test_button.click()
    if choice == "phrase":
        dialog.phrase_input.setText("召喚狐狸")
    else:
        dialog.language_combo.setCurrentIndex(1)
    dialog.set_test_result(True, "kon")
    assert restarts == [1]
    assert not dialog.samples and not dialog.candidate_checkboxes
    assert not dialog.pending and not dialog.apply_button.isEnabled()
    assert dialog.record_button.isEnabled()


def test_restart_cancels_pending_and_requests_three_new_utterances(dialog):
    restarts = []
    dialog.restart_requested.connect(lambda: restarts.append(1))
    dialog.set_samples(["kon"])
    dialog.record_button.click()
    assert dialog.restart_button.isEnabled()
    dialog.restart_button.click()
    assert restarts == [1] and dialog.samples == []
    assert not dialog.pending and "0 / 3" in dialog.progress_label.text()
    assert not dialog.apply_button.isEnabled()


@pytest.mark.parametrize("action", ["escape", "cancel", "close"])
def test_cancellation_emits_once_even_if_parent_closes_recursively(dialog, action):
    exits, calls = [], []
    dialog.exit_requested.connect(lambda: exits.append(1))
    dialog.exit_requested.connect(dialog.close)
    dialog.sample_requested.connect(lambda: calls.append(1))
    if action == "escape":
        QTest.keyClick(dialog, Qt.Key_Escape)
    elif action == "cancel":
        dialog.cancel_button.click()
    else:
        dialog.close()
    dialog.reject()
    dialog._request_sample()
    assert exits == [1] and calls == [] and dialog.result() == QDialog.Rejected


def test_chinese_phrase_and_language_are_direct_user_choices(application):
    dialog = PronunciationDialog(trigger_setting="召喚狐狸", language="zh")
    assert dialog.desired_phrase() == "召喚狐狸" and dialog.selected_language() == "zh"
    dialog.set_samples(["召喚狐狸。", "召喚狐狸", "召喚狐狸！"])
    dialog.candidate_checkboxes[0].setChecked(True)
    assert dialog.selected_aliases() == ("召喚狐狸",)
    dialog.close()


def test_plain_text_and_minimum_size_keep_controls_and_explanations_visible(dialog, application):
    dialog.phrase_input.setText("非常長的咒語文字")
    dialog.set_samples(["a whole pronunciation phrase", "召喚狐狸的完整咒語", "a different spoken phrase"])
    dialog.set_feedback("沒有辨識到勾選的結果。請確認辨識語言，再用同一種方式重試；也可以重新記錄三次。")
    dialog.resize(580, 650)
    application.processEvents()
    for label in (dialog.scope_label, dialog.instructions_label, dialog.feedback_label,
                  dialog.test_hint_label, dialog.capture_hint_label):
        assert label.textFormat() == Qt.PlainText
        needed = label.fontMetrics().boundingRect(
            QRect(0, 0, label.contentsRect().width(), 10000), Qt.TextWordWrap, label.text()).height()
        assert label.contentsRect().height() >= needed
    for button in (dialog.record_button, dialog.restart_button, dialog.cancel_button,
                   dialog.test_button, dialog.apply_button):
        assert button.isVisible()
        assert button.geometry().bottom() <= dialog.height()
