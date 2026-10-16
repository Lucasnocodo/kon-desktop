"""User-controlled spell pronunciation enrollment; audio and storage stay in the app."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .speech import normalize_utterance, parse_trigger_phrases


def _label(text, *, size=13, muted=False):
    label = QLabel(str(text))
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    label.setFont(QFont("PingFang TC", size))
    if muted:
        label.setStyleSheet("color: #b0a5bd;")
    return label


class PronunciationDialog(QDialog):
    """Emit requests; accept only an explicitly selected, freshly tested choice."""

    sample_requested = Signal()
    test_requested = Signal()
    save_requested = Signal()
    restart_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None, trigger_setting="kon", language="en"):
        super().__init__(parent)
        self.pending = False
        self.samples: list[str] = []
        self.candidate_checkboxes: list[QCheckBox] = []
        self._pending_mode = None
        self._awaiting_test = False
        self._test_configuration = None
        self._approved_configuration = None
        self._test_passed = False
        self._exit_emitted = False
        self._finished_successfully = False
        self._dispatching = False
        self.setWindowTitle("KON · 發音校正")
        self.setModal(True)
        self.setMinimumSize(580, 650)
        self.resize(640, 740)
        self.setStyleSheet("""
            QDialog { background: #111018; color: #eee6f1; }
            QLabel, QCheckBox { background: transparent; color: #eee6f1; }
            QFrame#pronunciationStage { background: #201b29; border: 1px solid #443549;
                border-radius: 12px; }
            QLineEdit, QComboBox { background: #201b29; color: #eee6f1;
                border: 1px solid #5b485e; border-radius: 8px; padding: 8px;
                font-family: 'PingFang TC'; font-size: 13px; }
            QLineEdit:focus, QComboBox:focus { border: 2px solid #ffbd91; }
            QComboBox QAbstractItemView { background: #201b29; color: #eee6f1;
                selection-background-color: #443040; }
            QPushButton { background: #2b2331; color: #eee6f1; border: 1px solid #5b485e;
                border-radius: 9px; padding: 9px 12px; min-height: 20px;
                font-family: 'PingFang TC'; font-size: 13px; }
            QPushButton:hover { background: #443040; }
            QPushButton:focus { border: 2px solid #ffbd91; }
            QPushButton#recordPronunciation, QPushButton#applyPronunciation {
                background: #f5ab80; color: #2b160e; border-color: #f5ab80; }
            QPushButton:disabled, QPushButton#recordPronunciation:disabled,
            QPushButton#applyPronunciation:disabled, QLineEdit:disabled, QComboBox:disabled {
                background: #29232e; color: #8c7d94; border-color: #3c3343; }
            QCheckBox { spacing: 9px; font-family: 'PingFang TC'; font-size: 14px; }
            QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #8c7d94;
                border-radius: 4px; background: #111018; }
            QCheckBox::indicator:checked { background: #f5ab80; border-color: #f5ab80; }
            QCheckBox::indicator:disabled { border-color: #3c3343; }
            QScrollBar:vertical { width: 8px; background: transparent; }
            QScrollBar::handle:vertical { background: #443549; border-radius: 4px; min-height: 32px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        try:
            primary = parse_trigger_phrases(trigger_setting)[0]
        except (TypeError, ValueError):
            primary = str(trigger_setting or "")
        self._build(primary, language)
        self.phrase_input.textChanged.connect(self._choice_edited)
        self.language_combo.currentIndexChanged.connect(self._choice_edited)
        self._refresh_samples()
        self._update_controls()
        self.phrase_input.setFocus(Qt.OtherFocusReason)

    def _build(self, primary, language):
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(24, 22, 24, 22)
        dialog_layout.setSpacing(12)
        content = QWidget(self)
        outer = QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        title = _label("決定我的咒語唸法", size=20)
        title.setFont(QFont("PingFang TC", 20, QFont.DemiBold))
        outer.addWidget(title)
        self.scope_label = _label(
            "先取個咒語名字，再用你習慣的方式唸三次。勾選你要用來發動的辨識結果。", muted=True)
        outer.addWidget(self.scope_label)

        fields = QHBoxLayout()
        phrase_column = QVBoxLayout()
        phrase_label = _label("咒語文字", size=12)
        self.phrase_input = QLineEdit(primary, self)
        self.phrase_input.setMaxLength(32)
        self.phrase_input.setPlaceholderText("例如 KON、空、召喚狐狸")
        self.phrase_input.setAccessibleName("咒語文字")
        phrase_label.setBuddy(self.phrase_input)
        phrase_column.addWidget(phrase_label)
        phrase_column.addWidget(self.phrase_input)
        fields.addLayout(phrase_column, 1)
        language_column = QVBoxLayout()
        language_label = _label("辨識語言", size=12)
        self.language_combo = QComboBox(self)
        self.language_combo.addItem("英語（English）", "en")
        self.language_combo.addItem("中文（國語）", "zh")
        index = self.language_combo.findData(language)
        self.language_combo.setCurrentIndex(max(0, index))
        self.language_combo.setAccessibleName("辨識語言")
        language_label.setBuddy(self.language_combo)
        language_column.addWidget(language_label)
        language_column.addWidget(self.language_combo)
        fields.addLayout(language_column)
        outer.addLayout(fields)

        stage = QFrame(self)
        stage.setObjectName("pronunciationStage")
        stage.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(16, 14, 16, 14)
        stage_layout.setSpacing(9)
        self.instructions_label = _label(
            "每次按下記錄，唸一次咒語，再稍停片刻。三次都使用同一種唸法。", muted=True)
        stage_layout.addWidget(self.instructions_label)
        self.progress_label = _label("", size=14)
        self.progress_label.setFont(QFont("PingFang TC", 14, QFont.DemiBold))
        stage_layout.addWidget(self.progress_label)
        self.sample_labels = []
        for number in range(1, 4):
            label = _label(f"第 {number} 次：尚未記錄", size=13)
            self.sample_labels.append(label)
            stage_layout.addWidget(label)
        outer.addWidget(stage)

        self.candidates_frame = QFrame(self)
        self.candidates_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.candidates_layout = QVBoxLayout(self.candidates_frame)
        self.candidates_layout.setContentsMargins(0, 0, 0, 0)
        self.candidates_layout.setSpacing(9)
        self.candidates_label = _label("勾選可用的辨識寫法", size=14)
        self.candidates_label.setFont(QFont("PingFang TC", 14, QFont.DemiBold))
        self.candidates_layout.addWidget(self.candidates_label)
        self.selection_label = _label("會接受咒語文字，以及你勾選的完整辨識結果。請至少選一個結果。", size=12, muted=True)
        self.candidates_layout.addWidget(self.selection_label)
        self.candidates_frame.hide()
        outer.addWidget(self.candidates_frame)

        self.feedback_label = _label("請選擇實際唸咒語的語言，準備好後開始記錄。")
        self.feedback_label.setStyleSheet("color: #ffd0ad;")
        self.feedback_label.setMinimumHeight(22)
        outer.addWidget(self.feedback_label)
        self.test_hint_label = _label(
            "安全測試會請你再唸一遍，確認勾選的結果能辨識。測試不會開啟或移動任何檔案。", size=12, muted=True)
        outer.addWidget(self.test_hint_label)
        self.capture_hint_label = _label(
            "只有按下記錄或測試後才會收集音訊；完成後即停止收集。", size=11, muted=True)
        outer.addWidget(self.capture_hint_label)
        outer.addStretch(1)
        self.content_scroll = QScrollArea(self)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content_scroll.setWidget(content)
        self.content_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: #111018; }")
        dialog_layout.addWidget(self.content_scroll, 1)

        recording_controls = QHBoxLayout()
        self.restart_button = self._button("重新記錄", self._request_restart)
        self.record_button = self._button("", self._request_sample)
        self.record_button.setObjectName("recordPronunciation")
        recording_controls.addWidget(self.restart_button)
        recording_controls.addStretch()
        recording_controls.addWidget(self.record_button)
        dialog_layout.addLayout(recording_controls)
        controls = QHBoxLayout()
        self.cancel_button = self._button("取消", self.reject)
        self.test_button = self._button("試唸咒語", self._request_test)
        self.apply_button = self._button("保存並套用", self._request_save)
        self.apply_button.setObjectName("applyPronunciation")
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        controls.addWidget(self.test_button)
        controls.addWidget(self.apply_button)
        dialog_layout.addLayout(controls)

    def _button(self, text, callback):
        button = QPushButton(text, self)
        # No dialog-wide recording shortcuts: Enter and Space remain text input.
        button.setAutoDefault(False)
        button.setDefault(False)
        button.setAutoRepeat(False)
        button.clicked.connect(callback)
        return button

    def desired_phrase(self) -> str:
        return self.phrase_input.text().strip()

    def selected_language(self) -> str:
        return self.language_combo.currentData()

    def selected_aliases(self) -> tuple[str, ...]:
        return tuple(checkbox.text() for checkbox in self.candidate_checkboxes if checkbox.isChecked())

    def _configuration(self):
        return self.desired_phrase(), self.selected_language(), self.selected_aliases()

    def _invalidate_test(self):
        self._test_passed = False
        self._awaiting_test = False
        self._test_configuration = None
        self._approved_configuration = None

    def _choice_edited(self, *_):
        has_work = bool(self.samples or self.pending or self._test_passed or self._awaiting_test)
        self._clear_collection()
        self.set_feedback("咒語或語言已變更，請用新的選擇重新記錄三次。")
        if has_work:
            self.restart_requested.emit()

    def _clear_collection(self):
        self.pending = False
        self._pending_mode = None
        self._invalidate_test()
        self.samples = []
        self._clear_candidates()
        self.candidates_frame.hide()
        self._refresh_samples()
        self._update_controls()

    def _clear_candidates(self):
        for checkbox in self.candidate_checkboxes:
            self.candidates_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.candidate_checkboxes = []

    def _refresh_samples(self):
        self.progress_label.setText(f"記錄進度  ·  {len(self.samples)} / 3 次")
        for index, label in enumerate(self.sample_labels):
            heard = self.samples[index] if index < len(self.samples) else "尚未記錄"
            label.setText(f"第 {index + 1} 次：{heard}")

    def set_samples(self, samples: list[str]):
        """Display up to three successful, independent utterances from the parent."""
        normalized = [normalize_utterance(sample) for sample in samples]
        normalized = [sample for sample in normalized if sample][:3]
        self.pending = False
        self._pending_mode = None
        if normalized != self.samples:
            self._invalidate_test()
            self.samples = normalized
            self._clear_candidates()
            if len(self.samples) == 3:
                for candidate in dict.fromkeys(self.samples):
                    checkbox = QCheckBox(candidate, self.candidates_frame)
                    checkbox.setAccessibleName(f"使用辨識結果：{candidate}")
                    checkbox.toggled.connect(self._selection_changed)
                    self.candidate_checkboxes.append(checkbox)
                    self.candidates_layout.addWidget(checkbox)
                self.set_feedback("三次已記錄。請勾選你要用來發動的結果，再做一次安全測試。")
        self.candidates_frame.setVisible(len(self.samples) == 3)
        self._refresh_samples()
        self._update_controls()

    def _selection_changed(self, *_):
        self._invalidate_test()
        self.set_feedback("選擇已更新，請再唸一次完成安全測試。" if self.selected_aliases()
                          else "請勾選至少一個你要使用的辨識結果。")
        self._update_controls()

    def set_capture_state(self, pending: bool, message: str = ""):
        if self._exit_emitted or self._finished_successfully:
            return
        self.pending = bool(pending)
        if not self.pending:
            self._pending_mode = None
        if message:
            self.set_feedback(message)
        self._update_controls()

    def set_feedback(self, text):
        self.feedback_label.setText(str(text or ""))

    def set_test_result(self, passed: bool, heard: str):
        """Only a result for the currently requested test can unlock Apply."""
        if (self._exit_emitted or self._finished_successfully or not self._awaiting_test
                or self._test_configuration != self._configuration()):
            return
        self.pending = False
        self._pending_mode = None
        self._awaiting_test = False
        normalized = normalize_utterance(str(heard or ""))
        # The named phrase is accepted as well as explicitly selected variants.
        accepted = (normalize_utterance(self.desired_phrase()), *self.selected_aliases())
        self._test_passed = bool(passed) and normalized in accepted
        self._approved_configuration = self._configuration() if self._test_passed else None
        if self._test_passed:
            self.set_feedback(f"測試通過：辨識為「{normalized}」。可以套用這個咒語唸法。")
        else:
            self.set_feedback(f"這次辨識為「{normalized or '未辨識到咒語'}」，尚未通過。請再試一次，或重新記錄。")
        self._update_controls()

    def _update_controls(self):
        alive = not self._exit_emitted and not self._finished_successfully
        idle = alive and not self.pending
        self.phrase_input.setEnabled(idle)
        self.language_combo.setEnabled(idle)
        self.record_button.setEnabled(idle and bool(self.desired_phrase()) and len(self.samples) < 3)
        self.record_button.setText("正在記錄…" if self.pending and self._pending_mode == "sample"
                                   else f"記錄第 {min(3, len(self.samples) + 1)} 次")
        self.restart_button.setEnabled(alive and bool(self.samples or self.pending))
        for checkbox in self.candidate_checkboxes:
            checkbox.setEnabled(idle)
        self.test_button.setEnabled(idle and len(self.samples) == 3 and bool(self.selected_aliases()))
        self.test_button.setText("正在測試…" if self.pending and self._pending_mode == "test" else "試唸咒語")
        approved = self._test_passed and self._approved_configuration == self._configuration()
        self.apply_button.setEnabled(idle and approved)
        self.apply_button.setText("正在套用…" if self.pending and self._pending_mode == "save" else "保存並套用")
        self.cancel_button.setEnabled(alive)

    def _request(self, mode, signal):
        if self.pending or self._dispatching or self._exit_emitted or self._finished_successfully:
            return
        self.pending = True
        self._pending_mode = mode
        self._dispatching = True
        self._update_controls()
        try:
            signal.emit()
        finally:
            self._dispatching = False

    def _request_sample(self):
        if self.desired_phrase() and len(self.samples) < 3:
            self._invalidate_test()
            self._request("sample", self.sample_requested)

    def _request_test(self):
        if (len(self.samples) != 3 or not self.selected_aliases() or self.pending
                or self._dispatching or self._exit_emitted or self._finished_successfully):
            return
        self._invalidate_test()
        self._test_configuration = self._configuration()
        self._awaiting_test = True
        self._request("test", self.test_requested)

    def _request_save(self):
        if self._test_passed and self._approved_configuration == self._configuration():
            self._request("save", self.save_requested)

    def _request_restart(self):
        if self._exit_emitted or self._finished_successfully or self._dispatching:
            return
        self._clear_collection()
        self.set_feedback("請用同一種唸法，重新記錄三次。")
        self.restart_requested.emit()

    def finish_success(self):
        if (self._exit_emitted or self._finished_successfully or not self._test_passed
                or self._approved_configuration != self._configuration()):
            return
        self._finished_successfully = True
        self.pending = False
        self._update_controls()
        self.accept()

    def reject(self):
        if self._finished_successfully:
            return
        if not self._exit_emitted:
            self._exit_emitted = True
            self.pending = False
            self._invalidate_test()
            self._update_controls()
            self.exit_requested.emit()
        super().reject()

    def closeEvent(self, event):
        if not self._finished_successfully:
            self.reject()
        event.accept()
