"""Optional comfortable hand-circle range setup; the app owns fresh XY capture."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from .i18n import QDialog, QFrame, QPushButton

from .gesture_calibration_dialog import _PreviewLabel, _label


class HandAimDialog(QDialog):
    """Record two positions in the supplied mirrored preview, without sensing."""

    sample_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step = 0
        self.pending = False
        self.ready = False
        self._required = 6
        self._exit_emitted = False
        self._finished_successfully = False
        self._sample_dispatching = False
        self.setWindowTitle("KON · 調整手部移動範圍（選用）")
        self.setModal(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(560, 640)
        self.resize(620, 700)
        self.setStyleSheet("""
            QDialog { background: #111018; color: #eee6f1; }
            QLabel { background: transparent; color: #eee6f1; }
            QFrame#rangeStage { background: #201b29; border: 1px solid #443549; border-radius: 12px; }
            QPushButton { background: #2b2331; color: #eee6f1; border: 1px solid #5b485e;
                border-radius: 9px; padding: 9px 16px; min-height: 20px;
                font-family: 'PingFang TC'; font-size: 13px; }
            QPushButton:hover { background: #443040; }
            QPushButton:focus { border: 2px solid #ffbd91; }
            QPushButton#recordRange { background: #f5ab80; color: #2b160e; border-color: #f5ab80; }
            QPushButton:disabled { background: #29232e; color: #8c7d94; border-color: #3c3343; }
        """)
        self._build()
        self._shortcuts = []
        for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(self._request_sample)
            self._shortcuts.append(shortcut)
        escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        escape.setContext(Qt.WidgetWithChildrenShortcut)
        escape.setAutoRepeat(False)
        escape.activated.connect(self.reject)
        self._shortcuts.append(escape)
        self.set_step(0)
        self.record_button.setFocus(Qt.OtherFocusReason)

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(10)
        title = _label("調整手部移動範圍", size=20)
        title.setFont(QFont("PingFang TC", 20, QFont.DemiBold))
        outer.addWidget(title)
        self.scope_label = _label("這是選用設定，未調整時可直接使用預設範圍。記錄兩個舒適位置，讓手圈的移動對應螢幕。", muted=True)
        outer.addWidget(self.scope_label)
        stage = QFrame(self)
        stage.setObjectName("rangeStage")
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(14, 12, 14, 12)
        stage_layout.setSpacing(5)
        self.step_label = _label("", size=14)
        self.step_label.setFont(QFont("PingFang TC", 14, QFont.DemiBold))
        self.instructions_label = _label("", muted=True)
        stage_layout.addWidget(self.step_label)
        stage_layout.addWidget(self.instructions_label)
        outer.addWidget(stage)
        self.preview_label = _PreviewLabel(self)
        outer.addWidget(self.preview_label, 1)
        self.preview_note_label = _label("位置以這裡的鏡像預覽為準：畫面左上 → 畫面右下。", size=11, muted=True)
        outer.addWidget(self.preview_note_label)
        self.progress_label = _label("", size=12)
        self.reason_label = _label("", muted=True)
        self.feedback_label = _label("", size=13)
        self.feedback_label.setStyleSheet("color: #ffd0ad;")
        self.feedback_label.setMinimumHeight(20)
        outer.addWidget(self.progress_label)
        outer.addWidget(self.reason_label)
        outer.addWidget(self.feedback_label)
        self.capture_hint_label = _label("按下記錄後會取樣新的位置；請停住手圈，直到畫面提示完成。", size=11, muted=True)
        outer.addWidget(self.capture_hint_label)
        controls = QHBoxLayout()
        controls.addStretch()
        self.cancel_button = QPushButton("取消（Esc）", self)
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.setAutoRepeat(False)
        self.cancel_button.clicked.connect(self.reject)
        self.record_button = QPushButton("", self)
        self.record_button.setObjectName("recordRange")
        self.record_button.setAutoDefault(False)
        self.record_button.setAutoRepeat(False)
        self.record_button.clicked.connect(self._request_sample)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.record_button)
        outer.addLayout(controls)

    def set_step(self, step=0):
        if step not in (0, 1) or isinstance(step, bool):
            raise ValueError("手部移動範圍只有左上與右下兩步。")
        self.step = step
        if step == 0:
            self.step_label.setText("左上位置  ·  1 / 2")
            self.instructions_label.setText("將手圈移到鏡像預覽的左上方，選擇能舒服維持的位置，再按記錄。")
            reason = "手圈到達左上位置後，停住片刻再記錄。"
        else:
            self.step_label.setText("右下位置  ·  2 / 2")
            self.instructions_label.setText("將手圈移到鏡像預覽的右下方。選擇比前一步更靠右、更靠下的舒適位置，再按記錄。")
            reason = "移到右下位置，這次會記錄新的位置畫面。"
        self.set_state(0, self._required, reason, False)

    def set_preview(self, image):
        # CameraWorker already supplies mirrored pixels; never mirror twice.
        self.preview_label.set_image(image)

    def set_state(self, count, required, reason, ready, pending=False):
        count, required = max(0, int(count)), max(1, int(required))
        self._required = required
        self.pending, self.ready = bool(pending), bool(ready)
        state = "正在記錄位置" if self.pending else "位置已穩定" if self.ready else "等待手圈停穩"
        self.progress_label.setText(f"{state}  ·  {count} / {required} 筆新影格")
        self.reason_label.setText(str(reason or ""))
        self._update_button()
        self._fit_content_height()

    def set_feedback(self, text):
        self.feedback_label.setText(str(text or ""))
        self._fit_content_height()

    def _fit_content_height(self):
        needed = self.layout().totalHeightForWidth(max(560, self.width()))
        minimum = max(640, needed)
        if self.minimumHeight() != minimum:
            self.setMinimumHeight(minimum)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "capture_hint_label"):
            self._fit_content_height()

    def _update_button(self):
        text = "正在記錄…" if self.pending else "記錄左上位置" if self.step == 0 else "記錄右下位置"
        self.record_button.setText(text + ("（空白鍵）" if not self.pending else ""))
        self.record_button.setEnabled(not self.pending and not self._finished_successfully and not self._exit_emitted)

    def _request_sample(self):
        if self.pending or self._sample_dispatching or self._exit_emitted or self._finished_successfully:
            return
        self._sample_dispatching = True
        self.pending = True
        self._update_button()
        try:
            self.sample_requested.emit()
        finally:
            self._sample_dispatching = False

    def finish_success(self):
        if self._exit_emitted or self._finished_successfully:
            return
        self._finished_successfully = True
        self._update_button()
        self.accept()

    def reject(self):
        if self._finished_successfully:
            return
        if not self._exit_emitted:
            self._exit_emitted = True
            self._update_button()
            self.exit_requested.emit()
        super().reject()

    def closeEvent(self, event):
        if not self._finished_successfully:
            self.reject()
        event.accept()
