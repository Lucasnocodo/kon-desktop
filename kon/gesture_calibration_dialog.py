"""Two fresh holds per personal gesture angle; sensing stays in the app."""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLayout, QProgressBar, QScrollArea, QSizePolicy, QVBoxLayout
from .i18n import QDialog, QFrame, QLabel, QPushButton, QWidget

from .gesture_calibration import STANDARD_GESTURE_ANGLES

_MATCH_PROMPT = "重新比出手勢後，這裡會顯示與第一次的比對結果"
_STANDARD_MATCH_PROMPT = "重新比出這個角度的手勢後，這裡會顯示與此角度第一次的比對結果"
_ANGLE_INSTRUCTIONS = (
    "讓手形正面朝向鏡頭，記錄中間角度。",
    "將手掌相對鏡頭稍微向上傾斜，記錄上方角度。",
    "將手掌相對鏡頭稍微向下傾斜，記錄下方角度。",
    "將手掌相對鏡頭稍微向左傾斜，記錄左側角度。",
    "將手掌相對鏡頭稍微向右傾斜，記錄右側角度。",
)


def _label(text, *, size=13, muted=False):
    label = QLabel(str(text))
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setFont(QFont("PingFang TC", size))
    if muted:
        label.setStyleSheet("color: #b0a5bd;")
    return label


class _PreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("手勢校正鏡頭預覽")
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setText("等待鏡頭畫面\n讓整隻手完整入鏡")
        self.setStyleSheet("background: #0c0b12; color: #b0a5bd; border-radius: 12px;")

    def set_image(self, image):
        self.image = image.copy() if isinstance(image, QImage) and not image.isNull() else None
        self._refresh()

    def _refresh(self):
        if self.image is None:
            self.clear()
            self.setText("等待鏡頭畫面\n讓整隻手完整入鏡")
        else:
            self.setPixmap(QPixmap.fromImage(self.image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class GestureCalibrationDialog(QDialog):
    """Emit user actions only. The parent collects, verifies, and saves samples."""

    sample_requested = Signal()
    exit_requested = Signal()
    save_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step = 0
        self.pending = False
        self.ready = False
        self._append_angle_count = None
        self._standard_angle_index = None
        self._required = 6
        self._exit_emitted = False
        self._finished_successfully = False
        self._sample_dispatching = False
        self._review_mode = False
        self._review_matched = False
        self._count = 0
        self.setWindowTitle("KON · 記錄召喚手勢")
        self.setModal(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(780, 620)
        self.resize(960, 700)
        self.setStyleSheet("""
            QDialog { background: #111018; color: #eee6f1; }
            QLabel { background: transparent; color: #eee6f1; }
            QFrame#gestureStage { background: #201b29; border: 1px solid #443549; border-radius: 12px; }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: #111018; }
            QLabel#angleCurrent { color: #ffcca8; background: #3b2c30; border: 1px solid #b77c5c; border-radius: 7px; padding: 7px 4px; }
            QLabel#angleComplete { color: #c8b7d4; background: #282230; border-radius: 7px; padding: 7px 4px; }
            QLabel#angleUpcoming { color: #a497ae; border: 1px solid #3d3345; border-radius: 7px; padding: 7px 4px; }
            QProgressBar { background: #29232e; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }
            QProgressBar::chunk { background: #f5ab80; border-radius: 4px; }
            QPushButton { background: #2b2331; color: #eee6f1; border: 1px solid #5b485e;
                border-radius: 9px; padding: 9px 16px; min-height: 20px;
                font-family: 'PingFang TC'; font-size: 13px; }
            QPushButton:hover { background: #443040; }
            QPushButton:focus { border: 2px solid #ffbd91; }
            QPushButton#recordGesture { background: #f5ab80; color: #2b160e; border-color: #f5ab80; }
            QPushButton:disabled { background: #29232e; color: #8c7d94; border-color: #3c3343; }
            QPushButton#recordGesture:disabled { background: #29232e; color: #8c7d94; border-color: #3c3343; }
        """)
        self._build()
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._fit_content_height)
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
        self.set_state(0, 6, "請比出要使用的召喚手形，再開始記錄。", False)
        self.record_button.setFocus(Qt.OtherFocusReason)

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(12)
        self.title_label = _label("記錄我的召喚手勢", size=20)
        self.title_label.setFont(QFont("PingFang TC", 20, QFont.DemiBold))
        outer.addWidget(self.title_label)
        self.scope_label = _label("已保存的九點瞄準校正會保留。這裡記錄你用來召喚狐狸的手形。", size=12, muted=True)
        outer.addWidget(self.scope_label)

        self.angle_track = QWidget(self)
        track_layout = QHBoxLayout(self.angle_track)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(6)
        self.angle_markers = []
        for index, angle in enumerate(STANDARD_GESTURE_ANGLES):
            marker = _label(f"{index + 1}  {angle}", size=12)
            marker.setAlignment(Qt.AlignCenter)
            marker.setWordWrap(False)
            track_layout.addWidget(marker, 1)
            self.angle_markers.append(marker)
        self.angle_track.hide()
        outer.addWidget(self.angle_track)

        body = QHBoxLayout()
        body.setSpacing(20)
        camera_column = QVBoxLayout()
        camera_column.setSpacing(9)
        self.preview_label = _PreviewLabel(self)
        camera_column.addWidget(self.preview_label, 1)
        self.shape_rule_label = _label("一個手形，依序記錄不同朝向。手指姿勢不變，只調整手掌角度。", size=12)
        camera_column.addWidget(self.shape_rule_label)
        self.capture_hint_label = _label("按下記錄後會等待清楚的新畫面。取樣期間請保持手形，直到畫面提示完成。", size=11, muted=True)
        camera_column.addWidget(self.capture_hint_label)
        body.addLayout(camera_column, 5)

        self.content_scroll = QScrollArea(self)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content_scroll.setMinimumWidth(300)
        self.content_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.content_scroll.setAccessibleName("目前校正步驟與辨識狀態")
        self.content_widget = QWidget(self.content_scroll)
        details = QVBoxLayout(self.content_widget)
        details.setContentsMargins(0, 0, 5, 0)
        details.setSpacing(12)
        details.setSizeConstraint(QLayout.SetMinimumSize)
        stage = QFrame(self)
        stage.setObjectName("gestureStage")
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(14, 12, 14, 12)
        stage_layout.setSpacing(5)
        self.angle_label = _label("", size=13)
        self.angle_label.hide()
        stage_layout.addWidget(self.angle_label)
        self.step_label = _label("", size=14)
        self.step_label.setFont(QFont("PingFang TC", 14, QFont.DemiBold))
        self.action_label = _label("", size=18)
        self.action_label.setFont(QFont("PingFang TC", 18, QFont.DemiBold))
        self.instructions_label = _label("", size=12, muted=True)
        stage_layout.addWidget(self.step_label)
        stage_layout.addWidget(self.action_label)
        stage_layout.addWidget(self.instructions_label)
        details.addWidget(stage)
        self.match_label = _label(_MATCH_PROMPT, size=13)
        self.match_label.setStyleSheet("color: #d6c3e3;")
        self.match_label.hide()
        details.addWidget(self.match_label)
        self.progress_label = _label("", size=12)
        self.sample_progress = QProgressBar(self)
        self.sample_progress.setRange(0, 6)
        self.sample_progress.setValue(0)
        self.sample_progress.setTextVisible(False)
        self.sample_progress.setAccessibleName("本次穩定取樣進度")
        self.reason_label = _label("", size=12, muted=True)
        self.feedback_label = _label("", size=13)
        self.feedback_label.setStyleSheet("color: #ffd0ad;")
        self.feedback_label.setMinimumHeight(20)
        details.addWidget(self.progress_label)
        details.addWidget(self.sample_progress)
        details.addWidget(self.reason_label)
        details.addWidget(self.feedback_label)
        details.addStretch(1)
        self.content_scroll.setWidget(self.content_widget)
        body.addWidget(self.content_scroll, 3)
        outer.addLayout(body, 1)
        controls = QHBoxLayout()
        self.keyboard_hint_label = _label("空白鍵／Enter 記錄　Esc 取消", size=11, muted=True)
        controls.addWidget(self.keyboard_hint_label, 1)
        self.retry_button = QPushButton("重新記錄", self)
        self.retry_button.setAutoDefault(False)
        self.retry_button.clicked.connect(self._request_retry)
        self.retry_button.hide()
        self.cancel_button = QPushButton("取消（Esc）", self)
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.setAutoRepeat(False)
        self.cancel_button.clicked.connect(self.reject)
        self.record_button = QPushButton("", self)
        self.record_button.setObjectName("recordGesture")
        self.record_button.setAutoDefault(False)
        self.record_button.setAutoRepeat(False)
        self.record_button.clicked.connect(self._request_sample)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.record_button)
        outer.addLayout(controls)
        self.setTabOrder(self.record_button, self.cancel_button)
        self.setTabOrder(self.cancel_button, self.retry_button)

    def set_step(self, step=0):
        if step not in (0, 1) or isinstance(step, bool):
            raise ValueError("召喚手勢記錄只有第一與第二步。")
        self.step = step
        self.pending = False
        self.ready = False
        self.match_label.setVisible(step == 1)
        self.match_label.setText(self._match_prompt())
        reason = self._apply_step_copy()
        self.set_state(0, self._required, reason, False)


    def set_review_mode(self, name):
        """Show a live practice stage; saving needs fresh evidence from the app."""
        self._review_mode = True
        self._review_matched = False
        self.pending = False
        self.ready = False
        self.title_label.setText(f"試做並儲存「{name}」")
        self.instructions_label.setText("放鬆手，再比出剛才的手形；確認辨識穩定後儲存。")
        self.action_label.setText("重新比出剛才的手形")
        self.shape_rule_label.setText("先放鬆手，再重新比出；看看目前手形能否穩定辨識。")
        self.step_label.setText("最後一步 · 試做確認")
        self.angle_label.hide()
        self.match_label.show()
        self.match_label.setText("等待試做；成功後即可儲存。")
        self.reason_label.setText("")
        self.feedback_label.setText("")
        self.progress_label.setText("手勢記錄已完成，請試做一次。")
        self.sample_progress.hide()
        self.retry_button.show()
        self.keyboard_hint_label.setText("Enter 儲存　Esc 取消")
        self.capture_hint_label.setText("試做成功後，按「儲存並使用」套用這個手勢。")
        self._update_angle_track()
        self._update_button()
        self._fit_content_height()

    def set_review_match(self, matched, hint=""):
        """Receive current practice evidence; stale or invalid frames must clear it."""
        if not self._review_mode:
            return
        self._review_matched = bool(matched)
        self.ready = self._review_matched
        self.match_label.setText(str(hint or ("試做成功，可以儲存。" if matched else "等待試做；成功後即可儲存。")))
        self.progress_label.setText("試做成功" if matched else "等待試做成功")
        self._update_button()
        self._fit_content_height()

    def leave_review_mode(self):
        """Return to capture before the parent starts a fresh enrollment."""
        self._review_mode = False
        self._review_matched = False
        self.pending = False
        self.ready = False
        self.retry_button.hide()
        self.sample_progress.show()
        self.keyboard_hint_label.setText("空白鍵／Enter 記錄　Esc 取消")
        self.capture_hint_label.setText("按下記錄後會等待清楚的新畫面。取樣期間請保持手形，直到畫面提示完成。")
        self.setWindowTitle("KON · 記錄召喚手勢")
        self.title_label.setText("記錄我的召喚手勢")
        self.scope_label.setText("已保存的九點瞄準校正會保留。這裡記錄你用來召喚狐狸的手形。")
        self.shape_rule_label.setText("一個手形，依序記錄不同朝向。手指姿勢不變，只調整手掌角度。")
        self._apply_step_copy()
        self._update_button()
        self._fit_content_height()
        self.match_label.setVisible(self.step == 1)
        self._update_angle_track()

    def _request_retry(self):
        if self._review_mode and not self.pending and not self._exit_emitted and not self._finished_successfully:
            self._review_matched = False
            self.ready = False
            self._update_button()
            self.retry_requested.emit()

    def set_standard_angle_mode(self, angle_index):
        """Show a required angle in the five-angle enrollment sequence."""
        if type(angle_index) is not int or not 0 <= angle_index < len(STANDARD_GESTURE_ANGLES):
            raise ValueError("標準手勢角度索引必須介於 0 與 4。")
        if self._review_mode:
            self.leave_review_mode()
        self._standard_angle_index = angle_index
        self._append_angle_count = None
        self.setWindowTitle("KON · 記錄五個召喚手勢角度")
        self.title_label.setText("記錄召喚手勢的五個角度")
        self.scope_label.setText("已保存的瞄準校正會保留。五個角度完成後一次保存；取消會保留原手勢。")
        sequence = " → ".join(STANDARD_GESTURE_ANGLES)
        self.angle_label.setText(f"角度 {angle_index + 1} / {len(STANDARD_GESTURE_ANGLES)}：{STANDARD_GESTURE_ANGLES[angle_index]}  ·  {sequence}")
        self.angle_label.show()
        self.angle_track.show()
        self._update_angle_track()
        if self.match_label.text() in (_MATCH_PROMPT, _STANDARD_MATCH_PROMPT):
            self.match_label.setText(self._match_prompt())
        self._apply_step_copy()
        self._update_button()
        self._fit_content_height()

    def set_append_angle_mode(self, existing_count):
        """Explain adding one angle without resetting live capture/comparison."""
        if type(existing_count) is not int or existing_count < 1:
            raise ValueError("已有手勢角度數量必須是正整數。")
        self._append_angle_count = existing_count
        self._standard_angle_index = None
        self.angle_label.hide()
        self.angle_track.hide()
        if self.match_label.text() in (_MATCH_PROMPT, _STANDARD_MATCH_PROMPT):
            self.match_label.setText(self._match_prompt())
        title = "增加手勢角度"
        self.setWindowTitle(f"KON · {title}")
        self.title_label.setText(title)
        self.scope_label.setText(f"已有{existing_count}個角度會保留。以同一隻手維持同樣手形，從另一方向記錄")
        self._apply_step_copy()
        self._update_button()
        self._fit_content_height()

    def _apply_step_copy(self):
        if self._review_mode:
            return "請完成試做後儲存。"
        self.action_label.setText("比出手形，保持不動" if self.step == 0 else "放開手，再比一次")
        if self._standard_angle_index is not None:
            angle = STANDARD_GESTURE_ANGLES[self._standard_angle_index]
            self.action_label.setText(("手形正面朝向鏡頭" if self._standard_angle_index == 0 else f"同一手形，向{angle}傾斜")
                                      if self.step == 0 else "放開手，再比一次")
            if self.step == 0:
                self.step_label.setText(f"{angle}角度 · 第一次記錄  ·  1 / 2")
                self.instructions_label.setText(
                    _ANGLE_INSTRUCTIONS[self._standard_angle_index]
                    + "同一隻手維持相同手形，只改變手與鏡頭之間的角度。不需要移動瞄準目標。")
                return f"請用同一隻手比出召喚手形，準備記錄{angle}角度。"
            self.step_label.setText(f"{angle}角度 · 確認一次  ·  2 / 2")
            self.instructions_label.setText(
                f"先放開手勢，再用同一隻手從剛才的{angle}角度重做相同手形。記錄新的畫面，只與此角度的第一次記錄比對。")
            return f"請先放開，再從同一個{angle}角度重新比出相同手形。"
        if self._append_angle_count is not None:
            if self.step == 0:
                self.step_label.setText("增加手勢角度  ·  1 / 2")
                self.instructions_label.setText("轉到常用的另一角度後記錄。以同一隻手維持相同手形，讓整隻手完整入鏡，保持直到完成。")
                return "轉到常用的另一角度後記錄。"
            self.step_label.setText("重做這個新角度，確認一次  ·  2 / 2")
            self.instructions_label.setText("先放開手勢，再用同一隻手從剛才的新角度重做相同手形。這次會記錄新的畫面，只與這個新角度的第一次記錄比對。")
            return "請先放開，再從同一個新角度重新比出相同手形。"
        if self.step == 0:
            self.step_label.setText("第一次記錄  ·  1 / 2")
            self.instructions_label.setText("比出你要用來召喚狐狸的手形，讓整隻手完整入鏡。按下記錄後，保持手形直到完成。")
            return "請比出要使用的召喚手形，再開始記錄。"
        else:
            self.step_label.setText("重新比出手勢，確認一次  ·  2 / 2")
            self.instructions_label.setText("先放開手勢，再用同一隻手重新比出相同手形。這次會記錄新的畫面，確認手勢可以再次辨識。")
            return "請先放開，再重新比出同一手形。"

    def set_preview(self, image):
        self.preview_label.set_image(image)

    def set_state(self, count, required, reason, ready, pending=False):
        count, required = max(0, int(count)), max(1, int(required))
        self._required = required
        self._count = count
        self.pending, self.ready = bool(pending), bool(ready)
        state = "正在取樣" if self.pending else "畫面已穩定" if self.ready else "等待手部穩定"
        self.progress_label.setText(f"{state}  ·  {count} / {required} 筆新影格")
        self.sample_progress.setRange(0, required)
        self.sample_progress.setValue(min(count, required))
        self.sample_progress.setAccessibleDescription(f"已取得 {min(count, required)} 筆，共需 {required} 筆穩定新畫面。")
        self.reason_label.setText(str(reason or ""))
        self._update_button()
        self._fit_content_height()

    def set_feedback(self, text):
        self.feedback_label.setText(str(text or ""))
        self._fit_content_height()

    def set_match_status(self, text):
        """Update live comparison separately from capture progress and errors."""
        self.match_label.setText(str(text or self._match_prompt()))
        self.match_label.setVisible(self.step == 1 or self._review_mode)
        self._fit_content_height()

    def _fit_content_height(self):
        # Long tracking errors scroll inside the side panel instead of making
        # the modal taller than the screen or hiding the cancel action.
        for label in (self.angle_label, self.step_label, self.action_label,
                      self.instructions_label, self.match_label, self.reason_label,
                      self.feedback_label, self.progress_label, self.scope_label,
                      self.shape_rule_label, self.capture_hint_label):
            width = max(40, label.contentsRect().width())
            needed = label.fontMetrics().boundingRect(QRect(0, 0, width, 10000), Qt.TextWordWrap, label.text()).height()
            label.setMinimumHeight(max(0, needed))
        self.content_widget.layout().activate()
        self.content_widget.setMinimumHeight(self.content_widget.layout().minimumSize().height())

    def _update_angle_track(self):
        for index, marker in enumerate(self.angle_markers):
            complete = self._review_mode or (self._standard_angle_index is not None and index < self._standard_angle_index)
            current = not self._review_mode and index == self._standard_angle_index
            marker.setObjectName("angleComplete" if complete else "angleCurrent" if current else "angleUpcoming")
            marker.setText(f"{'✓' if complete else index + 1}  {STANDARD_GESTURE_ANGLES[index]}")
            marker.setAccessibleName(f"{STANDARD_GESTURE_ANGLES[index]}角度，{'已完成' if complete else '目前步驟' if current else '尚未記錄'}")
            marker.style().unpolish(marker)
            marker.style().polish(marker)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "capture_hint_label"):
            self._fit_content_height()
        if hasattr(self, "_layout_timer"):
            self._layout_timer.start(0)

    def showEvent(self, event):
        super().showEvent(event)
        self._layout_timer.start(0)

    def _update_button(self):
        if self.pending:
            text = "正在儲存…" if self._review_mode else "正在記錄…"
        elif self._review_mode:
            text = "儲存並使用"
        elif self._standard_angle_index is not None:
            angle = STANDARD_GESTURE_ANGLES[self._standard_angle_index]
            text = f"記錄{angle}角度" if self.step == 0 else f"確認{angle}角度"
        elif self._append_angle_count is not None:
            text = "記錄這個角度" if self.step == 0 else "確認新角度"
        else:
            text = "記錄我的手勢" if self.step == 0 else "確認手勢"
        self.record_button.setText(text + ("（空白鍵）" if not self.pending and not self._review_mode else ""))
        enabled = not self.pending and not self._finished_successfully and not self._exit_emitted
        self.record_button.setEnabled(enabled and (not self._review_mode or self._review_matched))
        self.retry_button.setEnabled(enabled)

    def _match_prompt(self):
        return _STANDARD_MATCH_PROMPT if self._standard_angle_index is not None else _MATCH_PROMPT

    def _request_sample(self):
        if self.pending or self._sample_dispatching or self._exit_emitted or self._finished_successfully:
            return
        if self._review_mode:
            if not self._review_matched:
                return
            self._review_matched = False
            self.pending = True
            self._update_button()
            self.save_requested.emit()
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
        if not self._finished_successfully and not self._exit_emitted:
            self._exit_emitted = True
            self._update_button()
            self.exit_requested.emit()
        super().reject()

    def closeEvent(self, event):
        if not self._finished_successfully:
            self.reject()
        event.accept()
