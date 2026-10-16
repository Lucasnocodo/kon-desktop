"""KON desktop control room. Sensors start only from an explicit user action."""
from __future__ import annotations

from .paths import resource_root, state_root as user_state_dir
from .i18n import get_language, preferred_language, retranslate_all, save_language, set_language

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import median
import sys
import time

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLayout, QScrollArea, QVBoxLayout
from .i18n import QComboBox, QDoubleSpinBox, QFrame, QLabel, QLineEdit, QMainWindow, QMenu, QPushButton, QSpinBox, QSystemTrayIcon, QWidget

from .desktop import DesktopTarget, restore_receipt, scan_desktop, trash_target, validate_target_position
from .calibration import CalibrationBuffer
from .calibration_diagnostics import FitStatus, analyze_calibration
from .calibration_history import load_last_attempt, save_attempt
from .calibration_report import CalibrationReportDialog
from .calibration_store import compatible_configuration, gesture_profile_path, load_calibration, machine_key, save_calibration
from .cursor_motion import CursorMotion
from .aim_dwell import AimDwellWeight
from .gesture_calibration import (
    STANDARD_GESTURE_ANGLES, GestureCalibration, GestureEnrollmentBuffer,
    add_gesture_angle, extract_gesture_features,
)
from .gesture_calibration_dialog import GestureCalibrationDialog
from .hand_aim import HandAimCalibration, HandCursorSmoother
from .hand_aim_dialog import HandAimDialog
from .personal_calibration import PersonalCalibration
from .geometry import ScreenCalibration, TriggerGate
from .grid_selection import GridSelector
from .overlay import HeroWidget, ScreenOverlay
from .pronunciation import PronunciationProfile, validate_pronunciation_phrase
from .pronunciation_dialog import PronunciationDialog
from .speech import DEFAULT_TRIGGER_SETTING, SpeechWorker, parse_trigger_phrases
from .audio_devices import list_microphones, microphone_setting as encode_microphone
from .selection_settings import SelectionSettings, save_selection_settings
from .vision import CameraWorker, hand_circle_cursor

ROOT = resource_root()
CALIBRATION_POINTS = [(x, y) for y in (0.18, 0.5, 0.82) for x in (0.18, 0.5, 0.82)]
DEMO_DOCUMENTS = (
    ("草稿.txt", .25, .5), ("靈感.md", .5, .43), ("過期企劃.pdf", .75, .5),
    ("會議紀錄.md", .25, .24), ("待辦清單.txt", .5, .24), ("測試報告.pdf", .75, .24),
    ("備忘錄.txt", .25, .76), ("企劃草案.md", .5, .76), ("資料摘要.pdf", .75, .76),
)

STYLE = """
QWidget { background:#0d0c13; color:#eee9fa; font-family:'PingFang TC'; font-size:13px; }
QFrame#card { background:#15131f; border:1px solid #302b42; border-radius:18px; }
QLabel { background:transparent; }
QLabel#eyebrow { color:#dcb08f; font-size:10px; font-weight:600; letter-spacing:2px; }
QLabel#muted { color:#9990ad; }
QLabel#status { color:#cec2e6; background:#201a30; border:1px solid #3c3153; border-radius:10px; padding:12px; }
QPushButton { border:1px solid #443852; border-radius:10px; background:#211a30; padding:7px 15px; min-height:18px; font-weight:600; }
QPushButton:hover { background:#30253f; border-color:#a087c3; }
QPushButton:pressed { background:#49365f; }
QPushButton:disabled { color:#635b73; border-color:#2c2538; background:#17131f; }
QPushButton#primary { background:#f5ab80; border-color:#f5ab80; color:#2b160e; }
QPushButton#primary:hover { background:#ffd0ad; }
QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit { background:#201a2c; border:1px solid #443852; border-radius:8px; padding:7px; }
QComboBox::drop-down { border:none; width:24px; }
QComboBox QAbstractItemView { background:#211a30; selection-background-color:#473357; }
"""


class JobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function, self.signals = function, JobSignals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


@dataclass
class ActionState:
    serial: int
    target_id: str
    real_file: bool
    succeeded: bool = False
    animation_done: bool = False
    refreshing: bool = False


class Preview(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 280)
        self.image = None
        self.setStyleSheet("border-radius:12px; background:#100e17;")

    def set_image(self, image):
        self.image = image
        self._refresh()

    def _refresh(self):
        if self.image is not None:
            self.setPixmap(QPixmap.fromImage(self.image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


def label(text, object_name=None, size=None):
    item = QLabel(text)
    item.setWordWrap(True)
    item.setTextFormat(Qt.PlainText)
    if object_name:
        item.setObjectName(object_name)
    if size:
        font = item.font()
        font.setPointSize(size)
        item.setFont(font)
        item.setStyleSheet(f"font-size:{size}px; font-weight:600;")
    return item


def card():
    widget = QFrame()
    widget.setObjectName("card")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(12)
    return widget, layout


class Window(QMainWindow):
    def __init__(self, *, aim_mode=None):
        super().__init__()
        if aim_mode is None:
            aim_mode = self.saved_aim_mode()
        if aim_mode not in ("hand", "eye_ray"):
            raise ValueError("瞄準方式無效")
        self.aim_mode = aim_mode
        self.setWindowTitle("KON · 狐狸召喚")
        self.resize(1140 if get_language() == 'en' else 1040, 880)
        self.setMinimumSize(1100 if get_language() == 'en' else 960, 840)
        self.setStyleSheet(STYLE)
        self.screen = QApplication.primaryScreen()
        self.overlay = ScreenOverlay(self.screen)
        self.overlay.sample_requested.connect(self.capture_calibration)
        self.overlay.exit_requested.connect(self.cancel_calibration)
        self.overlay.bite.connect(self.on_bite)
        self.overlay.animation_done.connect(self.on_animation_done)
        self.camera = self.speech = None
        self.running = self.armed = self.testing = self.calibrating = False
        self.voice_active = False
        self.voice_armed_at = 0.0
        self.voice_target = None
        self.trigger_setting = DEFAULT_TRIGGER_SETTING
        self.trigger_display = DEFAULT_TRIGGER_SETTING
        self.trigger_language = None
        self.microphone_setting = None
        self.voice_trigger_mode = "loud_sound"
        self.pronunciation_profile = None
        self.pronunciation_calibrating = False
        self.pronunciation_dialog = None
        self.pronunciation_worker = None
        self.pronunciation_worker_owned = False
        self._retiring_pronunciation_workers = []
        self.pronunciation_samples = []
        self.pronunciation_request = None
        self.pronunciation_deadline = None
        self.pronunciation_serial = 0
        self.pronunciation_verification = None
        self.calibration = None
        self.selection_settings = SelectionSettings()
        self.stable_dwell_seconds = self.selection_settings.stable_seconds
        self.stable_dwell_multiplier = self.selection_settings.stable_multiplier
        self.hand_movement_speed_limit = self.selection_settings.movement_speed_limit
        self.gate = self.selection_gate()
        self.latest = None
        self.gesture_profile = None
        self.gesture_review_required = False
        self.gesture_review_profile = None
        self.gesture_review_passed = False
        self.gesture_review_released = False
        self.gesture_review_since = None
        self.gesture_review_count = 0
        self.gesture_review_timestamp = None
        self.gesture_calibrating = False
        self.gesture_dialog = None
        self.gesture_candidate = None
        self.gesture_append = False
        self.gesture_base_profile = None
        self.gesture_angle_index = 0
        self.gesture_staged_profile = None
        self.gesture_buffer = GestureEnrollmentBuffer()
        self.gesture_capture_deadline = None
        self.hand_smoother = HandCursorSmoother()
        self.aim_point = None
        self.cursor_motion = CursorMotion()
        self.cursor_timer = QTimer(self)
        self.cursor_timer.setTimerType(Qt.PreciseTimer)
        self.cursor_timer.setInterval(16)
        self.cursor_timer.timeout.connect(self.animate_cursor)
        self.grid_selector = GridSelector()
        self._selection_layout = None
        self._selection_frame_at = None
        self.lock_release_cells = self.selection_settings.release_cells
        self.hand_range_calibrating = False
        self.hand_range_dialog = None
        self.hand_range_points = []
        self.hand_range_deadline = None
        self.hand_range_buffer = CalibrationBuffer(min_span=.5, max_jitter=.012, hand_only=True)
        self.calibration_buffer = CalibrationBuffer(min_span=1.0)
        self.calibration_capture_deadline = None
        self.calibration_raw = []
        self.calibration_screen_points = list(CALIBRATION_POINTS)
        self.calibration_index = 0
        self.calibration_retrying = False
        self.calibration_observations = []
        self.calibration_point_details = []
        self.calibration_diagnostics = None
        self.calibration_report = None
        self.calibration_validating = False
        self.validation_raw = []
        self.validation_frames = []
        self.validation_point_details = []
        self.validation_resume_targets = False
        self.targets = []
        self.receipt = None
        self.action_serial = 0
        self.pending_action = None
        self.active_action = None
        self.pool = QThreadPool.globalInstance()
        self._jobs = set()
        self._build_ui()
        self._build_tray()
        self._load_preferences()
        self._restore_settings()
        self._load_calibration()
        self._load_calibration_attempt()
        self._load_gesture_profile()
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(100)
        self.watchdog.timeout.connect(self.check_freshness)
        self.watchdog.start()
        self.screen.geometryChanged.connect(self.screen_changed)

    def _build_ui(self):
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(30, 24, 30, 26)
        layout.setSpacing(18)
        top = QHBoxLayout()
        brand = QVBoxLayout()
        brand.addWidget(label("KON", size=34))
        subtitle = label("狐狸召喚  /  LOCAL EXPERIMENT 01", "eyebrow")
        subtitle.setWordWrap(False)
        brand.addWidget(subtitle)
        top.addLayout(brand)
        top.addStretch()
        language_controls = QVBoxLayout()
        language_controls.addWidget(label("Language / 語言", "muted"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("繁體中文", "zh-TW")
        self.language_combo.addItem("English", "en")
        self.language_combo.setAccessibleName("Interface language")
        self.language_combo.setCurrentIndex(self.language_combo.findData(get_language()))
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_controls.addWidget(self.language_combo)
        top.addLayout(language_controls)
        self.voice_instructions = label(f"移動手圈  →  鎖定檔案\n手勢維持  +  說出 {self.trigger_display}", "muted")
        top.addWidget(self.voice_instructions)
        layout.addLayout(top)
        columns = QHBoxLayout()
        columns.setSpacing(18)
        left, view = card()
        self.camera_heading = label("CAMERA / 手部追蹤" if self.aim_mode == "hand" else "CAMERA / 手部與慣用眼追蹤", "eyebrow")
        view.addWidget(self.camera_heading)
        self.hero = HeroWidget()
        view.addWidget(self.hero, 1)
        self.preview = Preview()
        self.preview.hide()
        view.addWidget(self.preview, 1)
        self.pose_label = label("讓手勢，成為召喚的入口。", size=19)
        view.addWidget(self.pose_label)
        self.gesture_hint_label = label("啟動鏡頭後，這裡會顯示各手指的辨識狀態。", "muted")
        view.addWidget(self.gesture_hint_label)
        self.gesture_instructions = label("移動手圈控制準星，不需要眼睛入鏡。\n可在第 03 步記錄你自己的召喚手勢。" if self.aim_mode == "hand" else
                                          "食指、小指翹起；中指、無名指壓住拇指。\n或在第 03 步記錄你自己的召喚手勢。", "muted")
        view.addWidget(self.gesture_instructions)
        self.camera_label = label("鏡頭尚未啟動", "muted")
        self.mic_label = label("語音尚未啟動 · 全程在本機處理", "muted")
        view.addWidget(self.camera_label)
        view.addWidget(self.mic_label)
        columns.addWidget(left, 3)
        right, controls = card()
        controls.setSpacing(8)
        controls.setSizeConstraint(QLayout.SetMinimumSize)
        controls.addWidget(label("SUMMONING SEQUENCE", "eyebrow"))
        controls.addWidget(label("01  建立連線", size=18))
        self.aim_method = QComboBox()
        self.aim_method.addItem("眼睛穿過手圈瞄準", "eye_ray")
        self.aim_method.addItem("手圈移動控制準星", "hand")
        self.aim_method.setAccessibleName("瞄準方式")
        self.aim_method.setToolTip("眼睛穿過手圈可指向檔案；手圈位置模式以移動手圈控制準星。暫停鏡頭後可切換，選擇會保存。")
        self.aim_method.setCurrentIndex(self.aim_method.findData(self.aim_mode))
        self.aim_method.currentIndexChanged.connect(self.aim_mode_changed)
        controls.addWidget(self.aim_method)
        self.aim_description = label("", "muted")
        controls.addWidget(self.aim_description)
        settings = QHBoxLayout()
        self.eye = QComboBox()
        self.eye.addItems(["右眼瞄準", "左眼瞄準"])
        self.eye.currentIndexChanged.connect(self.settings_changed)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 31)
        self.camera_index.setPrefix("鏡頭 ")
        self.camera_index.setAccessibleName("鏡頭編號")
        self.camera_index.valueChanged.connect(self.settings_changed)
        settings.addWidget(self.eye)
        self.eye.setVisible(self.aim_mode == "eye_ray")
        settings.addWidget(self.camera_index)
        controls.addLayout(settings)
        fov_row = QHBoxLayout()
        fov_label = label("鏡頭左右視野", "muted")
        fov_row.addWidget(fov_label)
        self.fov = QDoubleSpinBox()
        self.fov.setRange(20, 120)
        self.fov.setValue(60)
        self.fov.setToolTip("這是鏡頭左右能拍到的總角度（HFOV），不是鏡頭往下傾的角度。鏡頭俯拍時不用因此改這個數值。未知時 60° 只是估計；EMEET C960 1080P 在未裁切 16:9 畫面下約為 82.15°。")
        self.fov.setSuffix("°")
        self.fov.valueChanged.connect(self.settings_changed)
        fov_row.addWidget(self.fov)
        self.fov_controls = QWidget(right)
        fov_row.setContentsMargins(0, 0, 0, 0)
        self.fov_controls.setLayout(fov_row)
        self.fov_controls.setVisible(self.aim_mode == "eye_ray")
        controls.addWidget(self.fov_controls)
        self.start_button = QPushButton("啟動鏡頭與麥克風")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_sensors)
        controls.addWidget(self.start_button)
        controls.addSpacing(8)
        self.aim_heading = label("02  手圈移動範圍" if self.aim_mode == "hand" else "02  校正瞄準", size=18)
        controls.addWidget(self.aim_heading)
        self.cal_label = label("預設範圍 · 可直接開始" if self.aim_mode == "hand" else "尚未校正 · 9 個螢幕標記", "muted")
        controls.addWidget(self.cal_label)
        self.cal_button = QPushButton("調整手部移動範圍（選用）" if self.aim_mode == "hand" else "校正眼睛 → 手圈")
        self.cal_button.clicked.connect(self.begin_calibration)
        self.cal_button.setEnabled(False)
        controls.addWidget(self.cal_button)
        self.cal_report_button = QPushButton("查看校正誤差／重測單點")
        self.cal_report_button.clicked.connect(self.show_calibration_report)
        self.cal_report_button.hide()
        controls.addWidget(self.cal_report_button)
        self.validate_button = QPushButton("檢查瞄準精度（選用）")
        self.validate_button.clicked.connect(lambda: self.begin_validation())
        self.validate_button.hide()
        controls.addWidget(self.validate_button)
        controls.addSpacing(8)
        controls.addWidget(label("03  校正召喚手勢", size=18))
        self.gesture_profile_label = label("尚未記錄 · 目前使用預設 KON 手勢", "muted")
        controls.addWidget(self.gesture_profile_label)
        self.gesture_button = QPushButton("校正靈狐手圈")
        self.gesture_button.setToolTip("以手圈瞄準並發聲召喚靈狐。記錄中間、上、下、左、右五個角度，各兩次，再試做並儲存。")
        self.gesture_button.setEnabled(False)
        self.gesture_button.clicked.connect(lambda: self.begin_gesture_calibration(review=True))
        controls.addWidget(self.gesture_button)
        self.gesture_angle_button = QPushButton("補記其他角度")
        self.gesture_angle_button.setEnabled(False)
        self.gesture_angle_button.hide()
        self.gesture_angle_button.setToolTip("供未滿五個角度的靈狐手圈補記使用；重新記錄全部角度可按「校正靈狐手圈」。")
        self.gesture_angle_button.clicked.connect(lambda: self.begin_gesture_calibration(append=True))
        controls.addWidget(self.gesture_angle_button)
        controls.addSpacing(8)
        controls.addWidget(label("04  咒語觸發方式", size=18))
        controls.addWidget(label("麥克風輸入", "muted"))
        self.microphone_combo = QComboBox()
        self.microphone_combo.setAccessibleName("麥克風輸入裝置")
        self.microphone_combo.addItem("自動選擇麥克風（排除 iPhone／iPad）", None)
        self.microphone_combo.currentIndexChanged.connect(self.microphone_changed)
        self.microphone_combo.setToolTip("iPhone／iPad 保留作為攝影機；選擇外部攝影機、USB 麥克風或耳機的麥克風。暫停追蹤與發音校正後可修改。")
        controls.addWidget(self.microphone_combo)
        self.refresh_microphones_button = QPushButton("查看可用麥克風")
        self.refresh_microphones_button.clicked.connect(self.refresh_microphones)
        controls.addWidget(self.refresh_microphones_button)
        self.voice_trigger_mode_combo = QComboBox()
        self.voice_trigger_mode_combo.addItem("大聲短音即可", "loud_sound")
        self.voice_trigger_mode_combo.addItem("辨識咒語文字", "phrase")
        self.voice_trigger_mode_combo.setAccessibleName("咒語觸發方式")
        self.voice_trigger_mode_combo.setToolTip("大聲短音不需發音校正；鎖定後唸一個短音即可。此模式依音量偵測，其他大聲音也可能發動。")
        self.voice_trigger_mode_combo.currentIndexChanged.connect(self.voice_trigger_mode_changed)
        controls.addWidget(self.voice_trigger_mode_combo)
        self.pronunciation_label = label("由你決定咒語與唸法 · 錄三次，再試唸確認", "muted")
        controls.addWidget(self.pronunciation_label)
        self.pronunciation_button = QPushButton("校正我的咒語發音")
        self.pronunciation_button.clicked.connect(self.begin_pronunciation_calibration)
        controls.addWidget(self.pronunciation_button)
        voice_row = QHBoxLayout()
        voice_row.addWidget(label("辨識口令", "muted"))
        self.trigger_input = QLineEdit(self.trigger_setting)
        self.trigger_input.setAccessibleName("語音觸發發音")
        self.trigger_input.setMaximumWidth(185)
        self.trigger_input.setMaxLength(280)
        self.trigger_input.setPlaceholderText("kon 或 吃掉")
        self.trigger_input.setToolTip("可透過發音校正錄下自己的唸法，也可暫停感測器後直接輸入短口令，用逗號加入辨識詞。")
        self.trigger_input.editingFinished.connect(self.trigger_changed)
        voice_row.addWidget(self.trigger_input)
        controls.addLayout(voice_row)
        controls.addSpacing(8)
        controls.addWidget(label("05  鎖定與召喚", size=18))
        self.selection_build_controls(right)
        controls.addWidget(self.selection_settings_button)
        controls.addWidget(self.selection_settings_panel)
        controls.addSpacing(8)
        self.mode = QComboBox()
        self.mode.addItems(["演練 · 虛擬檔案", "桌面 · 移到垃圾桶"])
        self.mode.currentIndexChanged.connect(self.mode_changed)
        controls.addWidget(self.mode)
        self.test_button = QPushButton("開始演練")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self.begin_targets)
        controls.addWidget(self.test_button)
        self.demo_button = QPushButton("預覽狐狸吞噬動畫")
        self.demo_button.clicked.connect(self.preview_animation)
        controls.addWidget(self.demo_button)
        controls.addStretch()
        self.summon_caption = label(f"{self.selection_instruction()}，再說 {self.trigger_display}。", "muted")
        controls.addWidget(self.summon_caption)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setMinimumWidth(410 if get_language() == 'en' else 0)
        self.controls_scroll.setMaximumWidth(440 if get_language() == 'en' else 350)
        self.controls_scroll.setWidget(right)
        columns.addWidget(self.controls_scroll, 2)
        layout.addLayout(columns, 1)
        self.notice = label("移動手圈即可控制準星；啟動鏡頭後可直接演練，移動範圍與召喚手勢皆可自行調整。" if self.aim_mode == "hand" else
                            "啟動鏡頭後，依序校正瞄準與自己的召喚手勢；已保存的資料會自動沿用。", "status")
        layout.addWidget(self.notice)
        footer = QHBoxLayout()
        self.footnote = label("", "muted")
        self.footnote.setMinimumWidth(520)
        footer.addWidget(self.footnote)
        footer.addStretch()
        self.undo_button = QPushButton("復原上一個檔案")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo)
        footer.addWidget(self.undo_button)
        layout.addLayout(footer)
        self.refresh_aim_controls()

    def _build_tray(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        from PySide6.QtGui import QColor
        from .i18n import QPainter
        from PySide6.QtCore import QRectF
        from .overlay import draw_fox
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        draw_fox(painter, QRectF(1, 1, 30, 29))
        painter.end()
        self.setWindowIcon(QIcon(pixmap))
        self.tray = QSystemTrayIcon(QIcon(pixmap), self)
        self.tray.setToolTip("KON · 狐狸召喚")
        menu = QMenu()
        menu.addAction("返回控制台", self.return_to_control)
        menu.addAction("暫停並關閉鏡頭／麥克風", self.stop_sensors)
        menu.addSeparator()
        menu.addAction("結束 KON", self.close)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.return_to_control() if reason == QSystemTrayIcon.Trigger else None)
        self.tray.show()

    def config(self):
        rect = self.screen.geometry()
        config = {"eye": "right" if self.eye.currentIndex() == 0 else "left", "camera": self.camera_index.value(),
                "fov": self.fov.value(), "screen": [rect.width(), rect.height()],
                "screen_name": self.screen.name(), "screen_serial": self.screen.serialNumber(),
                "screen_origin": [rect.x(), rect.y()], "machine_id": machine_key()}
        return config

    @staticmethod
    def saved_aim_mode():
        try:
            saved = json.loads((user_state_dir(ROOT) / 'preferences.json').read_text())
            if isinstance(saved, dict) and saved.get("aim_mode") in ("eye_ray", "hand"):
                return saved["aim_mode"]
        except (OSError, ValueError, TypeError):
            pass
        return "eye_ray"

    def selection_gate(self):
        settings = self.selection_settings
        return TriggerGate(dwell_seconds=settings.window_seconds, keyword_max_age=2.5,
                           minimum_dwell_seconds=settings.minimum_score,
                           tracking_grace_seconds=settings.tracking_grace_seconds)

    def selection_instruction(self):
        settings = self.selection_settings
        return (f"最近 {settings.window_seconds:g} 秒內，加權停留最多且達門檻 "
                f"{settings.minimum_score:g} 秒即可鎖定")

    def selection_build_controls(self, parent):
        self.selection_settings_button = QPushButton("▸ 鎖定計算設定")
        self.selection_settings_button.setCheckable(True)
        self.selection_settings_button.setAccessibleName("展開或收合鎖定計算設定")
        self.selection_settings_button.setToolTip("調整停留時間、穩定加權與解除範圍。修改會取消目前鎖定，並自動保存。")
        self.selection_settings_panel = QWidget(parent)
        panel = QVBoxLayout(self.selection_settings_panel)
        panel.setContentsMargins(0, 4, 0, 8)
        panel.setSpacing(8)
        panel.addWidget(label("穩定後每秒可累積更多加權秒數。鏡頭開啟時也可修改；修改會取消目前鎖定。", "muted"))
        specifications = (
            ("window_seconds", "統計範圍（秒）", 2, .5,
             "只統計最近這段時間的停留；離開格子再回來仍可累積。"),
            ("minimum_score", "鎖定門檻（加權秒數）", 2, .1,
             "這是加權累積分數。穩定後使用較高權重，因此不必停留同樣長的實際秒數。"),
            ("stable_seconds", "同格穩定時間（秒）", 2, .05,
             "手部停止明顯移動，且實測準星持續位於同一格這麼久，後續停留才提高權重。"),
            ("stable_multiplier", "穩定後權重（倍）", 2, .1,
             "穩定判定完成後的停留倍率。例如 2 倍時，實際停留 0.1 秒可累積 0.2 加權秒。"),
            ("movement_speed_limit", "手部移動門檻（畫面比例／秒）", 3, .005,
             "0.04 表示每秒約移動畫面的 4%。超過門檻視為移動中，使用一般權重；低於門檻才判定穩定瞄準。數值較小需要更穩定的手部。"),
            ("release_cells", "解除範圍（周圍格數）", 0, 1,
             "準星移出鎖定格周圍這麼多格（包含對角）就解除。例如 2 表示周圍兩格。"),
            ("tracking_grace_seconds", "短暫追蹤中斷容許（秒）", 2, .05,
             "短於這個時間的追蹤中斷保留已累積分數；中斷期間不計分，語音與動作立即暫停。"),
        )
        self.selection_controls = {}
        for field, title, decimals, step, tooltip in specifications:
            control = QSpinBox() if field == "release_cells" else QDoubleSpinBox()
            lower, upper = SelectionSettings.RANGES[field]
            if field == "release_cells":
                control.setRange(int(lower), int(upper))
            else:
                control.setDecimals(decimals)
                control.setRange(lower, upper)
            control.setSingleStep(step)
            control.setKeyboardTracking(False)
            control.setValue(getattr(self.selection_settings, field))
            control.setAccessibleName(title)
            control.setToolTip(tooltip)
            control.setMaximumWidth(115)
            caption = label(title, "muted")
            caption.setBuddy(control)
            row = QHBoxLayout()
            row.addWidget(caption, 1)
            row.addWidget(control)
            panel.addLayout(row)
            self.selection_controls[field] = control
        self.selection_stability_caption = label("", "muted")
        panel.addWidget(self.selection_stability_caption)
        self.selection_settings_panel.hide()
        self.selection_settings_button.toggled.connect(self.selection_settings_toggled)
        for control in self.selection_controls.values():
            control.valueChanged.connect(self.selection_settings_changed)
        self.selection_sync_controls()

    def selection_settings_toggled(self, expanded):
        self.selection_settings_panel.setVisible(expanded)
        self.selection_settings_button.setText("▾ 鎖定計算設定" if expanded else "▸ 鎖定計算設定")

    def selection_sync_controls(self):
        settings = self.selection_settings
        for field, control in self.selection_controls.items():
            previous = control.blockSignals(True)
            if field == "minimum_score":
                control.setMaximum(settings.window_seconds * settings.stable_multiplier)
            control.setValue(getattr(settings, field))
            control.blockSignals(previous)

    def selection_apply_settings(self, settings):
        self.selection_settings = settings
        self.stable_dwell_seconds = settings.stable_seconds
        self.stable_dwell_multiplier = settings.stable_multiplier
        self.hand_movement_speed_limit = settings.movement_speed_limit
        self.lock_release_cells = settings.release_cells
        self.gate.dwell_seconds = settings.window_seconds
        self.gate.minimum_dwell_seconds = settings.minimum_score
        self.gate.tracking_grace_seconds = settings.tracking_grace_seconds
        self.selection_sync_controls()
        self.refresh_selection_copy()

    def selection_load_preferences(self):
        try:
            preferences = json.loads((user_state_dir(ROOT) / 'preferences.json').read_text(encoding="utf-8"))
            saved = preferences.get("selection_settings") if isinstance(preferences, dict) else None
        except (OSError, ValueError, TypeError):
            saved = None
        self.selection_apply_settings(SelectionSettings.from_mapping(saved))

    def selection_settings_changed(self, *_):
        values = {field: control.value() for field, control in self.selection_controls.items()}
        values["minimum_score"] = min(values["minimum_score"], values["window_seconds"] * values["stable_multiplier"])
        settings = SelectionSettings(**values)
        if settings == self.selection_settings:
            return
        self.set_voice(False)
        if self.overlay.eating is not None:
            self.return_to_control()
        else:
            self.clear_target_lock()
        self.selection_apply_settings(settings)
        message = "鎖定設定已套用並保存；目前鎖定與累積已清除，重新瞄準即可。"
        try:
            save_selection_settings(user_state_dir(ROOT) / 'preferences.json', settings)
        except (OSError, ValueError, TypeError) as exc:
            message = f"鎖定設定已套用，目前鎖定與累積已清除；這次無法保存：{exc}"
        self.tell(message)

    def refresh_selection_copy(self):
        settings = self.selection_settings
        self.selection_stability_caption.setText(
            f"同格穩定 {settings.stable_seconds:g} 秒後，後續停留以 {settings.stable_multiplier:g} 倍計分。")
        self.refresh_aim_controls()
        self.update_gesture_instructions()
        self.update_voice_labels()

    def refresh_aim_controls(self):
        hand = self.aim_mode == "hand"
        self.camera_heading.setText("CAMERA / 手部追蹤" if hand else "CAMERA / 手部與慣用眼追蹤")
        self.aim_description.setText(
            "移動手圈控制準星；不需要眼睛入鏡。" if hand else
            "用慣用眼穿過手圈指向檔案；眼睛與整隻手需入鏡。")
        self.eye.setVisible(not hand)
        self.fov_controls.setVisible(not hand)
        self.aim_heading.setText("02  手圈移動範圍" if hand else "02  校正瞄準")
        self.cal_button.setText("調整手部移動範圍（選用）" if hand else "校正眼睛 → 手圈")
        self.footnote.setText(
            f"手圈位置控制 · {self.selection_instruction()}" if hand else
            "眼睛穿過手圈瞄準 · 加強平滑 · 固定鏡頭與坐姿")

    def aim_mode_changed(self, *_):
        selected = self.aim_method.currentData()
        if selected not in ("eye_ray", "hand") or selected == self.aim_mode:
            return
        if self.running:
            previous = self.aim_method.blockSignals(True)
            self.aim_method.setCurrentIndex(self.aim_method.findData(self.aim_mode))
            self.aim_method.blockSignals(previous)
            self.tell("請先暫停鏡頭與麥克風，再切換瞄準方式。")
            return
        self.return_to_control()
        self.latest = None
        self.aim_mode = selected
        self.refresh_aim_controls()
        self.settings_changed()
        self.update_gesture_instructions()
        self.update_voice_labels()
        self.pose_label.setText("讓手勢，成為召喚的入口。")
        self.gesture_hint_label.setText("啟動鏡頭後，這裡會顯示各手指的辨識狀態。")
        message = ("已切換手圈位置控制；啟動鏡頭即可演練，移動範圍可自行調整。" if selected == "hand" else
                   "已切換眼睛穿過手圈瞄準；已沿用保存的校正，啟動鏡頭即可演練。" if self.calibration is not None else
                   "已切換眼睛穿過手圈瞄準；啟動鏡頭後先完成 9 點校正。")
        destination = user_state_dir(ROOT) / 'preferences.json'
        try:
            try:
                preferences = json.loads(destination.read_text())
                if not isinstance(preferences, dict):
                    preferences = {}
            except (OSError, ValueError):
                preferences = {}
            preferences["aim_mode"] = selected
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(preferences, ensure_ascii=False, indent=2))
            temporary.replace(destination)
        except OSError as exc:
            message += f" 這次切換已套用，但選擇無法儲存：{exc}"
        self.tell(message)

    def _load_calibration(self):
        self.hand_smoother.reset()
        if self.aim_mode == "hand":
            try:
                self.calibration = HandAimCalibration.load(self.hand_aim_path())
                self.cal_label.setText("已沿用 · 個人手部移動範圍")
            except (OSError, ValueError, TypeError):
                self.calibration = HandAimCalibration()
                self.cal_label.setText("預設範圍 · 可直接開始")
            self.cal_report_button.hide()
            self.validate_button.hide()
            return
        self.calibration = load_calibration(ROOT, self.config())
        if isinstance(self.calibration, PersonalCalibration):
            self.calibration_raw = list(self.calibration.raw_points)
            self.calibration_screen_points = list(self.calibration.screen_points)
            self.calibration_diagnostics = analyze_calibration(self.calibration_raw, self.calibration_screen_points)
            self.cal_report_button.show()
            self.update_personal_controls()
        elif self.calibration is not None:
            self.cal_label.setText(f"已保存並沿用 · RMS {self.calibration.rms_error * 100:.1f}%")
        if self.calibration is not None:
            self.cal_button.setText("重新校正眼睛 → 手圈")
            self.tell("已沿用這台機器的瞄準校正；可以直接啟動，或另外記錄自己的召喚手勢。")

    def _restore_settings(self):
        # Ignore retired acquisition settings and retain the last compatible
        # single-camera configuration without rewriting any saved profile.
        candidates = []
        for filename in ("camera-settings.json", "preferences.json", "calibration-meta.json"):
            try:
                saved = json.loads((user_state_dir(ROOT) / filename).read_text())
                if filename == "preferences.json":
                    saved = saved.get("camera_settings") if isinstance(saved, dict) else None
                candidates.append(saved)
            except (OSError, ValueError, TypeError):
                continue
        for saved in candidates:
            if not isinstance(saved, dict) or saved.get("acquisition_mode", "single") != "single":
                continue
            try:
                camera, fov = saved.get("camera"), saved.get("fov")
                if (saved.get("eye") not in ("right", "left")
                        or not isinstance(camera, int) or isinstance(camera, bool) or not 0 <= camera <= 31
                        or isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 20 <= fov <= 120):
                    continue
                values = ((self.eye, 0 if saved["eye"] == "right" else 1),
                          (self.camera_index, camera), (self.fov, fov))
                for control, value in values:
                    previous = control.blockSignals(True)
                    if control is self.eye:
                        control.setCurrentIndex(value)
                    else:
                        control.setValue(value)
                    control.blockSignals(previous)
                self.refresh_aim_controls()
                return
            except (TypeError, ValueError):
                continue

    def _load_calibration_attempt(self):
        if self.aim_mode == "hand":
            return
        attempt = load_last_attempt(ROOT, self.config())
        if attempt is None:
            try:
                old = json.loads((user_state_dir(ROOT) / 'calibration-last-attempt.json').read_text())
                if isinstance(old, dict) and old.get("version") == 1 and compatible_configuration(old.get("config"), self.config()):
                    attempt = old
            except (OSError, ValueError, TypeError):
                pass
        if attempt is None:
            return
        raw = attempt.get("raw_points", [])
        if not isinstance(raw, list) or len(raw) != 9:
            return
        targets = attempt.get("screen_points", [])
        if not isinstance(targets, list) or len(targets) != 9:
            return
        # Attempt history is diagnostic data, not the active saved profile.
        # A later failed/rejected attempt must not erase a completed mapping.
        if self.calibration is not None:
            if isinstance(self.calibration, PersonalCalibration) and (
                    tuple(map(tuple, raw)) == self.calibration.raw_points
                    and tuple(map(tuple, targets)) == self.calibration.screen_points):
                details = attempt.get("point_details", [])
                self.calibration_point_details = details if isinstance(details, list) else []
            return
        diagnostic = analyze_calibration(raw, targets)
        if diagnostic.status is FitStatus.INVALID_SAMPLES:
            return
        self.calibration_raw = [tuple(point) for point in raw]
        self.calibration_screen_points = [tuple(point) for point in targets]
        self.calibration_point_details = attempt.get("point_details", [])
        if not isinstance(self.calibration_point_details, list):
            self.calibration_point_details = []
        self.calibration_diagnostics = diagnostic
        self.cal_report_button.show()
        existing = self.calibration
        same_personal = isinstance(existing, PersonalCalibration) and (
            existing.raw_points == tuple(self.calibration_raw)
            and existing.screen_points == tuple(self.calibration_screen_points))
        if not same_personal:
            try:
                self.calibration = PersonalCalibration.fit(raw, targets, screen_size=self.config()["screen"])
            except ValueError:
                self.calibration = None
                self.cal_label.setText("校正點的區域無法區分 · 請重新記錄")
        self.update_personal_controls()

    def update_personal_controls(self):
        personal = isinstance(self.calibration, PersonalCalibration)
        self.validate_button.setVisible(personal)
        self.validate_button.setEnabled(personal and self.running)
        if personal:
            text = "已保存並沿用 · 9 點瞄準校正" + (" · 精度已驗證" if self.calibration.ready_for_live else "")
            self.cal_label.setText(text)

    def hand_aim_path(self):
        return user_state_dir(ROOT) / 'hand-aim' / f"{machine_key()}-camera-{self.camera_index.value()}.json"

    def begin_hand_range(self):
        if not self.running:
            self.tell("請先啟動鏡頭，再調整手部移動範圍。")
            return
        self.return_to_control()
        self.hand_range_calibrating = True
        self.hand_range_points = []
        self.hand_range_deadline = None
        self.hand_range_buffer.reset(time.monotonic())
        self.hand_range_dialog = HandAimDialog(self)
        self.hand_range_dialog.sample_requested.connect(self.capture_hand_range)
        self.hand_range_dialog.exit_requested.connect(self.cancel_hand_range)
        self.hand_range_dialog.finished.connect(lambda _result: self.cancel_hand_range())
        if self.preview.image is not None:
            self.hand_range_dialog.set_preview(self.preview.image)
        self.hand_range_dialog.show()
        self.refresh_hand_range()

    def capture_hand_range(self):
        if not self.running or not self.hand_range_calibrating:
            return
        if self.hand_range_deadline is not None:
            self.refresh_hand_range()
            return
        now = time.monotonic()
        self.hand_range_buffer.reset(now)
        self.hand_range_deadline = now + 5.0
        self.hand_range_dialog.set_feedback("正在記錄位置；請保持手圈約半秒，等待提示完成。")
        self.refresh_hand_range(now)

    def refresh_hand_range(self, now=None):
        if not self.hand_range_calibrating or self.hand_range_dialog is None:
            return
        now = time.monotonic() if now is None else now
        state = self.hand_range_buffer.evaluate(now)
        self.hand_range_dialog.set_state(state.count, state.required, state.reason, state.ready,
                                         pending=self.hand_range_deadline is not None)
        if self.hand_range_deadline is None:
            return
        if now >= self.hand_range_deadline:
            self.hand_range_deadline = None
            self.hand_range_dialog.set_feedback(f"尚未記錄：{state.reason}；調整後再按記錄。")
            self.hand_range_dialog.set_state(state.count, state.required, state.reason, state.ready, pending=False)
            return
        if not state.ready:
            return
        self.hand_range_deadline = None
        if not self.hand_range_points:
            self.hand_range_points.append(state.point)
            self.hand_range_buffer.reset(now)
            self.hand_range_dialog.set_step(1)
            self.hand_range_dialog.set_feedback("左上位置已記錄；把手圈移到舒適的右下位置，再記錄一次。")
            return
        try:
            model = HandAimCalibration.from_corners(self.hand_range_points[0], state.point)
            model.save(self.hand_aim_path())
        except (ValueError, OSError, TypeError) as exc:
            self.hand_range_dialog.set_feedback(f"尚未完成：{exc}；請重新記錄右下位置。")
            self.hand_range_buffer.reset(now)
            self.hand_range_dialog.set_state(0, state.required, "調整後再記錄", False, pending=False)
            return
        self.calibration = model
        self.hand_smoother.reset()
        self.cal_label.setText("已保存 · 個人手部移動範圍")
        dialog, self.hand_range_dialog = self.hand_range_dialog, None
        self.hand_range_calibrating = False
        dialog.finish_success()
        self.tell("手部移動範圍已保存，下次自動沿用；可以開始演練。")

    def cancel_hand_range(self):
        if not self.hand_range_calibrating:
            return
        self.hand_range_calibrating = False
        self.hand_range_deadline = None
        self.hand_range_points = []
        dialog, self.hand_range_dialog = self.hand_range_dialog, None
        if dialog is not None:
            dialog.reject()
        self.tell("已取消範圍調整，原本的手部移動範圍仍可沿用。")

    def _load_gesture_profile(self):
        try:
            self.gesture_profile = GestureCalibration.load(gesture_profile_path(ROOT))
            self.update_gesture_profile_label()
            self.update_gesture_instructions()
        except (OSError, ValueError, TypeError):
            self.gesture_profile = None
            self.update_gesture_profile_label()

    def update_gesture_profile_label(self):
        if self.gesture_profile is None:
            self.gesture_profile_label.setText("尚未記錄 · 目前使用預設 KON 手勢")
            self.gesture_angle_button.setEnabled(False)
            self.gesture_angle_button.hide()
            return
        count = getattr(self.gesture_profile, "angle_count", 1)
        self.gesture_profile_label.setText(f"靈狐手圈已保存 · {count} 個角度")
        self.gesture_angle_button.setEnabled(self.running and count < 5)
        self.gesture_angle_button.setVisible(count < 5)

    def update_gesture_instructions(self):
        gesture = "比出你記錄的召喚手勢" if self.gesture_profile else "比出 KON 手勢（或在第 03 步記錄自己的手勢）"
        self.gesture_instructions.setText(
            f"{gesture}，移動手圈控制準星。\n{self.selection_instruction()}；離開再回來仍累積。" if self.aim_mode == "hand" else
            f"{gesture}。\n將手圈放在慣用眼與螢幕目標之間。")

    def begin_gesture_calibration(self, *, append=False, review=False):
        if not self.running:
            self.tell("請先啟動鏡頭，再記錄自己的召喚手勢。")
            return
        existing = self.gesture_profile
        if append and (existing is None or getattr(existing, "angle_count", 1) >= 5):
            self.tell("請先記錄自己的召喚手勢；每個手勢最多保存五個角度，已滿時可重新校正。")
            return
        self.return_to_control()
        self.gesture_review_required = review
        self.gesture_review_profile = None
        self.gesture_review_passed = False
        self.gesture_review_released = False
        self.gesture_review_since = None
        self.gesture_review_count = 0
        self.gesture_review_timestamp = None
        self.gesture_calibrating = True
        self.gesture_candidate = None
        self.gesture_append = append
        self.gesture_base_profile = existing if append else None
        self.gesture_angle_index = 0
        self.gesture_staged_profile = None
        self.gesture_capture_deadline = None
        self.gesture_buffer.reset(time.monotonic())
        self.gesture_dialog = GestureCalibrationDialog(self)
        if self.aim_mode == "hand":
            self.gesture_dialog.scope_label.setText("這裡記錄召喚狐狸的手形；已保存的手部移動範圍會保留。")
        if append:
            self.gesture_dialog.set_append_angle_mode(getattr(self.gesture_base_profile, "angle_count", 1))
        else:
            self.gesture_dialog.set_standard_angle_mode(self.gesture_angle_index)
        self.gesture_dialog.sample_requested.connect(self.capture_gesture)
        self.gesture_dialog.save_requested.connect(self.save_reviewed_gesture)
        self.gesture_dialog.retry_requested.connect(self.retry_gesture_calibration)
        self.gesture_dialog.exit_requested.connect(self.cancel_gesture_calibration)
        self.gesture_dialog.finished.connect(lambda _result: self.cancel_gesture_calibration())
        if self.preview.image is not None:
            self.gesture_dialog.set_preview(self.preview.image)
        self.gesture_dialog.show()
        self.refresh_gesture_capture()

    def gesture_save_path(self):
        return gesture_profile_path(ROOT)

    def capture_gesture(self):
        if not self.running or not self.gesture_calibrating or self.gesture_review_profile is not None:
            return
        now = time.monotonic()
        if self.gesture_capture_deadline is not None:
            self.refresh_gesture_capture(now)
            return
        self.gesture_buffer.reset(now)
        self.gesture_capture_deadline = now + 8.0
        angle = ("這個新角度" if self.gesture_append
                 else f"「{STANDARD_GESTURE_ANGLES[self.gesture_angle_index]}」角度")
        self.gesture_dialog.set_feedback(f"正在記錄{angle}的新影格，請維持同一手形與角度約一秒。")
        # Wait for fresh observations after this explicit capture request.
        self.gesture_dialog.set_state(
            0, self.gesture_buffer.min_samples, "等待新的手勢影格，請保持同一手形與角度。", False, pending=True)

    def refresh_gesture_capture(self, now=None):
        if not self.gesture_calibrating or self.gesture_dialog is None:
            return
        now = time.monotonic() if now is None else now
        if self.gesture_review_profile is not None:
            if not self.gesture_review_passed and (self.latest is None or now - self.latest.timestamp > .3):
                self.gesture_review_since = None
                self.gesture_review_count = 0
                self.gesture_dialog.set_review_match(False, "等待新的手部畫面，再繼續試做。")
            return
        state = self.gesture_buffer.evaluate(now)
        pending = self.gesture_capture_deadline is not None
        self.gesture_dialog.set_state(state.count, state.required, state.reason, state.ready, pending=pending)
        if not pending:
            return
        if now >= self.gesture_capture_deadline:
            self.gesture_capture_deadline = None
            self.gesture_dialog.set_feedback(f"尚未記錄：{state.reason}；調整後再按記錄。")
            self.gesture_dialog.set_state(state.count, state.required, state.reason, state.ready, pending=False)
            return
        if not self.gesture_buffer.has_observation:
            return
        if not state.ready:
            return
        self.gesture_capture_deadline = None
        try:
            if self.gesture_candidate is None:
                candidate = GestureCalibration.fit(
                    state.samples, source=state.source, handedness=state.handedness)
                reference = self.gesture_base_profile if self.gesture_append else self.gesture_staged_profile
                if reference is not None and candidate.handedness != reference.handedness:
                    raise ValueError("各角度請使用相同的手，維持原來的手形")
                if reference is not None and candidate.source != reference.source:
                    raise ValueError("各角度請使用相同的追蹤資料")
                self.gesture_candidate = candidate
                self.gesture_buffer.reset(now)
                self.gesture_dialog.set_step(1)
                self.gesture_dialog.set_feedback("這個角度第一次已記錄。放鬆手，再用同一隻手從剛才的角度比出同樣手形，按確認。")
                self.gesture_dialog.set_state(0, state.required, "等待第二次新的手勢影格", False, pending=False)
                return
            profile = self.gesture_candidate.confirm(
                state.samples, source=state.source, handedness=state.handedness)
            if self.gesture_append:
                profile = add_gesture_angle(self.gesture_base_profile, profile)
            else:
                if self.gesture_staged_profile is not None:
                    profile = add_gesture_angle(self.gesture_staged_profile, profile)
                if self.gesture_angle_index < len(STANDARD_GESTURE_ANGLES) - 1:
                    completed = STANDARD_GESTURE_ANGLES[self.gesture_angle_index]
                    self.gesture_staged_profile = profile
                    self.gesture_angle_index += 1
                    self.gesture_candidate = None
                    self.gesture_buffer.reset(now)
                    self.gesture_dialog.set_standard_angle_mode(self.gesture_angle_index)
                    self.gesture_dialog.set_step(0)
                    next_angle = STANDARD_GESTURE_ANGLES[self.gesture_angle_index]
                    self.gesture_dialog.set_feedback(
                        f"「{completed}」角度已確認（{self.gesture_angle_index} / 5）。接著記錄「{next_angle}」；五個完成後一起保存。")
                    return
            if self.gesture_review_required:
                self.begin_gesture_review(profile)
                return
            profile.save(self.gesture_save_path())
        except (OSError, ValueError, TypeError) as exc:
            self.gesture_dialog.set_feedback(f"尚未完成：{exc}；請重新記錄這一步。")
            self.gesture_buffer.reset(now)
            self.gesture_dialog.set_state(0, state.required, "調整後再記錄", False, pending=False)
            return
        self.commit_gesture_profile(profile)

    def commit_gesture_profile(self, profile):
        self.gesture_profile = profile
        if self.camera is not None:
            self.camera.set_gesture_profile(profile)
        self.update_gesture_profile_label()
        self.update_gesture_instructions()
        dialog = self.gesture_dialog
        self.gesture_calibrating = False
        self.gesture_dialog = None
        self.gesture_candidate = None
        self.gesture_append = False
        self.gesture_base_profile = None
        self.gesture_angle_index = 0
        self.gesture_staged_profile = None
        self.gesture_review_profile = None
        dialog.finish_success()
        # Explicitly completing a new command enrollment starts a new,
        # disarmed session without carrying a previous command's latch.
        self.gate = self.selection_gate()
        self.tell(f"靈狐手圈已保存並套用 · {profile.angle_count} 個角度。瞄準校正保持原樣。")

    def begin_gesture_review(self, profile):
        self.gesture_review_profile = profile
        self.gesture_review_passed = False
        self.gesture_review_released = False
        self.gesture_review_since = None
        self.gesture_review_count = 0
        self.gesture_review_timestamp = None
        self.gesture_capture_deadline = None
        self.gesture_dialog.set_review_mode("靈狐手圈")
        self.gesture_dialog.set_feedback("試做只顯示辨識結果，通過後再按「儲存並使用」。")

    def observe_gesture_review(self, frame, now):
        if self.gesture_review_passed:
            return
        timestamp = frame.timestamp
        if not 0 <= now - timestamp <= .3 or (self.gesture_review_timestamp is not None
                                                       and timestamp <= self.gesture_review_timestamp):
            self.gesture_review_since = None
            self.gesture_review_count = 0
            self.gesture_dialog.set_review_match(False, "等待新的完整手部畫面。")
            return
        previous = self.gesture_review_timestamp
        self.gesture_review_timestamp = timestamp
        comparison = self.gesture_review_profile.match(
            frame.hand_world_points, source="world", handedness=frame.handedness)
        if not comparison.is_gao:
            if comparison.reason == "personal gesture mismatch" or frame.depth_status == "no_hand":
                self.gesture_review_released = True
            self.gesture_review_since = None
            self.gesture_review_count = 0
            self.gesture_dialog.set_review_match(False,
                "現在重新比出剛才的手圈。" if self.gesture_review_released else comparison.hint)
            return
        if not self.gesture_review_released:
            self.gesture_dialog.set_review_match(False, "先放鬆手，再重新比出剛才的手圈。")
            return
        if previous is None or timestamp - previous > .3 or self.gesture_review_since is None:
            self.gesture_review_since = timestamp
            self.gesture_review_count = 0
        self.gesture_review_count += 1
        self.gesture_review_passed = timestamp - self.gesture_review_since >= 1.0 and self.gesture_review_count >= 12
        self.gesture_dialog.set_review_match(self.gesture_review_passed,
            "手圈已穩定辨識，可以儲存並使用。" if self.gesture_review_passed else "手圈已辨識，再保持約一秒。")

    def save_reviewed_gesture(self):
        if not self.gesture_calibrating or self.gesture_review_profile is None or not self.gesture_review_passed:
            return
        profile = self.gesture_review_profile
        try:
            profile.save(self.gesture_save_path())
        except (OSError, ValueError, TypeError) as exc:
            self.gesture_dialog.set_state(0, 1, "原本保存的手勢仍保留。", False, pending=False)
            self.gesture_dialog.set_review_match(True, "試做已通過，可以重新儲存。")
            self.gesture_dialog.set_feedback(f"儲存失敗：{exc}")
            return
        self.commit_gesture_profile(profile)

    def retry_gesture_calibration(self):
        self.begin_gesture_calibration(append=self.gesture_append, review=self.gesture_review_required)

    def cancel_gesture_calibration(self):
        if not self.gesture_calibrating:
            return
        self.gesture_calibrating = False
        self.gesture_capture_deadline = None
        self.gesture_candidate = None
        self.gesture_append = False
        self.gesture_base_profile = None
        self.gesture_angle_index = 0
        self.gesture_staged_profile = None
        self.gesture_review_profile = None
        dialog, self.gesture_dialog = self.gesture_dialog, None
        if dialog is not None:
            dialog.reject()
        self.tell("已取消手勢校正，先前保存的手勢與瞄準資料仍可沿用。")

    def _load_preferences(self):
        self.selection_load_preferences()
        self.voice_trigger_mode = "loud_sound"
        try:
            saved = json.loads((user_state_dir(ROOT) / 'preferences.json').read_text())
            if not isinstance(saved, dict):
                raise ValueError("無效的偏好設定")
            self.voice_trigger_mode = "phrase" if saved.get("voice_trigger_mode") == "phrase" else "loud_sound"
            setting = saved.get("voice_trigger", DEFAULT_TRIGGER_SETTING)
            phrases = parse_trigger_phrases(setting)
            self.trigger_setting, self.trigger_display = setting, phrases[0]
            language = saved.get("voice_language")
            self.trigger_language = language if language in ("en", "zh") else None
            self.trigger_input.setText(setting)
            mic = saved.get("microphone")
            if isinstance(mic, dict) and isinstance(mic.get("name"), str) and mic["name"].strip():
                self.microphone_setting = dict(mic)
                self.microphone_combo.blockSignals(True)
                self.microphone_combo.addItem(f"已保存 · {mic['name']}", self.microphone_setting)
                self.microphone_combo.setCurrentIndex(self.microphone_combo.count() - 1)
                self.microphone_combo.blockSignals(False)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        try:
            self.pronunciation_profile = PronunciationProfile.load(ROOT)
        except (OSError, ValueError, TypeError, KeyError):
            self.pronunciation_profile = None
        self.voice_trigger_mode_combo.blockSignals(True)
        self.voice_trigger_mode_combo.setCurrentIndex(self.voice_trigger_mode_combo.findData(self.voice_trigger_mode))
        self.voice_trigger_mode_combo.blockSignals(False)
        self.update_voice_labels()

    def refresh_microphone_controls(self):
        enabled = not self.running and not self.pronunciation_calibrating
        enabled = enabled and not any(worker.isRunning() for worker in self._retiring_pronunciation_workers)
        running = getattr(self.speech, "isRunning", None)
        if callable(running) and running():
            enabled = False
        self.microphone_combo.setEnabled(enabled)
        self.refresh_microphones_button.setEnabled(enabled)

    def refresh_microphones(self):
        if self.running or self.pronunciation_calibrating:
            return
        self.refresh_microphones_button.setEnabled(False)
        def done(devices):
            selected = self.microphone_setting
            self.microphone_combo.blockSignals(True)
            self.microphone_combo.clear()
            self.microphone_combo.addItem("自動選擇麥克風（排除 iPhone／iPad）", None)
            index = 0
            for device in devices:
                setting = encode_microphone(device)
                self.microphone_combo.addItem(device.name, setting)
                if selected == setting:
                    index = self.microphone_combo.count() - 1
            if selected is not None and index == 0:
                self.microphone_combo.addItem(f"未連接 · {selected['name']}", selected)
                index = self.microphone_combo.count() - 1
            self.microphone_combo.setCurrentIndex(index)
            self.microphone_combo.blockSignals(False)
            self.refresh_microphone_controls()
            self.tell("已列出可用麥克風；iPhone／iPad 會保留給攝影機。" if devices else
                      "沒有可用的非 iPhone／iPad 麥克風。請連接外部攝影機的麥克風、USB 麥克風或耳機。")
        job = Job(list_microphones)
        self._jobs.add(job)
        job.signals.done.connect(lambda devices: (self._jobs.discard(job), done(devices)))
        job.signals.failed.connect(lambda reason: (self._jobs.discard(job), self.refresh_microphone_controls(), self.tell(reason)))
        self.pool.start(job)

    def microphone_changed(self, *_):
        if self.running or self.pronunciation_calibrating:
            previous = self.microphone_combo.blockSignals(True)
            index = next((i for i in range(self.microphone_combo.count())
                          if self.microphone_combo.itemData(i) == self.microphone_setting), 0)
            self.microphone_combo.setCurrentIndex(index)
            self.microphone_combo.blockSignals(previous)
            self.tell("請先暫停鏡頭與發音校正，再切換麥克風。")
            return
        value = self.microphone_combo.currentData()
        self.microphone_setting = dict(value) if isinstance(value, dict) else None
        destination = user_state_dir(ROOT) / 'preferences.json'
        try:
            try:
                payload = json.loads(destination.read_text())
                if not isinstance(payload, dict):
                    payload = {}
            except (OSError, ValueError):
                payload = {}
            payload["microphone"] = self.microphone_setting
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            temporary.replace(destination)
        except (OSError, ValueError) as exc:
            self.tell(f"麥克風選擇已套用，但無法儲存：{exc}")

    def update_voice_labels(self):
        aim_instruction = "移動手圈  →  鎖定檔案" if self.aim_mode == "hand" else "眼睛  →  手圈  →  目標"
        if self.voice_trigger_mode == "loud_sound":
            self.voice_instructions.setText(f"{aim_instruction}\n手勢維持  +  大聲唸一個短音")
            self.summon_caption.setText(f"{self.selection_instruction()}，再大聲唸一個短音。")
            self.pronunciation_label.setText("目前使用大聲短音 · 不必辨識成指定文字，也不需發音校正" +
                                             ("\n已保存你的咒語唸法，可切換文字辨識沿用。" if self.pronunciation_profile else ""))
            return
        self.voice_instructions.setText(f"{aim_instruction}\n手勢維持  +  說出 {self.trigger_display}")
        self.summon_caption.setText(f"{self.selection_instruction()}，再說 {self.trigger_display}。")
        self.pronunciation_label.setText(
            "已保存並沿用 · 你的咒語唸法，已通過試唸" if self.pronunciation_profile else
            "由你決定咒語與唸法 · 錄三次，再試唸確認")

    def voice_trigger_instruction(self):
        return "大聲唸一個短音即可" if self.voice_trigger_mode == "loud_sound" else f"請說 {self.trigger_display}"

    def voice_trigger_mode_changed(self, *_):
        mode = self.voice_trigger_mode_combo.currentData()
        if mode == self.voice_trigger_mode:
            return
        self.return_to_control()
        self.voice_trigger_mode = mode
        if self.running and self.speech is not None and self.speech.isRunning():
            self.speech.set_active(False)
            self.speech.configure_trigger(self.trigger_setting, language=self.trigger_language,
                                          accepted_phrases=self.pronunciation_phrases(), trigger_mode=mode)
        self.update_voice_labels()
        destination = user_state_dir(ROOT) / 'preferences.json'
        try:
            destination.parent.mkdir(exist_ok=True)
            try:
                preferences = json.loads(destination.read_text())
                if not isinstance(preferences, dict):
                    preferences = {}
            except (OSError, ValueError):
                preferences = {}
            preferences["voice_trigger_mode"] = mode
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(preferences, ensure_ascii=False, indent=2))
            temporary.replace(destination)
        except OSError as exc:
            self.tell(f"觸發方式已套用，但設定無法儲存：{exc}")
            return
        self.tell("已改為大聲短音：鎖定後大聲唸一個短音即可，不必辨識成指定文字。" if mode == "loud_sound" else
                  f"已改為辨識咒語文字：鎖定後請說 {self.trigger_display}。")

    def pronunciation_phrases(self):
        profile = self.pronunciation_profile
        return tuple(dict.fromkeys((profile.phrase, *profile.aliases))) if profile is not None else None

    def begin_pronunciation_calibration(self):
        self.return_to_control()
        self.pronunciation_calibrating = True
        self.refresh_microphone_controls()
        self.trigger_input.setEnabled(False)
        self.pronunciation_samples = []
        self.pronunciation_verification = None
        language = self.trigger_language or ("zh" if any("\u3400" <= char <= "\u9fff" for char in self.trigger_display) else "en")
        dialog = PronunciationDialog(self, trigger_setting=self.trigger_display, language=language)
        self.pronunciation_dialog = dialog
        dialog.sample_requested.connect(self.capture_pronunciation)
        dialog.test_requested.connect(lambda: self.capture_pronunciation(test=True))
        dialog.save_requested.connect(self.save_pronunciation)
        dialog.restart_requested.connect(self.restart_pronunciation)
        dialog.exit_requested.connect(self.cancel_pronunciation_calibration)
        dialog.finished.connect(lambda _result: self.cancel_pronunciation_calibration())
        dialog.show()
        self.tell("設定咒語，照你想使用的唸法錄三次，再勾選辨識結果並試唸。")

    def _connect_pronunciation_worker(self, worker):
        if getattr(worker, "_pronunciation_app_connected", False):
            return
        worker._pronunciation_app_connected = True
        worker.sample.connect(lambda request, text: self.on_pronunciation_sample(worker, request, text))
        worker.sample_failed.connect(lambda request, message: self.on_pronunciation_failure(worker, request, message))

    def _get_pronunciation_worker(self, language):
        if self.pronunciation_worker is not None and self.pronunciation_worker.isRunning():
            return self.pronunciation_worker
        if any(worker.isRunning() for worker in self._retiring_pronunciation_workers):
            raise RuntimeError("麥克風正在停止，請稍候再按錄音。")
        if not self.running and self.speech is not None and self.speech.isRunning():
            raise RuntimeError("先前的麥克風正在停止，請稍候再按錄音。")
        if self.running and self.speech is not None and self.speech.isRunning():
            worker = self.speech
            self.pronunciation_worker_owned = False
        else:
            worker = SpeechWorker(ROOT, parent=self, trigger_setting=self.trigger_setting, language=language,
                                  microphone_setting=self.microphone_setting)
            self.pronunciation_worker_owned = True
            worker.status.connect(lambda message: self._pronunciation_status(worker, message))
            worker.error.connect(lambda message: self._pronunciation_worker_error(worker, message))
        self.pronunciation_worker = worker
        self._connect_pronunciation_worker(worker)
        if self.pronunciation_worker_owned:
            worker.start()
        return worker

    def _pronunciation_status(self, worker, message):
        if self.pronunciation_calibrating and worker is self.pronunciation_worker:
            self.mic_label.setText(self.microphone_status_text(worker, message))

    @staticmethod
    def microphone_status_text(worker, message):
        device = getattr(worker, "input_device", None)
        name = getattr(device, "name", "")
        return f"{name} · {message}" if name and name not in message else message

    def _pronunciation_worker_error(self, worker, message):
        if worker is self.pronunciation_worker and self.pronunciation_calibrating:
            if self.pronunciation_request is not None:
                self.on_pronunciation_failure(worker, self.pronunciation_request[0], message)
            else:
                self.pronunciation_dialog.set_feedback(message)

    def _pronunciation_choices(self):
        dialog = self.pronunciation_dialog
        return (validate_pronunciation_phrase(dialog.desired_phrase()), dialog.selected_language(),
                tuple(dialog.selected_aliases()))

    def capture_pronunciation(self, *, test=False):
        dialog = self.pronunciation_dialog
        if not self.pronunciation_calibrating or dialog is None or self.pronunciation_request is not None:
            return
        try:
            phrase, language, aliases = self._pronunciation_choices()
            if test:
                if len(self.pronunciation_samples) != 3 or not aliases:
                    raise ValueError("請先錄三次，並勾選要接受的辨識結果。")
            elif len(self.pronunciation_samples) >= 3:
                return
            worker = self._get_pronunciation_worker(language)
            self.pronunciation_serial += 1
            request = self.pronunciation_serial
            self.pronunciation_request = (request, bool(test), phrase, language, aliases)
            self.pronunciation_deadline = time.monotonic() + 30.0
            self.pronunciation_verification = None
            dialog.set_capture_state(True, "請照你的唸法說一次咒語，說完稍停。")
            if not worker.request_sample(request, language=language):
                raise RuntimeError("麥克風正在停止，請稍候再按錄音。")
        except (ValueError, TypeError, RuntimeError) as exc:
            self.pronunciation_request = None
            self.pronunciation_deadline = None
            dialog.set_capture_state(False)
            dialog.set_feedback(str(exc))

    def on_pronunciation_sample(self, worker, request, text):
        pending = self.pronunciation_request
        dialog = self.pronunciation_dialog
        if (not self.pronunciation_calibrating or dialog is None or worker is not self.pronunciation_worker
                or pending is None or request != pending[0]):
            return
        self.pronunciation_request = None
        self.pronunciation_deadline = None
        dialog.set_capture_state(False)
        try:
            heard = validate_pronunciation_phrase(text)
        except (ValueError, TypeError):
            dialog.set_feedback(f"這次未取得清楚的短口令：{text or '沒有辨識到聲音'}。請再錄一次，說完稍停。")
            if pending[1]:
                dialog.set_test_result(False, text)
            return
        if pending[1]:
            try:
                current = self._pronunciation_choices()
            except (ValueError, TypeError):
                current = None
            passed = current == pending[2:] and heard in (pending[2], *pending[4])
            self.pronunciation_verification = (*current, heard) if passed else None
            dialog.set_test_result(passed, heard)
        else:
            self.pronunciation_samples.append(heard)
            dialog.set_samples(self.pronunciation_samples)
            dialog.set_feedback("三次已錄好。勾選你要接受的辨識結果，再按「試唸咒語」。" if len(self.pronunciation_samples) == 3 else
                                f"聽到：{heard}。請用同樣唸法再錄一次。")

    def on_pronunciation_failure(self, worker, request, message):
        if (not self.pronunciation_calibrating or worker is not self.pronunciation_worker
                or self.pronunciation_request is None or self.pronunciation_request[0] != request):
            return
        self.pronunciation_request = None
        self.pronunciation_deadline = None
        self.pronunciation_verification = None
        self.pronunciation_dialog.set_capture_state(False)
        self.pronunciation_dialog.set_feedback(f"這次尚未錄好：{message}")

    def restart_pronunciation(self):
        if not self.pronunciation_calibrating:
            return
        self.pronunciation_request = None
        self.pronunciation_deadline = None
        self.pronunciation_verification = None
        self.pronunciation_samples = []
        if self.pronunciation_worker is not None:
            self.pronunciation_worker.cancel_sample()

    def check_pronunciation_capture(self, now=None):
        now = time.monotonic() if now is None else now
        if self.pronunciation_deadline is not None and now >= self.pronunciation_deadline:
            worker, request = self.pronunciation_worker, self.pronunciation_request[0]
            worker.cancel_sample()
            self.on_pronunciation_failure(worker, request, "沒有在時間內取得咒語，請靠近麥克風再錄一次。")

    def save_pronunciation(self):
        if not self.pronunciation_calibrating or self.pronunciation_dialog is None or self.pronunciation_request is not None:
            return
        dialog = self.pronunciation_dialog
        try:
            choices = self._pronunciation_choices()
            verification = self.pronunciation_verification
            if verification is None or verification[:3] != choices:
                raise ValueError("請先以目前勾選的辨識結果通過一次試唸。")
            profile = PronunciationProfile.create(*choices[:2], self.pronunciation_samples, choices[2], verification[3])
            profile.save(ROOT)
        except (OSError, ValueError, TypeError) as exc:
            dialog.set_capture_state(False)
            dialog.set_feedback(f"尚未保存：{exc}")
            return
        self.pronunciation_profile = profile
        self.trigger_setting, self.trigger_display, self.trigger_language = profile.trigger_setting, profile.phrase, profile.language
        self.trigger_input.setText(self.trigger_setting)
        if self.running and self.speech is not None and self.speech.isRunning():
            self.speech.configure_trigger(self.trigger_setting, language=self.trigger_language,
                                          accepted_phrases=self.pronunciation_phrases(), trigger_mode=self.voice_trigger_mode)
        self.update_voice_labels()
        self._end_pronunciation_calibration()
        dialog.finish_success()
        self.gate = self.selection_gate()
        self.tell(f"咒語「{self.trigger_display}」的唸法已保存，重開會沿用。" +
                  ("目前使用大聲短音，鎖定後大聲唸一個短音即可。" if self.voice_trigger_mode == "loud_sound" else
                   "鎖定目標後照剛才的唸法發動。"))

    def _end_pronunciation_calibration(self):
        self.pronunciation_calibrating = False
        self.pronunciation_dialog = None
        self.pronunciation_request = None
        self.pronunciation_deadline = None
        self.pronunciation_samples = []
        self.pronunciation_verification = None
        worker, self.pronunciation_worker = self.pronunciation_worker, None
        owned, self.pronunciation_worker_owned = self.pronunciation_worker_owned, False
        self.trigger_input.setEnabled(not self.running)
        if worker is not None:
            worker.cancel_sample()
            if owned:
                self._retiring_pronunciation_workers.append(worker)
                worker.finished.connect(lambda: self._release_pronunciation_worker(worker))
                worker.stop()
                if not worker.isRunning():
                    self._release_pronunciation_worker(worker)
        if not self.running:
            self.mic_label.setText("語音尚未啟動 · 全程在本機處理")
        self.refresh_microphone_controls()

    def _release_pronunciation_worker(self, worker):
        if worker in self._retiring_pronunciation_workers:
            self._retiring_pronunciation_workers.remove(worker)
        self.refresh_microphone_controls()

    def cancel_pronunciation_calibration(self):
        if not self.pronunciation_calibrating:
            return
        dialog = self.pronunciation_dialog
        self._end_pronunciation_calibration()
        if dialog is not None:
            dialog.reject()
        self.tell("已取消發音校正，原先的咒語與校正資料仍可沿用。")

    def trigger_changed(self):
        if self.running or self.pronunciation_calibrating:
            return False
        try:
            setting = self.trigger_input.text().strip()
            phrases = parse_trigger_phrases(setting)
        except (ValueError, TypeError) as exc:
            self.trigger_input.setText(self.trigger_setting)
            self.tell(str(exc))
            return False
        changed = setting != self.trigger_setting
        self.trigger_setting, self.trigger_display = setting, phrases[0]
        if changed:
            self.pronunciation_profile = None
            self.trigger_language = None
        self.update_voice_labels()
        destination = user_state_dir(ROOT) / 'preferences.json'
        try:
            destination.parent.mkdir(exist_ok=True)
            try:
                preferences = json.loads(destination.read_text())
                if not isinstance(preferences, dict):
                    preferences = {}
            except (OSError, ValueError):
                preferences = {}
            preferences.update(voice_trigger=setting, voice_trigger_mode=self.voice_trigger_mode, camera_settings={
                "eye": "right" if self.eye.currentIndex() == 0 else "left",
                "camera": self.camera_index.value(), "fov": self.fov.value(),
            })
            if changed:
                preferences.pop("voice_language", None)
                preferences.pop("voice_pronunciation", None)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(preferences, ensure_ascii=False, indent=2))
            temporary.replace(destination)
            camera_path = user_state_dir(ROOT) / 'camera-settings.json'
            camera_temporary = camera_path.with_suffix(".tmp")
            camera_temporary.write_text(json.dumps(preferences["camera_settings"], indent=2))
            camera_temporary.replace(camera_path)
        except OSError as exc:
            self.tell(f"這次觸發詞已套用，但設定無法儲存：{exc}")
        return True

    def settings_changed(self, *_):
        if not hasattr(self, "cal_label"):
            return
        self.calibration = None
        self.calibration_diagnostics = None
        self.calibration_raw.clear()
        self.cal_report_button.hide()
        self.validate_button.hide()
        self.cal_label.setText("尚未校正 · 9 個螢幕標記")
        self._load_calibration()
        self._load_calibration_attempt()
        self.test_button.setEnabled(self.running and self.calibration is not None)

    def screen_changed(self, *_):
        self.return_to_control()
        self.settings_changed()
        self.overlay.setGeometry(self.screen.geometry())

    def tell(self, text):
        self.notice.setText(text)

    def job(self, function, done):
        job = Job(function)
        self._jobs.add(job)
        def finished(result):
            self._jobs.discard(job)
            done(result)
        def failed(message):
            self._jobs.discard(job)
            self.return_to_control()
            self.undo_button.setEnabled(self.receipt is not None)
            self.tell(message)
        job.signals.done.connect(finished)
        job.signals.failed.connect(failed)
        self.pool.start(job)

    def toggle_sensors(self):
        if self.running:
            self.stop_sensors()
            return
        self.cancel_pronunciation_calibration()
        if any(worker.isRunning() for worker in self._retiring_pronunciation_workers):
            self.tell("發音校正的麥克風正在停止，請稍候再啟動。")
            return
        if any(worker is not None and worker.isRunning() for worker in (self.camera, self.speech)):
            self.tell("感測器正在停止，請稍候。")
            return
        if not self.trigger_changed():
            return
        self.running = True
        self.gate = self.selection_gate()
        self.latest = None
        self.camera = CameraWorker(self.camera_index.value(), self.fov.value(), self.config()["eye"],
                                   gesture_profile=self.gesture_profile, aim_mode=self.aim_mode)
        self.speech = SpeechWorker(ROOT, trigger_setting=self.trigger_setting, language=self.trigger_language,
                                   accepted_phrases=self.pronunciation_phrases(), trigger_mode=self.voice_trigger_mode,
                                   microphone_setting=self.microphone_setting)
        finished = getattr(self.speech, "finished", None)
        if finished is not None:
            finished.connect(self.refresh_microphone_controls)
        self.camera.frame.connect(self.on_frame)
        self.camera.tracking.connect(self.on_tracking)
        self.camera.status.connect(self.camera_label.setText)
        self.camera.error.connect(self.sensor_error)
        speech = self.speech
        self.speech.status.connect(lambda message: self.mic_label.setText(self.microphone_status_text(speech, message))
                                   if speech is self.speech else None)
        self.speech.error.connect(self.speech_error)
        self.speech.transcript.connect(lambda text: self.tell(f"聽到：{text or '（未辨識）'}"))
        self.speech.keyword.connect(self.on_keyword)
        self.start_button.setText("暫停鏡頭與麥克風")
        for control in (self.aim_method, self.eye, self.camera_index, self.fov, self.trigger_input):
            control.setEnabled(False)
        self.cal_button.setEnabled(True)
        self.gesture_button.setEnabled(True)
        self.update_gesture_profile_label()
        self.test_button.setEnabled(self.calibration is not None)
        self.update_personal_controls()
        self.refresh_microphone_controls()
        self.camera.start()
        self.speech.start()
        if self.aim_mode == "hand":
            self.tell("移動手圈即可控制準星，不需眼睛入鏡。可直接演練，或自行調整舒適的手部移動範圍。")
        elif self.calibration is not None:
            self.tell("已沿用保存的瞄準校正，可以直接開始；校正自己的召喚手勢不會改動瞄準資料。")
        else:
            self.tell("請允許鏡頭與麥克風；初次使用先完成 9 點瞄準校正。")

    def stop_sensors(self):
        self.cancel_hand_range()
        self.cancel_pronunciation_calibration()
        self.cancel_gesture_calibration()
        if self.calibrating:
            self.cancel_calibration()
        self.running = self.armed = self.testing = self.calibrating = False
        self.calibration_capture_deadline = None
        self.calibration_validating = False
        self.validation_resume_targets = False
        self.action_serial += 1
        self.pending_action = None
        self.active_action = None
        self.clear_target_lock()
        self.overlay.grid_enabled = False
        self.set_voice(False)
        self.overlay.cancel()
        for worker in (self.camera, self.speech):
            if worker:
                worker.stop()
        self.start_button.setText("啟動鏡頭與麥克風")
        for control in (self.aim_method, self.eye, self.camera_index, self.fov, self.trigger_input):
            control.setEnabled(True)
        self.refresh_microphone_controls()
        self.cal_button.setEnabled(False)
        self.gesture_button.setEnabled(False)
        self.gesture_angle_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.validate_button.setEnabled(False)
        self.pose_label.setText("鏡頭已暫停")
        self.gesture_hint_label.setText("重新啟動鏡頭後會檢查手形；已保存的瞄準校正可沿用。")
        self.showNormal()
        self.tell("已暫停；鏡頭與麥克風正在關閉。")

    def sensor_error(self, message):
        self.stop_sensors()
        self.tell(message)

    def speech_error(self, message):
        self.armed = False
        self.set_voice(False)
        if self.pronunciation_calibrating and self.speech is self.pronunciation_worker:
            self._pronunciation_worker_error(self.speech, message)
        else:
            self.return_to_control()
        self.tell(message)

    def on_frame(self, image):
        if self.running and not self.testing:
            self.hero.hide()
            self.preview.show()
            self.preview.set_image(image)
            if self.calibrating:
                self.overlay.set_calibration_preview(image)
            if self.gesture_calibrating and self.gesture_dialog is not None:
                self.gesture_dialog.set_preview(image)
            if self.hand_range_calibrating and self.hand_range_dialog is not None:
                self.hand_range_dialog.set_preview(image)

    def set_camera_preview(self, enabled):
        setter = getattr(self.camera, "set_preview_enabled", None)
        if callable(setter):
            setter(enabled)

    def present_cursor(self, point, *, locked=False):
        """Animate pixels only; selection and action freshness use aim_point."""
        now = time.monotonic()
        if point is None:
            self.cursor_timer.stop()
            self.cursor_motion.clear()
            self.overlay.set_cursor_point(None)
            return
        if locked or not self.testing or not self.overlay.isVisible() or self.overlay.eating is not None:
            self.cursor_motion.snap(point, now)
            self.cursor_timer.stop()
        else:
            self.cursor_motion.push(point, now)
        self.overlay.set_cursor_point(self.cursor_motion.sample(now))
        if self.cursor_motion.is_animating and self.testing and self.overlay.isVisible() and self.overlay.eating is None:
            if not self.cursor_timer.isActive():
                self.cursor_timer.start()
        else:
            self.cursor_timer.stop()

    def animate_cursor(self):
        """A display tick never supplies a camera sample or advances dwell."""
        now = time.monotonic()
        if (not self.running or not self.testing or not self.overlay.isVisible()
                or self.overlay.eating is not None):
            self.cursor_timer.stop()
            return
        if self.latest is None or not 0 <= now - self.latest.timestamp <= .35:
            self.aim_point = None
            target = self.locked_target()
            self.present_cursor(target.center if target else None, locked=True)
            return
        self.overlay.set_cursor_point(self.cursor_motion.sample(now))
        if not self.cursor_motion.is_animating:
            self.cursor_timer.stop()

    def locked_target(self):
        identity = getattr(self.gate, "locked_target", None)
        return next((target for target in self.targets if target.id == identity), None)

    @staticmethod
    def observed_gesture_release(frame):
        # Delay/status events contain no actual image even though their
        # timestamp is recent. Only an observed absence or valid hand-shape
        # mismatch establishes release, rather than uncertain tracking.
        if frame.depth_status == "no_hand":
            return True
        if frame.gesture_reason == "personal gesture mismatch":
            return True
        if frame.gesture_reason not in ("finger pattern", "thumb contact or finger margin"):
            return False
        try:
            if len(frame.hand_world_points) == 21:
                extract_gesture_features(frame.hand_world_points, source="world")
            else:
                width, height = frame.inference_size or (1, 1)
                extract_gesture_features(frame.hand_points, source="normalized", aspect_ratio=width / height)
        except (ValueError, TypeError, ZeroDivisionError):
            return False
        return True

    def clear_target_lock(self):
        clear = getattr(self.gate, "clear_lock", None)
        if callable(clear):
            clear(time.monotonic())
        self.hand_smoother.reset()
        self.reset_dwell_motion()
        self.grid_selector.reset()
        self._selection_layout = None
        self._selection_frame_at = None
        self.aim_point = None
        self.present_cursor(None)
        self.overlay.target_id = None
        self.overlay.progress = 0.0
        self.overlay.grid_active_cell = self.overlay.grid_ambiguous_cell = None

    def reset_dwell_motion(self):
        self._aim_dwell_motion = AimDwellWeight()
        self.aim_weight_status = "等待手部移動資料 · 一般權重"

    def motion_dwell_weight(self, frame, point, screen_size):
        if not hasattr(self, "_aim_dwell_motion"):
            self.reset_dwell_motion()
        hand_position = frame.hand_cursor
        if hand_position is None:
            hand_position = hand_circle_cursor(frame.hand_points)
        cell = self.grid_selector.cell_for(point, screen_size)
        weight = self._aim_dwell_motion.observe(
            frame.timestamp, hand_position, cell,
            stable_seconds=self.stable_dwell_seconds,
            multiplier=self.stable_dwell_multiplier,
            speed_limit=self.hand_movement_speed_limit,
            max_gap=self.gate.pose_max_age,
        )
        motion = self._aim_dwell_motion
        self.aim_weight_status = (f"穩定瞄準 ×{weight:g}" if motion.stable else
                                 "移動中 · 一般權重" if motion.moving else
                                 "瞄準中 · 等待同格穩定" if motion.moving is False else
                                 "等待手部移動資料 · 一般權重")
        return weight

    def aim_releases_lock(self, point, screen_size):
        target = self.locked_target()
        if target is None:
            return False
        locked_cell = self.grid_selector.cell_for(target.center, screen_size)
        aimed_cell = self.grid_selector.cell_for(point, screen_size)
        return (locked_cell is not None and aimed_cell is not None
                and max(abs(a - b) for a, b in zip(locked_cell, aimed_cell)) > self.lock_release_cells)

    def show_paused_target_lock(self, reason="追蹤暫時中斷，恢復後即可繼續"):
        self.reset_dwell_motion()
        locked = self.locked_target()
        candidate = next((target for target in self.targets
                          if target.id == getattr(self.gate, "leading_target", None)), None)
        target = locked or candidate
        self.aim_point = None
        self.present_cursor(locked.center if locked else None, locked=True)
        self.overlay.target_id = target.id if target else None
        self.overlay.progress = 1.0 if locked else getattr(self.gate, "progress", 0.0)
        self.overlay.grid_active_cell = (self.grid_selector.cell_for(target.center, (self.overlay.width(), self.overlay.height()))
                                         if target else None)
        self.overlay.grid_ambiguous_cell = None
        self.overlay.hint = (f"已鎖定 {locked.name}\n{reason}" if locked else
                             f"{candidate.name} 加權累積 {self.gate.leading_dwell_seconds:.1f} 秒已保留\n{reason}" if candidate else reason)
        self.overlay.update()

    def on_tracking(self, frame):
        if not self.running:
            return
        now = time.monotonic()
        if frame.aiming_mode != self.aim_mode:
            self.gate.invalidate_tracking(now)
            self.set_voice(False)
            self.hand_smoother.reset()
            self.grid_selector.reset()
            self.show_paused_target_lock("等待目前瞄準方式的追蹤資料")
            return
        if (not any((self.calibrating, self.gesture_calibrating, self.hand_range_calibrating,
                     self.pronunciation_calibrating))
                and self._selection_frame_at is not None and frame.timestamp <= self._selection_frame_at):
            self.gate.invalidate_tracking(now)
            self.set_voice(False)
            self.hand_smoother.reset()
            self.grid_selector.reset()
            self.show_paused_target_lock("等待新的追蹤畫面")
            return
        self.latest = frame
        fresh = 0 <= now - frame.timestamp <= 0.3
        recognized = "你的召喚手勢已辨識" if self.gesture_profile else "KON 手勢已辨識"
        self.pose_label.setText(recognized if fresh and frame.pose_ok else "等待召喚手勢")
        if not fresh:
            gesture_hint = "追蹤畫面延遲，等待新的手部資料。"
        elif frame.pose_ok:
            gesture_hint = (f"召喚手勢已通過；{self.selection_instruction()}，離開再回來可繼續累積。" if self.aim_mode == "hand" else
                            "召喚手勢已通過；將手圈移到慣用眼與目標之間。")
        elif not frame.hand_detected and len(frame.hand_points) != 21:
            gesture_hint = "尚未偵測到手部；請讓整隻手完整入鏡。"
        else:
            gesture_hint = getattr(frame, "gesture_hint", "") or (
                "已追蹤到手部，手形尚未通過；可在第 03 步記錄自己的召喚手勢。")
        self.gesture_hint_label.setText(gesture_hint)
        if self.hand_range_calibrating:
            usable = (0 <= now - frame.timestamp <= .35 and frame.aiming_mode == "hand"
                      and frame.hand_cursor is not None)
            self.hand_range_buffer.observe(frame.timestamp, frame.hand_cursor if usable else None, reason=frame.message)
            self.refresh_hand_range(now)
            return
        if self.pronunciation_calibrating:
            return
        if self.gesture_calibrating:
            if self.gesture_review_profile is not None:
                self.observe_gesture_review(frame, now)
                return
            points = getattr(frame, "hand_world_points", ())
            enrollment_fresh = 0 <= now - frame.timestamp <= self.gesture_buffer.max_age
            self.gesture_buffer.observe(
                frame.timestamp, points if enrollment_fresh and len(points) == 21 else None,
                source="world", handedness=getattr(frame, "handedness", "unknown"),
            )
            if self.gesture_candidate is not None and self.gesture_dialog is not None:
                if enrollment_fresh and len(points) == 21:
                    comparison = self.gesture_candidate.match(points, source="world", handedness=frame.handedness)
                    self.gesture_dialog.set_match_status(
                        "與第一次記錄相符；保持這個手形，再按確認。" if comparison.is_gao else comparison.hint)
                else:
                    self.gesture_dialog.set_match_status("等待新的完整手部畫面，再比對第一次記錄。")
            self.refresh_gesture_capture(now)
            return
        if self.calibrating:
            # Aiming calibration does not execute commands. It needs a usable
            # eye-through-aperture ray, independently of the command pose label.
            usable = 0 <= now - frame.timestamp <= 0.35 and frame.raw_hit is not None
            reason = frame.message if usable or now - frame.timestamp <= 0.35 else "鏡頭畫面延遲，請稍候新的追蹤畫面。"
            accepted = self.calibration_buffer.observe(frame.timestamp, frame.raw_hit if usable else None, reason=reason)
            if accepted and usable and self.calibration_capture_deadline is not None:
                self.calibration_observations.append(frame)
                self.calibration_observations = [item for item in self.calibration_observations[-256:]
                                                 if frame.timestamp - item.timestamp <= 2.0]
            self.refresh_calibration_capture(now)
            return
        if not fresh:
            self.gate.invalidate_tracking(now)
            self.set_voice(False)
            self.hand_smoother.reset()
            self.grid_selector.reset()
            self.show_paused_target_lock()
            return
        self._selection_frame_at = frame.timestamp
        point = None
        sample = frame.hand_cursor if self.aim_mode == "hand" else frame.raw_hit
        if frame.pose_ok and sample is not None and self.calibration is not None:
            normalized = self.calibration.predict(sample)
            if normalized is not None and all(0 <= axis <= 1 for axis in normalized):
                normalized = self.hand_smoother.update(frame.timestamp, normalized,
                                                       (self.overlay.width(), self.overlay.height()))
                if normalized is not None:
                    point = (normalized[0] * self.overlay.width(), normalized[1] * self.overlay.height())
        screen_size = (self.overlay.width(), self.overlay.height())
        aimed_target = self.grid_selector.pick(self.targets, point, screen_size) if self.testing else None
        if (self.grid_selector.layout_signature is not None
                and self.grid_selector.layout_signature != self._selection_layout):
            self.gate.invalidate_tracking(now, discard_dwell=True)
            self._selection_layout = self.grid_selector.layout_signature
            self.reset_dwell_motion()
        if getattr(self.gate, "locked_target", None) is not None and self.locked_target() is None:
            self.clear_target_lock()
        previous_lock = getattr(self.gate, "locked_target", None)
        movement_released = self.aim_releases_lock(point, screen_size)
        if movement_released:
            self.gate.clear_lock(now)
            self.reset_dwell_motion()
        point_usable = point is not None and (not self.testing or self.grid_selector.sample_valid)
        if frame.pose_ok and not point_usable and frame.raw_pose_ok is not False:
            malformed_layout = (self.testing and point is not None
                                and self.grid_selector.layout_signature is None)
            self.gate.invalidate_tracking(now, discard_dwell=malformed_layout)
            self.reset_dwell_motion()
            ready = False
        else:
            dwell_weight = self.motion_dwell_weight(frame, point, screen_size)
            ready = self.gate.update(
                now, frame.pose_ok and point_usable, aimed_target.id if aimed_target else None, self.armed,
                release_observed=self.observed_gesture_release(frame),
                observed_at=frame.timestamp,
                dwell_weight=dwell_weight,
                raw_pose_ok=frame.raw_pose_ok,
            )
        if self.aim_releases_lock(point, screen_size):
            self.gate.clear_lock(now)
            self.reset_dwell_motion()
            ready = False
        if previous_lock is not None and getattr(self.gate, "locked_target", None) is None:
            self.hand_smoother.reset()
            self.reset_dwell_motion()
        locked = self.locked_target()
        candidate = next((target for target in self.targets
                          if target.id == getattr(self.gate, "leading_target", None)), None)
        target = locked or candidate or aimed_target
        self.aim_point = point
        self.present_cursor(locked.center if locked else point, locked=locked is not None)
        self.overlay.target_id = target.id if target else None
        self.overlay.progress = 1.0 if locked else self.gate.progress
        self.overlay.grid_active_cell = (self.grid_selector.cell_for(target.center, screen_size)
                                         if target else self.grid_selector.active_cell)
        self.overlay.grid_ambiguous_cell = None if locked else self.grid_selector.ambiguous_cell
        self.set_voice(ready, target.id if target else None)
        self.gesture_hint_label.setText(f"{gesture_hint}\n{self.aim_weight_status}")
        if self.testing:
            outside = (fresh and frame.pose_ok and frame.raw_hit is not None
                       and isinstance(self.calibration, PersonalCalibration) and point is None)
            missing = "瞄準位置超出個人校正範圍，請返回控制台重新校正。" if outside else frame.message
            if locked:
                self.overlay.hint = (f"已鎖定 {locked.name} · {self.voice_trigger_instruction()}\n移出鎖定格周圍 {self.lock_release_cells} 格、張開手或移出鏡頭可解除"
                                     if ready else f"已鎖定 {locked.name}\n等待清楚的召喚手勢與追蹤畫面")
            else:
                if point is None:
                    self.overlay.hint = (f"{candidate.name} 加權累積 {self.gate.leading_dwell_seconds:.1f} 秒已保留\n{missing}"
                                         if candidate else missing)
                elif self.grid_selector.ambiguous_cell is not None:
                    self.overlay.hint = "這格有多個檔案，請把圖示分到不同格再讀取桌面"
                elif target:
                    self.overlay.hint = (f"累積最多：{target.name} · {self.gate.leading_dwell_seconds:.1f}／{self.gate.minimum_dwell_seconds:g} 加權秒\n"
                                         f"{self.aim_weight_status} · {self.selection_instruction()}；離開再回來可接著累積")
                else:
                    self.overlay.hint = f"將準星移到有檔案的格子，{self.selection_instruction()}"
            self.overlay.update()
        triggered = self.gate.take_trigger(now=time.monotonic())
        if triggered:
            self.summon(triggered)

    def check_freshness(self):
        if self.pronunciation_calibrating:
            self.check_pronunciation_capture()
            return
        if self.hand_range_calibrating:
            self.refresh_hand_range()
            return
        if self.gesture_calibrating:
            self.refresh_gesture_capture(time.monotonic())
            return
        if self.calibrating:
            self.refresh_calibration_capture(time.monotonic())
            return
        if self.running and self.latest is not None and time.monotonic() - self.latest.timestamp > 0.35:
            self.pose_label.setText("等待新的手部追蹤")
            self.gesture_hint_label.setText("追蹤暫時中斷；已保存的瞄準校正可沿用。")
            self.gate.invalidate_tracking(time.monotonic())
            self.set_voice(False)
            self.hand_smoother.reset()
            self.grid_selector.reset()
            self.show_paused_target_lock()

    def set_voice(self, enabled, target_id=None):
        if self.pronunciation_calibrating:
            return
        changed = enabled != self.voice_active or (enabled and target_id != self.voice_target)
        if not changed:
            return
        if self.speech:
            if self.voice_active:
                self.speech.set_active(False)
            self.speech.set_active(bool(enabled))
        self.voice_active, self.voice_target = bool(enabled), target_id if enabled else None
        if enabled:
            self.voice_armed_at = time.monotonic()

    def on_keyword(self, captured_at, text):
        now = time.monotonic()
        if self.pronunciation_calibrating or not self.voice_active or captured_at < self.voice_armed_at or now - captured_at > 2.5:
            return
        target = self.gate.hear_keyword(now, captured_at=captured_at)
        if target:
            self.summon(target)

    def begin_calibration(self):
        if self.aim_mode == "hand":
            self.begin_hand_range()
            return
        if not self.running:
            return
        self.return_to_control()
        self.calibrating = True
        self.calibration_validating = False
        self.calibration_capture_deadline = None
        self.calibration_buffer = CalibrationBuffer(min_span=1.0)
        self.calibration_buffer.reset(time.monotonic())
        self.calibration_raw.clear()
        margin_x = min(.1, 28 / self.overlay.width())
        margin_y = min(.1, 40 / self.overlay.height())
        self.calibration_screen_points = [(x, y) for y in (margin_y, .5, 1 - margin_y)
                                          for x in (margin_x, .5, 1 - margin_x)]
        self.calibration = None
        self.test_button.setEnabled(False)
        self.validate_button.hide()
        self.calibration_index = 0
        self.calibration_retrying = False
        self.calibration_observations.clear()
        self.calibration_point_details.clear()
        self.calibration_diagnostics = None
        self.cal_report_button.hide()
        self.overlay.calibration_feedback = "用慣用眼穿過手圈瞄準標記；按記錄後保持約 1 秒。手勢未通過也能校正位置。"
        self.showMinimized()
        self.overlay.set_calibration(self.calibration_screen_points[0], 0)
        if self.preview.image is not None:
            self.overlay.set_calibration_preview(self.preview.image)
        self.refresh_calibration_capture()

    def capture_calibration(self):
        if not self.calibrating:
            return
        now = time.monotonic()
        if self.calibration_capture_deadline is not None:
            self.refresh_calibration_capture(now)
            return
        # Source frames must have been captured after this explicit request.
        # A ready pre-click hold may still have been aimed at the previous mark.
        self.calibration_buffer.reset(now)
        self.calibration_observations.clear()
        self.calibration_capture_deadline = now + 6.0
        self.overlay.calibration_feedback = "正在重新取樣；穿過手圈看圓點並保持至少 1 秒，暫時不要看鏡頭預覽。"
        self.refresh_calibration_capture(now)

    def refresh_calibration_capture(self, now=None):
        if not self.calibrating:
            return
        now = time.monotonic() if now is None else now
        state = self.calibration_buffer.evaluate(now)
        self.overlay.set_calibration_state(state.count, state.required, state.reason, state.ready)
        deadline = self.calibration_capture_deadline
        if deadline is not None:
            if now >= deadline:
                self.calibration_capture_deadline = None
                reason = "取樣時間已結束。" if state.ready else state.reason
                self.overlay.calibration_feedback = f"尚未記錄：{reason} 調整後再按空白鍵或點「記錄」。"
            elif state.ready:
                self.commit_calibration_point(state.point, now)
                return
            else:
                self.overlay.calibration_feedback = f"正在取樣 · {state.count}/{state.required} · 保持至少 1 秒（剩餘 {deadline - now:.1f} 秒）"
        self.overlay.update()

    def commit_calibration_point(self, point, now):
        if point is None or not self.calibrating or self.calibration_capture_deadline is None:
            return
        self.calibration_capture_deadline = None
        if self.calibration_validating:
            self.commit_validation_point(point, now)
            return
        index = self.calibration_index if self.calibration_retrying else len(self.calibration_raw)
        if self.calibration_retrying:
            self.calibration_raw[index] = point
        else:
            self.calibration_raw.append(point)
        while len(self.calibration_point_details) <= index:
            self.calibration_point_details.append(None)
        self.calibration_point_details[index] = self.calibration_point_summary(now)
        self.calibration_buffer.reset(now)
        step = len(self.calibration_raw)
        if step < 9 and not self.calibration_retrying:
            self.calibration_index = step
            self.overlay.calibration_feedback = f"第 {step} 點已記錄。瞄準新的標記後，再記錄下一點。"
            self.overlay.set_calibration(self.calibration_screen_points[step], step)
            state = self.calibration_buffer.evaluate(now)
            self.overlay.set_calibration_state(state.count, state.required, state.reason, state.ready)
            return
        diagnostic = analyze_calibration(self.calibration_raw, self.calibration_screen_points)
        self.calibration_diagnostics = diagnostic
        self.cal_report_button.show()
        history_warning = ""
        try:
            personal = PersonalCalibration.fit(self.calibration_raw, self.calibration_screen_points,
                                               screen_size=self.config()["screen"])
        except ValueError as exc:
            diagnostic = replace(diagnostic, status=FitStatus.DEGENERATE_MAPPING,
                                 reason=f"個人校正區域無法建立：{exc}",
                                 calibration=None, suggested_point_index=None)
            self.calibration_diagnostics = diagnostic
            try:
                save_attempt(ROOT, self.calibration_raw, self.calibration_screen_points, diagnostic,
                             self.config(), self.calibration_point_details)
            except (OSError, ValueError) as history_error:
                history_warning = f"\n校正診斷無法儲存：{history_error}"
            self.calibration = None
            self.cal_label.setText("校正區域無法區分 · 請重新記錄")
            self.test_button.setEnabled(False)
            self.validate_button.hide()
            self.tell(f"個人校正未建立：{exc}{history_warning}")
            self.cancel_calibration()
            return
        try:
            save_attempt(ROOT, self.calibration_raw, self.calibration_screen_points, diagnostic,
                         self.config(), self.calibration_point_details, personal=personal)
        except (OSError, ValueError) as exc:
            history_warning = f"\n校正診斷無法儲存：{exc}"
        try:
            save_calibration(ROOT, personal, self.config())
            self.calibration = personal
            self.update_personal_controls()
            self.test_button.setEnabled(True)
            self.tell(f"你的九點瞄準校正已保存，下次開啟會自動沿用；可以開始演練，或另外校正召喚手勢。{history_warning}")
        except (ValueError, OSError) as exc:
            self.calibration = None
            self.cal_label.setText("校正未通過 · 請重新校正")
            self.test_button.setEnabled(False)
            self.validate_button.hide()
            self.tell(f"個人對應無法儲存：{exc}")
        self.cancel_calibration()

    def calibration_point_summary(self, now):
        frames = self.calibration_observations
        def xyz_median(attribute):
            values = [getattr(frame, attribute) for frame in frames if getattr(frame, attribute) is not None]
            return [median(point[axis] for point in values) for axis in range(3)] if values else None
        depths = [frame.eye_xyz[2] - frame.aperture_xyz[2] for frame in frames
                  if frame.eye_xyz is not None and frame.aperture_xyz is not None]
        amplification = [frame.eye_xyz[2] / (frame.eye_xyz[2] - frame.aperture_xyz[2]) for frame in frames
                         if frame.eye_xyz is not None and frame.aperture_xyz is not None
                         and frame.eye_xyz[2] > frame.aperture_xyz[2]]
        rates = [frame.source_fps for frame in frames if frame.source_fps is not None]
        gesture_failures = {}
        for frame in frames:
            if not frame.pose_ok:
                reason = frame.gesture_reason or "unspecified"
                gesture_failures[reason] = gesture_failures.get(reason, 0) + 1
        return {
            "captured_at_monotonic": now, "source_sample_count": len(frames),
            "source_time_span": frames[-1].timestamp - frames[0].timestamp if frames else 0,
            "eye_xyz_median": xyz_median("eye_xyz"),
            "aperture_xyz_median": xyz_median("aperture_xyz"),
            "depth_gap_median": median(depths) if depths else None,
            "ray_amplification_median": median(amplification) if amplification else None,
            "source_fps_median": median(rates) if rates else None,
            "source_image_size": getattr(frames[-1], "source_size", None) if frames else None,
            "inference_image_size": getattr(frames[-1], "inference_size", None) if frames else None,
            "gesture_recognized_frames": sum(frame.pose_ok for frame in frames),
            "gesture_failure_reasons": gesture_failures,
        }

    def show_calibration_report(self):
        if self.calibration_diagnostics is None:
            return
        personal = self.calibration if isinstance(self.calibration, PersonalCalibration) else None
        self.calibration_report = CalibrationReportDialog(
            self.calibration_diagnostics, (self.overlay.width(), self.overlay.height()), self,
            personal=personal)
        self.calibration_report.retry_requested.connect(self.retry_calibration_point)
        self.calibration_report.validate_requested.connect(lambda: self.begin_validation())
        self.calibration_report.show()

    def retry_calibration_point(self, index):
        if not self.running:
            self.tell("請先啟動鏡頭，再查看校正誤差並選擇重測點。")
            return
        if (self.calibration_diagnostics is None or len(self.calibration_raw) != 9
                or not isinstance(index, int) or not 0 <= index < 9):
            return
        self.return_to_control()
        self.calibrating = True
        self.calibration_validating = False
        self.calibration_retrying = True
        self.calibration_index = index
        self.calibration_capture_deadline = None
        self.calibration_buffer = CalibrationBuffer(min_span=1.0)
        self.calibration_buffer.reset(time.monotonic())
        self.calibration_observations.clear()
        self.overlay.calibration_feedback = f"重測第 {index + 1} 點；其餘八點保留。瞄準圓點後按空白鍵。"
        self.showMinimized()
        self.overlay.set_calibration(self.calibration_screen_points[index], index)
        if self.preview.image is not None:
            self.overlay.set_calibration_preview(self.preview.image)
        self.refresh_calibration_capture()

    def begin_validation(self, *, resume_targets=False):
        if not self.running or not isinstance(self.calibration, PersonalCalibration):
            self.tell("請先啟動鏡頭並建立九點個人對應，再驗證實際瞄準。")
            return
        left, top, right, bottom = self.calibration.coverage_bounds
        if not all(left <= x <= right and top <= y <= bottom
                   for x, y in self.calibration.validation_targets):
            self.tell("已保存的九點仍可沿用；目前映射範圍未涵蓋所有驗證點。如需檢查邊緣，可自行重新記錄靠近邊緣的九點。")
            return
        self.return_to_control()
        self.calibration.clear_validation()
        self.update_personal_controls()
        try:
            save_calibration(ROOT, self.calibration, self.config())
        except OSError as exc:
            self.tell(f"無法保存新的驗證狀態：{exc}")
            return
        self.calibrating = self.calibration_validating = True
        self.calibration_retrying = False
        self.validation_resume_targets = resume_targets
        self.validation_raw = []
        self.validation_frames = []
        self.validation_point_details = []
        self.calibration_index = 0
        self.calibration_buffer = CalibrationBuffer(min_span=1.0)
        self.calibration_buffer.reset(time.monotonic())
        self.calibration_observations.clear()
        targets = self.calibration.validation_targets
        self.overlay.calibration_feedback = "個人對應已固定；現在以新位置確認精度。每點看圓點約一秒，再記錄。"
        self.showMinimized()
        self.overlay.set_calibration(targets[0], 0, total=len(targets), title="實際瞄準驗證")
        if self.preview.image is not None:
            self.overlay.set_calibration_preview(self.preview.image)
        self.refresh_calibration_capture()

    def commit_validation_point(self, point, now):
        model = self.calibration
        if not isinstance(model, PersonalCalibration):
            self.cancel_calibration()
            return
        self.validation_raw.append(point)
        self.validation_frames.append([list(frame.raw_hit) for frame in self.calibration_observations])
        self.validation_point_details.append(self.calibration_point_summary(now))
        self.calibration_buffer.reset(now)
        self.calibration_observations.clear()
        step, targets = len(self.validation_raw), model.validation_targets
        if step < len(targets):
            self.calibration_index = step
            self.overlay.calibration_feedback = f"驗證第 {step} 點已記錄；繼續瞄準新圓點並記錄。"
            self.overlay.set_calibration(targets[step], step, total=len(targets), title="實際瞄準驗證")
            self.refresh_calibration_capture(now)
            return
        report = model.validate(self.validation_raw, targets, screen_size=self.config()["screen"],
                                frame_samples=self.validation_frames)
        resume = self.validation_resume_targets
        try:
            save_calibration(ROOT, model, self.config())
            save_attempt(ROOT, self.calibration_raw, self.calibration_screen_points,
                         self.calibration_diagnostics, self.config(), self.calibration_point_details,
                         personal=model, verification_details=self.validation_point_details)
        except (OSError, ValueError) as exc:
            self.tell(f"驗證結果無法儲存：{exc}")
            self.cancel_calibration()
            self.update_personal_controls()
            return
        self.cancel_calibration()
        self.update_personal_controls()
        if model.ready_for_live:
            self.tell(f"個人瞄準驗證通過：平均 {report.rms_pixels:.1f} px、最大 {report.max_pixels:.1f} px；可使用已校正範圍內的桌面檔案。")
            if resume:
                self.begin_targets()
        else:
            self.tell(f"九點個人對應仍已採用，可先演練；實際瞄準驗證尚未通過：{report.reason}")

    def calibrated_target(self, target):
        if not isinstance(self.calibration, PersonalCalibration):
            return True
        x, y = target.center[0] / self.overlay.width(), target.center[1] / self.overlay.height()
        left, top, right, bottom = self.calibration.coverage_bounds
        return left <= x <= right and top <= y <= bottom

    def cancel_calibration(self):
        self.calibrating = False
        self.calibration_validating = False
        self.validation_resume_targets = False
        self.calibration_capture_deadline = None
        self.overlay.calibration_point = None
        self.overlay.cancel()
        if self.calibration is None:
            previous_notice = self.notice.text()
            self._load_calibration()
            if self.calibration is not None:
                self.test_button.setEnabled(self.running)
                self.tell(f"{previous_notice} 先前完成的瞄準校正已保留並沿用。")
        self.showNormal()
        self.activateWindow()

    def mode_changed(self, *_):
        self.return_to_control()
        self.test_button.setText(("開始演練", "讀取桌面並準備召喚")[self.mode.currentIndex()])
        self.tell(("演練只使用虛擬檔案。",
                   "桌面模式會將鎖定檔案移到垃圾桶。請先在演練確認準星；第一版限主螢幕一般檔案。")[self.mode.currentIndex()])

    def begin_targets(self):
        if not self.running or self.calibration is None:
            self.tell("請先啟動鏡頭與麥克風。" if self.aim_mode == "hand" else
                      "請先啟動感測器並完成瞄準校正。")
            return
        personal = isinstance(self.calibration, PersonalCalibration)
        if self.speech is None or not self.speech.isRunning():
            self.tell("語音監聽未就緒，請重新啟動感測器。")
            return
        self.return_to_control()
        if self.mode.currentIndex() == 0:
            w, h = self.overlay.width(), self.overlay.height()
            targets = []
            for index, (name, x, y) in enumerate(DEMO_DOCUMENTS):
                cell = self.grid_selector.cell_for((x * w, y * h), (w, h))
                left, top, width, height = self.grid_selector.cell_bounds(cell, (w, h))
                center = (left + width / 2, top + height / 2)
                targets.append(DesktopTarget(f"demo-{index}", name, center, radius=52))
            self.activate_targets(targets)
        else:
            self.tell("正在讀取 Finder 的桌面圖示位置；首次請允許控制 Finder。")
            serial = self.action_serial
            self.job(lambda: scan_desktop((self.overlay.width(), self.overlay.height())),
                     lambda targets: self.activate_targets(targets) if serial == self.action_serial else None)

    def activate_targets(self, targets):
        if self.mode.currentIndex() == 1:
            if not targets:
                self.tell("沒有讀到可操作的主螢幕桌面檔案。請確認桌面上有一般檔案並使用一般圖示排列；資料夾、符號連結與隱藏檔不支援。")
                return
            outside = sum(not self.calibrated_target(target) for target in targets)
            if outside:
                self.tell(f"已讀到 {len(targets)} 個桌面檔案，其中 {outside} 個在已校正範圍外；狐狸召喚限已校正範圍。")
        if not targets:
            self.tell("沒有可用的演練檔案。")
            return
        self.clear_target_lock()
        self.targets = self.overlay.targets = targets
        self.overlay.grid_enabled = True
        self.overlay.grid_columns = self.grid_selector.columns
        self.overlay.grid_rows = self.grid_selector.rows
        self.overlay.practice = self.mode.currentIndex() == 0
        self.set_camera_preview(False)
        self.overlay.hint = f"{self.selection_instruction()}後{self.voice_trigger_instruction()}"
        self.armed = self.testing = True
        self.overlay.show()
        self.showMinimized()

    def return_to_control(self, *, stop_sound=True):
        self.set_camera_preview(True)
        self.cancel_hand_range()
        self.cancel_pronunciation_calibration()
        self.cancel_gesture_calibration()
        if self.calibrating:
            self.cancel_calibration()
        self.armed = self.testing = self.calibrating = False
        self.calibration_validating = False
        self.validation_resume_targets = False
        self.calibration_capture_deadline = None
        self.action_serial += 1
        self.pending_action = None
        self.active_action = None
        self.clear_target_lock()
        self.overlay.grid_enabled = False
        self.set_voice(False)
        self.overlay.cancel(stop_sound=stop_sound)
        self.overlay.calibration_point = None
        self.gate.invalidate_tracking(time.monotonic())
        self.showNormal()
        self.activateWindow()

    def summon(self, target_id):
        target = next((value for value in self.targets if value.id == target_id), None)
        if target is None or not self.armed or self.overlay.eating is not None:
            return
        if (target.snapshot is not None and not self.overlay.practice
                and isinstance(self.calibration, PersonalCalibration)
                and not self.calibrated_target(target)):
            self.return_to_control()
            self.tell("圖示超出已保存的瞄準範圍，請調整圖示位置後再瞄準。")
            return
        self.armed = False
        self.set_voice(False)
        serial = self.action_serial
        def play(checked_target):
            if serial != self.action_serial:
                return
            real_file = bool(checked_target.snapshot and not self.overlay.practice)
            self.active_action = ActionState(serial, checked_target.id, real_file)
            self.pending_action = (serial, checked_target.id, real_file)
            self.overlay.hint = f"狐狸已召喚 · {checked_target.name}"
            self.overlay.play(checked_target)
        if target.snapshot is not None and not self.overlay.practice:
            point = target.center if self.locked_target() is target else self.aim_point
            self.overlay.hint = f"正在確認 {target.name} 的圖示位置…"
            self.job(lambda: validate_target_position(target, point, (self.overlay.width(), self.overlay.height())), play)
        else:
            play(target)

    def preview_animation(self):
        self.return_to_control()
        self.overlay.practice = True
        self.overlay.targets = []
        self.overlay.hint = "動畫預覽 · 虛擬檔案"
        self.overlay.play(DesktopTarget("preview", "練習檔案.txt", (self.overlay.width() * .5, self.overlay.height() * .5)))

    def on_bite(self, target):
        if target is None:
            return
        pending = self.pending_action
        if target.snapshot is None:
            self.targets = [value for value in self.targets if value.id != target.id]
            self.overlay.targets = [value for value in self.overlay.targets if value.id != target.id]
            if getattr(self.gate, "locked_target", None) == target.id:
                self.clear_target_lock()
            state = self.active_action
            if state is not None and state.serial == self.action_serial and state.target_id == target.id:
                state.succeeded = True
                self.continue_after_action(state)
            self.tell(f"狐狸吞掉了 {target.name}（演練）")
            return
        if pending is None or not pending[2]:
            return
        serial, target_id, _ = pending
        if serial != self.action_serial or target_id != target.id or target.snapshot is None:
            return
        self.pending_action = None
        self.job(lambda: self.trash_if_current(serial, target.snapshot),
                 lambda receipt: self.file_trashed(receipt, serial=serial, target_id=target_id))

    def trash_if_current(self, serial, snapshot):
        if serial != self.action_serial:
            return None
        return trash_target(snapshot)

    def file_trashed(self, receipt, *, serial=None, target_id=None):
        if receipt is None:
            if self.active_action is not None and self.active_action.serial == serial:
                self.return_to_control()
                self.tell("檔案處理未完成，請重新確認後再開始。")
            return
        self.receipt = receipt
        self.undo_button.setEnabled(True)
        self.targets = [target for target in self.targets if target.id != str(receipt.original)]
        self.overlay.targets = self.targets
        if getattr(self.gate, "locked_target", None) == str(receipt.original):
            self.clear_target_lock()
        self.tell(f"{receipt.original.name} 已移到垃圾桶，可使用「復原上一個檔案」。")
        state = self.active_action
        if (state is not None and state.serial == serial == self.action_serial
                and state.target_id == target_id == str(receipt.original)):
            state.succeeded = True
            self.continue_after_action(state)

    def on_animation_done(self):
        if self.overlay.eating is not None:
            return
        self.pending_action = None
        state = self.active_action
        if state is not None and self.testing and state.serial == self.action_serial:
            state.animation_done = True
            self.continue_after_action(state)
        elif not self.testing:
            self.overlay.hide()

    def continue_after_action(self, state):
        """A confirmed success and a finished animation authorize one next selection."""
        if (state is not self.active_action or state.serial != self.action_serial
                or not self.running or not self.testing
                or not state.succeeded or not state.animation_done or state.refreshing):
            return
        state.refreshing = True
        self.set_voice(False)
        if state.real_file:
            # Finder can rearrange icons after a deletion. Never reuse positions
            # or snapshots from before the successful move to Trash.
            self.overlay.hint = "正在更新桌面檔案位置…"
            self.job(lambda: scan_desktop((self.overlay.width(), self.overlay.height())),
                     lambda targets: self.resume_after_action(state, targets))
        else:
            self.resume_after_action(state, self.targets)

    def resume_after_action(self, state, targets):
        if (state is not self.active_action or state.serial != self.action_serial
                or not self.running or not self.testing):
            return
        if not self.gate.complete_action(state.target_id, time.monotonic()):
            self.return_to_control(stop_sound=False)
            self.tell("這次召喚狀態已取消，請重新開始。")
            return
        self.active_action = None
        if not targets:
            self.return_to_control(stop_sound=False)
            self.tell("這一輪檔案已處理完成。")
            return
        self.activate_targets(targets)
        if self.armed:
            self.overlay.hint = (f"保持手勢，瞄準下一個檔案 · {self.selection_instruction()}\n"
                                 f"鎖定後稍停，再{self.voice_trigger_instruction()}")

    def undo(self):
        if self.receipt is None:
            return
        self.return_to_control()
        receipt = self.receipt
        self.undo_button.setEnabled(False)
        def restored(_):
            self.receipt = None
            self.tell(f"已復原 {receipt.original.name}。")
        self.job(lambda: restore_receipt(receipt), restored)

    def change_language(self, *_):
        language = self.language_combo.currentData()
        if language == get_language():
            return
        try:
            save_language(language, ROOT)
        except OSError as exc:
            self.tell(f"無法儲存語言設定：{exc}")
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(self.language_combo.findData(get_language()))
            self.language_combo.blockSignals(False)
            return
        set_language(language)
        self.controls_scroll.setMinimumWidth(410 if language == 'en' else 0)
        self.controls_scroll.setMaximumWidth(440 if language == 'en' else 350)
        self.setMinimumWidth(1100 if language == 'en' else 960)
        if language == 'en' and self.width() < 1140:
            self.resize(1140, self.height())
        retranslate_all()
        self.overlay.update()

    def closeEvent(self, event: QCloseEvent):
        self.stop_sensors()
        for worker in (self.camera, self.speech, *self._retiring_pronunciation_workers):
            if worker and not worker.wait(2500):
                self.tell("感測器仍在停止，請稍候再關閉。")
                event.ignore()
                return
        self.tray.hide()
        self.overlay.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, help="Render UI without opening sensors")
    parser.add_argument("--language", choices=("en", "zh-TW"), help="Interface language (saved preference or system language by default)")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    set_language(preferred_language(ROOT, args.language))
    app.setApplicationName("KON")
    app.setOrganizationName("KON Local")
    app.setFont(QFont("PingFang TC", 13))
    window = Window()
    window.show()
    if args.screenshot:
        def save():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            window.close()
            app.quit()
        QTimer.singleShot(300, save)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
