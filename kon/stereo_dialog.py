"""Explicit dual-camera checkerboard calibration, without opening cameras on construction."""
from __future__ import annotations

from .paths import resource_root, state_root

import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .gesture_calibration_dialog import _PreviewLabel, _label
from .stereo import ScreenPlane, checkerboard_object_points, detect_checkerboard, solve_stereo_calibration
from .stereo_capture import DualCapture, list_camera_sources
from .stereo_samples import MAX_SAMPLES, StereoSampleSession, save_diagnostic_report, validate_sample

_COLUMNS, _ROWS = 9, 6
_MINIMUM_PAIRS = 15
_MAX_SKEW_S, _MAX_AGE_S = .020, .3
CHECKPOINT_DIR = state_root() / 'stereo-samples'


class _FreshSampleRequest:
    """The request time belongs to the UI click, not to delayed frame processing."""

    def __init__(self):
        self._lock = threading.Lock()
        self._at = None

    @property
    def pending_at(self):
        with self._lock:
            return self._at

    def request(self, now=None):
        with self._lock:
            if self._at is not None:
                return False
            self._at = time.monotonic() if now is None else float(now)
            return True

    def cancel(self):
        with self._lock:
            self._at = None

    def finish(self, token):
        with self._lock:
            if self._at is None or self._at != token:
                return False
            self._at = None
            return True

    def is_fresh(self, pair, now=None):
        at = self.pending_at
        now = time.monotonic() if now is None else float(now)
        stamps = (float(pair.primary.timestamp), float(pair.secondary.timestamp))
        return (at is not None and all(math.isfinite(t) and at <= t <= now for t in stamps)
                and now - min(stamps) <= _MAX_AGE_S
                and abs(stamps[0] - stamps[1]) <= _MAX_SKEW_S)


class _CalibrationPreview(_PreviewLabel):
    def _refresh(self):
        if self.image is None:
            self.clear()
            self.setText("啟動預覽後顯示畫面\n讓完整棋盤同時出現在兩個鏡頭")
        else:
            super()._refresh()


def _preview_image(pixels, corners=None):
    pixels = pixels.copy()
    if corners is not None:
        cv2.drawChessboardCorners(pixels, (_COLUMNS, _ROWS),
                                  np.asarray(corners, np.float32).reshape(-1, 1, 2), True)
    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


def checkerboard_print_image(square_mm=25., pixels_per_square=256):
    """PNG metadata keeps square dimensions physical when printed at 100%."""
    square_mm = float(square_mm)
    if not math.isfinite(square_mm) or not 5 <= square_mm <= 100:
        raise ValueError("棋盤格邊長必須介於 5 與 100 mm。")
    if type(pixels_per_square) is not int or pixels_per_square < 16:
        raise ValueError("棋盤格解析度不足。")
    side = pixels_per_square
    margin = round(side * 10 / square_mm)
    image = np.full(((_ROWS + 1) * side + margin * 2,
                     (_COLUMNS + 1) * side + margin * 2), 255, np.uint8)
    for row in range(_ROWS + 1):
        for column in range(_COLUMNS + 1):
            if (row + column) % 2 == 0:
                y, x = margin + row * side, margin + column * side
                image[y:y + side, x:x + side] = 0
    height, width = image.shape
    result = QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()
    dots_per_meter = round(side * 1000 / square_mm)
    result.setDotsPerMeterX(dots_per_meter)
    result.setDotsPerMeterY(dots_per_meter)
    return result


class _CaptureWorker(QThread):
    previews = Signal(object, object, float)
    sample_ready = Signal(object)
    sample_failed = Signal(str)
    failed = Signal(str)

    def __init__(self, primary_source, secondary_source, primary_id='', secondary_id='', parent=None):
        super().__init__(parent)
        self.primary_source = primary_source
        self.secondary_source = secondary_source
        self.primary_id, self.secondary_id = primary_id, secondary_id
        self.requests = _FreshSampleRequest()
        self._stop_event = threading.Event()

    def request_sample(self, now=None):
        return self.requests.request(now)

    def stop(self):
        self.requests.cancel()
        self._stop_event.set()
        self.requestInterruption()

    def _sources(self):
        # Native discovery enumerates device names/IDs without opening capture sessions.
        sources = {item.index: item for item in list_camera_sources()}
        resolved = []
        for index, expected in ((self.primary_source, self.primary_id),
                                (self.secondary_source, self.secondary_id)):
            source = sources.get(index)
            device_id = getattr(source, 'device_id', '')
            if expected and device_id != expected:
                raise RuntimeError("鏡頭編號已改變，請重新選擇主鏡頭與副鏡頭。")
            resolved.append(source if source is not None else index)
        return resolved

    def run(self):
        capture = None
        try:
            primary, secondary = self._sources()
            self.primary_id = getattr(primary, 'device_id', '')
            self.secondary_id = getattr(secondary, 'device_id', '')
            if self.isInterruptionRequested():
                return
            capture = DualCapture(primary, secondary, max_skew_s=_MAX_SKEW_S,
                                  max_age_s=_MAX_AGE_S, stop_event=self._stop_event)
            capture.start()
            last_preview = 0.
            while not self.isInterruptionRequested():
                pair = capture.take(timeout=.05)
                now = time.monotonic()
                token = self.requests.pending_at
                if token is not None and now - token > 5:
                    if self.requests.finish(token):
                        self.sample_failed.emit("未取得新的清楚棋盤格畫面；請讓完整棋盤同時出現在兩個鏡頭，再記錄。")
                if pair is None:
                    if capture.failure:
                        raise RuntimeError(str(capture.failure))
                    continue
                if now - last_preview < .12 and token is None:
                    continue
                last_preview = now
                left = detect_checkerboard(pair.primary.pixels, _COLUMNS, _ROWS)
                right = detect_checkerboard(pair.secondary.pixels, _COLUMNS, _ROWS)
                if self.isInterruptionRequested():
                    break
                skew = abs(pair.primary.timestamp - pair.secondary.timestamp)
                self.previews.emit(_preview_image(pair.primary.pixels, left),
                                   _preview_image(pair.secondary.pixels, right), skew)
                # Do not use a cached preview, a pair read before the click, or an expired pair.
                if (token is not None and left is not None and right is not None
                        and self.requests.is_fresh(pair) and self.requests.finish(token)):
                    self.sample_ready.emit({
                        'primary': np.asarray(left, np.float32).reshape(-1, 2).copy(),
                        'secondary': np.asarray(right, np.float32).reshape(-1, 2).copy(),
                        'primary_size': tuple(reversed(pair.primary.pixels.shape[:2])),
                        'secondary_size': tuple(reversed(pair.secondary.pixels.shape[:2])),
                        'primary_id': self.primary_id, 'secondary_id': self.secondary_id,
                        'primary_name': getattr(primary, 'name', ''),
                        'secondary_name': getattr(secondary, 'name', ''),
                    })
        except Exception as error:
            if not self.isInterruptionRequested():
                self.failed.emit(f"雙鏡頭無法使用：{error}")
        finally:
            self.requests.cancel()
            if capture is not None:
                warned = False
                while not capture.close():
                    if not warned:
                        self.failed.emit("鏡頭仍在結束連線，請等候鏡頭釋放後再使用。")
                        warned = True
                    # Keep the dialog alive until both camera owners have released them.
                    self.msleep(100)


class _SolveWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, samples, square_mm, primary_source, secondary_source, parent=None, *, checkpoint_path=None):
        super().__init__(parent)
        self.samples = samples
        self.square_mm = square_mm
        self.primary_source, self.secondary_source = primary_source, secondary_source
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.diagnostics_path = None

    def run(self):
        try:
            points = checkerboard_object_points(_COLUMNS, _ROWS, self.square_mm / 1000)
            first = self.samples[0]
            result = solve_stereo_calibration(
                [points.copy() for _ in self.samples],
                [sample['primary'] for sample in self.samples],
                [sample['secondary'] for sample in self.samples],
                first['primary_size'], first['secondary_size'],
                primary_source=self.primary_source, secondary_source=self.secondary_source,
                primary_id=first['primary_id'], secondary_id=first['secondary_id'])
            if not self.isInterruptionRequested():
                self.succeeded.emit(result)
        except Exception as error:
            diagnostic_note = ''
            diagnostics = getattr(error, 'diagnostics', None)
            if diagnostics is not None and self.checkpoint_path is not None:
                path = self.checkpoint_path.with_name(f'{self.checkpoint_path.stem}.diagnostics-{uuid4().hex}.json')
                try:
                    save_diagnostic_report(path, diagnostics, sample_checkpoint=self.checkpoint_path)
                    self.diagnostics_path = path
                    diagnostic_note = f'\n數值診斷已保存：{path}'
                except (OSError, ValueError, TypeError) as save_error:
                    diagnostic_note = f'\n數值診斷無法保存：{save_error}'
            if not self.isInterruptionRequested():
                self.failed.emit(f"校正未完成：{error}。請補拍不同位置、距離與傾斜角度。{diagnostic_note}")


def _spin(label, value, minimum, maximum, suffix, decimals=1):
    field = QDoubleSpinBox()
    field.setAccessibleName(label)
    field.setRange(minimum, maximum)
    field.setDecimals(decimals)
    field.setSuffix(suffix)
    field.setValue(value)
    field.setKeyboardTracking(False)
    return field


class StereoCalibrationDialog(QDialog):
    """Return a calibration and measured screen plane only after explicit acceptance."""

    calibrated = Signal(object, object)

    def __init__(self, primary_source: int, secondary_source: int, parent=None, *,
                 initial_plane=None, screen_size_mm=None, primary_id='', secondary_id='', checkpoint_dir=None):
        super().__init__(parent)
        self.primary_source, self.secondary_source = int(primary_source), int(secondary_source)
        self.primary_id, self.secondary_id = str(primary_id), str(secondary_id)
        self.calibration_result = self.screen_plane_result = None
        self._calibration = None
        self._samples = []
        self._capture_worker = self._solve_worker = None
        self._capture_stopping = False
        self._record_pending = False
        self._solve_pending = False
        self._finish_code = None
        self._pending_accept = None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
        self._new_sample_session()
        self.setWindowTitle("KON · 雙鏡頭深度校正")
        self.setModal(True)
        self.resize(960, 850)
        self.setMinimumSize(660, 540)
        self.setStyleSheet("""
            QDialog { background: #111018; color: #eee6f1; }
            QLabel, QCheckBox, QGroupBox { color: #eee6f1; background: transparent; }
            QScrollArea { border: none; }
            QScrollArea > QWidget > QWidget { background: #111018; }
            QGroupBox { border: 1px solid #443549; border-radius: 9px; margin-top: 12px; padding: 13px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QDoubleSpinBox { background: #201b29; color: #eee6f1; border: 1px solid #5b485e; padding: 5px; }
            QPushButton { background: #2b2331; color: #eee6f1; border: 1px solid #5b485e;
                border-radius: 8px; padding: 9px 13px; min-height: 18px; }
            QPushButton:hover { background: #443040; }
            QPushButton:focus, QDoubleSpinBox:focus, QCheckBox:focus { border: 2px solid #ffbd91; }
            QPushButton#acceptStereo { background: #f5ab80; color: #2b160e; }
            QPushButton:disabled { background: #29232e; color: #8c7d94; border-color: #3c3343; }
        """)
        self._build(initial_plane, screen_size_mm)
        self._update_controls()

    def _new_sample_session(self):
        self._sample_session_id = uuid4().hex
        self._sample_created_at = datetime.now(timezone.utc).isoformat()
        self.checkpoint_path = self.checkpoint_dir / f'{self._sample_session_id}.json'

    def _sample_snapshot(self):
        return StereoSampleSession(self.square_mm.value(), self.primary_source, self.secondary_source,
                                   tuple(self._samples), self._sample_session_id, self._sample_created_at)

    def _checkpoint_samples(self):
        try:
            self._sample_snapshot().save(self.checkpoint_path)
        except (OSError, ValueError, TypeError) as error:
            message = f'角點仍保留於本次視窗，但自動保存失敗：{error}'
            self.checkpoint_label.setText(message)
            return message
        self.checkpoint_label.setText(f'已自動保存 {len(self._samples)} 組角點：{self.checkpoint_path}')
        return None

    def _save_samples(self):
        if not self.save_samples_button.isEnabled():
            return
        filename, _ = QFileDialog.getSaveFileName(self, '儲存雙鏡頭棋盤取樣', str(self.checkpoint_path),
                                                '棋盤角點 JSON (*.json)')
        if not filename:
            return
        path = Path(filename).with_suffix('.json')
        try:
            self._sample_snapshot().save(path)
        except (OSError, ValueError, TypeError) as error:
            self.status_label.setText(f'取樣未儲存；本次角點仍在視窗中：{error}')
            return
        self.status_label.setText(f'已儲存 {len(self._samples)} 組角點，不含影像：{path}')

    def _load_samples(self):
        # Loading never changes the current camera owners or interrupts capture.
        if (self._capture_worker is not None or self._capture_stopping or self._record_pending or
                self._solve_pending or self._solve_worker is not None or self._finish_code is not None):
            self.status_label.setText('請先暫停雙鏡頭預覽並等候計算結束，再載入取樣。')
            return
        filename, _ = QFileDialog.getOpenFileName(self, '載入雙鏡頭棋盤取樣', str(self.checkpoint_dir),
                                                '棋盤角點 JSON (*.json)')
        if not filename:
            return
        try:
            session = StereoSampleSession.load(filename)
            sources = {source.index: source for source in list_camera_sources()}
            primary, secondary = sources.get(self.primary_source), sources.get(self.secondary_source)
            session.require_matching_sources(primary, secondary)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            self.status_label.setText(f'未載入，保留本次取樣：{error}')
            return
        confirmed = QMessageBox.question(
            self, '確認沿用同一組實體鏡頭配置',
            f'載入 {len(session.samples)} 組角點，每格 {session.square_mm:g} mm。\n'
            '請確認兩台實體裝置、選定鏡頭、支架位置、方向、縮放、裁切、鏡像與視野皆未改變。'
            'Camo 裝置識別無法確認背後的 iPhone／iPad 或鏡頭是否已換過。\n'
            '確認相同才取代本次角點；載入後仍須重新計算校正並核對螢幕量測。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirmed != QMessageBox.Yes:
            self.status_label.setText('已取消載入；本次取樣未改動。')
            return
        # All validation and explicit confirmation precede every state mutation.
        self._samples = [validate_sample(sample) for sample in session.samples]
        previous = self.square_mm.blockSignals(True)
        self.square_mm.setValue(session.square_mm)
        self.square_mm.blockSignals(previous)
        self.primary_id, self.secondary_id = primary.device_id, secondary.device_id
        self._calibration = None
        self.measurements_confirmed.setChecked(False)
        self._new_sample_session()
        warning = self._checkpoint_samples()
        self.progress_label.setText(f'已記錄 {len(self._samples)} / {_MINIMUM_PAIRS} 組不同角度')
        self.status_label.setText('已載入角點，尚未套用校正。重新啟動預覽後，新取樣須符合原裝置與影像尺寸。' +
                                  (f'\n{warning}' if warning else ''))
        self._update_controls()

    def _build(self, initial_plane, screen_size_mm):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        content = QVBoxLayout(body)
        content.setSpacing(12)
        content.addWidget(_label("外部攝影機 ＋ 副鏡頭 · 深度校正", size=20))
        self.instructions_label = _label(
            "先用 USB 連接副鏡頭：iPhone 可用接續互通，iPad 請先透過 Camo Studio 連線。"
            "固定兩台鏡頭並關閉人物居中、縮放與自動切換鏡頭。"
            "可用貼平硬板的紙本或平整螢幕棋盤；固定圖樣大小、縮放與朝向，完整棋盤及白邊都要入鏡。"
            "在畫面中央、四周與不同距離傾斜棋盤，停穩後記錄至少 15 個角度。", muted=True)
        content.addWidget(self.instructions_label)
        board_row = QHBoxLayout()
        board_label = _label("9 × 6 個內角點 · 每格邊長")
        board_label.setWordWrap(False)
        board_row.addWidget(board_label)
        self.square_mm = _spin("實際棋盤格邊長", 25, 5, 100, " mm", 2)
        self.square_mm.setMaximumWidth(140)
        self.square_mm.valueChanged.connect(self._board_changed)
        board_row.addWidget(self.square_mm)
        self.export_button = QPushButton("儲存列印棋盤…")
        self.export_button.clicked.connect(self._export_board)
        board_row.addWidget(self.export_button)
        board_row.addStretch()
        content.addLayout(board_row)
        content.addWidget(_label("棋盤共 10 × 7 格。紙本以 100% 列印；螢幕圖樣固定大小。量測整排十格寬度 ÷ 10，填入實際單格毫米數。", size=11, muted=True))
        previews = QHBoxLayout()
        for label, attribute, index in (("外部攝影機", 'primary_preview', self.primary_source),
                                         ("副鏡頭", 'secondary_preview', self.secondary_source)):
            tile = QVBoxLayout()
            tile.addWidget(_label(f"{label} · 鏡頭 {index}"))
            preview = _CalibrationPreview()
            preview.setMinimumSize(280, 170)
            preview.setAccessibleName(f"{label}校正預覽")
            preview._refresh()
            setattr(self, attribute, preview)
            tile.addWidget(preview)
            previews.addLayout(tile)
        content.addLayout(previews)
        self.sync_label = _label("以電腦收到影格的時間配對，並非同步曝光；記錄時請讓棋盤停穩。", size=11, muted=True)
        content.addWidget(self.sync_label)
        capture_row = QHBoxLayout()
        self.start_button = QPushButton("啟動雙鏡頭預覽")
        self.start_button.clicked.connect(self._toggle_preview)
        self.record_button = QPushButton("記錄此角度")
        self.record_button.clicked.connect(self._request_sample)
        self.reset_button = QPushButton("清除本次記錄")
        self.reset_button.clicked.connect(self._reset_samples)
        self.solve_button = QPushButton("計算雙鏡頭校正")
        self.solve_button.clicked.connect(self._begin_solve)
        for button in (self.start_button, self.record_button, self.reset_button, self.solve_button):
            button.setAutoDefault(False)
            capture_row.addWidget(button)
        content.addLayout(capture_row)
        self.progress_label = _label(f"已記錄 0 / {_MINIMUM_PAIRS} 組不同角度")
        self.progress_label.setAccessibleName("雙鏡頭校正取樣數")
        content.addWidget(self.progress_label)
        sample_row = QHBoxLayout()
        self.save_samples_button = QPushButton("儲存取樣…")
        self.save_samples_button.clicked.connect(self._save_samples)
        self.load_samples_button = QPushButton("載入取樣…")
        self.load_samples_button.clicked.connect(self._load_samples)
        for button in (self.save_samples_button, self.load_samples_button):
            button.setAutoDefault(False)
            sample_row.addWidget(button)
        sample_row.addStretch()
        content.addLayout(sample_row)
        self.checkpoint_label = _label("每組角點自動保存，不含影像；載入前需暫停預覽並確認兩鏡配置未變。", size=11, muted=True)
        self.checkpoint_label.setAccessibleName("雙鏡頭取樣保存狀態")
        content.addWidget(self.checkpoint_label)
        self.status_label = _label("準備好後啟動預覽。原有校正會在完成並套用後才更新。", muted=True)
        self.status_label.setAccessibleName("雙鏡頭校正狀態")
        content.addWidget(self.status_label)
        geometry = QGroupBox("螢幕與主鏡頭的實際位置")
        fields = QFormLayout(geometry)
        fields.addRow(_label(
            "量測螢幕可見範圍與外部攝影機鏡頭中心。以你面向螢幕時的左上角為起點；"
            "X 向右、Y 向下為正；鏡頭在螢幕上方時 Y 為負。鏡頭朝向你的距離為正。"
            "建議主鏡頭置中、光軸垂直螢幕。角度請填實測值；改動鏡頭或螢幕後需重新校正。", size=11, muted=True))
        fields.addRow(_label("角度正向：俯仰向下、偏航朝你面向螢幕時的右方、旋轉為你面向螢幕時看到的順時針。", size=11, muted=True))
        width, height = (screen_size_mm or (530, 300))
        width = width if math.isfinite(width) and width > 0 else 530
        height = height if math.isfinite(height) and height > 0 else 300
        values = {'width_m': width / 1000, 'height_m': height / 1000,
                  'camera_x_m': width / 2000, 'camera_y_m': -.02, 'camera_forward_m': .01,
                  'pitch_degrees': 0, 'yaw_degrees': 0, 'roll_degrees': 0}
        if initial_plane is not None:
            measurements = getattr(initial_plane, 'measurements', None)
            if callable(measurements):
                values.update(measurements())
            elif isinstance(measurements, dict):
                values.update(measurements)
        self.plane_fields = {}
        for key, title in (("width_m", "螢幕可見寬度"), ("height_m", "螢幕可見高度"),
                           ("camera_x_m", "鏡頭 X（距螢幕左緣）"), ("camera_y_m", "鏡頭 Y（距螢幕上緣）"),
                           ("camera_forward_m", "鏡頭在螢幕前方的距離"),
                           ("pitch_degrees", "俯仰 pitch"), ("yaw_degrees", "偏航 yaw"),
                           ("roll_degrees", "旋轉 roll")):
            angular = key.endswith('degrees')
            positive = key in ('width_m', 'height_m')
            minimum = -75 if angular else 10 if positive else 0 if key == 'camera_forward_m' else -300
            maximum = (75 if angular else 200 if key == 'height_m'
                       else 50 if key == 'camera_forward_m' else 300)
            field = _spin(title, values[key] if angular else values[key] * 100,
                          minimum, maximum, " °" if angular else " cm", 1)
            self.plane_fields[key] = field
            field.valueChanged.connect(self._measurement_changed)
            fields.addRow(title, field)
        self.measurements_confirmed = QCheckBox("已確認螢幕尺寸、鏡頭位置與角度")
        self.measurements_confirmed.toggled.connect(self._update_controls)
        fields.addRow(self.measurements_confirmed)
        content.addWidget(geometry)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        footer = QHBoxLayout()
        footer.addStretch()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.accept_button = QPushButton("套用深度校正")
        self.accept_button.setObjectName('acceptStereo')
        self.accept_button.clicked.connect(self.accept)
        for button in (self.cancel_button, self.accept_button):
            button.setAutoDefault(False)
            footer.addWidget(button)
        outer.addLayout(footer)

    def _update_controls(self, *_):
        capture_active = self._capture_worker is not None
        busy = self._solve_pending or self._solve_worker is not None or self._finish_code is not None
        self.start_button.setEnabled(not busy and not self._capture_stopping)
        self.start_button.setText("暫停預覽" if capture_active else "啟動雙鏡頭預覽")
        self.record_button.setEnabled(capture_active and not self._capture_stopping
                                      and not self._record_pending and not busy and len(self._samples) < MAX_SAMPLES)
        self.reset_button.setEnabled(bool(self._samples) and not busy and not self._record_pending)
        self.solve_button.setEnabled(len(self._samples) >= _MINIMUM_PAIRS and not busy
                                     and not self._record_pending and not self._capture_stopping)
        self.square_mm.setEnabled(not busy and not self._record_pending)
        self.save_samples_button.setEnabled(bool(self._samples) and not busy and not self._record_pending
                                            and not self._capture_stopping)
        self.load_samples_button.setEnabled(not capture_active and not busy and not self._record_pending
                                            and not self._capture_stopping)
        self.accept_button.setEnabled(self._calibration is not None and not busy
                                      and self.measurements_confirmed.isChecked())
        self.cancel_button.setEnabled(self._finish_code is None)

    def _toggle_preview(self):
        if self._capture_worker is not None:
            self._stop_capture()
            return
        if self.primary_source == self.secondary_source:
            self.status_label.setText("請選擇兩個不同的鏡頭。")
            return
        self._capture_stopping = False
        worker = _CaptureWorker(self.primary_source, self.secondary_source,
                                self.primary_id, self.secondary_id, self)
        self._capture_worker = worker
        worker.previews.connect(self._set_previews)
        worker.sample_ready.connect(self._record_pair)
        worker.sample_failed.connect(self._sample_failed)
        worker.failed.connect(self._capture_failed)
        worker.finished.connect(self._capture_finished)
        self.status_label.setText("正在開啟兩台鏡頭，請稍候…")
        self._update_controls()
        worker.start()

    def _stop_capture(self):
        if self._capture_worker is not None:
            self._capture_stopping = True
            self._record_pending = False
            self._capture_worker.stop()
            self.status_label.setText("正在釋放兩台鏡頭…")
            self._update_controls()

    def _capture_finished(self):
        worker = self._capture_worker
        self._capture_worker = None
        self._capture_stopping = False
        self._record_pending = False
        if worker is not None:
            worker.deleteLater()
        if self._finish_code is not None:
            self._finish_when_stopped()
        elif self._solve_pending:
            self._start_solver()
        self._update_controls()

    def _set_previews(self, primary, secondary, skew):
        if self._capture_stopping or self._finish_code is not None:
            return
        self.primary_preview.set_image(primary)
        self.secondary_preview.set_image(secondary)
        self.sync_label.setText(f"接收時間差 {skew * 1000:.0f} ms（上限 20 ms）；非同步曝光。記錄時請讓棋盤停穩。")

    def _request_sample(self):
        worker = self._capture_worker
        if worker is None or not self.record_button.isEnabled():
            return
        if worker.request_sample(time.monotonic()):
            self._record_pending = True
            self.status_label.setText("等待按下記錄後的新畫面…請停穩棋盤，直到顯示已記錄。")
            self._update_controls()

    def _record_pair(self, sample):
        if not self._record_pending or self._capture_stopping or self._finish_code is not None:
            return
        self._record_pending = False
        if len(self._samples) >= MAX_SAMPLES:
            self.status_label.setText(f"已達 {MAX_SAMPLES} 組取樣上限；請先計算校正或另開新取樣。")
            self._update_controls()
            return
        try:
            sample = validate_sample(sample)
        except (ValueError, TypeError) as error:
            self.status_label.setText(f"這組棋盤資料未採用：{error}")
            self._update_controls()
            return
        if self._samples:
            first = self._samples[0]
            if any(sample[key] != first[key] for key in ('primary_size', 'secondary_size', 'primary_id', 'secondary_id')):
                self._stop_capture()
                self.status_label.setText("鏡頭或畫面尺寸已改變。原取樣已保留；請固定設定後另開新取樣，勿混用。")
                self._update_controls()
                return
            for old in self._samples:
                distances = [np.mean(np.sum(((sample[key] - old[key]) / np.asarray(sample[size])) ** 2, axis=1))
                             for key, size in (('primary', 'primary_size'), ('secondary', 'secondary_size'))]
                if math.sqrt(sum(distances) / 2) < .028:
                    self.status_label.setText("此角度與已有記錄太接近；請改變棋盤位置、距離或傾斜角度，再停穩記錄。")
                    self._update_controls()
                    return
        self._samples.append(sample)
        self._calibration = None
        warning = self._checkpoint_samples()
        self.progress_label.setText(f"已記錄 {len(self._samples)} / {_MINIMUM_PAIRS} 組不同角度")
        self.status_label.setText("已記錄。移動棋盤到另一位置或角度，再停穩記錄。"
                                  if len(self._samples) < _MINIMUM_PAIRS else "已收集足夠角度，可計算校正；也可以繼續補拍。")
        if warning:
            self.status_label.setText(self.status_label.text() + '\n' + warning)
        self._update_controls()

    def _sample_failed(self, message):
        if self._finish_code is None and not self._capture_stopping:
            self._record_pending = False
            self.status_label.setText(message)
            self._update_controls()

    def _capture_failed(self, message):
        if self._finish_code is None:
            self._record_pending = False
            self.status_label.setText(message)
            self._update_controls()

    def _reset_samples(self):
        self._samples.clear()
        self._calibration = None
        self._new_sample_session()
        self.checkpoint_label.setText("先前的取樣檔仍保留；新記錄會自動保存至另一個工作階段。")
        self.progress_label.setText(f"已記錄 0 / {_MINIMUM_PAIRS} 組不同角度")
        self.status_label.setText("本次記錄已清除，請重新記錄不同位置與角度。")
        self._update_controls()

    def _board_changed(self, *_):
        if hasattr(self, 'status_label'):
            self._reset_samples()

    def _measurement_changed(self, *_):
        if hasattr(self, 'measurements_confirmed'):
            self.measurements_confirmed.setChecked(False)
            self._update_controls()

    def _make_screen_plane(self):
        values = {key: field.value() if key.endswith('degrees') else field.value() / 100
                  for key, field in self.plane_fields.items()}
        return ScreenPlane.from_measurements(**values)

    def _begin_solve(self):
        if not self.solve_button.isEnabled():
            return
        self._solve_pending = True
        self._calibration = None
        if self._capture_worker is not None:
            self._stop_capture()
        else:
            self._start_solver()
        self._update_controls()

    def _start_solver(self):
        if self._finish_code is not None:
            return
        self._solve_pending = False
        worker = _SolveWorker(list(self._samples), self.square_mm.value(),
                              self.primary_source, self.secondary_source, self,
                              checkpoint_path=self.checkpoint_path)
        self._solve_worker = worker
        worker.succeeded.connect(self._solved)
        worker.failed.connect(self._solve_failed)
        worker.finished.connect(self._solve_finished)
        self.status_label.setText("正在計算鏡頭參數與相對位置…取消後會等計算安全結束。")
        self._update_controls()
        worker.start()

    def _solved(self, calibration):
        if self._finish_code is None:
            self._calibration = calibration
            self.status_label.setText(f"校正成功 · 重投影 RMS {calibration.rms:.3f} px。"
                                      "請核對下方螢幕量測值，確認後套用；此數值不代表實際瞄準誤差。")
            self._update_controls()

    def _solve_failed(self, message):
        if self._finish_code is None:
            self.status_label.setText(message)

    def _solve_finished(self):
        worker = self._solve_worker
        self._solve_worker = None
        if worker is not None:
            worker.deleteLater()
        self._finish_when_stopped()
        self._update_controls()

    def _export_board(self):
        filename, _ = QFileDialog.getSaveFileName(self, "儲存 9 × 6 內角點列印棋盤",
                                                f"KON-checkerboard-{self.square_mm.value():g}mm.png", "PNG 圖片 (*.png)")
        if not filename:
            return
        filename = str(Path(filename).with_suffix('.png'))
        if checkerboard_print_image(self.square_mm.value()).save(filename, 'PNG'):
            self.status_label.setText("棋盤已儲存。請以實際大小／100% 列印，量測單格實際邊長後填入設定。")
        else:
            self.status_label.setText("棋盤儲存失敗，請選擇可寫入的位置。")

    def accept(self):
        if (self._calibration is None or not self.measurements_confirmed.isChecked()
                or self._solve_worker is not None or self._solve_pending):
            self.status_label.setText("請完成雙鏡頭校正並確認螢幕尺寸、鏡頭位置與角度，再套用。")
            return
        try:
            plane = self._make_screen_plane()
        except (ValueError, TypeError) as error:
            self.status_label.setText(f"請修正螢幕量測值：{error}")
            return
        self._pending_accept = (self._calibration, plane)
        self.done(QDialog.Accepted)

    def done(self, result):
        if result == QDialog.Accepted and self._pending_accept is None:
            self.accept()
            return
        if self._finish_code is not None:
            return
        self._finish_code = result
        self._solve_pending = False
        self._stop_capture()
        if self._solve_worker is not None:
            self._solve_worker.requestInterruption()
            self.status_label.setText("正在安全結束校正計算，請稍候…")
        self._update_controls()
        self._finish_when_stopped()

    def _finish_when_stopped(self):
        if (self._finish_code is None or self._capture_worker is not None
                or self._solve_worker is not None):
            return
        result = self._finish_code
        if result == QDialog.Accepted and self._pending_accept is not None:
            self.calibration_result, self.screen_plane_result = self._pending_accept
            self.calibrated.emit(self.calibration_result, self.screen_plane_result)
        super().done(result)

    def closeEvent(self, event):
        if self._capture_worker is not None or self._solve_worker is not None:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
