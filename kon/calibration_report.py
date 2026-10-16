"""Read-only calibration and independent validation reporting.

This dialog neither opens a sensor nor records samples. Affine residuals are
reference-only when personal calibration is supplied. The application owns
point rerecording, fresh validation capture, and file-mode eligibility.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
)

if TYPE_CHECKING:
    from .calibration_diagnostics import FitDiagnostics

_POINTS = tuple((x, y) for y in (.18, .5, .82) for x in (.18, .5, .82))
_LOCATIONS = ("左上", "上中", "右上", "左中", "中央", "右中", "左下", "下中", "右下")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _pair(value):
    try:
        if len(value) != 2:
            return None
        pair = tuple(_finite(part) for part in value)
    except TypeError:
        return None
    return pair if all(part is not None for part in pair) else None


def _sequence(diagnostics, name):
    value = getattr(diagnostics, name, ())
    try:
        return tuple(value) if value is not None else ()
    except TypeError:
        return ()


def _label(text, *, size=11, muted=False):
    widget = QLabel(str(text))
    widget.setTextFormat(Qt.PlainText)
    widget.setWordWrap(True)
    widget.setFont(QFont("PingFang TC", size))
    if muted:
        widget.setStyleSheet("color: #b0a6b9;")
    return widget


class CalibrationReportDialog(QDialog):
    retry_requested = Signal(int)
    validate_requested = Signal()

    def __init__(self, diagnostics: FitDiagnostics, screen_size=None, parent=None, *, personal=None):
        super().__init__(parent)
        self.diagnostics = diagnostics
        self.personal = personal
        self.validation = getattr(personal, "validation", None)
        self.validation_targets = _sequence(personal, "validation_targets") if personal is not None else ()
        self.validation_count = len(self.validation_targets)
        self.validation_rms_limit = _finite(getattr(self.validation, "max_rms_pixels", 30)) or 30
        self.validation_point_limit = _finite(getattr(self.validation, "max_point_pixels", 40)) or 40
        self._retry_emitted = False
        self._validation_emitted = False
        supplied = _pair(screen_size)
        if supplied is None or min(supplied) <= 0:
            geometry = self.screen().geometry()
            supplied = float(geometry.width()), float(geometry.height())
        self.screen_size = supplied
        self.setWindowTitle("KON · 校正擬合診斷" if personal is None else "KON · 個人瞄準與精度檢查")
        self.setModal(True)
        self.setMinimumSize(640, 540)
        self.resize(min(760, max(640, round(supplied[0] - 48))),
                    min(720 if personal is not None else 650, max(540, round(supplied[1] - 70))))
        self.setStyleSheet("""
            QDialog { background: #111018; color: #eee6f1; }
            QLabel { background: transparent; color: #eee6f1; }
            QFrame#fitMetric { background: #201c28; border: 1px solid #3d3448; border-radius: 10px; }
            QTableWidget { background: #17141f; alternate-background-color: #1d1926;
                color: #eee6f1; border: 1px solid #3c3446; border-radius: 8px;
                gridline-color: #302938; selection-background-color: #5a3b2b;
                selection-color: #fff1e3; font-family: 'PingFang TC'; font-size: 13px; }
            QHeaderView::section { background: #28212e; color: #cfc2d6;
                border: none; padding: 8px 5px; font-family: 'PingFang TC'; font-size: 12px; }
            QPushButton { background: #2b2331; color: #eee6f1; border: 1px solid #5b485e;
                border-radius: 8px; padding: 8px 16px; min-height: 20px;
                font-family: 'PingFang TC'; font-size: 13px; }
            QPushButton:hover { background: #443040; }
            QPushButton:focus { border: 2px solid #ffbd91; }
            QPushButton#retryPoint { background: #f0a678; color: #2b180f; border-color: #f0a678; }
            QPushButton#validateAim { background: #f0a678; color: #2b180f; border-color: #f0a678; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #25202d; color: #baaac4; border-radius: 6px;
                padding: 6px 12px; margin-right: 5px; }
            QTabBar::tab:selected { background: #513829; color: #ffe0c3; }
            QPushButton:disabled { background: #29232e; color: #817588; border-color: #3c3343; }
        """)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 20, 22, 20)
        outer.setSpacing(12)
        passed = bool(getattr(self.diagnostics, "passed", False))
        count = getattr(self.diagnostics, "sample_count", 0)
        if self.personal is None:
            summary = f"{'校正擬合已通過' if passed else '校正擬合尚未通過'} · {count} / 9 點已記錄"
        elif self.validation is None:
            summary = "9點個人對應已建立 · 精度尚未檢查"
        elif bool(getattr(self.validation, "passed", False)):
            summary = ("9點個人對應已建立 · 精度檢查通過"
                       if bool(getattr(self.validation, "frame_samples_checked", False)) else
                       "9點個人對應已建立 · 尚待逐幀檢查")
        else:
            summary = "9點個人對應已建立 · 精度檢查偏差較大"
        self.summary_label = _label(summary, size=17)
        self.summary_label.setFont(QFont("PingFang TC", 17, QFont.DemiBold))
        outer.addWidget(self.summary_label)
        self.rms_limit = _finite(getattr(self.diagnostics, "max_rms_error", .08)) or .08
        self.point_limit = _finite(getattr(self.diagnostics, "max_point_error", .16)) or .16
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        if self.personal is None:
            self.rms_label = self._metric(metrics, "整體擬合殘差 RMS", getattr(self.diagnostics, "rms_error", None), self.rms_limit)
            self.max_error_label = self._metric(metrics, "最大單點擬合殘差", getattr(self.diagnostics, "max_error", None), self.point_limit)
        else:
            self.rms_label = self._metric(metrics, "全新樣本實測 RMS", getattr(self.validation, "rms_pixels", None), self.validation_rms_limit, pixels=True)
            self.max_error_label = self._metric(metrics, "全新樣本最大誤差", getattr(self.validation, "max_pixels", None), self.validation_point_limit, pixels=True)
        outer.addLayout(metrics)
        reason = getattr(self.diagnostics, "reason", "") or (
            "已記錄樣本符合門檻，請先在演練模式檢查實際瞄準。" if passed else
            "選取偏差較大的點重新記錄；其餘點保留，不需全部重做。")
        if self.personal is not None:
            if self.validation is None:
                reason = f"這9點已採用並保存，可以演練或操作已校正範圍內的桌面圖示。{self.validation_count} 點精度檢查是選用項目，重開會沿用你的個人對應。"
            else:
                reason = str(getattr(self.validation, "reason", "") or "實測結果已更新。")
                if bool(getattr(self.validation, "passed", False)) and not bool(getattr(self.validation, "frame_samples_checked", False)):
                    reason = "中位點偏差符合參考值；這次尚未檢查連續影格的抖動。"
                reason += " 精度檢查是選用項目，結果不會清除已保存的九點；可繼續操作已校正範圍內的桌面圖示，或重新記錄選取點改善瞄準。"
        self.reason_label = _label(reason)
        outer.addWidget(self.reason_label)

        self.table = QTableWidget(9, 4 if self.personal is None else 5, self)
        self.table.setObjectName("calibrationPointTable")
        headers = ("校正點", "記錄目標位置", "擬合殘差", "換算像素殘差") if self.personal is None else (
            "校正點", "個人對應目標", "採用狀態", "舊模型參考偏差", "參考像素偏差")
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setFocusPolicy(Qt.StrongFocus)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        if self.personal is not None:
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._populate_rows()
        self.tabs = None
        self.validation_table = None
        if self.personal is None:
            outer.addWidget(self.table, 1)
        else:
            self.tabs = QTabWidget(self)
            self.tabs.addTab(self.table, "已採用的9點")
            self.validation_table = self._build_validation_table()
            self.tabs.addTab(self.validation_table, f"精度檢查（{self.validation_count}點）")
            outer.addWidget(self.tabs, 1)
        self.selection_label = _label("", muted=True)
        self.selection_label.setMinimumHeight(24)
        outer.addWidget(self.selection_label)
        scope = "以上是已記錄校正樣本的擬合殘差，並非之後每次瞄準的精準度。像素值依目前螢幕座標換算。" if self.personal is None else (
            f"9點採用表示已建立個人對應，精準度仍以實際瞄準為準。舊模型偏差僅供參考，不是採用門檻；選用精度檢查的參考值為 RMS ≤ {self.validation_rms_limit:.0f} px、單點與逐幀 P95 ≤ {self.validation_point_limit:.0f} px。")
        self.scope_label = _label(scope, size=10, muted=True)
        outer.addWidget(self.scope_label)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.close_button = QPushButton("關閉", self)
        self.close_button.clicked.connect(self.reject)
        controls.addWidget(self.close_button)
        self.retry_button = QPushButton("重測選取點" if self.personal is None else "重新記錄選取點", self)
        self.retry_button.setObjectName("retryPoint")
        self.retry_button.setDefault(self.personal is None)
        self.retry_button.setAutoRepeat(False)
        self.retry_button.clicked.connect(self._retry_selected)
        controls.addWidget(self.retry_button)
        self.validate_button = None
        if self.personal is not None:
            self.retry_button.setObjectName("rerecordPoint")
            self.validate_button = QPushButton(f"檢查瞄準精度（{self.validation_count}點，選用）", self)
            self.validate_button.setObjectName("validateAim")
            self.validate_button.setDefault(True)
            self.validate_button.setAutoRepeat(False)
            self.validate_button.setEnabled(self.validation_count > 0)
            self.validate_button.clicked.connect(self._request_validation)
            controls.addWidget(self.validate_button)
        outer.addLayout(controls)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        suggested = getattr(self.diagnostics, "suggested_point_index", None) if self.personal is None else None
        self.suggested_point_index = suggested if isinstance(suggested, int) and not isinstance(suggested, bool) and 0 <= suggested < 9 else None
        self.table.setCurrentCell(self.suggested_point_index if self.suggested_point_index is not None else 0, 0)
        if self.tabs is not None:
            self.tabs.currentChanged.connect(self._selection_changed)
            if self.validation is not None:
                self.tabs.setCurrentIndex(1)
        self._selection_changed()

    def _metric(self, layout, title, value, limit, *, pixels=False):
        box = QFrame(self)
        box.setObjectName("fitMetric")
        contents = QVBoxLayout(box)
        contents.setContentsMargins(12, 10, 12, 10)
        contents.setSpacing(3)
        contents.addWidget(_label(title, size=10, muted=True))
        number = _finite(value)
        if pixels:
            text = "尚未檢查" if number is None else f"{number:.1f} px"
            metric_text = f"{text}   ／ 參考 {limit:.0f} px"
        else:
            text = "—" if number is None else f"{number * 100:.1f}%"
            metric_text = f"{text}   ／ 門檻 {limit * 100:.1f}%"
        result = _label(metric_text, size=16)
        if number is not None:
            result.setStyleSheet(f"color: {'#ffaf82' if number > limit else '#c1dca1'};")
        contents.addWidget(result)
        layout.addWidget(box, 1)
        return result

    def _populate_rows(self):
        targets = (_sequence(self.personal, "screen_points") or _sequence(self.personal, "target_points")) if self.personal is not None else ()
        targets = targets or _sequence(self.diagnostics, "target_points")
        predictions = _sequence(self.diagnostics, "predicted_points")
        errors = _sequence(self.diagnostics, "point_errors")
        suggested = getattr(self.diagnostics, "suggested_point_index", None) if self.personal is None else None
        self._pixel_vectors = []
        for index in range(9):
            target = _pair(targets[index]) if index < len(targets) else _POINTS[index]
            target = target or _POINTS[index]
            prediction = _pair(predictions[index]) if index < len(predictions) else None
            error = _finite(errors[index]) if index < len(errors) else None
            if error is None and prediction is not None:
                error = math.dist(prediction, target)
            pixels = None if prediction is None else (
                (prediction[0] - target[0]) * self.screen_size[0],
                (prediction[1] - target[1]) * self.screen_size[1])
            self._pixel_vectors.append(pixels)
            row = (f"{index + 1}{' · 建議' if index == suggested else ''}",
                   f"{_LOCATIONS[index]} ({target[0] * 100:.0f}%, {target[1] * 100:.0f}%)",
                   "—" if error is None else f"{error * 100:.1f}%",
                   "—" if pixels is None else f"{math.hypot(*pixels):.1f} px")
            if self.personal is not None:
                row = row[:2] + ("已採用",) + row[2:]
            for column, text in enumerate(row):
                item = QTableWidgetItem(text)
                if column >= (2 if self.personal is None else 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if index == suggested:
                    item.setForeground(QColor("#ffc49c"))
                self.table.setItem(index, column, item)

    def _build_validation_table(self):
        table = QTableWidget(self.validation_count, 5, self)
        table.setHorizontalHeaderLabels(("實測點", "全新樣本目標", "中位點偏差", "逐幀 P95", "新影格數"))
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(30)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for column in (2, 3, 4):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        targets = _sequence(self.validation, "target_points") or self.validation_targets
        errors = _sequence(self.validation, "point_pixel_errors")
        p95_errors = _sequence(self.validation, "point_p95_pixels")
        frame_counts = _sequence(self.validation, "point_frame_counts")
        for index in range(self.validation_count):
            target = _pair(targets[index]) if index < len(targets) else None
            error = _finite(errors[index]) if index < len(errors) else None
            p95 = _finite(p95_errors[index]) if index < len(p95_errors) else None
            count = _finite(frame_counts[index]) if index < len(frame_counts) else None
            values = (str(index + 1), "—" if target is None else f"({target[0] * 100:.0f}%, {target[1] * 100:.0f}%)",
                      "尚未檢查" if error is None else f"{error:.1f} px",
                      "尚未檢查" if p95 is None else f"{p95:.1f} px",
                      "—" if count is None else f"{count:.0f}")
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                value = error if column == 2 else p95 if column == 3 else None
                if value is not None:
                    item.setForeground(QColor("#ffaf82" if value > self.validation_point_limit else "#c1dca1"))
                if column >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 3:
                    item.setToolTip(f"95% 的新影格，瞄準偏差在此像素值以內；精度檢查參考值為 {self.validation_point_limit:.0f} px。")
                table.setItem(index, column, item)
        return table

    def _selection_changed(self):
        index = self.table.currentRow()
        validation_tab = self.tabs is not None and self.tabs.currentIndex() == 1
        self.retry_button.setEnabled(0 <= index < 9 and not self._retry_emitted and not self._validation_emitted and not validation_tab)
        if validation_tab:
            self.selection_label.setText("這些是全新實測點，與已採用的9點不同。可選用「檢查瞄準精度」重新檢查。")
            return
        if not 0 <= index < 9:
            self.selection_label.setText("請先選取要重測的校正點。")
            return
        vector = self._pixel_vectors[index]
        if vector is None:
            detail = "尚無可計算的擬合殘差"
        else:
            detail = f"擬合位置差：x {vector[0]:+.1f} px、y {vector[1]:+.1f} px"
        if self.personal is None:
            self.selection_label.setText(f"第 {index + 1} 點 · {detail}。按「重測選取點」只重記這一點。")
        else:
            self.selection_label.setText(f"第 {index + 1} 點已採用。舊模型偏差不代表此點記錄錯誤；重新記錄會更新個人對應，也可再檢查精度。")

    def _retry_selected(self):
        index = self.table.currentRow()
        if not 0 <= index < 9 or self._retry_emitted or self._validation_emitted or self.tabs is not None and self.tabs.currentIndex() == 1:
            return
        self._retry_emitted = True
        self.retry_button.setEnabled(False)
        self.retry_requested.emit(index)
        self.accept()

    def _request_validation(self):
        if self.personal is None or self.validation_count < 1 or self._validation_emitted or self._retry_emitted:
            return
        self._validation_emitted = True
        self.validate_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.validate_requested.emit()
        self.accept()
