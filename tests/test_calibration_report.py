"""Report rendering/retry interactions; no sensing, recording, or file actions."""
import math
import os
from types import SimpleNamespace
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from kon.calibration_diagnostics import FitDiagnostics, FitStatus, analyze_calibration
from kon.calibration_report import CalibrationReportDialog
from kon.personal_calibration import PersonalCalibration

TARGETS = tuple((x, y) for y in (.18, .5, .82) for x in (.18, .5, .82))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def observed_failure():
    residuals = [(0.06, 0.08), (.05, .05), (-.054, .072), (.03, -.06),
                 (-.064, .048), (.08, -.025), (-.065, math.sqrt(.00299811)),
                 (.10, .0733), (.03, -.06)]
    predictions = tuple((target[0] + residual[0], target[1] + residual[1])
                        for target, residual in zip(TARGETS, residuals))
    return FitDiagnostics(
        FitStatus.HIGH_RESIDUAL,
        "整體擬合殘差超過門檻；先重測建議點，其餘校正資料保留。",
        sample_count=9, rms_error=math.sqrt(sum(math.hypot(*p) ** 2 for p in residuals) / 9),
        max_error=max(math.hypot(*p) for p in residuals), condition_number=2.1,
        target_points=TARGETS, predicted_points=predictions,
        residual_vectors=tuple(residuals), point_errors=tuple(math.hypot(*p) for p in residuals),
        suggested_point_index=7,
    )


@pytest.fixture
def report(app):
    dialog = CalibrationReportDialog(observed_failure(), screen_size=(1920, 1080))
    yield dialog
    dialog.close()


def test_observed_metrics_limits_and_suggested_selection(report):
    assert "8.7%" in report.rms_label.text() and "8.0%" in report.rms_label.text()
    assert "12.4%" in report.max_error_label.text() and "16.0%" in report.max_error_label.text()
    assert report.table.rowCount() == 9
    assert report.table.currentRow() == 7
    assert "建議" in report.table.item(7, 0).text()
    assert report.reason_label.textFormat() == Qt.PlainText
    assert "已記錄校正樣本" in report.scope_label.text()


def test_pixel_residual_scales_both_screen_axes_independently(report):
    # sqrt((.06 * 1920)^2 + (.08 * 1080)^2) = 144 px, not 192 px.
    assert report.table.item(0, 2).text() == "10.0%"
    assert report.table.item(0, 3).text() == "144.0 px"
    report.table.setCurrentCell(0, 0)
    assert "x +115.2 px" in report.selection_label.text()
    assert "y +86.4 px" in report.selection_label.text()


def test_click_retry_emits_selected_zero_based_index_and_closes_once(report, app):
    retries = []
    report.retry_requested.connect(retries.append)
    report.show()
    app.processEvents()
    QTest.mouseClick(report.retry_button, Qt.LeftButton)
    report.retry_button.click()
    report._retry_selected()
    assert retries == [7]
    assert report.result() == QDialog.Accepted
    assert not report.isVisible()


def test_keyboard_row_change_then_enter_retries_that_point(report, app):
    retries = []
    report.retry_requested.connect(retries.append)
    report.show()
    report.table.setFocus()
    app.processEvents()
    QTest.keyClick(report.table, Qt.Key_Up)
    assert report.table.currentRow() == 6
    QTest.keyClick(report, Qt.Key_Return)
    assert retries == [6]
    assert not report.isVisible()


def test_escape_only_closes_without_retry(report, app):
    retries = []
    report.retry_requested.connect(retries.append)
    report.show()
    app.processEvents()
    QTest.keyClick(report, Qt.Key_Escape)
    assert retries == []
    assert report.result() == QDialog.Rejected


def test_degenerate_fit_keeps_nine_selectable_rows_with_unknown_errors(app):
    diagnostics = FitDiagnostics(FitStatus.COLLINEAR_RAYS,
        "瞄準線資料接近一直線，無法計算校正；請選取一點重新記錄。",
        sample_count=9, target_points=TARGETS)
    dialog = CalibrationReportDialog(diagnostics, screen_size=(1280, 800))
    retries = []
    dialog.retry_requested.connect(retries.append)
    assert "—" in dialog.rms_label.text() and "—" in dialog.max_error_label.text()
    assert dialog.table.rowCount() == 9
    assert all(dialog.table.item(i, 2).text() == dialog.table.item(i, 3).text() == "—" for i in range(9))
    dialog.table.setCurrentCell(8, 0)
    dialog.retry_button.click()
    assert retries == [8]
    dialog.close()


def test_passed_fit_uses_recorded_sample_scope_without_runtime_guarantee(app):
    targets = np.asarray(TARGETS)
    diagnostics = analyze_calibration(targets * [.4, .25] - [.2, .1], targets)
    assert diagnostics.passed
    dialog = CalibrationReportDialog(diagnostics, screen_size=(1280, 800))
    assert "已通過" in dialog.summary_label.text()
    assert "並非之後每次瞄準" in dialog.scope_label.text()
    assert "0.0%" in dialog.rms_label.text()
    dialog.close()


def test_selection_without_row_cannot_dispatch_retry(report):
    retries = []
    report.retry_requested.connect(retries.append)
    report.table.clearSelection()
    report.table.setCurrentCell(-1, -1)
    report._selection_changed()
    report._retry_selected()
    assert not report.retry_button.isEnabled() and not retries


def personal_fixture(validation=None, *, ready=False, targets=9):
    validation_targets = ((.05, .05), (.95, .05), (.05, .95), (.95, .95),
                          (.34, .34), (.66, .34), (.34, .66), (.66, .66), (.5, .82))[:targets]
    return SimpleNamespace(sample_count=9, target_points=TARGETS,
        validation_targets=validation_targets, validation=validation,
        validated=bool(validation and validation.passed), ready_for_live=ready,
        full_screen_coverage=True, coverage_reason="")


def validation_fixture(*, passed=True, frames_checked=True):
    errors = (12, 15, 17, 20, 14, 18, 23, 26, 27.3 if passed else 47.8)
    return SimpleNamespace(passed=passed, reason="全新樣本的實測結果已計算。",
        sample_count=9, rms_pixels=math.sqrt(sum(value ** 2 for value in errors) / len(errors)),
        max_pixels=27.3 if passed else 47.8,
        target_points=personal_fixture().validation_targets,
        point_pixel_errors=errors,
        frame_samples_checked=frames_checked,
        point_p95_pixels=(15, 18, 22, 24, 18, 21, 27, 29, 32.8 if passed else 54.4) if frames_checked else (),
        point_frame_counts=(12,) * 9 if frames_checked else ())


def test_personal_candidate_adopts_nine_points_without_affine_rejection(app):
    dialog = CalibrationReportDialog(observed_failure(), screen_size=(1920, 1080), personal=personal_fixture())
    assert "9點個人對應已建立" in dialog.summary_label.text()
    assert "精度尚未檢查" in dialog.summary_label.text()
    assert "尚未檢查" in dialog.rms_label.text() and "30 px" in dialog.rms_label.text()
    assert "尚未檢查" in dialog.max_error_label.text() and "40 px" in dialog.max_error_label.text()
    assert "可以演練" in dialog.reason_label.text() and "選用項目" in dialog.reason_label.text()
    assert "桌面圖示" in dialog.reason_label.text() and "還需通過" not in dialog.reason_label.text()
    assert dialog.table.columnCount() == 5
    assert all(dialog.table.item(i, 2).text() == "已採用" for i in range(9))
    assert all("建議" not in dialog.table.item(i, 0).text() for i in range(9))
    assert dialog.suggested_point_index is None and dialog.table.currentRow() == 0
    assert "參考" in dialog.table.horizontalHeaderItem(3).text()
    assert "不是採用門檻" in dialog.scope_label.text()
    dialog.close()


def test_personal_validation_count_is_taken_from_current_targets(app):
    for total in (7, 9):
        dialog = CalibrationReportDialog(observed_failure(), personal=personal_fixture(targets=total))
        assert f"（{total}點，選用）" in dialog.validate_button.text()
        assert dialog.validation_table.rowCount() == total
        dialog.close()


def test_personal_validate_action_emits_once_and_does_not_rerecord(app):
    dialog = CalibrationReportDialog(observed_failure(), personal=personal_fixture())
    actions = []
    dialog.validate_requested.connect(lambda: actions.append("validate"))
    dialog.retry_requested.connect(lambda index: actions.append(("retry", index)))
    dialog.show()
    app.processEvents()
    QTest.keyClick(dialog, Qt.Key_Return)
    dialog._request_validation()
    dialog._retry_selected()
    assert actions == ["validate"]
    assert dialog.result() == QDialog.Accepted and not dialog.isVisible()


@pytest.mark.parametrize("passed", [True, False])
def test_personal_metrics_use_actual_fresh_validation_in_pixels(app, passed):
    validation = validation_fixture(passed=passed)
    dialog = CalibrationReportDialog(observed_failure(), personal=personal_fixture(validation, ready=passed))
    assert "精度檢查通過" in dialog.summary_label.text() if passed else "精度檢查偏差較大" in dialog.summary_label.text()
    assert f"{validation.rms_pixels:.1f} px" in dialog.rms_label.text()
    assert f"{validation.max_pixels:.1f} px" in dialog.max_error_label.text()
    assert "%" not in dialog.rms_label.text() + dialog.max_error_label.text()
    assert "選用項目" in dialog.reason_label.text()
    assert "可繼續操作已校正範圍內的桌面圖示" in dialog.reason_label.text()
    assert "尚未開放" not in dialog.reason_label.text()
    assert "不會清除已保存的九點" in dialog.reason_label.text()
    assert dialog.tabs.currentIndex() == 1
    assert not dialog.retry_button.isEnabled()
    assert dialog.validation_table.item(8, 2).text() == f"{validation.point_pixel_errors[8]:.1f} px"
    assert dialog.validation_table.item(8, 3).text() == f"{validation.point_p95_pixels[8]:.1f} px"
    assert dialog.validation_table.item(8, 4).text() == "12"
    dialog.tabs.setCurrentIndex(0)
    dialog.table.setCurrentCell(5, 0)
    actions = []
    dialog.retry_requested.connect(actions.append)
    dialog.retry_button.click()
    assert actions == [5]
    dialog.close()


def test_median_only_validation_does_not_claim_completed_validation(app):
    validation = validation_fixture(frames_checked=False)
    dialog = CalibrationReportDialog(observed_failure(), personal=personal_fixture(validation))
    assert "尚待逐幀檢查" in dialog.summary_label.text()
    assert "精度檢查通過" not in dialog.summary_label.text()
    assert "桌面圖示" in dialog.reason_label.text() and "尚未開放" not in dialog.reason_label.text()
    assert "選用項目" in dialog.reason_label.text() and "實際使用仍需" not in dialog.reason_label.text()
    assert dialog.validation_table.item(0, 3).text() == "尚未檢查"
    dialog.close()


def test_real_personal_model_reports_adopted_targets_and_independent_validation(app):
    targets = np.asarray(TARGETS)
    raw = targets * [.4, .3] - [.2, .15]
    personal = PersonalCalibration.fit(raw, targets, screen_size=(1920, 1080))
    fresh_targets = np.asarray(personal.validation_targets)
    fresh_raw = fresh_targets * [.4, .3] - [.2, .15]
    personal.validate(fresh_raw, fresh_targets, screen_size=(1920, 1080),
                      frame_samples=[[point] * 6 for point in fresh_raw])
    assert personal.ready_for_live
    dialog = CalibrationReportDialog(observed_failure(), screen_size=(1920, 1080), personal=personal)
    assert "精度檢查通過" in dialog.summary_label.text()
    assert "0.0 px" in dialog.rms_label.text()
    assert "桌面圖示" in dialog.reason_label.text()
    assert dialog.validation_table.rowCount() == len(personal.validation_targets) == 9
    assert all(dialog.validation_table.item(i, 4).text() == "6" for i in range(9))
    assert dialog.table.item(8, 1).text() == "右下 (82%, 82%)"
    dialog.close()


def test_personal_report_uses_actual_stricter_validation_limits(app):
    validation = validation_fixture()
    validation.max_rms_pixels, validation.max_point_pixels = 20, 25
    dialog = CalibrationReportDialog(observed_failure(), personal=personal_fixture(validation))
    assert "參考 20 px" in dialog.rms_label.text()
    assert "參考 25 px" in dialog.max_error_label.text()
    assert "≤ 25 px" in dialog.scope_label.text()
    assert "25 px" in dialog.validation_table.item(0, 3).toolTip()
    dialog.close()


@pytest.mark.parametrize("validation", [None, validation_fixture(passed=False)])
def test_personal_report_nine_rows_fit_without_scroll_at_default_desktop_size(app, validation):
    dialog = CalibrationReportDialog(observed_failure(), screen_size=(1920, 1080),
                                     personal=personal_fixture(validation))
    dialog.show()
    app.processEvents()
    table = dialog.table if validation is None else dialog.validation_table
    assert table.verticalScrollBar().maximum() == 0
    assert table.horizontalScrollBar().maximum() == 0
    dialog.close()
